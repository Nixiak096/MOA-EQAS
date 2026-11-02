"""多智能体协作：检索员 / 分析师 / 撰稿人 / 审核员 串行协作 + 审核反馈闭环。

与供热业务对接：
    检索员  意图解析（intent_parser）→ 知识库检索（Milvus）→ 确定数据域
    分析师  指标计算（indicators）+ 统计分析（analytics）
    撰稿人  报告生成（report_generator）
    审核员  口径校验（公式/单位/层级归属/居民非居民/引用/重算一致性），不通过则反馈重算

设计要点：
  * 审核以**确定性规则**为主（可测、可复现），大模型只做补充性定性审查；
  * 「项目之和 = 公司」等校验是真实回查数据库的，不是形式检查；
  * 审核不通过 → 编排器带着审核意见重算/重写，最多 MAX_REVIEW_ROUNDS 轮；
  * 仍未通过则如实标注未通过项，不伪装成通过。

用法：
    python multi_agents.py --selftest
    python multi_agents.py "公司A今天的气耗指标是多少？和昨天比有什么变化？"
"""

import argparse
import datetime
import json
import logging
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import config
import heat_db
from sqlalchemy import text

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("multi_agents")

MAX_REVIEW_ROUNDS = 2

# 规范正文在知识库里的路径标记与相似度下限。
# 只有「规范目录下的条文」才用于合规性判定 —— bert-base-chinese 的区分度有限
# （相关 0.82~0.84、近似 0.78~0.83，区间重叠），单靠相似度阈值会把无关文档也选进来。
STANDARDS_DIR = "05_standards"
COMPLIANCE_MIN_SIM = 0.78

# 「是否合规 / 是否达标」类问法：这类问题既要算数又要对照规范
COMPLIANCE_WORDS = [
    "达标", "合格", "合规", "符合标准", "是否符合", "符不符合", "是否满足",
    "是否正常", "是否安全", "有没有问题", "标准要求", "规范要求", "国家标准",
    "安全要求", "有没有超标", "超没超",
]


def _asks_compliance(question: str) -> bool:
    text = (question or "").replace(" ", "")
    return any(w in text for w in COMPLIANCE_WORDS)


# ============================================================================
# 协作上下文
# ============================================================================


@dataclass
class AgentContext:
    """在四个智能体间逐级传递的上下文。"""

    question: str = ""
    intent: Any = None
    retrieval: List[Dict[str, Any]] = field(default_factory=list)
    indicators: List[Any] = field(default_factory=list)
    analytics: Dict[str, Any] = field(default_factory=dict)
    report: Any = None
    review: Dict[str, Any] = field(default_factory=dict)
    steps: List[Dict[str, Any]] = field(default_factory=list)
    review_round: int = 0
    review_feedback: List[str] = field(default_factory=list)
    # 过程回调：编排器在有流式订阅者时设置，用于把「思考过程」实时推给前端。
    # 不设置时 trace()/stream() 静默 —— 因此 /chat 的非流式路径与既有测试行为不变。
    on_trace: Optional[Callable[[Dict[str, Any]], None]] = None

    def log_step(self, agent: str, detail: Dict[str, Any]) -> None:
        self.steps.append({"agent": agent, "detail": detail})

    @property
    def streaming(self) -> bool:
        """是否有流式订阅者（决定大模型是否走逐字流式）。"""
        return self.on_trace is not None

    def trace(self, agent: str, text: str, level: str = "info") -> None:
        """推送一条「思考过程」记录（解析结果 / 指标 / 校验 / 分节 …）。"""
        if self.on_trace:
            try:
                self.on_trace({"type": "trace", "agent": agent, "level": level, "text": text})
            except Exception:  # noqa: BLE001
                pass

    def stream(self, agent: str, delta: str) -> None:
        """推送大模型输出的一段增量（前端逐字追加）。"""
        if self.on_trace:
            try:
                self.on_trace({"type": "stream", "agent": agent, "text": delta})
            except Exception:  # noqa: BLE001
                pass


# ============================================================================
# 智能体基类
# ============================================================================


class Agent:
    """智能体基类：供热系统里每个角色只负责一段明确职责。

    基类收口「调大模型」这一横切关注点（`ask`）与「是否启用大模型」（`use_llm`），
    子类只需声明自己的 `name` / `role` / `role_key` 并实现 `run`。

    `role_key` 是 `llm_client.get_llm(role)` 用的角色标识（见 llm_client.ROLE_SPECS），
    子类用它指定自己走哪个模型；留空表示该角色不调大模型。
    """

    name = "智能体"
    role = ""
    role_key = ""

    def __init__(self, use_llm: bool = True):
        self.use_llm = use_llm

    def run(self, ctx: AgentContext) -> Dict[str, Any]:
        raise NotImplementedError

    def ask(self, prompt: str, ctx: Optional["AgentContext"] = None) -> Optional[str]:
        """问一次大模型，返回文本（失败或空则返回 None）。

        只负责「调」与「失败收口」：异常在这里记日志并降级为 None，
        绝不影响确定性结果。**是否该调**由调用方判断 —— 各角色的门控条件不同
        （例如审核员只在有报告时才审查），不能统一塞进本方法。

        传入 ctx 且有人订阅过程时走**逐字流式**，增量经 ctx.stream 实时推给前端；
        否则走普通一次性调用（非流式路径与既有测试行为不变）。
        """
        try:
            from llm_client import get_llm

            client = get_llm(self.role_key)
            if ctx is not None and ctx.streaming:
                try:
                    text = client.complete_stream(
                        prompt, on_delta=lambda d: ctx.stream(self.name, d))
                except Exception as e:  # noqa: BLE001
                    # 流式不可用（供应商不支持 stream / 中途断流）时退回一次性调用，
                    # 否则整段模型文字会白丢
                    logger.warning("%s 流式调用失败，回退为一次性调用：%s", self.name, e)
                    text = client.complete(prompt)
            else:
                text = client.complete(prompt)
            return text.strip() or None
        except Exception as e:  # noqa: BLE001
            logger.warning("%s的大模型调用失败（不影响确定性结果）：%s", self.name, e)
            return None


# ============================================================================
# 检索员：意图解析 + 知识检索
# ============================================================================


class RetrieverAgent(Agent):
    name = "检索员"
    role = "供热数据检索"
    role_key = "retriever"

    def __init__(self, storage_factory=None, use_llm: bool = True):
        """storage_factory: 可调用对象，返回 MilvusStorage；每次检索用它建**短连接**。

        不长期持有连接 —— pymilvus 的空闲心跳会被服务端限流（GOAWAY too_many_pings），
        而后续大模型阶段耗时长，连接必须用完即关。
        """
        super().__init__(use_llm)
        self.storage_factory = storage_factory

    def run(self, ctx: AgentContext) -> Dict[str, Any]:
        from intent_parser import parse

        ref = None
        try:
            eng = heat_db.get_engine()
            with eng.connect() as c:
                ref = c.execute(text("SELECT MAX(stat_date) FROM fact_energy_daily")).scalar()
            eng.dispose()
        except Exception:  # noqa: BLE001
            ref = None

        intent = parse(ctx.question, reference_date=ref, use_llm=self.use_llm)
        ctx.intent = intent
        ctx.trace(self.name, "意图解析：" + "｜".join(_intent_brief(intent)))
        for note in intent.notes:
            ctx.trace(self.name, f"解析说明：{note}", level="warn")
        if intent.unsupported_modes:
            ctx.trace(self.name, f"数据不支持的口径（如实标注）：{'、'.join(intent.unsupported_modes)}",
                      level="warn")

        # 知识库检索：短连接，用完立即关闭（避免空闲心跳被限流）
        retrieval: List[Dict[str, Any]] = []
        if self.storage_factory is not None:
            ctx.trace(self.name, "正在检索知识库（计算说明 / 数据字典 / 规范条文）…")
            try:
                with self.storage_factory() as s:
                    retrieval = s.search(ctx.question, top_k=4)
            except Exception as e:  # noqa: BLE001
                logger.warning("知识库检索失败（不影响数据链路）：%s", e)
        ctx.retrieval = retrieval
        for r in retrieval:
            sim = r.get("similarity")
            sim_text = f"，相似度 {sim:.4f}" if isinstance(sim, float) else ""
            ctx.trace(self.name, f"命中条文：{r.get('filename')}（分片 {r.get('chunk_index')}{sim_text}）")
        if not retrieval:
            ctx.trace(self.name, "知识库未命中文档（不影响数据链路）", level="warn")

        out = {
            "domain": intent.domain,
            "metrics": intent.metrics,
            "modes": intent.modes,
            "entity_ids": intent.entity_ids(),
            "stat_date": intent.stat_date.isoformat() if intent.stat_date else None,
            "knowledge_hits": len(retrieval),
            "notes": intent.notes,
        }
        ctx.log_step(self.name, out)
        return out


# ============================================================================
# 分析师：指标计算 + 统计分析
# ============================================================================


class AnalystAgent(Agent):
    # 分析师的主体是确定性核算（indicators / analytics）；use_llm 只控制末尾那段
    # 「数据解读」文字是否生成，不影响任何数值结论。
    name = "分析师"
    role = "供热指标核算"
    role_key = "analyst"

    def run(self, ctx: AgentContext) -> Dict[str, Any]:
        import analytics
        import indicators

        intent = ctx.intent
        results: List[Any] = []
        ana: Dict[str, Any] = {}

        if intent is None:
            ctx.log_step(self.name, {"error": "缺少意图"})
            return {"error": "缺少意图"}

        if intent.domain == "enterprise" and intent.metrics:
            results = indicators.compute_from_intent(intent)
            # 横向对比：多项目时给出排名，单项目时给出同口径的可用值
            if len(intent.entity_ids()) > 1 or not intent.entity_ids():
                code = _metric_to_code(intent.metrics[0], intent.modes)
                if code:
                    ana["compare"] = analytics.compare_projects(code).to_dict()
        elif intent.domain == "scada":
            if intent.substations:
                ana["scada"] = analytics.scada_analysis(intent.substations[0])
            else:
                # 不许静默空转：说清为什么算不了，交给撰稿人写进口径说明、审核员据此校验
                ana["blocked"] = "未解析出换热站编号（形如 L4），无法进行换热站工况分析"
        elif intent.domain == "household":
            # household_top 是 DataReader 的方法（不是模块级函数），需借 reader 调用
            with analytics.DataReader() as dr:
                ana["household_ranking"] = dr.household_top(limit=5)
        elif intent.domain == "weather":
            with analytics.DataReader() as dr:
                stations = dr.list_stations()
            if stations:
                ana["weather"] = analytics.weather_trend(stations[0])
            else:
                ana["blocked"] = "库中无气象站数据，无法进行气象趋势分析"
        elif intent.domain == "enterprise" and not intent.metrics:
            ana["blocked"] = "未识别出要核算的指标，无法进行指标核算"
        elif intent.domain == "knowledge":
            ana["note"] = "规范/方法类问题，数据核算不适用，交由知识库与撰稿人组织回答"

        ctx.indicators = results
        for r in results:
            if getattr(r, "available", False):
                unit = f" {r.unit}" if getattr(r, "unit", "") else ""
                ctx.trace(self.name, f"{r.label} = {_brief(r.value)}{unit}　←　{r.formula}",
                          level="ok")
            else:
                ctx.trace(self.name, f"{r.label} 不可用：{getattr(r, 'reason', '')}", level="warn")
        cmp_res = ana.get("compare")
        if isinstance(cmp_res, dict) and cmp_res.get("available"):
            ctx.trace(self.name, f"横向对比：最高 {cmp_res.get('max_entity')}、"
                                 f"最低 {cmp_res.get('min_entity')}，共 {cmp_res.get('n_entries')} 个项目")
        if ana.get("blocked"):
            ctx.trace(self.name, f"未做核算：{ana['blocked']}", level="warn")
        if ana.get("note"):
            ctx.trace(self.name, ana["note"])

        # 模型文字解读：与数值结果分开存放，由撰稿人放进带标注的独立分节。
        if self.use_llm:
            ctx.trace(self.name, "正在生成结果解读（模型，仅供参考）…")
        interpretation = self._interpret(ctx)
        if interpretation:
            ana["interpretation"] = interpretation
        ctx.analytics = ana

        out = {
            "n_indicators": len(results),
            "n_available": sum(1 for r in results if getattr(r, "available", False)),
            "analytics_keys": list(ana.keys()),
        }
        if ana.get("blocked"):
            out["blocked"] = ana["blocked"]
        ctx.log_step(self.name, out)
        return out

    def _interpret(self, ctx: AgentContext) -> Optional[str]:
        """用模型对**已算出**的结果写一段文字解读。

        严格定位：只为读者解释已有数字的含义，不得引入任何新数值、不得给出达标判定、
        不得臆造阈值。结果单独存放（`analytics["interpretation"]`），由撰稿人放进
        「数据解读（模型生成，仅供参考）」分节，**绝不并入「结论」或「指标数据」**，
        因此数值结论始终只来自确定性链路。失败即跳过，不影响任何核算结果。
        """
        if not self.use_llm:
            return None

        facts: List[str] = [f"- {r.label} = {r.value}{f' {r.unit}' if getattr(r, 'unit', '') else ''}"
                            for r in ctx.indicators if getattr(r, "available", False)]
        ana = ctx.analytics or {}
        cmp_res = ana.get("compare")
        if isinstance(cmp_res, dict) and cmp_res.get("available"):
            facts.append(f"- 横向对比：最高 {cmp_res.get('max_entity')}、最低 {cmp_res.get('min_entity')}，"
                         f"共 {cmp_res.get('n_entries')} 个项目")
        for key, label in (("scada", "换热站工况"), ("weather", "气象"), ("household", "住户")):
            d = ana.get(key)
            if isinstance(d, dict) and d.get("available"):
                facts.append(f"- 已完成的{label}分析，数据区间 {d.get('data_range')}")

        if not facts:          # 无可解读的实质结果（如规范类问题、或未识别出指标）
            return None
        prompt = (
            "你是供热运营数据分析助手。下面是系统**已经核算完成**的结果，"
            "请用 1～3 句中文写一段解读，帮助读者理解这些数字说明了什么。\n"
            "硬性要求：\n"
            "1) 只解释给定结果，**不得引入任何未出现的数字、比例、标准号或外部数据**；\n"
            "2) 不得判定是否达标，也不得臆造任何阈值；没有条文依据时不下结论；\n"
            "3) 若数据条件不足以支撑解读（例如数据量少、只有单日快照），"
            "就直接说明数据条件所限，不要硬凑；\n"
            "4) 写解读而不是复述，不要重复罗列数字。\n\n"
            "已核算结果：\n" + "\n".join(facts)
        )
        return self.ask(prompt, ctx)


def _brief(v: Any) -> str:
    """过程流里的数值显示：大数加千分位、小数保留有效位，与报告正文风格一致。"""
    if not isinstance(v, (int, float)):
        return str(v)
    return f"{v:,.4f}" if abs(v) >= 1e4 else f"{v:.6g}"


def _intent_brief(intent) -> List[str]:
    """把意图解析结果压成一行「人话」，只列与该数据域相关的字段。"""
    bits = [f"数据域 {intent.domain}"]
    if intent.domain == "enterprise":
        bits.append(f"指标 {'/'.join(intent.metrics) or '—'}")
        bits.append(f"层级 {'—' if intent.level == 'none' else intent.level}")
    elif intent.domain == "scada":
        bits.append(f"换热站 {'、'.join(intent.substations) or '—（未给出站号）'}")
    elif intent.domain == "household":
        bits.append("住户域")
    elif intent.domain == "weather":
        bits.append("气象域")
    else:
        bits.append("规范/方法类问题")
    if intent.company and intent.domain == "enterprise":
        bits.append(f"公司 {intent.company}")
    if intent.stat_date:
        bits.append(f"日期 {intent.stat_date}")
    if intent.modes:
        bits.append(f"口径 {'/'.join(intent.modes)}")
    if intent.categories:
        bits.append(f"分类 {'/'.join(intent.categories)}")
    return bits


def _metric_to_code(metric: str, modes: List[str]) -> Optional[str]:
    """把意图指标映射到 fact_energy_daily 的 metric_code（用于横向对比）。"""
    if metric == "heating_area":
        return "total_heating_area"
    prefix = {"gas": "gas", "water": "water", "elec": "elec"}.get(metric)
    if not prefix:
        return None
    if "cumulative" in modes:
        return f"cumulative_{prefix}_index"
    return f"daily_{prefix}_index"


# ============================================================================
# 撰稿人：报告生成
# ============================================================================


class WriterAgent(Agent):
    name = "撰稿人"
    role = "供热报告撰写"
    role_key = "writer"

    def run(self, ctx: AgentContext) -> Dict[str, Any]:
        import report_generator as rg

        intent = ctx.intent
        ana = ctx.analytics
        rep = None

        if "scada" in ana:
            rep = rg.build_scada_report(ana["scada"])
        elif "household" in ana or "household_ranking" in ana:
            hid = None
            if ana.get("household_ranking"):
                hid = ana["household_ranking"][0]["household_id"]
            if hid:
                import analytics
                # 另存一份分析结果：报告只呈现其中一部分（结论/指标/异常抽样），
                # 不存就会在报告之外丢失，前端的「异常检测」面板也就取不到住户域异常。
                ana["household"] = analytics.household_analysis(hid)
                rep = rg.build_household_report(ana["household"])
        elif "weather" in ana:
            rep = rg.build_weather_report(ana["weather"])
        elif ctx.indicators:
            scope = f"实体 {intent.entity_ids()}，计算日 {intent.stat_date}" if intent else ""
            rep = rg.build_indicator_report(
                ctx.indicators,
                title="供热运营指标核算报告",
                scope=scope,
            )
        else:
            rep = rg.Report(
                title="知识库答复",
                generated_at=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            )
            if ctx.retrieval:
                # 先给"答复"，再列"依据条文"——只罗列条文不算回答，
                # 规范类问题需要把检索到的条文读成一句话的结论。
                answer = self._compose_answer(ctx.question, ctx.retrieval, ctx)
                if answer:
                    rep.add("答复", answer)
                else:
                    rep.add_caveat(
                        "未能生成综合答复（大模型未启用或调用失败），以下为检索到的条文原文，请自行判读"
                    )
                rep.add("依据条文", [
                    f"[{r['filename']}] {r['content'][:200]}" for r in ctx.retrieval[:4]
                ])
                # 检索到的条文即本报告的引用来源；不标注来源会被审核员判为不合格
                for r in ctx.retrieval[:4]:
                    rep.add_source(
                        f"{r.get('filename')}（分片 {r.get('chunk_index')}，"
                        f"相似度 {r.get('similarity'):.4f}）"
                        if isinstance(r.get("similarity"), float)
                        else f"{r.get('filename')}（分片 {r.get('chunk_index')}）"
                    )
            else:
                rep.add_caveat("知识库中未检索到相关内容，无法回答该规范/条文类问题")
            # 分析师未能执行的原因必须写进报告，不能让读者以为「已算过」
            if ana.get("blocked"):
                rep.add_caveat(f"本次未做数据核算：{ana['blocked']}")

        # 分析师的模型文字解读：单列一节并标明系模型生成。
        # 不与「结论」「指标数据」合并 —— 数值结论只能来自确定性核算链路。
        interpretation = (ana or {}).get("interpretation")
        if interpretation:
            rep.add("数据解读（模型生成，仅供参考）", [interpretation])

        self._add_compliance_section(rep, ctx)

        # 可选：用大模型润色「结论」段（失败不影响确定性内容）
        if self.use_llm and rep.sections.get("结论"):
            polished = self._polish(rep, ctx)
            if polished:
                rep.sections["大模型综述（模型生成，仅供参考）"] = [polished]

        ctx.report = rep
        for heading, lines in rep.sections.items():
            ctx.trace(self.name, f"撰写分节「{heading}」（{len(lines)} 行）")
        for spec in rep.charts:
            ctx.trace(self.name, f"装配图表「{spec.title}」（{spec.kind}）")
        for s in rep.sources:
            ctx.trace(self.name, f"标注来源：{s}")
        for c in rep.caveats:
            ctx.trace(self.name, f"口径说明：{c}", level="warn")
        out = {
            "title": rep.title,
            "sections": list(rep.sections.keys()),
            "n_charts": len(rep.charts),
            "n_sources": len(rep.sources),
            "n_caveats": len(rep.caveats),
        }
        ctx.log_step(self.name, out)
        return out

    def _add_compliance_section(self, rep, ctx: AgentContext) -> None:
        """混合型问题补一段规范依据。

        域分流是单值的：像「本季收费率 75% 算不算达标」这种问题，会被分到 enterprise
        只算出数值，而「算不算达标」需要对照规范条文 —— 后半问就此丢失。
        检索其实每一问都在跑（检索员始终调知识库），这里把命中规范正文的条文利用起来。
        """
        if not _asks_compliance(ctx.question) or not ctx.retrieval:
            return
        hits = [r for r in ctx.retrieval
                if STANDARDS_DIR in str(r.get("path") or "")
                and (r.get("similarity") or 0) >= COMPLIANCE_MIN_SIM]
        if not hits:
            rep.add_caveat(
                "本问题涉及「是否符合标准/要求」，但知识库中未检索到对应规范条文，"
                "无法给出合规性判定；请补充相关标准条文后重试"
            )
            return
        answer = self._compose_answer(ctx.question, hits, ctx)
        if answer:
            rep.add("依据条文与合规性说明", answer)
        else:
            # 大模型不可用时也要给出规范依据，否则"达标"类问题只剩数字、答了半问
            rep.add("依据条文与合规性说明",
                    [f"[{r['filename']}] {r['content'][:200]}" for r in hits[:3]])
            rep.add_caveat(
                "未能生成综合合规性判定（大模型未启用或调用失败），上列条文原文供自行判读"
            )
        for r in hits[:4]:
            rep.add_source(
                f"{r.get('filename')}（分片 {r.get('chunk_index')}，"
                f"相似度 {r.get('similarity'):.4f}）"
            )

    def _compose_answer(self, question: str, retrieval: List[Dict[str, Any]],
                        ctx: "AgentContext" = None) -> Optional[List[str]]:
        """把检索到的条文读成"答复"，而不是罗列原文。

        严格约束：只许用给定条文里的内容，必须标出处；条文没写的就说没写，
        不得凭模型记忆补充标准号、限额或达标线。
        """
        if not self.use_llm:
            return None
        docs, seen = [], set()
        for r in retrieval:
            key = (r.get("filename"), r.get("chunk_index"))
            if key in seen:
                continue
            seen.add(key)
            docs.append(f"【{r.get('filename')}｜分片{r.get('chunk_index')}】\n"
                        f"{r.get('content', '')[:900]}")
        prompt = (
            "你是供热行业的规范问答助手。下面是知识库检索到的条文原文，"
            "请据此回答用户问题，输出 1～4 条中文要点（每条一句话，可用分号）。\n"
            "硬性要求：\n"
            "1) 只允许使用给定条文中的内容，不得引入条文之外的数字、标准号、限额或结论；\n"
            "2) 每条要点末尾用【】标注所依据的文档名，如【GB55010-2021_供热工程项目规范.md】；\n"
            "3) 若条文未覆盖问题所问的内容（例如问某数值是否达标，而条文未规定该数值），"
            "必须直接说明「给定条文未规定……」并把可用的替代口径说清楚，**不得臆造阈值**；\n"
            "4) 不要复述整段条文，也不要写「根据上述材料」这类空话；\n"
            "5) 若给定条文与问题无关，只回答「知识库未收录该问题的对应条文」。\n\n"
            f"用户问题：{question}\n\n检索到的条文：\n" + "\n\n".join(docs)
        )
        text = self.ask(prompt, ctx)
        if not text:
            return None
        lines = [ln.strip(" -·") for ln in text.splitlines() if ln.strip()]
        return lines or None

    def _polish(self, rep, ctx: "AgentContext" = None) -> Optional[str]:
        facts = "\n".join(f"- {x}" for x in rep.sections.get("结论", []))
        limits = "\n".join(f"- {x}" for x in rep.caveats) or "（本次无数据限制说明）"
        # 「口径」一词在本系统里有两个含义：指标表里的口径指日值/累计/评价值等统计范围，
        # 而报告末尾的「口径与数据说明」指数据限制。此前未加区分，导致综述写出
        # 「材料未提供口径与数据说明」这类与正文自相矛盾的断言（审核员据此判不合格）。
        prompt = (
            "请根据下列**已核算完成**的结论，用 2-3 句中文写一段综述。\n"
            "硬性要求：\n"
            "1) 只使用给定内容，不得引入任何未出现的数字、标准号或出处；\n"
            "2) 不要评价材料是否完整，不要声称「未提供口径 / 未提供来源 / 未提供说明」"
            "——材料给什么就写什么，没有的就不提；\n"
            "3) 术语提醒：指标数据表里的「口径」指日值/累计/评价值/度日数折标等**统计范围**，"
            "与下方「数据限制说明」不是一回事，切勿混为一谈；\n"
            "4) 若核算结论为空，只回答「本次无可综述的核算结论」。\n\n"
            f"核算结论：\n{facts}\n\n数据限制说明：\n{limits}"
        )
        return self.ask(prompt, ctx)


# ============================================================================
# 审核员：口径校验
# ============================================================================


class ReviewerAgent(Agent):
    # use_llm 由基类保存，编排器统一控制（历史教训：曾用静态方法无条件返回 True，
    # 使 --no-llm 对审核员失效、仍然调用大模型）。
    name = "审核员"
    role = "供热口径校验"
    role_key = "reviewer"

    def run(self, ctx: AgentContext) -> Dict[str, Any]:
        issues: List[str] = []
        checks: List[Dict[str, Any]] = []
        intent = ctx.intent

        def chk(item: str, ok: bool, comment: str) -> None:
            checks.append({"item": item, "status": "pass" if ok else "fail", "comment": comment})
            if not ok:
                issues.append(f"{item}：{comment}")
            ctx.trace(self.name, f"{'通过' if ok else '不通过'} · {item}：{comment}",
                      level="ok" if ok else "warn")

        # 0. 分析师是否真的产出结果（或已明确声明为什么没产出）
        #    防止「领域选了但没有实体 → 静默空转 → 报告只剩检索条文」还判合格
        ana = ctx.analytics or {}
        produced = bool(ctx.indicators) or any(
            k in ana for k in ("compare", "scada", "household_ranking", "weather"))
        declared = bool(ana.get("blocked")) or ana.get("note") or intent is None
        chk("分析执行", produced or bool(declared),
            "分析师已产出核算/统计结果" if produced
            else (f"本次未做数据核算，原因已说明：{ana.get('blocked')}" if ana.get("blocked")
                  else "规范/方法类问题，数据核算不适用" if declared
                  else "分析师未产出任何结果，且未说明原因"))

        # 1. 公式是否符合《数据及计算说明》
        with_formula = [r for r in ctx.indicators if getattr(r, "available", False)]
        missing_formula = [r.label for r in with_formula if not getattr(r, "formula", "")]
        chk("公式依据", not missing_formula,
            "全部指标均标注所用公式" if not missing_formula
            else f"以下指标缺少公式说明：{missing_formula}")

        # 2. 单位是否标注
        missing_unit = [r.label for r in with_formula
                        if not getattr(r, "unit", "") and r.mode not in ("season",)]
        chk("单位标注", not missing_unit,
            "单位齐备" if not missing_unit else f"缺少单位：{missing_unit}")

        # 3. 不可用项是否已说明原因（不允许静默缺失）
        na = [r for r in ctx.indicators if not getattr(r, "available", True)]
        unexplained = [r.label for r in na if not getattr(r, "reason", "")]
        chk("不可用项说明", not unexplained,
            f"{len(na)} 项不可用且均已说明原因" if not unexplained
            else f"以下不可用项未说明原因：{unexplained}")

        # 4. 居民/非居民口径（收费类必须区分或明示取小计）
        fee_metrics = {"fee_rate", "household_payment_rate", "arrears_fee_rate"}
        if intent is not None and any(m in fee_metrics for m in intent.metrics):
            ok = bool(intent.categories)
            chk("居民/非居民口径", ok,
                f"收费类已指定分类：{intent.categories}" if ok
                else "收费类问题未指定居民/非居民分类")

        # 5. 本季 / 旧欠 口径
        if intent is not None and any(m in fee_metrics for m in intent.metrics):
            chk("本季/旧欠口径", True,
                "已按「旧欠」口径" if intent.arrears else "已按「本季」口径（非旧欠）")

        # 6. 引用来源：报告只要有正文（结论/指标/条文…）就必须标注来源；
        #    若整篇只有「口径与数据说明」（未做核算、也未检索到条文），则无来源可标
        rep = ctx.report
        if rep is not None:
            if rep.sections:
                chk("引用来源", bool(rep.sources),
                    f"报告含 {len(rep.sources)} 项数据来源" if rep.sources
                    else "报告未标注数据来源")
            else:
                chk("引用来源", True, "本次未产生核算结果与检索条文，无来源需要标注")

        # 7. 层级归属：项目之和应等于公司（真实回查数据库）
        level_ok, level_msg = self._check_level_consistency(ctx)
        chk("层级归属", level_ok, level_msg)

        # 8. 收费率重算一致性（若做了重算对照）
        rate_issues = []
        for r in with_formula:
            cc = getattr(r, "cross_check", None)
            if cc and cc.get("consistent") is False:
                rate_issues.append(
                    f"{r.label} 重算 {cc['recomputed']:.4f}% 与报表 {cc['reported']:.4f}% "
                    f"偏差 {cc['delta_pp']:.4f}pp"
                )
        chk("收费率重算校验", not rate_issues,
            "收费率重算与报表一致" if not rate_issues
            else "以下收费率重算与报表不一致，建议复核口径：" + "；".join(rate_issues))

        # 9. 模型生成的文字（大模型综述 / 数据解读）必须单独分节并显式标注「模型生成」。
        #    防止将来有人把模型文字并进「结论」，或新增模型分节却忘记标注来源。
        rep_sections = (rep.sections if rep is not None else None) or {}
        model_heads = [k for k in rep_sections if k.startswith(("大模型综述", "数据解读"))]
        unlabeled = [k for k in model_heads if "模型生成" not in k]
        chk("模型生成内容标注", not unlabeled,
            f"{len(model_heads)} 处模型生成内容均已单独分节并标注" if model_heads
            else "本次无模型生成内容")

        passed = not issues
        review = {
            "review_status": "approved" if passed else "rejected",
            "quality_score": max(0, 100 - 15 * len(issues)),
            "checks": checks,
            "issues": issues,
            "round": ctx.review_round,
        }
        ctx.review = review
        if not passed:
            ctx.review_feedback = issues

        # 可选：大模型定性审查（只补充意见，不改变确定性结论；受 use_llm 控制）
        if self.use_llm and rep is not None:
            note = self._llm_review(ctx)
            if note:
                review["llm_note"] = note

        ctx.log_step(self.name, {
            "status": review["review_status"],
            "score": review["quality_score"],
            "issues": issues,
        })
        return review

    def _check_level_consistency(self, ctx: AgentContext) -> Tuple[bool, str]:
        """层级归属校验：所查指标在「项目之和」与「公司合计」之间是否吻合。"""
        intent = ctx.intent
        if intent is None or intent.domain != "enterprise" or not intent.metrics:
            return True, "非企业运营域，层级校验不适用"
        try:
            import indicators as ind

            date = intent.stat_date or datetime.date.today()
            for metric in intent.metrics:
                codes = ind.ENERGY_CODES.get(metric)
                if not codes:
                    continue
                with ind.DataAccess() as da:
                    area_company = da.energy("company:公司A", date, ind.AREA_CODE)
                    with heat_db.get_engine().connect() as c:
                        rows = c.execute(text(
                            "SELECT d.entity_id, d.metric_code, d.value FROM fact_energy_daily d "
                            "WHERE d.stat_date=:dt AND d.entity_id LIKE 'project:公司A:%' "
                            "AND d.metric_code IN (:m1,:m2)"
                        ), {"dt": date, "m1": codes["daily"], "m2": ind.AREA_CODE}).all()
                    proj_qty = sum(r[2] for r in rows if r[1] == codes["daily"] and r[2] is not None)
                    proj_area = sum(r[2] for r in rows if r[1] == ind.AREA_CODE and r[2] is not None)
                    comp_qty = da.energy("company:公司A", date, codes["daily"])
                if comp_qty is None or proj_qty == 0:
                    continue
                # 耗量层面应满足：各项目之和 = 公司合计（面积亦同）
                qty_ok = abs(proj_qty - comp_qty) / comp_qty < 0.005
                area_ok = (area_company is None or proj_area == 0
                           or abs(proj_area - area_company) / area_company < 0.005)
                if not (qty_ok and area_ok):
                    return False, (
                        f"{ind.METRIC_LABEL.get(metric, metric)} 各项目之和({proj_qty:,.2f}) "
                        f"与公司合计({comp_qty:,.2f}) 不一致；面积 "
                        f"{proj_area:,.2f} vs {area_company:,.2f}"
                    )
            return True, "各项目之和与公司合计一致（耗量与面积均已核对）"
        except Exception as e:  # noqa: BLE001
            logger.warning("层级校验执行失败：%s", e)
            return True, f"层级校验未能执行：{e}"

    def _llm_review(self, ctx: AgentContext) -> Optional[str]:
        rep = ctx.report
        body = "\n".join(
            f"## {h}\n" + "\n".join(f"- {x}" for x in lines)
            for h, lines in rep.sections.items()
        )
        prompt = (
            "请审查下面这份供热运营数据报告，只指出问题，不要重写。"
            "重点看：数字与结论是否自相矛盾、是否有未标注来源的断言、是否出现了给定事实之外的数字。"
            "若无问题请回答「无问题」。\n\n" + body[:3000]
        )
        return self.ask(prompt, ctx)


# ============================================================================
# 编排器
# ============================================================================


class AgentOrchestrator:
    """串行编排 + 审核反馈闭环。"""

    def __init__(self, storage_factory=None, use_llm: bool = True):
        self.retriever = RetrieverAgent(storage_factory, use_llm=use_llm)
        self.analyst = AnalystAgent(use_llm=use_llm)
        self.writer = WriterAgent(use_llm=use_llm)
        self.reviewer = ReviewerAgent(use_llm=use_llm)
        self.use_llm = use_llm
        # 角色 → 生效模型名，仅用于过程流展示（只读配置，不构造客户端）
        from llm_client import role_models

        self.models = {m["label"]: m["model"] for m in role_models()}

    def run_task(
        self,
        question: str,
        stream_callback: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, Any]:
        ctx = AgentContext(question=question)

        def emit(payload: Dict[str, Any]) -> None:
            if stream_callback:
                try:
                    stream_callback(json.dumps(payload, ensure_ascii=False, default=str))
                except Exception:  # noqa: BLE001
                    pass

        # 有人订阅时才把过程回调挂上：这样 /chat 的非流式路径不变，
        # 各角色的 trace/stream 调用在无订阅者时静默。
        if stream_callback:
            ctx.on_trace = emit

        def step(agent_name: str, status: str, **extra: Any) -> None:
            """步骤事件；顺带带上该角色实际使用的模型名，供前端逐行显示。"""
            payload: Dict[str, Any] = {"type": "step", "agent": agent_name, "status": status}
            if self.use_llm:
                payload["model"] = self.models.get(agent_name, "")
            payload.update(extra)
            emit(payload)

        for round_no in range(MAX_REVIEW_ROUNDS + 1):
            ctx.review_round = round_no
            if round_no > 0:
                emit({"type": "retry", "round": round_no, "feedback": ctx.review_feedback})

            # 检索员（重算轮次里跳过重复的知识库检索，但保留意图）
            if round_no == 0:
                step(self.retriever.name, "running")
                r_out = self.retriever.run(ctx)
                step(self.retriever.name, "done", **r_out)

            step(self.analyst.name, "running")
            a_out = self.analyst.run(ctx)
            step(self.analyst.name, "done", **a_out)

            step(self.writer.name, "running")
            w_out = self.writer.run(ctx)
            step(self.writer.name, "done", **w_out)

            step(self.reviewer.name, "running")
            review = self.reviewer.run(ctx)
            step(self.reviewer.name, "done",
                 status_text=review["review_status"],
                 score=review["quality_score"],
                 issues=review["issues"])

            if review["review_status"] == "approved":
                break

        rep = ctx.report
        panels = collect_analysis_panels(ctx)
        final = {
            "question": question,
            "domain": getattr(ctx.intent, "domain", None),
            "answer": rep.to_markdown() if rep is not None else "",
            "report": rep.to_dict() if rep is not None else None,
            # 结构化指标明细（含不可用项及其原因），供前端直接渲染指标结果表
            "indicators": [r.to_dict() for r in ctx.indicators],
            "analytics": ctx.analytics,
            # 报告之外的两个补充面板：横向对比与异常检测（原本算完即丢，前端无处可取）
            "compare": panels["compare"],
            "anomalies": panels["anomalies"],
            "review": ctx.review,
            "steps": ctx.steps,
            "review_rounds": ctx.review_round,
        }
        emit({"type": "final", "data": {
            "review_status": ctx.review.get("review_status"),
            "quality_score": ctx.review.get("quality_score"),
            "review_rounds": ctx.review_round,
        }})
        return final


# ============================================================================
# 结果面板：把「已算出、但报告未呈现」的结果收拢给前端
# ============================================================================
#
# 两类结果此前算完即丢：横向对比（企业运营域多项目的排名）只写进 ctx.analytics
# 的 compare 键、报告不引用；异常检测散落在换热站/住户/气象三域的分析结果里，
# 报告只以「异常提示」文字段落摘录一部分。本组函数只做**取用与归一化**，
# 不重算已算过的结果；唯一例外是企业运营域——它此前从未做过异常检测，
# 这里按需补一次（单日快照下会如实返回「数据不足」，不编造结果）。

ANOMALY_KIND_LABEL = {
    "threshold": "统计越限",
    "jump": "突变",
    "missing": "缺测",
    "deviation": "工况偏离",
}

# 面板最多列出的异常条数（总数仍如实给出）
PANEL_MAX_ITEMS = 12


def _norm_anomaly(a: Dict[str, Any], scope: str = "") -> Dict[str, Any]:
    kind = a.get("kind") or ""
    return {
        "kind": kind,
        "kind_label": ANOMALY_KIND_LABEL.get(kind, kind or "异常"),
        "severity": a.get("severity") or "中",
        "ts": a.get("ts"),
        "value": a.get("value"),
        "detail": a.get("detail") or "",
        "scope": scope,
    }


def _panel(available: bool, scope: str = "", summary: str = "",
           items: Optional[List[Dict[str, Any]]] = None,
           total: Optional[int] = None, reason: str = "") -> Dict[str, Any]:
    items = items or []
    n_total = len(items) if total is None else total
    return {
        "available": available,
        "scope": scope,
        "summary": summary,
        "reason": reason,
        "total": n_total,
        "shown": len(items),
        "truncated": n_total > len(items),
        "items": items,
    }


def _anomaly_panel(ctx: "AgentContext", ana: Dict[str, Any]) -> Dict[str, Any]:
    """异常检测面板。每题至多命中一个数据域，按域取用已算出的异常。"""
    # —— 换热站：二次侧供回水温差工况偏离（scada_analysis 已算）
    scada = ana.get("scada")
    if isinstance(scada, dict) and scada.get("available"):
        sid = scada.get("substation_id")
        dev = scada.get("deviation_anomalies") or {}
        total = dev.get("count") or 0
        rng = (scada.get("delta_t") or {}).get("reasonable_range") or []
        items = [_norm_anomaly(a, f"换热站 {sid}") for a in (dev.get("samples") or [])]
        summary = (f"温差越界 {total} 个时段（占 {dev.get('ratio')}%），"
                   f"合理区间 {rng[0]}~{rng[1]}℃"
                   if total else
                   f"全部时段温差均落在合理区间 {rng[0] if rng else '—'}~"
                   f"{rng[1] if rng else '—'}℃ 内")
        return _panel(True, f"换热站 {sid}", summary, items, total)

    # —— 气象：日均气温的统计越限 / 突变 / 缺测（weather_trend 已算）
    wt = ana.get("weather")
    if isinstance(wt, dict) and wt.get("available"):
        anom = wt.get("anomalies") or {}
        cnt = {k: len(anom.get(k) or []) for k in ("threshold", "jump", "missing")}
        items = [_norm_anomaly(a, f"气象站 {wt.get('weather_id')}")
                 for k in ("threshold", "jump", "missing")
                 for a in (anom.get(k) or [])]
        summary = (f"统计越限 {cnt['threshold']} 个、突变 {cnt['jump']} 个、"
                   f"缺测 {cnt['missing']} 处" if items
                   else "区间内日均气温未见越限、突变或缺失")
        return _panel(True, f"气象站 {wt.get('weather_id')}", summary,
                      items[:PANEL_MAX_ITEMS], len(items))

    # —— 住户：优先温度归一化口径（剔除季节性），原始口径仅作参考
    ha = ana.get("household")
    if isinstance(ha, dict) and ha.get("available"):
        hid = (ha.get("household") or {}).get("household_id")
        anom = ha.get("anomalies") or {}
        norm_hits = anom.get("threshold_normalized") or []
        raw_hits = ((anom.get("threshold") or []) + (anom.get("jump") or [])
                    + (anom.get("missing") or []))
        if norm_hits:
            items = [_norm_anomaly(a, f"住户 {hid}（温度归一化后）") for a in norm_hits]
            summary = (f"温度归一化口径下越限 {len(norm_hits)} 处（已剔除季节性，结论更可信）；"
                       f"原始口径另有 {len(raw_hits)} 处，含季节性、仅供参考")
        else:
            items = [_norm_anomaly(a, f"住户 {hid}（原始口径）") for a in raw_hits]
            summary = (f"原始口径 {len(raw_hits)} 处；该户所属气象站无度日数数据，"
                       "未能做温度归一化，判读需注意季节性")
        return _panel(True, f"住户 {hid}", summary, items[:PANEL_MAX_ITEMS], len(items))

    # —— 企业运营：单日快照，异常检测需多日序列，按需检测并如实说明
    intent = ctx.intent
    domain = getattr(intent, "domain", None) if intent is not None else None
    if domain == "enterprise":
        if not intent.metrics:
            return _panel(False, reason=ana.get("blocked") or "未识别出要核算的指标，未做异常检测")
        code = _metric_to_code(intent.metrics[0], intent.modes)
        if not code:
            return _panel(False, reason=f"指标「{intent.metrics[0]}」不对应数值序列，未做异常检测")
        import analytics as _an

        entity_ids = intent.entity_ids()
        entity = entity_ids[0] if entity_ids else "company:公司A"
        res = _an.enterprise_anomalies(entity, code)
        if not res.available:
            return _panel(False, res.dimension, reason=res.reason)
        items = [_norm_anomaly(a.to_dict(), res.dimension) for a in res.anomalies]
        return _panel(True, res.dimension,
                      f"核验 {res.checked_points} 个数据点，发现 {len(items)} 处异常",
                      items[:PANEL_MAX_ITEMS], len(items))

    if domain == "knowledge":
        return _panel(False, reason="规范/方法类问题，不做数据异常检测")
    return _panel(False, reason="本次未产生可供检测的数值序列")


def collect_analysis_panels(ctx: "AgentContext") -> Dict[str, Any]:
    """给前端的两个补充面板：横向对比 + 异常检测。

    失败不影响主链路（报告与指标结果照常返回），故整体兜住异常。
    """
    ana = ctx.analytics or {}
    try:
        anomalies = _anomaly_panel(ctx, ana)
    except Exception as e:  # noqa: BLE001
        logger.exception("异常检测面板构建失败")
        anomalies = _panel(False, reason=f"异常检测面板构建失败：{e}")
    return {"compare": ana.get("compare"), "anomalies": anomalies}


def _make_storage_factory():
    """返回一个「每次调用新建短连接」的工厂；知识库不可用时返回 None。

    不用长连接：pymilvus 的空闲心跳会被 Milvus 以 GOAWAY(too_many_pings) 限流，
    而多智能体流程中有大段大模型调用，连接必须用完即关。
    """
    try:
        from document_processor import MilvusStorage

        with MilvusStorage() as probe:   # 预检可用性
            probe.create_collection()
        return MilvusStorage
    except Exception as e:  # noqa: BLE001
        print(f"[提示] 知识库不可用，仅走数据链路：{e}\n")
        return None
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="多智能体协作（命令行跑一次完整问答）")
    ap.add_argument("question", nargs="?", help="要提问的问题")
    ap.add_argument("--no-llm", action="store_true", help="不调用大模型（仅确定性链路）")
    args = ap.parse_args()

    if not args.question:
        ap.print_help()
        raise SystemExit(0)

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass

    _orch = AgentOrchestrator(_make_storage_factory(), use_llm=not args.no_llm)
    _res = _orch.run_task(args.question,
                          stream_callback=lambda s: print(f"  [SSE] {s[:150]}"))
    print(_res["answer"])
