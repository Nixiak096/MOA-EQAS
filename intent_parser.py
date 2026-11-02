"""供热意图解析：把自然语言问题解析为结构化查询意图。

归属：检索员的第一步动作（不另设独立角色）。

解析维度：
    领域 domain      企业运营 / 换热站 / 住户 / 气象 / 规范知识
    指标 metric      气耗 / 水耗 / 电耗 / 收费率 / 户缴率 / 供热面积 …
    层级 level       集团级 / 公司级 / 项目级
    日期 date        今天 / 昨天 / 具体日期 / 采暖季
    口径 mode        日值 / 累计 / 评价值 / 环比 / 度日数折标
    分类 category    小计 / 居民 / 非居民（可多选）
    比较 compare     环比 / 同比 / 周比

四类问题必须先分流到对应数据域，否则问「换热站温差」会被当成公司级指标查询，
问「国家标准要求」会被当成指标计算 —— 答案会答非所问。

实现方式：
    规则解析（确定性、可审计）+ 可选大模型兜底（parse_with_llm）。
    规则命中的字段直接采信，便于审核员核验口径。

用法：
    python intent_parser.py            # 用 docx 中的示例问题自测
    python intent_parser.py "公司A今天的气耗指标是多少？"
"""

import argparse
import datetime
import json
import re
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import config

# ============================================================================
# 词表
# ============================================================================

# 数据域：先分流，再解析指标
DOMAIN_KEYWORDS: Dict[str, List[str]] = {
    "scada": ["换热站", "供回水", "供回水温差", "一次侧", "二次侧", "水力", "scada", "站级"],
    "household": ["住户", "分户", "热泵", "能效", "单户", "居民户"],
    "weather": ["气象", "气象站", "温度站", "室外温度", "度日数本身"],
}
# 换热站编号：不能用 \b —— Python 中 CJK 也算 \w，'L4换热站' 里 4 与 换 之间无词边界
SUBSTATION_RE = re.compile(r"(?<![A-Za-z])L\s?(\d{1,3})(?![0-9])")

# 规范/知识类（无指标、问条文或要求）
KNOWLEDGE_WORDS = [
    "国家标准", "国标", "规范", "标准规定", "条文", "条例", "规程", "技术标准",
    "gb", "cjj", "要求是什么", "怎么规定的", "如何规定",
]

# 指标别名（按业务语义归并）
METRIC_ALIASES: Dict[str, List[str]] = {
    "gas": ["气耗", "燃气", "天然气", "用气量", "耗气"],
    "water": ["水耗", "耗水", "用水量", "耗水量"],
    "elec": ["电耗", "耗电", "用电量", "耗电量"],
    "fee_rate": ["本季收费率", "收费率", "收缴率", "收费"],
    "household_payment_rate": ["户缴率"],
    "arrears_fee_rate": ["旧欠收费率"],
    "heating_area": ["总供热面积", "供热面积", "收费面积", "面积"],
}

# 本季 / 旧欠 口径（审核员需校验，避免口径混淆）
ARREARS_WORDS = ["旧欠", "陈欠", "历史欠费"]

# 能源类指标（适用日值/累计/环比/度日数折标等口径）
ENERGY_METRICS = {"gas", "water", "elec", "heating_area"}

# 口径
MODE_ALIASES: Dict[str, List[str]] = {
    "degree_day": ["度日数折标", "度日数", "折标"],
    "evaluation": ["评价值", "评价"],
    "cumulative": ["累计", "累积"],
    "daily": ["日值", "当日", "今日", "今天", "日"],
}
MODE_ORDER = ["degree_day", "evaluation", "cumulative", "daily"]

# 比较口径（环比/同比/周比）：既可能直接说「环比」，也可能说「和昨天比」
COMPARE_PATTERNS: Dict[str, List[str]] = {
    "mom": ["环比", "和昨天比", "与昨天比", "比昨天", "较昨日", "和昨日比", "比昨日", "较昨天"],
    "yoy": ["同比", "和去年同期", "与去年同期", "比去年同期"],
    "wow": ["和上周", "较上周", "上周同期", "与上周", "比上周"],
}

# 层级
GROUP_WORDS = ["集团", "全集团", "集团级"]
PROJECT_WORDS = ["项目级", "项目"]

# 分类（可多选）
CATEGORY_ALIASES: Dict[str, List[str]] = {
    "居民": ["居民"],
    "非居民": ["非居民"],
    "小计": ["小计", "合计", "总计", "整体", "公司整体"],
}

# 相对日期
RELATIVE_DAYS = {
    "大前天": -3,
    "前天": -2,
    "昨天": -1, "昨日": -1,
    "今天": 0, "今日": 0, "当日": 0, "本日": 0,
}

SEASON_WORDS = ["本季", "采暖季", "供暖季", "供热季"]
ALL_PROJECT_WORDS = ["各项目", "所有项目", "全部项目", "各个项目"]

PROJECT_RE = re.compile(r"项目\s*([1-9１-９一二三四五六七八九])")
COMPANY_RE = re.compile(r"公司\s*([A-Za-zＡ-Ｚａ-ｚ0-9一二三四五六七八九十]+)")
DATE_RE = re.compile(r"(\d{4})[-/年](\d{1,2})[-/月](\d{1,2})日?")
MONTH_DAY_RE = re.compile(r"(?<!\d)(\d{1,2})月(\d{1,2})日")

_CN_DIGITS = {"一": "1", "二": "2", "三": "3", "四": "4", "五": "5",
              "六": "6", "七": "7", "八": "8", "九": "9"}


@dataclass
class QueryIntent:
    """结构化查询意图。"""

    raw: str = ""
    domain: str = "enterprise"       # enterprise|scada|household|weather|knowledge
    metrics: List[str] = field(default_factory=list)
    modes: List[str] = field(default_factory=list)
    level: str = "company"           # group | company | project
    company: Optional[str] = None
    projects: List[str] = field(default_factory=list)
    categories: List[str] = field(default_factory=list)  # 小计/居民/非居民（可多选）
    arrears: bool = False            # True=旧欠口径，False=本季
    substations: List[str] = field(default_factory=list)  # 换热站编号（scada 域）
    stat_date: Optional[datetime.date] = None
    relative_date: Optional[str] = None
    season: Optional[str] = None
    compare: Optional[str] = None    # mom | yoy | wow
    unsupported_modes: List[str] = field(default_factory=list)  # 数据不支持的口径
    notes: List[str] = field(default_factory=list)

    def wants(self, mode: str) -> bool:
        return mode in self.modes

    def entity_ids(self) -> List[str]:
        """按层级展开为实体 ID（对应 dim_entity.entity_id）。非企业域返回空。"""
        comp = self.company
        if self.level == "group":
            return [f"group:{comp or '集团'}"]
        if self.level == "project":
            return [f"project:{comp}:{p}" for p in self.projects] if self.projects else []
        if self.level == "company":
            return [f"company:{comp}"] if comp else []
        return []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "raw": self.raw,
            "domain": self.domain,
            "metrics": self.metrics,
            "modes": self.modes,
            "level": self.level,
            "company": self.company,
            "projects": self.projects,
            "categories": self.categories,
            "arrears": self.arrears,
            "substations": self.substations,
            "stat_date": self.stat_date.isoformat() if self.stat_date else None,
            "relative_date": self.relative_date,
            "season": self.season,
            "compare": self.compare,
            "unsupported_modes": self.unsupported_modes,
            "notes": self.notes,
        }


def _norm(text: str) -> str:
    return re.sub(r"\s+", "", text or "")


def _hit(text: str, words: List[str]) -> Optional[str]:
    return next((w for w in words if w in text), None)


def _cn_num(s: str) -> str:
    s = s.strip().translate(str.maketrans("１２３４５６７８９", "123456789"))
    return _CN_DIGITS.get(s, s)


# ============================================================================
# 规则解析
# ============================================================================


def parse_rules(
    question: str,
    reference_date: Optional[datetime.date] = None,
    default_company: Optional[str] = "公司A",
) -> QueryIntent:
    """规则解析。reference_date 通常传库中最新统计日期，用于解析「今天/昨天」。"""
    text = _norm(question)
    intent = QueryIntent(raw=question)
    intent.company = default_company

    # ---- 1. 领域分流 ----
    subs = SUBSTATION_RE.findall(text)
    if subs or _hit(text, DOMAIN_KEYWORDS["scada"]):
        intent.domain = "scada"
        intent.substations = [f"L{n}" for n in subs]
    elif _hit(text, DOMAIN_KEYWORDS["household"]):
        intent.domain = "household"
    elif _hit(text, DOMAIN_KEYWORDS["weather"]):
        intent.domain = "weather"

    # ---- 2. 指标 ----
    for metric, words in METRIC_ALIASES.items():
        if _hit(text, words) and metric not in intent.metrics:
            intent.metrics.append(metric)
    # 「户缴率」命中时，别被「收费」误加 fee_rate
    if "household_payment_rate" in intent.metrics and "收费率" not in text and "收缴率" not in text:
        intent.metrics = [m for m in intent.metrics if m != "fee_rate"]
    # 泛指能耗
    if not intent.metrics and any(w in text for w in ["能耗", "耗量", "单耗"]) \
            and intent.domain == "enterprise":
        intent.metrics = ["gas", "water", "elec"]
        intent.notes.append("未指明具体能源，按气/水/电三类展开")

    # ---- 3. 规范/知识类判定 ----
    # 无指标 + 命中规范词，或问「是什么/怎么规定」的纯知识问题
    if intent.domain == "enterprise" and not intent.metrics:
        if _hit(text, KNOWLEDGE_WORDS) or re.search(r"(多少|是什么|怎么办|如何|要求)", text):
            if not intent.substations and not any(w in text for w in ["气耗", "水耗", "电耗", "收费"]):
                intent.domain = "knowledge"
                intent.notes.append("判为规范/知识类问题，走知识库检索")

    # ---- 4. 口径 ----
    for mode in MODE_ORDER:
        if _hit(text, MODE_ALIASES[mode]) and mode not in intent.modes:
            intent.modes.append(mode)

    # ---- 5. 比较口径 ----
    # 比较短语里常含相对日期（「和昨天比」），需先剔除，否则主日期会被误取成比较基准日
    date_text = text
    for kind, words in COMPARE_PATTERNS.items():
        matched = _hit(text, words)
        if not matched:
            continue
        intent.compare = kind
        for w in sorted(words, key=len, reverse=True):
            date_text = date_text.replace(w, "")
        if kind == "mom":
            if "mom" not in intent.modes:
                intent.modes.append("mom")
        else:
            label = {"yoy": "同比", "wow": "周比"}[kind]
            intent.unsupported_modes.append(kind)
            intent.notes.append(f"命中「{label}」，但当前数据仅单日快照，无法计算")
        break

    if (
        any(m in intent.modes for m in ("mom", "evaluation"))
        and "daily" not in intent.modes
        and any(m in ENERGY_METRICS for m in intent.metrics)
    ):
        intent.modes.append("daily")

    # 收费率等为采暖季比率口径，不套用日值
    if not intent.modes and intent.domain == "enterprise":
        if intent.metrics and not any(m in ENERGY_METRICS for m in intent.metrics):
            intent.notes.append("收费类为采暖季比率口径，不适用日值口径")
        else:
            intent.modes = ["daily"]
            intent.notes.append("未指明口径，默认按日值")

    # ---- 6. 分类 ----
    for cat, words in CATEGORY_ALIASES.items():
        if _hit(text, words) and cat not in intent.categories:
            intent.categories.append(cat)
    if not intent.categories and any(
        m in intent.metrics for m in ("fee_rate", "household_payment_rate", "arrears_fee_rate")
    ):
        intent.categories = ["小计"]
        intent.notes.append("收费类未指明居民/非居民，默认取小计")

    # ---- 7. 本季 / 旧欠 ----
    intent.arrears = bool(_hit(text, ARREARS_WORDS))
    if intent.arrears:
        intent.notes.append("命中「旧欠」口径（区别于「本季」）")

    # ---- 8. 层级与实体 ----
    proj_hits = PROJECT_RE.findall(text)
    if proj_hits:
        intent.level = "project"
        intent.projects = [f"项目{_cn_num(p)}" for p in proj_hits]
        intent.notes.append(f"命中项目：{intent.projects}")
    elif _hit(text, ALL_PROJECT_WORDS):
        intent.level = "project"
        intent.notes.append("指代全部项目，项目列表待补全")
    elif _hit(text, GROUP_WORDS):
        intent.level = "group"

    m = COMPANY_RE.search(text)
    if m:
        intent.company = f"公司{m.group(1)}"
    elif intent.level == "group":
        intent.company = None

    # ---- 9. 日期（用剔除比较短语后的 date_text，避免取成比较基准日）----
    for word in sorted(RELATIVE_DAYS, key=len, reverse=True):
        if word in date_text:
            intent.relative_date = word
            if reference_date:
                intent.stat_date = reference_date + datetime.timedelta(days=RELATIVE_DAYS[word])
            intent.notes.append(f"相对日期「{word}」→ {intent.stat_date or '待定'}")
            break

    if intent.stat_date is None:
        m = DATE_RE.search(date_text)
        if m:
            y, mo, d = (int(x) for x in m.groups())
            try:
                intent.stat_date = datetime.date(y, mo, d)
            except ValueError:
                intent.notes.append(f"日期非法：{m.group(0)}")
        else:
            m = MONTH_DAY_RE.search(date_text)
            if m and reference_date:
                try:
                    intent.stat_date = datetime.date(
                        reference_date.year, int(m.group(1)), int(m.group(2))
                    )
                    intent.notes.append(f"月日补全年份 → {intent.stat_date}")
                except ValueError:
                    pass

    if intent.stat_date is None and _hit(text, SEASON_WORDS):
        intent.season = config.HEATING_SEASON
        intent.notes.append(f"命中采暖季口径：{intent.season}")

    # 未指明具体日期时一律取库中最新统计日期。
    # 注意「本季」这类季节口径也要取具体 stat_date：季内数据在库中是「截至最新统计日」的
    # 快照，若留着 None，下游会退化成 today() 从而查无数据（季节只作口径标签，仍存 intent.season）。
    if intent.stat_date is None and intent.domain in ("enterprise", "scada", "household"):
        intent.stat_date = reference_date
        if reference_date:
            scope = f"（{intent.season} 季内最新快照）" if intent.season else ""
            intent.notes.append(f"未指明日期，取最新统计日期 {reference_date}{scope}")

    # ---- 10. 领域与层级一致性 ----
    # 换热站 / 住户 / 气象 / 规范域与公司-项目层级无关，清掉误带的企业维度
    if intent.domain in ("scada", "household", "weather", "knowledge"):
        intent.level = "none"
        intent.company = None
        intent.projects = []
        intent.categories = []

    if intent.domain == "scada" and not intent.substations:
        intent.notes.append("未解析出换热站编号（形如 L4）")

    return intent


# ============================================================================
# 大模型兜底（可选，需配置密钥）
# ============================================================================

_LLM_SYSTEM = """你是供热运营数据查询的意图解析器。把用户问题解析为 JSON，字段：
domain: enterprise/scada/household/weather/knowledge
metrics: 数组，取值 gas/water/elec/fee_rate/household_payment_rate/arrears_fee_rate/heating_area
modes: 数组，取值 daily/cumulative/evaluation/mom/degree_day
level: group/company/project
company: 字符串或 null
projects: 数组，如 ["项目3"]
categories: 数组，取值 小计/居民/非居民
substations: 数组，如 ["L4"]
season: 字符串或 null
只输出 JSON，不要解释。不确定的字段用 null 或空数组，不要臆造。"""


def parse_with_llm(question: str) -> Optional[Dict[str, Any]]:
    """用大模型解析，返回原始字典；未配置密钥时返回 None。"""
    try:
        from llm_client import get_llm

        return get_llm("retriever").chat_json(question, system=_LLM_SYSTEM)
    except Exception:  # noqa: BLE001
        return None


def parse(
    question: str,
    reference_date: Optional[datetime.date] = None,
    default_company: Optional[str] = "公司A",
    use_llm: bool = False,
) -> QueryIntent:
    """统一入口：规则优先，可选大模型补全未命中的字段。"""
    intent = parse_rules(question, reference_date, default_company)
    if not use_llm:
        return intent

    raw = parse_with_llm(question)
    if not isinstance(raw, dict):
        intent.notes.append("大模型解析不可用，仅用规则结果")
        return intent

    if not intent.metrics and isinstance(raw.get("metrics"), list):
        intent.metrics = [m for m in raw["metrics"] if isinstance(m, str)]
        if intent.metrics:
            intent.notes.append("指标由大模型补全")
    if isinstance(raw.get("modes"), list):
        for m in raw["modes"]:
            if isinstance(m, str) and m not in intent.modes:
                intent.modes.append(m)
    if isinstance(raw.get("projects"), list) and not intent.projects:
        intent.projects = [str(p) for p in raw["projects"]]
        if intent.projects:
            intent.level = "project"
    if isinstance(raw.get("categories"), list) and not intent.categories:
        intent.categories = [str(c) for c in raw["categories"]]
    if isinstance(raw.get("substations"), list) and not intent.substations:
        intent.substations = [str(s) for s in raw["substations"]]
    # 领域分流：大模型只能「补齐」，不能推翻规则已识别的数据域。
    # scada/household/weather 三个域没有具体实体就取不到数（会静默产出空报告），
    # 因此只有当大模型同时给出该域所需的实体时，才采纳它的判定。
    llm_domain = raw.get("domain")
    if llm_domain in ("enterprise", "scada", "household", "weather", "knowledge") \
            and intent.domain == "enterprise":
        if llm_domain == "enterprise":
            pass
        elif llm_domain == "scada":
            if intent.substations:
                intent.domain = "scada"
            else:
                intent.notes.append(
                    "大模型判为换热站域，但未给出站号，保留规则判定 " + intent.domain)
        elif llm_domain == "knowledge":
            if not intent.metrics:
                intent.domain = "knowledge"
                intent.notes.append("规则未识别出指标，采纳大模型的知识类判定")
            else:
                intent.notes.append(
                    "大模型判为知识类，但规则已识别出指标，保留数据域判定 " + intent.domain)
        else:
            # 大模型输出的字段里没有 household_id / weather_id，无法据此取数，不采纳
            intent.notes.append(
                f"大模型判为 {llm_domain} 域，但未给出该域所需实体，保留规则判定 {intent.domain}")
    return intent


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="供热意图解析（命令行看解析结果）")
    ap.add_argument("question", nargs="?", help="要解析的问题")
    ap.add_argument("--reference-date", help="参考日期 YYYY-MM-DD（默认取库中最新统计日期）")
    ap.add_argument("--use-llm", action="store_true", help="启用大模型兜底（需配置密钥）")
    args = ap.parse_args()

    if not args.question:
        ap.print_help()
        raise SystemExit(0)

    ref = None
    if args.reference_date:
        ref = datetime.date.fromisoformat(args.reference_date)
    else:
        try:
            from sqlalchemy import text

            import heat_db

            eng = heat_db.get_engine()
            with eng.connect() as c:
                ref = c.execute(text("SELECT MAX(stat_date) FROM fact_energy_daily")).scalar()
            eng.dispose()
        except Exception:  # noqa: BLE001
            ref = None

    intent = parse(args.question, reference_date=ref, use_llm=args.use_llm)
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    print(json.dumps(intent.to_dict(), ensure_ascii=False, indent=2))
