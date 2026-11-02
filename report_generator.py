"""报表生成：把指标计算结果与统计分析结果组织成图文报告。

按课题要求，报告须「含数据、结论、来源」。本模块产出的报告显式携带：

    sections  分节正文（结论 / 指标数据 / 统计分析 / 异常提示）
    charts    图表规格（折线 / 柱状），供前端渲染，不在此处画图
    sources   数据来源清单（供审核员核验引用）
    caveats   口径限制与数据缺失说明（如「单日快照」「演示站点」「缺基准值」）

不可用（available=False）的结果不会被写进结论，而是进入 caveats 说明原因 ——
不猜测、不编造，保证报告可被审核员逐项回溯。

用法：
    python report_generator.py --selftest
"""

import argparse
import datetime
import json
import logging
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import config
from indicators import IndicatorResult

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("report_generator")


@dataclass
class ChartSpec:
    kind: str = "line"          # line | bar
    title: str = ""
    x: List[Any] = field(default_factory=list)
    series: List[Dict[str, Any]] = field(default_factory=list)
    y_unit: str = ""
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return vars(self)


@dataclass
class Report:
    title: str = ""
    generated_at: str = ""
    subtitle: str = ""
    sections: Dict[str, List[str]] = field(default_factory=dict)
    charts: List[ChartSpec] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)
    caveats: List[str] = field(default_factory=list)

    def add(self, heading: str, lines: List[str]) -> None:
        if lines:
            self.sections.setdefault(heading, []).extend(lines)

    def add_source(self, s: Optional[str]) -> None:
        if s and s not in self.sources:
            self.sources.append(s)

    def add_caveat(self, c: Optional[str]) -> None:
        if c and c not in self.caveats:
            self.caveats.append(c)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "subtitle": self.subtitle,
            "generated_at": self.generated_at,
            "sections": self.sections,
            "charts": [c.to_dict() for c in self.charts],
            "sources": self.sources,
            "caveats": self.caveats,
        }

    def to_markdown(self) -> str:
        L: List[str] = [f"# {self.title}"]
        if self.subtitle:
            L.append(f"*{self.subtitle}*")
        L.append(f"\n生成时间：{self.generated_at}")
        for heading, lines in self.sections.items():
            L.append(f"\n## {heading}")
            for ln in lines:
                L.append(f"- {ln}" if not ln.startswith(("|", " ", ">")) else ln)
        if self.caveats:
            L.append("\n## 口径与数据说明")
            for c in self.caveats:
                L.append(f"- {c}")
        if self.sources:
            L.append("\n## 数据来源")
            for s in self.sources:
                L.append(f"- {s}")
        return "\n".join(L)


def _fmt(v: Optional[float], digits: int = 4) -> str:
    if v is None:
        return "—"
    if abs(v) >= 1e4:
        return f"{v:,.{min(digits, 3)}f}"
    return f"{v:.{digits}g}"


def _fmt_anomaly(h: Dict[str, Any], *, with_kind: bool = False) -> str:
    """一条异常的描述行；`with_kind` 为真时额外标出异常类型。"""
    return (f"{h['ts']}（{h['kind']}）：{h['detail']}" if with_kind
            else f"{h['ts']}：{h['detail']}")


def _anomaly_section(
    hits: List[Dict[str, Any]],
    *,
    heading: str = "异常提示",
    with_kind: bool = False,
    limit: Optional[int] = None,
    total: Optional[int] = None,
) -> Optional[Tuple[str, List[str]]]:
    """把异常列表组织成一个分节（标题, 行）。

    传入 `limit` 时只列前若干条，并在标题里如实说明是抽样、以及总条数；
    未传 `limit` 则全部列出。无异常时返回 None（调用方据此跳过该分节）。
    """
    if not hits:
        return None
    shown = hits if limit is None else hits[:limit]
    n_total = len(hits) if total is None else total
    if n_total > len(shown):
        heading = f"{heading}（共 {n_total} 条，下列为抽样示意 {len(shown)} 条）"
    return heading, [_fmt_anomaly(h, with_kind=with_kind) for h in shown]


# ============================================================================
# 指标类报告
# ============================================================================

MODE_LABEL = {
    "daily": "日值", "evaluation": "评价值", "mom": "环比",
    "cumulative": "累计", "cumulative_evaluation": "累计评价值",
    "season_cumulative": "标准采暖季累计",
    "season_cumulative_evaluation": "标准采暖季累计评价值",
    "degree_day": "度日数折标", "season": "采暖季",
}


def build_indicator_report(
    results: List[IndicatorResult],
    title: str = "供热运营指标核算报告",
    scope: str = "",
) -> Report:
    rep = Report(
        title=title,
        subtitle=scope,
        generated_at=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )

    ok = [r for r in results if r.available]
    na = [r for r in results if not r.available]

    # ---- 结论 ----
    concl: List[str] = []
    for r in ok:
        unit = f" {r.unit}" if r.unit and not r.unit.startswith("按源") else ""
        concl.append(f"{r.label} = **{_fmt(r.value)}{unit}**")
    if concl:
        rep.add("结论", concl)

    # ---- 指标数据（含公式与输入，供审核员核验）----
    rows = ["| 指标 | 口径 | 计算值 | 所用公式 |", "|---|---|---|---|"]
    for r in ok:
        unit = "" if r.unit.startswith("按源") else r.unit
        rows.append(
            f"| {r.label} | {MODE_LABEL.get(r.mode, r.mode)} | {_fmt(r.value)} {unit} "
            f"| {r.formula} |"
        )
    if len(rows) > 2:
        rep.add("指标数据", rows)

    # ---- 输入明细（逐项标注单位，便于读者自行核对）----
    detail: List[str] = []
    for r in ok:
        if not r.inputs:
            continue
        units = getattr(r, "input_units", None) or {}
        kv = "、".join(
            f"{k}{f'({units[k]})' if units.get(k) else ''}"
            f"={_fmt(v) if isinstance(v, (int, float)) else v}"
            for k, v in r.inputs.items()
        )
        detail.append(f"{r.label}：{kv}")
    rep.add("计算输入", detail)

    # ---- 口径校验（收费率重算对照，审核员关注点）----
    checks: List[str] = []
    for r in ok:
        cc = r.cross_check
        if not cc:
            continue
        if cc.get("recomputed") is not None:
            flag = "一致" if cc["consistent"] else "★不一致，建议复核"
            method = f"重算方法：{cc['method']}；" if cc.get("method") else ""
            checks.append(
                f"{r.label}：{method}重算 {_fmt(cc['recomputed'])}% vs 报表原值 "
                f"{_fmt(cc['reported'])}%，偏差 {cc['delta_pp']:.4f} 个百分点 — {flag}"
            )
            if cc.get("inputs"):
                units = cc.get("input_units") or {}
                kv = "、".join(
                    f"{k}{f'({units[k]})' if units.get(k) else ''}"
                    f"={_fmt(v) if isinstance(v, (int, float)) else v}"
                    for k, v in cc["inputs"].items()
                )
                checks.append(f"　└ {kv}")
        elif cc.get("note"):
            checks.append(f"{r.label}：{cc['note']}")
    rep.add("口径校验", checks)

    # ---- 图表 ----
    bars = [r for r in ok if r.mode in ("daily", "cumulative", "evaluation")]
    if bars:
        rep.charts.append(ChartSpec(
            kind="bar",
            title="指标计算值对比",
            x=[r.label for r in bars],
            series=[{"name": "计算值", "data": [r.value for r in bars]}],
            note="来自 fact_energy_daily / fact_fee_season",
        ))

    # ---- 来源与说明 ----
    for r in results:
        rep.add_source(r.source)
    for r in na:
        rep.add_caveat(f"{r.label} 未能计算：{r.reason}")
    if any(r.mode == "degree_day" for r in ok):
        rep.add_caveat(
            f"度日数折标所用温度取自气象站 {config.DEGREE_DAY_STATION}；"
            "该站点为匿名化编号、非本项目实测区域，仅作演示/替代温度源，需替换为真实站点温度"
        )
    if any(r.mode == "mom" for r in na):
        rep.add_caveat("环比需要「昨日」数据，当前企业运营数据为单日快照，故无法计算")
    # 公式名称：《数据及计算说明》用「日耗指标值」这类通用写法，本报告按所查能源写成
    # 「日气耗指标值」等具体名称，避免同一份报告里公式名与指标名不一致
    if any(r.metric in ("gas", "water", "elec") for r in ok):
        rep.add_caveat(
            "指标公式依据《数据及计算说明》；其中「日耗指标值」「日耗量」等通用写法，"
            "本报告按所查能源替换为「日气耗指标值」「日气耗量」等具体名称，公式关系不变"
        )
    # 收费率的两级聚合口径不同，必须说明，否则读者会以为项目收费率能算平均得到公司收费率
    if any(r.metric == "fee_rate" and (r.cross_check or {}).get("method", "").startswith("户数口径")
           for r in ok):
        rep.add_caveat(
            "源报表的收费率在两级采用不同聚合口径：项目级为「累计本季实收金额 ÷ 本季应收金额」，"
            "公司级为「各项目收费率按本季应收户数加权」。因此公司级收费率并不等于各项目收费率的"
            "简单平均，也不等于公司级金额之比（非居民因户数少且分布悬殊，两种口径相差约 1.8 个百分点）"
        )
    return rep


# ============================================================================
# 换热站报告
# ============================================================================


def build_scada_report(sa: Dict[str, Any]) -> Report:
    """换热站运行工况分析报告。"""
    sid = sa.get("substation_id", "?")
    rep = Report(
        title=f"换热站 {sid} 运行工况分析报告",
        generated_at=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )
    if not sa.get("available"):
        rep.add_caveat(f"数据不可用：{sa.get('reason')}")
        return rep

    dt = sa.get("delta_t", {}).get("stats") or {}
    dr = sa["data_range"]
    rep.subtitle = f"数据区间 {str(dr[0])[:16]} ~ {str(dr[1])[:16]}，共 {sa['n_points']} 条"
    dev = sa["deviation_anomalies"]
    rep.add("结论", [
        f"二次侧供回水温差均值 {_fmt(dt.get('mean'), 3)}℃，"
        f"范围 [{_fmt(dt.get('minimum'), 3)}, {_fmt(dt.get('maximum'), 3)}]℃",
        f"调节特性相关系数 {sa.get('curve_relation')}；{sa.get('curve_interpretation')}",
        f"温差越界时段 {dev['count']} 个（占 {dev['ratio']}%）",
    ])
    rep.add("指标数据", [
        "| 指标 | 数值 |", "|---|---|",
        f"| 数据点数 | {sa['n_points']} |",
        f"| 温差均值 | {_fmt(dt.get('mean'), 3)} ℃ |",
        f"| 温差中位数 | {_fmt(dt.get('median'), 3)} ℃ |",
        f"| 温差标准差 | {_fmt(dt.get('std'), 3)} ℃ |",
        f"| 合理区间 | {sa['delta_t']['reasonable_range'][0]} ~ "
        f"{sa['delta_t']['reasonable_range'][1]} ℃ |",
    ])

    section = _anomaly_section(dev.get("samples") or [], limit=8, total=dev["count"])
    if section:
        rep.add(*section)

    curve = sa.get("supply_temp_curve") or []
    if curve:
        rep.charts.append(ChartSpec(
            kind="line",
            title="二次侧供水温度随室外温度的调节特性",
            x=[c["t_amb_bin"] for c in curve],
            series=[{"name": "二次侧供水温度均值", "data": [round(c["t_sup_sec_mean"], 2) for c in curve]}],
            y_unit="℃",
            note="横轴：室外温度分箱（5℃ 一档）；纵轴：该档内二次侧供水温度均值",
        ))
    if dev["count"]:
        rep.add_caveat(
            "温差合理区间为工程经验值（当前 "
            f"{sa['delta_t']['reasonable_range'][0]}~{sa['delta_t']['reasonable_range'][1]}℃），"
            "应按本站实际工况标定后使用"
        )
    rep.add_source(f"fact_scada_hourly（substation_id={sid}）")
    rep.add_source("dim_substation（供热面积）")
    return rep


def build_household_report(ha: Dict[str, Any]) -> Report:
    """住户能耗画像报告。"""
    rep = Report(generated_at=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    if not ha.get("available"):
        rep.title = f"住户 {ha.get('household', {}).get('household_id', '?')} 能耗分析报告"
        rep.add_caveat(f"数据不可用：{ha.get('reason')}")
        return rep

    info = ha["household"]
    rep.title = f"住户 {info['household_id']} 能耗画像报告"
    rep.subtitle = (f"{info.get('building_type') or '—'} / {info.get('living_area') or '—'} m² / "
                    f"热泵 {info.get('heatpump_type') or '—'} / "
                    f"末端 {'地暖' if info.get('floor_heating') else '散热器'}")

    st = ha.get("stats") or {}
    hp = ha.get("heatpump_stats") or {}
    hn = ha.get("hdd_normalized") or {}
    hn_st = hn.get("stats") or {}
    slope = ha["trend"].get("slope_per_unit")
    rep.add("结论", [
        f"统计点数 {ha['n_points']}，趋势方向「{ha['trend']['direction']}」"
        + (f"（斜率 {_fmt(slope, 4)} kWh/日）" if slope is not None else ""),
        f"日均用电 {_fmt(st.get('mean'), 4)} kWh，其中热泵日均 {_fmt(hp.get('mean'), 4)} kWh",
        f"单位面积年用电估算 {_fmt(ha.get('kwh_per_m2_per_year'), 4)} kWh/m²·年"
        if ha.get("kwh_per_m2_per_year") else "单位面积年用电：缺建筑面积，无法估算",
        (f"温度归一化后（按所属气象站 {hn.get('weather_id')} 的 SIA 供暖度日数）"
         f"单位度日数用电均值 {_fmt(hn_st.get('mean'), 4)} kWh/℃·日"
         if hn_st.get("mean") is not None else
         "温度归一化：该户所属气象站无度日数数据，未做归一化"),
    ])
    rep.add("指标数据", [
        "| 指标 | 数值 |", "|---|---|",
        f"| 日均用电 | {_fmt(st.get('mean'), 4)} kWh |",
        f"| 最大日用电 | {_fmt(st.get('maximum'), 4)} kWh |",
        f"| 最小日用电 | {_fmt(st.get('minimum'), 4)} kWh |",
        f"| 标准差 | {_fmt(st.get('std'), 4)} kWh |",
        f"| 热泵日均 | {_fmt(hp.get('mean'), 4)} kWh |",
        f"| 单位度日数用电（归一化） | {_fmt(hn_st.get('mean'), 4)} kWh/℃·日 |",
    ])

    an = ha.get("anomalies", {})
    # 优先给出归一化后的异常（剔除季节性），原始口径的越限会提示其偏差
    for section in (
        _anomaly_section(an.get("threshold_normalized") or [],
                         heading="异常提示（温度归一化后，剔除季节性）"),
        _anomaly_section((an.get("threshold") or []) + (an.get("jump") or [])
                         + (an.get("missing") or []),
                         heading="异常提示（原始口径，含季节性，仅供参考）", with_kind=True),
    ):
        if section:
            rep.add(*section)

    monthly = ha.get("monthly") or []
    if monthly:
        rep.charts.append(ChartSpec(
            kind="bar",
            title="逐月用电量",
            x=[m["month"] for m in monthly],
            series=[{"name": "月用电量", "data": [round(m["kwh_total"], 1) for m in monthly]}],
            y_unit="kWh",
        ))

    rep.add_source(f"fact_household_meter（household_id={info['household_id']}）")
    rep.add_source("dim_household（建筑与热泵属性）")
    if hn.get("weather_id"):
        rep.add_source(f"fact_weather_daily（weather_id={hn['weather_id']}，用于度日数归一化）")
    if an.get("threshold"):
        rep.add_caveat(
            "供热用电具强季节性，直接对原始日用电量做「均值±kσ」越限会把每个采暖季高峰判为异常；"
            "应以温度归一化（单位度日数用电量）后的结果为准"
        )
    if info.get("smd_daily_available") is False:
        rep.add_caveat("该住户标记为「无日粒度数据」，统计结果可能不完整")
    return rep


def build_weather_report(wt: Dict[str, Any]) -> Report:
    """气象站温度趋势报告。"""
    rep = Report(generated_at=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    if not wt.get("available"):
        rep.title = "气象分析报告"
        rep.add_caveat(f"数据不可用：{wt.get('reason')}")
        return rep

    wid = wt["weather_id"]
    tr = wt["trend"]
    dr = wt["data_range"]
    rep.title = f"气象站 {wid} 温度趋势报告"
    rep.subtitle = f"数据范围 {dr[0][:10]} ~ {dr[1][:10]}"
    rep.add("结论", [
        f"区间内日均气温趋势「{tr['direction']}」，斜率 {_fmt(tr['slope_per_unit'], 4)} ℃/日",
        f"SIA 供暖度日数累计 {_fmt(wt['hdd_sia_sum'], 5)} ℃·日",
        f"统计越限 {len(wt['anomalies']['threshold'])} 个、突变 {len(wt['anomalies']['jump'])} 个、"
        f"缺失 {len(wt['anomalies']['missing'])} 处",
    ])
    st = tr.get("stats") or {}
    rep.add("指标数据", [
        "| 指标 | 数值 |", "|---|---|",
        f"| 区间点数 | {tr['n_points']} |",
        f"| 日均气温均值 | {_fmt(st.get('mean'), 4)} ℃ |",
        f"| 最高 | {_fmt(st.get('maximum'), 4)} ℃ |",
        f"| 最低 | {_fmt(st.get('minimum'), 4)} ℃ |",
        f"| 标准差 | {_fmt(st.get('std'), 4)} ℃ |",
    ])
    # 绘图用抽稀序列：trend.to_dict() 不带全量点，weather_trend 另外给出 chart_points
    pts = wt.get("chart_points") or []
    if pts:
        step = wt.get("chart_step") or 1
        rep.charts.append(ChartSpec(
            kind="line",
            title="日均气温趋势",
            x=[p["ts"][:10] for p in pts],
            series=[{"name": "日均气温", "data": [round(p["value"], 2) for p in pts]}],
            y_unit="℃",
            note=(f"为控制点数，按每 {step} 天抽样显示" if step > 1 else "全部数据点"),
        ))

    anom = wt["anomalies"]
    section = _anomaly_section((anom.get("threshold") or []) + anom["jump"],
                               with_kind=True, limit=8)
    if section:
        rep.add(*section)

    rep.add_source(f"fact_weather_daily（weather_id={wid}）")
    rep.add_caveat(
        "该站点为匿名化编号、无地理位置信息，非本项目实测区域，仅作演示/替代温度源"
    )
    return rep
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="报表生成：打印一份示例指标报告（JSON）")
    args = ap.parse_args()

    import indicators
    from intent_parser import parse

    _intent = parse("公司A今天的气耗指标是多少？", reference_date=datetime.date(2023, 12, 16))
    _rep = build_indicator_report(indicators.compute_from_intent(_intent))
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    print(json.dumps(_rep.to_dict(), ensure_ascii=False, indent=2))
