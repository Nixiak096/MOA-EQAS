"""统计分析、异常检测与对比分析。

面向四个数据域各自的可用能力（不硬造数据）：

    企业运营域  单日快照 + 多项目           -> 横向对比（项目间指标排名）
    气象域      8 站点逐时/逐日 2018-2024   -> 温度趋势、度日数累计、缺失检测
    换热站域    5 站逐小时 SCADA            -> 供回水温差工况、供水温度调节特性、工况偏离
    住户域      1408 户 逐日/逐月/累计      -> 分户能耗画像、逐月趋势、异常户识别

异常检测四类（对应课题任务的「异常检测」）：
    threshold  统计越限（均值 ± k × 标准差）
    jump       突变（相邻时点变化率超阈）
    missing    缺失（应有数据的时间点为空）
    deviation  工况偏离（换热站供回水温差越界）

数据不足时返回 available=False 并说明原因，不给假结论。

用法：
    python analytics.py --selftest
"""

import argparse
import datetime
import json
import logging
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sqlalchemy import text

import config
import heat_db

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("analytics")

ENERGY_LABEL = {"daily_gas_index": "日气耗指标值", "cumulative_gas_index": "累计气耗指标值",
                "daily_water_index": "日水耗指标值", "daily_elec_index": "日电耗指标值",
                "total_heating_area": "总供热面积", "daily_gas": "日气耗量",
                "cumulative_gas": "累计气耗量"}


@dataclass
class SeriesStats:
    n: int = 0
    mean: float = 0.0
    median: float = 0.0
    minimum: float = 0.0
    maximum: float = 0.0
    std: float = 0.0


@dataclass
class TrendResult:
    dimension: str = ""
    metric: str = ""
    label: str = ""
    unit: str = ""
    start: Optional[str] = None
    end: Optional[str] = None
    stats: Optional[SeriesStats] = None
    slope_per_unit: Optional[float] = None
    direction: str = ""
    points: List[Dict[str, Any]] = field(default_factory=list)
    available: bool = True
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "dimension": self.dimension, "metric": self.metric, "label": self.label,
            "unit": self.unit, "start": self.start, "end": self.end,
            "slope_per_unit": self.slope_per_unit, "direction": self.direction,
            "available": self.available, "reason": self.reason, "n_points": len(self.points),
        }
        if self.stats:
            d["stats"] = vars(self.stats)
        return d


@dataclass
class Anomaly:
    kind: str = ""            # threshold | jump | missing | deviation
    severity: str = ""        # 高 | 中 | 低
    ts: Optional[str] = None
    value: Optional[float] = None
    detail: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return vars(self)


@dataclass
class AnomalyResult:
    dimension: str = ""
    metric: str = ""
    label: str = ""
    anomalies: List[Anomaly] = field(default_factory=list)
    checked_points: int = 0
    available: bool = True
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dimension": self.dimension, "metric": self.metric, "label": self.label,
            "checked_points": self.checked_points, "available": self.available,
            "reason": self.reason, "count": len(self.anomalies),
            "anomalies": [a.to_dict() for a in self.anomalies],
        }


@dataclass
class CompareResult:
    metric: str = ""
    label: str = ""
    unit: str = ""
    stat_date: Optional[str] = None
    entries: List[Dict[str, Any]] = field(default_factory=list)
    max: Optional[float] = None
    min: Optional[float] = None
    mean: Optional[float] = None
    max_entity: Optional[str] = None
    min_entity: Optional[str] = None
    available: bool = True
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric": self.metric, "label": self.label, "unit": self.unit,
            "stat_date": self.stat_date, "available": self.available,
            "reason": self.reason, "n_entries": len(self.entries),
            "entries": self.entries,
            "max": self.max, "min": self.min, "mean": self.mean,
            "max_entity": self.max_entity, "min_entity": self.min_entity,
        }


# ============================================================================
# 取数层
# ============================================================================


class DataReader:
    def __init__(self, engine=None):
        self._engine = engine or heat_db.get_engine()
        self._conn = None

    def __enter__(self) -> "DataReader":
        self._conn = self._engine.connect()
        return self

    def __exit__(self, *exc: Any) -> None:
        if self._conn is not None:
            self._conn.close()
        self._engine.dispose()

    # ---- 实体 ----

    def list_entities(self, level: str = "project",
                      company: str = "公司A") -> List[Tuple[str, str]]:
        rows = self._conn.execute(
            text(
                "SELECT entity_id, display_name FROM dim_entity "
                "WHERE entity_level=:lv AND (company=:c OR :c IS NULL) "
                "ORDER BY display_name"
            ),
            {"lv": level, "c": company if level != "group" else None},
        ).all()
        return [(r[0], r[1]) for r in rows]

    # ---- 时序通用 ----

    def _series_rows(self, sql: str, params: Dict[str, Any], start: Optional[Any] = None,
                     end: Optional[Any] = None, ts_col: str = "ts") -> List[Any]:
        """取时序行：按需追加时间区间条件，并按时间排序。

        气象 / 换热站 / 住户三处查询此前各自重写这段区间拼装，现收在此处。
        """
        p = dict(params)
        if start:
            sql += f" AND {ts_col}>=:r_start"
            p["r_start"] = start
        if end:
            sql += f" AND {ts_col}<=:r_end"
            p["r_end"] = end
        return self._conn.execute(text(sql + f" ORDER BY {ts_col}"), p).all()

    # ---- 企业运营 ----

    def energy_series(self, entity_id: str, metric_code: str) -> List[Tuple[Any, float]]:
        rows = self._conn.execute(
            text(
                "SELECT stat_date, value FROM fact_energy_daily "
                "WHERE entity_id=:e AND metric_code=:m AND value IS NOT NULL "
                "ORDER BY stat_date"
            ),
            {"e": entity_id, "m": metric_code},
        ).all()
        return [(r[0], float(r[1])) for r in rows]

    def energy_at(self, entity_id: str, stat_date: datetime.date,
                  metric_code: str) -> Optional[float]:
        row = self._conn.execute(
            text(
                "SELECT value FROM fact_energy_daily "
                "WHERE entity_id=:e AND stat_date=:d AND metric_code=:m"
            ),
            {"e": entity_id, "d": stat_date, "m": metric_code},
        ).first()
        return float(row[0]) if row and row[0] is not None else None

    def latest_stat_date(self) -> Optional[datetime.date]:
        row = self._conn.execute(
            text("SELECT MAX(stat_date) FROM fact_energy_daily")
        ).first()
        return row[0] if row and row[0] is not None else None

    # ---- 气象 ----

    def weather_daily(self, weather_id: str, start: Optional[datetime.date] = None,
                      end: Optional[datetime.date] = None) -> List[Tuple[Any, Dict[str, float]]]:
        rows = self._series_rows(
            "SELECT ts, temperature_avg, hdd_sia, hdd_us, temperature_max, temperature_min "
            "FROM fact_weather_daily WHERE weather_id=:w",
            {"w": weather_id}, start, end)
        return [(r[0], {"temperature_avg": r[1], "hdd_sia": r[2], "hdd_us": r[3],
                        "temperature_max": r[4], "temperature_min": r[5]}) for r in rows]

    def weather_range(self, weather_id: str) -> Tuple[Optional[Any], Optional[Any]]:
        row = self._conn.execute(
            text("SELECT MIN(ts), MAX(ts) FROM fact_weather_daily WHERE weather_id=:w"),
            {"w": weather_id},
        ).first()
        return (row[0], row[1]) if row else (None, None)

    def list_stations(self) -> List[str]:
        rows = self._conn.execute(
            text("SELECT weather_id FROM dim_weather_station ORDER BY weather_id")
        ).all()
        return [r[0] for r in rows]

    # ---- 换热站 ----

    def scada_series(self, substation_id: str, start: Optional[Any] = None,
                     end: Optional[Any] = None) -> List[Dict[str, Any]]:
        rows = self._series_rows(
            "SELECT ts, t_amb, t_sup_prim, t_ret_prim, t_sup_sec, t_ret_sec, delta_e "
            "FROM fact_scada_hourly WHERE substation_id=:s",
            {"s": substation_id}, start, end)
        return [
            {"ts": r[0], "t_amb": r[1], "t_sup_prim": r[2], "t_ret_prim": r[3],
             "t_sup_sec": r[4], "t_ret_sec": r[5], "delta_e": r[6]}
            for r in rows
        ]

    def scada_range(self, substation_id: str) -> Tuple[Optional[Any], Optional[Any]]:
        row = self._conn.execute(
            text("SELECT MIN(ts), MAX(ts) FROM fact_scada_hourly WHERE substation_id=:s"),
            {"s": substation_id},
        ).first()
        return (row[0], row[1]) if row else (None, None)

    def list_substations(self) -> List[str]:
        rows = self._conn.execute(
            text("SELECT substation_id FROM dim_substation ORDER BY substation_id")
        ).all()
        return [r[0] for r in rows]

    # ---- 住户 ----

    def household_meter(self, household_id: str, resolution: str = "daily",
                        start: Optional[Any] = None,
                        end: Optional[Any] = None) -> List[Tuple[Any, float, Optional[float]]]:
        rows = self._series_rows(
            "SELECT ts, kwh_received_total, kwh_received_heatpump "
            "FROM fact_household_meter WHERE household_id=:h AND resolution=:r",
            {"h": household_id, "r": resolution}, start, end)
        return [(r[0], r[1], r[2]) for r in rows]

    def household_info(self, household_id: str) -> Optional[Dict[str, Any]]:
        row = self._conn.execute(
            text(
                "SELECT household_id, group_label, weather_id, building_type, living_area, "
                "residents, heatpump_type, floor_heating, has_pv, smd_daily_available "
                "FROM dim_household WHERE household_id=:h"
            ),
            {"h": household_id},
        ).first()
        if not row:
            return None
        keys = ["household_id", "group_label", "weather_id", "building_type",
                "living_area", "residents", "heatpump_type", "floor_heating",
                "has_pv", "smd_daily_available"]
        return dict(zip(keys, row))

    def household_top(self, limit: int = 10, resolution: str = "daily") -> List[Dict[str, Any]]:
        """按日均用电量排名（单条 SQL 聚合，避免逐户取数）。"""
        rows = self._conn.execute(
            text(
                "SELECT household_id, COUNT(*) n_days, AVG(kwh_received_total) avg_kwh "
                "FROM fact_household_meter WHERE resolution=:r AND kwh_received_total IS NOT NULL "
                "GROUP BY household_id HAVING n_days >= 30 "
                "ORDER BY avg_kwh DESC LIMIT :lim"
            ),
            {"r": resolution, "lim": limit},
        ).all()
        return [{"household_id": r[0], "n_days": r[1], "avg_kwh": float(r[2])} for r in rows]


# ============================================================================
# 统计与趋势
# ============================================================================


def _stats(values: Sequence[float]) -> SeriesStats:
    return SeriesStats(
        n=len(values),
        mean=float(np.mean(values)),
        median=float(np.median(values)),
        minimum=float(np.min(values)),
        maximum=float(np.max(values)),
        std=float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
    )


def trend_from_points(dimension: str, metric: str, label: str,
                      points: List[Tuple[Any, float]], unit: str = "") -> TrendResult:
    """对时序做趋势判断：最小二乘斜率 + 方向。"""
    res = TrendResult(dimension=dimension, metric=metric, label=label, unit=unit)
    if not points:
        res.available = False
        res.reason = "无数据点"
        return res
    if len(points) < 3:
        res.available = False
        res.reason = f"数据点仅 {len(points)} 个（<3），不足以判断趋势"
        return res

    xs = np.arange(len(points), dtype=float)
    ys = np.array([p[1] for p in points], dtype=float)
    slope = float(np.polyfit(xs, ys, 1)[0])
    res.stats = _stats(ys.tolist())
    res.slope_per_unit = slope
    res.start = str(points[0][0])
    res.end = str(points[-1][0])
    res.points = [{"ts": str(t), "value": v} for t, v in points]

    # 方向：斜率相对均值足够小则判平稳（阈值 0.5%/点）
    ref = abs(res.stats.mean) if res.stats.mean else 1.0
    rel = abs(slope) / ref
    if rel < 0.005:
        res.direction = "平稳"
    else:
        res.direction = "上升" if slope > 0 else "下降"
    return res


# ============================================================================
# 异常检测
# ============================================================================


def _sample_spread(items: List[Any], n: int) -> List[Any]:
    """在序列上等间隔取样，避免只取到开头一段（如异常集中在起始日）。"""
    if len(items) <= n:
        return items
    step = len(items) / n
    return [items[int(i * step)] for i in range(n)]


def detect_threshold(points: List[Tuple[Any, float]], sigmas: Optional[float] = None,
                     label: str = "") -> List[Anomaly]:
    """统计越限：超出均值 ± k×标准差。"""
    k = config.ANOMALY_SIGMA if sigmas is None else sigmas
    if len(points) < 5:
        return []
    vals = [p[1] for p in points]
    mean = float(np.mean(vals))
    std = float(np.std(vals, ddof=1))
    if std == 0:
        return []
    out: List[Anomaly] = []
    for ts, v in points:
        dev = abs(v - mean) / std
        if dev > k:
            sev = "高" if dev > k + 1 else "中"
            out.append(Anomaly(
                kind="threshold", severity=sev, ts=str(ts), value=v,
                detail=f"{label or '数值'} {v:.4g} 偏离均值 {mean:.4g} 达 {dev:.2f}σ（阈值 {k}σ）",
            ))
    return out


def detect_jump(points: List[Tuple[Any, float]], pct: Optional[float] = None,
                label: str = "", abs_delta: Optional[float] = None,
                unit: str = "") -> List[Anomaly]:
    """突变：相邻时点变化超阈。

    pct 为相对变化率阈值（适合能耗类）；abs_delta 为绝对变化阈值（适合气温类，
    因为气温的日间自然波动常超 30%，用相对率会大量误报）。
    """
    thr = config.ANOMALY_JUMP_PCT if pct is None else pct
    if len(points) < 2:
        return []
    out: List[Anomaly] = []
    for (t0, v0), (t1, v1) in zip(points, points[1:]):
        if abs_delta is not None:
            chg_abs = v1 - v0
            if abs(chg_abs) <= abs_delta:
                continue
            sev = "高" if abs(chg_abs) > 2 * abs_delta else "中"
            out.append(Anomaly(
                kind="jump", severity=sev, ts=str(t1), value=v1,
                detail=f"{label or '数值'}由 {v0:.4g}{unit} 变为 {v1:.4g}{unit}，"
                       f"变化 {chg_abs:+.2f}{unit}（绝对阈值 ±{abs_delta}{unit}）",
            ))
            continue
        if v0 == 0:
            continue
        chg = (v1 - v0) / abs(v0) * 100
        if abs(chg) > thr:
            sev = "高" if abs(chg) > 2 * thr else "中"
            out.append(Anomaly(
                kind="jump", severity=sev, ts=str(t1), value=v1,
                detail=f"{label or '数值'}由 {v0:.4g} 变为 {v1:.4g}，变化 {chg:+.2f}%（阈值 ±{thr}%）",
            ))
    return out


def detect_missing(points: List[Tuple[Any, float]], expected_gap_days: int = 1,
                   label: str = "") -> List[Anomaly]:
    """缺失：相邻数据点间隔超过期望间隔（仅适用于日期型序列）。"""
    if len(points) < 2:
        return []
    out: List[Anomaly] = []
    for (t0, _), (t1, _) in zip(points, points[1:]):
        try:
            d0 = t0.date() if hasattr(t0, "date") else t0
            d1 = t1.date() if hasattr(t1, "date") else t1
            gap = (d1 - d0).days
        except (AttributeError, TypeError):
            continue
        if gap > expected_gap_days:
            out.append(Anomaly(
                kind="missing", severity="中", ts=str(t1), value=None,
                detail=f"{label or '序列'}在 {d0} 与 {d1} 之间缺 {gap - expected_gap_days} 天数据",
            ))
    return out


def scada_delta_t_anomalies(rows: List[Dict[str, Any]],
                            t_min: Optional[float] = None,
                            t_max: Optional[float] = None) -> List[Anomaly]:
    """工况偏离：二次侧供回水温差越界（过低疑水力失调，过高疑流量不足）。"""
    lo = config.SCADA_DELTA_T_MIN if t_min is None else t_min
    hi = config.SCADA_DELTA_T_MAX if t_max is None else t_max
    out: List[Anomaly] = []
    for r in rows:
        sup, ret = r.get("t_sup_sec"), r.get("t_ret_sec")
        if sup is None or ret is None:
            continue
        dt = sup - ret
        if dt < lo:
            out.append(Anomaly(kind="deviation", severity="中", ts=str(r["ts"]), value=dt,
                               detail=f"二次侧供回水温差 {dt:.2f}℃，低于 {lo}℃ — 疑水力失调/大流量小温差"))
        elif dt > hi:
            out.append(Anomaly(kind="deviation", severity="高", ts=str(r["ts"]), value=dt,
                               detail=f"二次侧供回水温差 {dt:.2f}℃，高于 {hi}℃ — 疑流量不足"))
    return out


# ============================================================================
# 企业运营：多项目横向对比
# ============================================================================


def compare_projects(metric_code: str = "daily_gas_index",
                     company: str = "公司A",
                     stat_date: Optional[datetime.date] = None) -> CompareResult:
    label = ENERGY_LABEL.get(metric_code, metric_code)
    res = CompareResult(metric=metric_code, label=label, unit="按源单位")
    with DataReader() as dr:
        if stat_date is None:
            stat_date = dr.latest_stat_date()
        if stat_date is None:
            res.available = False
            res.reason = "库中无企业运营数据"
            return res
        res.stat_date = stat_date.isoformat()
        entries = []
        for eid, name in dr.list_entities("project", company):
            v = dr.energy_at(eid, stat_date, metric_code)
            if v is not None:
                entries.append({"entity_id": eid, "label": name, "value": v})
    if not entries:
        res.available = False
        res.reason = f"{stat_date} 各项目均无「{label}」数据"
        return res
    res.entries = sorted(entries, key=lambda e: e["value"], reverse=True)
    vals = [e["value"] for e in res.entries]
    res.max = max(vals)
    res.min = min(vals)
    res.mean = sum(vals) / len(vals)
    res.max_entity = res.entries[0]["label"]
    res.min_entity = res.entries[-1]["label"]
    return res


# ============================================================================
# 气象：趋势 + 度日数累计 + 缺失
# ============================================================================


def weather_trend(weather_id: str, start: Optional[datetime.date] = None,
                  end: Optional[datetime.date] = None) -> Dict[str, Any]:
    """气象趋势。返回值里的 chart_points 是**抽稀后**的绘图序列。

    TrendResult.to_dict() 刻意不带全量 points（上万点会让接口响应过大），
    所以这里单独给出抽稀序列供图表使用；抽稀步长一并返回，便于如实标注。
    """
    with DataReader() as dr:
        lo, hi = dr.weather_range(weather_id)
        if lo is None:
            return {"available": False, "reason": f"气象站 {weather_id} 无数据"}
        data = dr.weather_daily(weather_id, start, end)
    if not data:
        return {"available": False, "reason": "指定区间无气象数据"}
    pts = [(t, d["temperature_avg"]) for t, d in data if d["temperature_avg"] is not None]
    tr = trend_from_points(f"weather:{weather_id}", "temperature_avg",
                           f"{weather_id} 日均气温", pts, unit="℃")
    hdd = sum(d["hdd_sia"] for _, d in data if d["hdd_sia"] is not None)
    step = max(1, len(pts) // 120)
    return {
        "available": True,
        "weather_id": weather_id,
        "data_range": [str(lo), str(hi)],
        "trend": tr.to_dict(),
        "chart_points": [{"ts": str(t), "value": v} for t, v in pts[::step]],
        "chart_step": step,
        "hdd_sia_sum": hdd,
        "anomalies": {
            "threshold": [a.to_dict() for a in detect_threshold(pts, label="日均气温")],
            # 气温用绝对变化阈值：日间自然波动常超 30%，相对率会大量误报
            "jump": [a.to_dict() for a in detect_jump(
                pts, label="日均气温", abs_delta=config.ANOMALY_TEMP_JUMP_C, unit="℃")],
            "missing": [a.to_dict() for a in detect_missing(
                [(t, 0.0) for t, _ in data], expected_gap_days=1, label="气温序列")],
        },
    }


# ============================================================================
# 换热站：温差工况 + 供水温度调节特性 + 工况异常
# ============================================================================


def scada_analysis(substation_id: str, start: Optional[Any] = None,
                   end: Optional[Any] = None) -> Dict[str, Any]:
    with DataReader() as dr:
        lo, hi = dr.scada_range(substation_id)
        if lo is None:
            return {"available": False, "reason": f"换热站 {substation_id} 无 SCADA 数据"}
        rows = dr.scada_series(substation_id, start, end)
    if not rows:
        return {"available": False, "reason": "指定区间无 SCADA 数据"}

    deltas = [(r["ts"], r["t_sup_sec"] - r["t_ret_sec"])
              for r in rows if r["t_sup_sec"] is not None and r["t_ret_sec"] is not None]

    # 供水温度随室外温度的调节特性：按室外温度分箱取二次侧供水温度均值
    bins: Dict[int, List[float]] = {}
    for r in rows:
        if r["t_amb"] is None or r["t_sup_sec"] is None:
            continue
        b = int(np.floor(r["t_amb"] / 5.0) * 5)
        bins.setdefault(b, []).append(r["t_sup_sec"])
    curve = [
        {"t_amb_bin": b, "t_sup_sec_mean": float(np.mean(v)), "n": len(v)}
        for b, v in sorted(bins.items()) if len(v) >= 5
    ]
    # 正常调节特性：室外温度越低，二次侧供水温度越高（负相关）
    curve_relation = None
    if len(curve) >= 3:
        xs = np.array([c["t_amb_bin"] for c in curve], dtype=float)
        ys = np.array([c["t_sup_sec_mean"] for c in curve], dtype=float)
        curve_relation = float(np.corrcoef(xs, ys)[0, 1])

    deviations = scada_delta_t_anomalies(rows)
    return {
        "available": True,
        "substation_id": substation_id,
        "data_range": [str(lo), str(hi)],
        "n_points": len(rows),
        "delta_t": {
            "stats": vars(_stats([v for _, v in deltas])) if deltas else None,
            "unit": "℃",
            "reasonable_range": [config.SCADA_DELTA_T_MIN, config.SCADA_DELTA_T_MAX],
        },
        "supply_temp_curve": curve,
        "curve_relation": curve_relation,
        "curve_interpretation": (
            "室外温度越低、二次侧供水温度越高，符合正常运行调节特性"
            if curve_relation is not None and curve_relation < -0.3 else
            "供水温度与室外温度未见明显负相关，建议核查调节曲线设置"
            if curve_relation is not None else "数据不足，无法判断调节特性"
        ),
        "deviation_anomalies": {
            "count": len(deviations),
            "ratio": round(len(deviations) / len(rows) * 100, 2) if rows else 0,
            "samples": [a.to_dict() for a in _sample_spread(deviations, 10)],
        },
    }


# ============================================================================
# 住户：分户画像 + 趋势 + 异常
# ============================================================================


def _to_date(t: Any) -> Any:
    return t.date() if hasattr(t, "date") else t


def household_analysis(household_id: str, resolution: str = "daily") -> Dict[str, Any]:
    with DataReader() as dr:
        info = dr.household_info(household_id)
        rows = dr.household_meter(household_id, resolution=resolution)
        weather_rows = []
        wid = (info or {}).get("weather_id")
        if wid:
            weather_rows = dr.weather_daily(wid)   # 该户所属气象站逐日度日数
    if info is None:
        return {"available": False, "reason": f"住户 {household_id} 不存在"}
    if not rows:
        return {"available": False, "reason": f"住户 {household_id} 无 {resolution} 电表数据",
                "household": info}

    pts = [(t, v) for t, v, _ in rows if v is not None]
    hp_pts = [(t, v) for t, _, v in rows if v is not None]
    tr = trend_from_points(f"household:{household_id}", resolution,
                           f"{household_id} 用电量", pts, unit="kWh")

    # ---- 温度归一化：单位度日数用电量（kWh/℃·日）----
    # 供热能耗有强季节性，直接对原始序列做统计越限会把每个采暖季判为异常；
    # 用所属气象站的 SIA 供暖度日数归一化后再检测，才是可比口径。
    hdd_by_date = {_to_date(t): d.get("hdd_sia")
                   for t, d in weather_rows if d.get("hdd_sia") is not None}
    norm_pts: List[Tuple[Any, float]] = []
    hdd_floor = config.HDD_MIN_FOR_NORMALIZATION
    for t, v in pts:
        hdd = hdd_by_date.get(_to_date(t))
        if hdd is not None and hdd >= hdd_floor:   # 分母过小会放大成假异常
            norm_pts.append((t, v / hdd))
    norm_tr = (trend_from_points(f"household:{household_id}", resolution,
                                 f"{household_id} 单位度日数用电量", norm_pts,
                                 unit="kWh/℃·日") if norm_pts else None)

    # 逐月聚合
    by_month: Dict[str, List[float]] = {}
    for t, v in pts:
        by_month.setdefault(str(t)[:7], []).append(v)
    monthly = [{"month": m, "kwh_total": sum(vs), "n_days": len(vs)}
               for m, vs in sorted(by_month.items())]

    living_area = info.get("living_area")
    return {
        "available": True,
        "household": info,
        "n_points": len(pts),
        "trend": tr.to_dict(),
        "monthly": monthly[-12:],
        "stats": vars(_stats([v for _, v in pts])) if pts else None,
        "heatpump_stats": vars(_stats([v for _, v in hp_pts])) if hp_pts else None,
        "kwh_per_m2_per_year": (
            sum(v for _, v in pts) / living_area * 365 / len(pts)
            if living_area and pts else None
        ),
        "hdd_normalized": {
            "weather_id": wid,
            "n_points": len(norm_pts),
            "trend": norm_tr.to_dict() if norm_tr else None,
            "stats": vars(_stats([v for _, v in norm_pts])) if norm_pts else None,
            "unit": "kWh/℃·日",
        },
        "anomalies": {
            # 原始口径（含季节性，仅供参考）
            "threshold": [a.to_dict() for a in
                          _sample_spread(detect_threshold(pts, label="日用电量"), 8)],
            "jump": [a.to_dict() for a in
                     _sample_spread(detect_jump(pts, label="日用电量"), 8)],
            "missing": [a.to_dict() for a in
                        _sample_spread(detect_missing(
                            [(t, 0.0) for t, _ in pts], expected_gap_days=1,
                            label="日用电量"), 8)],
            # 温度归一化口径（剔除季节性后再判异常，结论更可信）
            "threshold_normalized": [a.to_dict() for a in
                                     _sample_spread(detect_threshold(
                                         norm_pts, label="单位度日数用电量"), 8)],
        },
    }


# ============================================================================
# 企业运营：异常检测（数据不足时如实说明）
# ============================================================================


def enterprise_anomalies(entity_id: str, metric_code: str) -> AnomalyResult:
    label = ENERGY_LABEL.get(metric_code, metric_code)
    res = AnomalyResult(dimension=entity_id, metric=metric_code, label=label)
    with DataReader() as dr:
        pts = dr.energy_series(entity_id, metric_code)
    res.checked_points = len(pts)
    if len(pts) < 5:
        res.available = False
        res.reason = (f"库中该实体仅 {len(pts)} 个日期点（<5），"
                      "异常检测需多日序列；当前企业运营数据为单日快照")
        return res
    res.anomalies = (detect_threshold(pts, label=label)
                     + detect_jump(pts, label=label)
                     + detect_missing(pts, expected_gap_days=1, label=label))
    return res
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="统计分析 / 异常检测 / 对比分析（命令行查看某个域的原始分析结果）")
    ap.add_argument("--json", help="输出指定分析的 JSON：weather:<id> | scada:<id> | household:<id>")
    args = ap.parse_args()

    if not args.json:
        ap.print_help()
        raise SystemExit(0)

    kind, _, key = args.json.partition(":")
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    _ANALYSES = {"weather": weather_trend, "scada": scada_analysis,
                 "household": household_analysis}
    fn = _ANALYSES.get(kind)
    if fn is None:
        print("未知类型，用法：--json weather:8jB | scada:L4 | household:<id>")
        raise SystemExit(1)
    print(json.dumps(fn(key), ensure_ascii=False, indent=2, default=str))
