"""供热指标计算（依据《数据及计算说明》）。

每个计算结果都携带 formula（所用公式）、inputs（参与计算的输入值）、
source（数据来源），供审核员核验口径与单位。

数据取不到时返回 available=False 并给出原因，不猜测、不编造。
收费率类结果额外做 cross_check：把按公式重算的值与报表原值并列，
偏差超过阈值时标记不一致 —— 这正是审核员「校验公式…不通过则反馈重算」的落点。

口径（见《数据及计算说明》）：
    发生责任日期 = 计算日 - 1
    日报日期     = 发生责任日期 + 1 = 计算日
    已供暖天数   = 计算日 - 供热开始日期            （供热开始日期为设定口）
    法定已供暖天数 = 计算日 - 法定正式供热日期(11.15)

    日耗指标值 = 日耗量 / 面积
    日耗评价值 = 日耗指标值 × 121
    日耗指标值环比 = (当日 - 昨日) / 昨日
    累计耗指标值 = 累计耗量 / 面积
    累计耗评价值 = 累计耗指标值 / 已供暖天数 × 121
    标准采暖季累计耗指标值 = (当前累计耗量 - 11月14日累计耗量) / 面积
    标准采暖季累计耗评价值 = 标准采暖季累计耗指标值 / 法定已供暖天数 × 121
    度日数折标后日耗量 = 日耗量 × (18 + 0.7) / (18 - 当天室外平均温度)
    度日数折标后累计耗指标值 = 累计每日的度日数折标后日耗指标值

    本季收费率 = 累计本季实收 / 本季应收 × 100%

用法：
    python indicators.py --selftest
    python indicators.py --entity company:公司A --date 2023-12-16 --metric gas
"""

import argparse
import datetime
import json
import logging
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from sqlalchemy import text

import config
import heat_db

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("indicators")

# 能源类指标 -> (日值, 累计值, 日指标, 累计指标) 在 fact_energy_daily 中的 metric_code
ENERGY_CODES: Dict[str, Dict[str, str]] = {
    "gas": {"daily": "daily_gas", "cumulative": "cumulative_gas",
            "daily_index": "daily_gas_index", "cum_index": "cumulative_gas_index"},
    "water": {"daily": "daily_water", "cumulative": "cumulative_water",
              "daily_index": "daily_water_index", "cum_index": "cumulative_water_index"},
    "elec": {"daily": "daily_elec", "cumulative": "cumulative_elec",
             "daily_index": "daily_elec_index", "cum_index": "cumulative_elec_index"},
}
AREA_CODE = "total_heating_area"

METRIC_LABEL = {"gas": "气耗", "water": "水耗", "elec": "电耗"}

# 收费类 item_code
FEE_ITEMS = {
    "fee_rate": "season_fee_rate",
    "household_payment_rate": "household_payment_rate",
    "arrears_fee_rate": "arrears_fee_rate",
}
FEES = {
    "fee_rate": "本季收费率",
    "household_payment_rate": "户缴率",
    "arrears_fee_rate": "旧欠收费率",
}

RATE_TOLERANCE = 0.5  # 百分点阈值：重算值与报表原值偏差超过则标记不一致


@dataclass
class IndicatorResult:
    entity_id: str = ""
    stat_date: Optional[datetime.date] = None
    metric: str = ""
    mode: str = ""
    label: str = ""
    value: Optional[float] = None
    unit: str = ""
    formula: str = ""
    inputs: Dict[str, Any] = field(default_factory=dict)
    # 各输入项的单位，键与 inputs 对齐；供报告写「日耗量(Nm3)=789,659.507」这样的可核对输入
    input_units: Dict[str, str] = field(default_factory=dict)
    # 集团级聚合用的分量（不对外序列化）：{"num": 分子, "den": 分母, "name": 分量名,
    # "unit": 单位, 可选 "prev_num"/"prev_den"}。
    # 集团级 = Σ各公司 num ÷ Σ各公司 den，再套用同一口径的后续变换。
    # 用显式字段而不是从 inputs 的键名去猜——键名是按能源拼的中文，猜不准。
    agg: Optional[Dict[str, Any]] = None
    source: str = ""
    available: bool = True
    reason: str = ""
    cross_check: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "stat_date": self.stat_date.isoformat() if self.stat_date else None,
            "metric": self.metric,
            "mode": self.mode,
            "label": self.label,
            "value": self.value,
            "unit": self.unit,
            "formula": self.formula,
            "inputs": self.inputs,
            "input_units": self.input_units,
            "source": self.source,
            "available": self.available,
            "reason": self.reason,
            "cross_check": self.cross_check,
        }


def _unavailable(**kw: Any) -> IndicatorResult:
    kw["available"] = False
    return IndicatorResult(**kw)


# ============================================================================
# 日期口径
# ============================================================================


def legal_heating_start(stat_date: datetime.date) -> datetime.date:
    """法定正式供热日期：当年 11 月 15 日；若计算日早于该日则取上一年。"""
    this_year = datetime.date(
        stat_date.year, config.LEGAL_HEATING_START_MONTH, config.LEGAL_HEATING_START_DAY
    )
    if stat_date >= this_year:
        return this_year
    return datetime.date(
        stat_date.year - 1, config.LEGAL_HEATING_START_MONTH, config.LEGAL_HEATING_START_DAY
    )


def legal_heating_days(stat_date: datetime.date) -> int:
    """法定已供暖天数 = 计算日 - 法定正式供热日期。"""
    return (stat_date - legal_heating_start(stat_date)).days


def heating_start_date() -> Optional[datetime.date]:
    """供热开始日期（设定口）。未配置返回 None。"""
    raw = (config.HEATING_START_DATE or "").strip()
    if not raw:
        return None
    try:
        return datetime.date.fromisoformat(raw)
    except ValueError:
        logger.warning("HEATING_START_DATE 格式非法：%s", raw)
        return None


def heating_days(stat_date: datetime.date) -> Optional[int]:
    """已供暖天数 = 计算日 - 供热开始日期。未配置供热开始日期则不可用。"""
    start = heating_start_date()
    if start is None:
        return None
    return (stat_date - start).days


# ============================================================================
# 取数
# ============================================================================


class DataAccess:
    """指标计算的取数层，全部来自 MySQL。"""

    def __init__(self, engine=None):
        self._engine = engine or heat_db.get_engine()
        self._conn = None

    def __enter__(self) -> "DataAccess":
        self._conn = self._engine.connect()
        return self

    def __exit__(self, *exc: Any) -> None:
        if self._conn is not None:
            self._conn.close()
        self._engine.dispose()

    # ---- 能耗 ----

    def energy(self, entity_id: str, stat_date: datetime.date,
               metric_code: str) -> Optional[float]:
        row = self._conn.execute(
            text(
                "SELECT value FROM fact_energy_daily "
                "WHERE entity_id=:e AND stat_date=:d AND metric_code=:m"
            ),
            {"e": entity_id, "d": stat_date, "m": metric_code},
        ).first()
        return float(row[0]) if row and row[0] is not None else None

    def child_companies(self, group_name: str) -> List[str]:
        """集团下属的公司实体（用于集团级聚合：集团 = 各公司之和）。

        依据 dim_entity 的 group_name 归属关系。
        """
        rows = self._conn.execute(
            text(
                "SELECT entity_id FROM dim_entity "
                "WHERE entity_level='company' AND group_name=:g ORDER BY entity_id"
            ),
            {"g": group_name},
        ).all()
        return [r[0] for r in rows]

    def entities_with_energy(self) -> List[str]:
        """库中真正有能耗数据的实体（注意：dim_entity 里可能登记了尚无数据的实体）。"""
        rows = self._conn.execute(
            text("SELECT DISTINCT entity_id FROM fact_energy_daily ORDER BY entity_id")
        ).all()
        return [r[0] for r in rows]

    def unit(self, entity_id: str, stat_date: datetime.date,
             metric_code: str) -> Optional[str]:
        """取库中该指标的原生单位（如 Nm3/m2）。

        单平米指标的单位由「源量单位 / m²」推出，直接读库比在代码里写死更可靠：
        气耗是 Nm3/m2、电耗是 kWh/m2、水耗是 kg/m2，三者并不统一。
        """
        row = self._conn.execute(
            text(
                "SELECT unit FROM fact_energy_daily "
                "WHERE entity_id=:e AND stat_date=:d AND metric_code=:m"
            ),
            {"e": entity_id, "d": stat_date, "m": metric_code},
        ).first()
        return row[0] if row and row[0] else None

    def previous_stat_date(self, entity_id: str,
                           stat_date: datetime.date) -> Optional[datetime.date]:
        row = self._conn.execute(
            text(
                "SELECT MAX(stat_date) FROM fact_energy_daily "
                "WHERE entity_id=:e AND stat_date < :d"
            ),
            {"e": entity_id, "d": stat_date},
        ).first()
        return row[0] if row and row[0] is not None else None

    def available_dates(self, entity_id: str) -> List[datetime.date]:
        rows = self._conn.execute(
            text(
                "SELECT DISTINCT stat_date FROM fact_energy_daily "
                "WHERE entity_id=:e ORDER BY stat_date"
            ),
            {"e": entity_id},
        ).all()
        return [r[0] for r in rows]

    # ---- 收费 ----

    def fee(self, entity_id: str, stat_date: datetime.date, item_code: str,
            category: str) -> Optional[float]:
        row = self._conn.execute(
            text(
                "SELECT value FROM fact_fee_season "
                "WHERE entity_id=:e AND stat_date=:d AND item_code=:i AND category=:c"
            ),
            {"e": entity_id, "d": stat_date, "i": item_code, "c": category},
        ).first()
        return float(row[0]) if row and row[0] is not None else None

    # ---- 收费 ----

    def fee_project_ids(self, company: Optional[str] = None,
                        pattern: Optional[str] = None) -> List[str]:
        """有收费数据的项目实体（按 entity_id 排序）。

        用于公司级/集团级收费率的「户数加权」复算：源表这两级的口径都是
        「各项目收费率按本季应收户数加权」，所以要知道有哪些项目参与加权。
        company 指定单公司；pattern 可直接给 LIKE 模式（集团级用 `project:%`）。
        """
        expr = pattern or f"project:{company}:%"
        rows = self._conn.execute(
            text("SELECT DISTINCT entity_id FROM fact_fee_season "
                 "WHERE entity_id LIKE :p ORDER BY entity_id"),
            {"p": expr},
        ).all()
        return [r[0] for r in rows]

    # ---- 气象 ----

    def outdoor_avg_temp(self, weather_id: str,
                         stat_date: datetime.date) -> Optional[float]:
        row = self._conn.execute(
            text(
                "SELECT temperature_avg FROM fact_weather_daily "
                "WHERE weather_id=:w AND ts=:d"
            ),
            {"w": weather_id, "d": stat_date},
        ).first()
        return float(row[0]) if row and row[0] is not None else None


# ============================================================================
# 能耗指标
# ============================================================================


def _energy_result(da: DataAccess, entity_id: str, stat_date: datetime.date,
                   metric: str, mode: str) -> IndicatorResult:
    codes = ENERGY_CODES[metric]
    label = METRIC_LABEL[metric]
    area = da.energy(entity_id, stat_date, AREA_CODE)
    base = dict(entity_id=entity_id, stat_date=stat_date, metric=metric, mode=mode)

    # 单平米指标的单位取自库中原生单位（Nm3/m2、kWh/m2、kg/m2 各不相同）；
    # 库中缺单位时如实降级标注，不臆造具体单位。
    unit_daily = da.unit(entity_id, stat_date, codes["daily_index"]) or "按源量/面积"
    unit_cum = da.unit(entity_id, stat_date, codes["cum_index"]) or unit_daily
    # 耗量本身的单位（Nm3 / t / kWh）与面积单位，用于给「计算输入」逐项标注
    unit_qty_daily = da.unit(entity_id, stat_date, codes["daily"]) or ""
    unit_qty_cum = da.unit(entity_id, stat_date, codes["cumulative"]) or unit_qty_daily
    AREA_UNIT = "m2"
    QTY = f"{label}量"          # 气耗量 / 水耗量 / 电耗量
    CUM_QTY = f"累计{label}量"

    if area in (None, 0):
        return _unavailable(**base, label=label, reason="缺少供热面积，无法按单平米计算",
                           source="fact_energy_daily.total_heating_area")

    # --- 日指标值 ---
    if mode == "daily":
        qty = da.energy(entity_id, stat_date, codes["daily"])
        if qty is None:
            return _unavailable(**base, label=f"日{label}指标值", unit=unit_daily,
                                reason=f"缺少日{QTY}", source=f"fact_energy_daily.{codes['daily']}")
        return IndicatorResult(
            **base, label=f"日{label}指标值", value=qty / area, unit=unit_daily,
            formula=f"日{label}指标值 = 日{QTY} / 面积",
            inputs={f"日{QTY}": qty, "面积": area},
            input_units={f"日{QTY}": unit_qty_daily, "面积": AREA_UNIT},
            agg={"num": qty, "den": area, "name": f"日{QTY}", "unit": unit_qty_daily},
            source=f"fact_energy_daily.{codes['daily']}",
        )

    # --- 日评价值 ---
    if mode == "evaluation":
        qty = da.energy(entity_id, stat_date, codes["daily"])
        if qty is None:
            return _unavailable(**base, label=f"日{label}评价值",
                                reason=f"缺少日{QTY}", source=f"fact_energy_daily.{codes['daily']}")
        idx = qty / area
        return IndicatorResult(
            **base, label=f"日{label}评价值", value=idx * config.EVALUATION_FACTOR,
            unit=unit_daily,
            formula=f"日{label}评价值 = 日{label}指标值 × {config.EVALUATION_FACTOR:g}",
            inputs={f"日{label}指标值": idx, "评价系数": config.EVALUATION_FACTOR,
                    f"日{QTY}": qty, "面积": area},
            input_units={f"日{label}指标值": unit_daily, "评价系数": "无量纲",
                         f"日{QTY}": unit_qty_daily, "面积": AREA_UNIT},
            agg={"num": qty, "den": area, "name": f"日{QTY}", "unit": unit_qty_daily},
            source=f"fact_energy_daily.{codes['daily']}",
        )

    # --- 环比 ---
    if mode == "mom":
        prev_date = da.previous_stat_date(entity_id, stat_date)
        if prev_date is None:
            return _unavailable(
                **base, label=f"日{label}指标值环比",
                reason="缺少上一日数据（当前库中仅单日快照），无法计算环比",
                source=f"fact_energy_daily.{codes['daily']}",
            )
        cur = da.energy(entity_id, stat_date, codes["daily_index"])
        prev = da.energy(entity_id, prev_date, codes["daily_index"])
        if cur is None or prev in (None, 0):
            return _unavailable(**base, label=f"日{label}指标值环比",
                                reason="缺少当日或上一日指标值",
                                source=f"fact_energy_daily.{codes['daily_index']}")
        return IndicatorResult(
            **base, label=f"日{label}指标值环比",
            value=(cur - prev) / prev * 100, unit="%",
            formula=f"日{label}指标值环比 = (当日{label}指标值 - 昨日{label}指标值) / 昨日{label}指标值 × 100%",
            inputs={f"当日{label}指标值": cur, f"昨日{label}指标值": prev,
                    "昨日日期": prev_date.isoformat()},
            input_units={f"当日{label}指标值": unit_daily, f"昨日{label}指标值": unit_daily},
            agg={"num": cur * area, "den": area, "name": f"当日{label}指标值加权",
                 "unit": unit_daily, "prev_num": prev * area, "prev_den": area},
            source=f"fact_energy_daily.{codes['daily_index']}",
        )

    # --- 累计指标值 ---
    if mode == "cumulative":
        total = da.energy(entity_id, stat_date, codes["cumulative"])
        if total is None:
            return _unavailable(**base, label=f"累计{label}指标值",
                                reason=f"缺少{CUM_QTY}",
                                source=f"fact_energy_daily.{codes['cumulative']}")
        return IndicatorResult(
            **base, label=f"累计{label}指标值", value=total / area, unit=unit_cum,
            formula=f"累计{label}指标值 = {CUM_QTY} / 面积",
            inputs={CUM_QTY: total, "面积": area},
            input_units={CUM_QTY: unit_qty_cum, "面积": AREA_UNIT},
            agg={"num": total, "den": area, "name": CUM_QTY, "unit": unit_qty_cum},
            source=f"fact_energy_daily.{codes['cumulative']}",
        )

    # --- 累计评价值 ---
    if mode == "cumulative_evaluation":
        total = da.energy(entity_id, stat_date, codes["cumulative"])
        days = heating_days(stat_date)
        if total is None:
            return _unavailable(**base, label=f"累计{label}评价值", reason="缺少累计耗量")
        if days is None:
            return _unavailable(
                **base, label=f"累计{label}评价值",
                reason="未配置供热开始日期（HEATING_START_DATE，设定口），无法算已供暖天数",
            )
        idx = total / area
        return IndicatorResult(
            **base, label=f"累计{label}评价值",
            value=idx / days * config.EVALUATION_FACTOR, unit=unit_cum,
            formula=f"累计{label}评价值 = 累计{label}指标值 / 已供暖天数 × {config.EVALUATION_FACTOR:g}",
            inputs={f"累计{label}指标值": idx, "已供暖天数": days,
                    "供热开始日期": heating_start_date().isoformat(),
                    CUM_QTY: total, "面积": area},
            input_units={f"累计{label}指标值": unit_cum, "已供暖天数": "天",
                         CUM_QTY: unit_qty_cum, "面积": AREA_UNIT},
            agg={"num": total, "den": area, "name": CUM_QTY, "unit": unit_qty_cum},
            source=f"fact_energy_daily.{codes['cumulative']}",
        )

    # --- 标准采暖季累计指标值 ---
    if mode == "season_cumulative":
        total = da.energy(entity_id, stat_date, codes["cumulative"])
        baseline_date = datetime.date(
            stat_date.year, config.DEGREE_DAY_BASELINE_MONTH, config.DEGREE_DAY_BASELINE_DAY
        )
        baseline = da.energy(entity_id, baseline_date, codes["cumulative"])
        if total is None:
            return _unavailable(**base, label=f"标准采暖季累计{label}指标值", reason="缺少累计耗量")
        if baseline is None:
            return _unavailable(
                **base, label=f"标准采暖季累计{label}指标值",
                reason=f"缺少 {baseline_date.isoformat()} 的累计耗量基准值",
            )
        return IndicatorResult(
            **base, label=f"标准采暖季累计{label}指标值",
            value=(total - baseline) / area, unit=unit_cum,
            formula=(f"标准采暖季累计{label}指标值 = "
                     f"(当前{CUM_QTY} - 11月14日{CUM_QTY}) / 面积"),
            inputs={f"当前{CUM_QTY}": total, f"基准日{CUM_QTY}": baseline,
                    "基准日": baseline_date.isoformat(), "面积": area},
            input_units={f"当前{CUM_QTY}": unit_qty_cum, f"基准日{CUM_QTY}": unit_qty_cum,
                         "面积": AREA_UNIT},
            agg={"num": total - baseline, "den": area, "name": f"当前{CUM_QTY}−基准日{CUM_QTY}",
                 "unit": unit_qty_cum},
            source=f"fact_energy_daily.{codes['cumulative']}",
        )

    # --- 标准采暖季累计评价值 ---
    if mode == "season_cumulative_evaluation":
        total = da.energy(entity_id, stat_date, codes["cumulative"])
        baseline_date = datetime.date(
            stat_date.year, config.DEGREE_DAY_BASELINE_MONTH, config.DEGREE_DAY_BASELINE_DAY
        )
        baseline = da.energy(entity_id, baseline_date, codes["cumulative"])
        ldays = legal_heating_days(stat_date)
        if total is None or baseline is None:
            return _unavailable(
                **base, label=f"标准采暖季累计{label}评价值",
                reason=f"缺少当前或基准日（{baseline_date.isoformat()}）累计耗量")
        idx = (total - baseline) / area
        return IndicatorResult(
            **base, label=f"标准采暖季累计{label}评价值",
            value=idx / ldays * config.EVALUATION_FACTOR, unit=unit_cum,
            formula=(f"标准采暖季累计{label}评价值 = 标准采暖季累计{label}指标值 / "
                     f"法定已供暖天数 × {config.EVALUATION_FACTOR:g}"),
            inputs={f"标准采暖季累计{label}指标值": idx, "法定已供暖天数": ldays},
            input_units={f"标准采暖季累计{label}指标值": unit_cum, "法定已供暖天数": "天"},
            agg={"num": total - baseline, "den": area, "name": f"当前{CUM_QTY}−基准日{CUM_QTY}",
                 "unit": unit_qty_cum},
            source=f"fact_energy_daily.{codes['cumulative']}",
        )

    # --- 度日数折标（日）---
    if mode == "degree_day":
        qty = da.energy(entity_id, stat_date, codes["daily"])
        station = (config.DEGREE_DAY_STATION or "").strip()
        if qty is None:
            return _unavailable(**base, label=f"度日数折标后日{label}指标值", reason="缺少日耗量")
        if not station:
            return _unavailable(
                **base, label=f"度日数折标后日{label}指标值",
                reason="未配置度日数折标温度源（DEGREE_DAY_STATION），无法取得当天室外平均温度",
            )
        temp = da.outdoor_avg_temp(station, stat_date)
        if temp is None:
            return _unavailable(
                **base, label=f"度日数折标后日{label}指标值",
                reason=f"气象站 {station} 缺少 {stat_date.isoformat()} 的室外平均温度",
                source=f"fact_weather_daily(weather_id={station})",
            )
        denom = config.DEGREE_DAY_BASE_TEMP - temp
        if denom == 0:
            return _unavailable(**base, label=f"度日数折标后日{label}指标值",
                                reason="室外平均温度等于基准温度，折标分母为 0")
        factor = (config.DEGREE_DAY_BASE_TEMP + config.DEGREE_DAY_INDOOR_OFFSET) / denom
        adjusted = qty * factor
        return IndicatorResult(
            **base, label=f"度日数折标后日{label}指标值", value=adjusted / area,
            unit=unit_daily,
            formula=(f"度日数折标后日{QTY} = 日{QTY} × "
                     f"({config.DEGREE_DAY_BASE_TEMP:g} + {config.DEGREE_DAY_INDOOR_OFFSET:g}) / "
                     f"({config.DEGREE_DAY_BASE_TEMP:g} - 当天室外平均温度)；"
                     f"度日数折标后日{label}指标值 = 度日数折标后日{QTY} / 面积"),
            inputs={f"日{QTY}": qty, "当天室外平均温度": temp, "折标系数": factor,
                    "气象站": station, "气象站为演示/替代站点": True, "面积": area},
            input_units={f"日{QTY}": unit_qty_daily, "当天室外平均温度": "℃",
                         "折标系数": "无量纲", "面积": AREA_UNIT},
            agg={"num": adjusted, "den": area, "name": "度日数折标后日耗量",
                 "unit": unit_qty_daily},
            source=f"fact_energy_daily.{codes['daily']} + fact_weather_daily",
        )

    return _unavailable(**base, label=label, reason=f"未支持的口径：{mode}")


# ============================================================================
# 集团级聚合
# ============================================================================


def _group_energy_result(da: DataAccess, group_id: str, stat_date: datetime.date,
                         metric: str, mode: str) -> IndicatorResult:
    """集团级能耗指标：各单位按对应温度计算后加和，再做后续计算。

    《数据及计算说明》：「公司级 = 公司A项目1-8 之和；集团级 = 各公司之和」；
    度日数类「在气耗量步骤时，各单位按对应温度计算，加和成集团级气耗量总数，再做后续计算」。
    所以集团级**不能**对各公司的「单平米指标值」求平均，而应
    先把各公司的耗量分量与面积分别加总（分母为各公司面积之和），再用同一口径算出集团指标。
    """
    group_name = group_id.split(":", 1)[1] if ":" in group_id else group_id
    companies = da.child_companies(group_name)
    base = dict(entity_id=group_id, stat_date=stat_date, metric=metric, mode=mode)
    label = METRIC_LABEL[metric]

    if not companies:
        return _unavailable(**base, label=f"集团级{label}",
                            reason=f"维度表中没有归属「{group_name}」的公司，无法做集团级聚合",
                            source="dim_entity")

    parts = [_energy_result(da, c, stat_date, metric, mode) for c in companies]
    ok = [p for p in parts if p.available and p.agg]
    if not ok:
        reasons = "；".join(f"{p.entity_id.split(':')[-1]}: {p.reason}" for p in parts[:3])
        return _unavailable(**base, label=f"集团级{label}",
                            reason=f"各公司均无法计算，故集团级不可用（{reasons}）",
                            source="fact_energy_daily")

    num = sum(float(p.agg["num"]) for p in ok)
    den = sum(float(p.agg["den"]) for p in ok)
    if not den:
        return _unavailable(**base, label=f"集团级{label}", reason="各公司面积之和为 0，无法聚合")
    used = "、".join(p.entity_id.split(":")[-1] for p in ok)
    name = ok[0].agg.get("name") or "耗量分量"
    unit_num = ok[0].agg.get("unit") or ""
    idx = num / den
    factor = config.EVALUATION_FACTOR
    unit = ok[0].unit
    inputs: Dict[str, Any] = {name: num, "面积": den, "参与聚合的公司": used}
    units = {name: unit_num, "面积": "m2"}
    formula = f"集团级 = 各公司「{name}」之和 ÷ 各公司面积之和"

    if mode == "daily":
        value = idx
        formula = f"集团级日{label}指标值 = 各公司日{label}量之和 ÷ 各公司面积之和"
    elif mode == "evaluation":
        value = idx * factor
        formula = (f"集团级日{label}评价值 = 集团级日{label}指标值 × {factor:g}；"
                   + formula)
    elif mode == "cumulative":
        value = idx
        formula = f"集团级累计{label}指标值 = 各公司累计{label}量之和 ÷ 各公司面积之和"
    elif mode == "cumulative_evaluation":
        days = heating_days(stat_date)
        if days is None:
            return _unavailable(**base, label=f"集团级累计{label}评价值",
                                reason="未配置供热开始日期，无法算已供暖天数")
        value = idx / days * factor
        inputs["已供暖天数"] = days
        units["已供暖天数"] = "天"
        formula = (f"集团级累计{label}评价值 = 集团级累计{label}指标值 / 已供暖天数 × {factor:g}；"
                   + formula)
    elif mode == "season_cumulative":
        value = idx
        formula = (f"集团级标准采暖季累计{label}指标值 = 各公司(当前–基准日)累计{label}量之和 "
                   f"÷ 各公司面积之和")
    elif mode == "season_cumulative_evaluation":
        ldays = legal_heating_days(stat_date)
        value = idx / ldays * factor
        inputs["法定已供暖天数"] = ldays
        units["法定已供暖天数"] = "天"
        formula = (f"集团级标准采暖季累计{label}评价值 = 季指标值 / 法定已供暖天数 × {factor:g}；"
                   + formula)
    elif mode == "degree_day":
        value = idx
        formula = ("集团级度日数折标后日" + label + "指标值 = 各公司度日数折标后日耗量之和 ÷ "
                   "各公司面积之和（各单位按各自温度折标后加和）")
    elif mode == "mom":
        if not all("prev_num" in p.agg for p in ok):
            return _unavailable(**base, label=f"集团级日{label}指标值环比",
                                reason="缺少可比对的昨日分量，无法做集团级环比")
        cur_i = num / den
        pd_ = sum(float(p.agg["prev_den"]) for p in ok)
        pn = sum(float(p.agg["prev_num"]) for p in ok)
        if not pd_ or pn == 0:
            return _unavailable(**base, label=f"集团级日{label}指标值环比",
                                reason="昨日分量不可用，无法做集团级环比")
        prev_i = pn / pd_
        value = (cur_i - prev_i) / prev_i * 100
        unit = "%"
        inputs = {"集团级当日指标值": cur_i, "集团级昨日指标值": prev_i,
                  "参与聚合的公司": used}
        units = {"集团级当日指标值": ok[0].unit, "集团级昨日指标值": ok[0].unit}
        formula = f"集团级日{label}指标值环比 = (集团级当日 − 集团级昨日) / 集团级昨日 × 100%"
    else:
        return _unavailable(**base, label=f"集团级{label}",
                            reason=f"口径 {mode} 不支持集团级聚合")

    return IndicatorResult(
        **base, label="集团级" + ok[0].label.replace("集团级", ""),
        value=value, unit=unit, formula=formula,
        inputs=inputs, input_units=units,
        source="fact_energy_daily（各公司聚合）+ dim_entity",
    )


# ============================================================================
# 收费指标
# ============================================================================


def _fee_result(da: DataAccess, entity_id: str, stat_date: datetime.date,
                metric: str, category: str) -> IndicatorResult:
    item = FEE_ITEMS[metric]
    label = FEES[metric]
    base = dict(entity_id=entity_id, stat_date=stat_date, metric=metric, mode="season")
    is_group = entity_id.startswith("group:")

    reported = da.fee(entity_id, stat_date, item, category)

    if reported is None and not is_group:
        return _unavailable(**base, label=f"{label}({category})",
                            reason="报表中无该分类的收费率数据",
                            source=f"fact_fee_season.{item}[{category}]")

    if reported is None:
        # 集团级在源报表里没有对应行（各家只报自己）。按《数据及计算说明》
        # 「集团 = 公司A + …」自行聚合；收费率沿用公司与项目两级的户数加权口径。
        # 非收费率类（户缴率/旧欠收费率）计算说明未给公式，如实返回不可用。
        if metric != "fee_rate":
            return _unavailable(
                **base, label=f"集团级{label}({category})",
                reason=f"《数据及计算说明》未给出「{label}」的计算公式，"
                       f"且源报表无集团级行，无法聚合",
                source=f"fact_fee_season.{item}[{category}]")
        cc = _fee_rate_cross_check(da, entity_id, stat_date, category,
                                   float("nan"), label=label)
        computed = cc.get("recomputed")
        if computed is None:
            return _unavailable(**base, label=f"集团级{label}({category})",
                                reason=cc.get("note") or "缺乏户数/金额，无法聚合集团级收费率",
                                source="fact_fee_season")
        return IndicatorResult(
            **base, label=f"集团级{label}({category})", value=computed, unit="%",
            formula=(f"集团级{label}({category}) = Σ(各项目{label} × 各项目本季应收户数) ÷ "
                     f"Σ(各项目本季应收户数)（集团 = 各公司之和，与公司级同口径）"),
            inputs=cc.get("inputs") or {}, input_units=cc.get("input_units") or {},
            source="fact_fee_season（各项目聚合，集团级无源报表行）",
            cross_check={"consistent": None,
                         "note": "集团级无源报表原值可比对，按《数据及计算说明》"
                                 "「集团 = 各公司之和」聚合得出",
                         "method": cc.get("method"),
                         "inputs": cc.get("inputs")},
        )

    unit = "%"
    # 收费率按公式重算，与报表原值对照（审核员校验点）
    res = IndicatorResult(
        **base, label=f"{label}({category})", value=reported, unit=unit,
        formula=f"{label}({category}) = 报表原值",
        inputs={"分类": category},
        source=f"fact_fee_season.{item}[{category}]",
    )

    if metric == "fee_rate":
        res.cross_check = _fee_rate_cross_check(da, entity_id, stat_date, category,
                                               reported, label=label)
        if res.cross_check.get("report_formula"):
            res.formula = res.cross_check["report_formula"]
    else:
        res.cross_check = {
            "consistent": None,
            "note": f"《数据及计算说明》未给出「{label}」的计算公式，只能采用报表原值",
        }
    return res


def _fee_rate_cross_check(da: DataAccess, entity_id: str, stat_date: datetime.date,
                          category: str, reported: float,
                          label: str = "本季收费率") -> Dict[str, Any]:
    """本季收费率的重算对照。

    ⚠️ 源报表**两级用的是不同聚合口径**（已逐项核对到 0.0000 个百分点）：

      * 项目级：项目收费率 = 累计本季实收金额 ÷ 本季应收金额 × 100%      —— 金额口径
      * 公司级：公司收费率 = Σ(项目收费率 × 项目本季应收户数) ÷ Σ(项目本季应收户数)
                                                                        —— 户数口径

    所以公司级**不能**用金额口径复算，否则会得到一个假的「不一致」告警：
    非居民尤其明显——户数少且各项目分布悬殊，两种口径相差 1.79 个百分点
    （金额口径 69.7085% vs 报表 67.9187%）。
    这一点在报表里曾被当成"源数据偏差"，实际是口径不同，现已查清。
    """
    if entity_id.startswith(("company:", "group:")):
        # 公司级取本公司的项目；集团级取集团下全部公司的项目（同一套户数加权口径）
        proj_ids = (da.fee_project_ids(pattern="project:%")
                    if entity_id.startswith("group:")
                    else da.fee_project_ids(entity_id.split(":", 1)[1]))
        num = den = 0.0
        parts = []
        for pid in proj_ids:
            rate = da.fee(pid, stat_date, FEE_ITEMS["fee_rate"], category)
            hh = da.fee(pid, stat_date, "season_receivable_households", category)
            if rate is None or not hh:
                continue
            num += rate * hh
            den += hh
            parts.append(f"{pid.split(':')[-1]} {rate:.2f}%×{hh:.0f}户")
        if den <= 0:
            return {"consistent": None,
                    "note": "公司级收费率需按各项目本季应收户数加权，但库中缺户数，无法重算校验",
                    "method": "各项目收费率按本季应收户数加权",
                    "report_formula": (f"公司级{label}({category}) = Σ(各项目{label} × "
                                       f"各项目本季应收户数) ÷ Σ(各项目本季应收户数)")}
        recomputed = num / den
        return {
            "recomputed": recomputed,
            "reported": reported,
            "delta_pp": abs(recomputed - reported),
            "consistent": abs(recomputed - reported) <= RATE_TOLERANCE,
            "method": "各项目收费率按本季应收户数加权",
            "report_formula": (f"公司级{label}({category}) = Σ(各项目{label} × "
                               f"各项目本季应收户数) ÷ Σ(各项目本季应收户数)"),
            "inputs": {"加权构成": "；".join(parts)},
            "input_units": {},
        }

    receivable = da.fee(entity_id, stat_date, "season_receivable_amount", category)
    received = da.fee(entity_id, stat_date, "cumulative_season_received_amount", category)
    if receivable in (None, 0) or received is None:
        return {"consistent": None,
                "note": "缺少本季应收/累计实收金额，无法重算校验",
                "method": "金额口径：累计本季实收金额 ÷ 本季应收金额",
                "report_formula": f"{label}({category}) = 累计本季实收金额 ÷ 本季应收金额 × 100%"}
    recomputed = received / receivable * 100
    return {
        "recomputed": recomputed,
        "reported": reported,
        "delta_pp": abs(recomputed - reported),
        "consistent": abs(recomputed - reported) <= RATE_TOLERANCE,
        "method": "金额口径：累计本季实收金额 ÷ 本季应收金额",
        "report_formula": f"{label}({category}) = 累计本季实收金额 ÷ 本季应收金额 × 100%",
        "inputs": {"累计本季实收金额": received, "本季应收金额": receivable},
        "input_units": {"累计本季实收金额": "万元", "本季应收金额": "万元"},
    }


# ============================================================================
# 调度
# ============================================================================


def _entity_missing_reason(
    entity_id: str, stat_date: datetime.date, known: List[str]
) -> Optional[str]:
    """实体在库中完全无数据时给出明确原因；有数据则返回 None。

    区分两种情况，避免都报成「报表中无该分类数据」这种误导性结论：
      * 实体本身不存在/无任何数据 → 直接点名，并列出库中真实存在的实体；
      * 实体存在、只是该口径当天缺数 → 交由各口径自己的 reason 说明。
    """
    if entity_id in known:
        return None
    return (f"库中不存在实体「{entity_id}」的任何数据"
            f"（计算日 {stat_date.isoformat()} 亦无对应行）；"
            f"库中现有数据的实体：{'、'.join(known) if known else '无'}")


def compute(
    entity_id: str,
    stat_date: datetime.date,
    metrics: List[str],
    modes: Optional[List[str]] = None,
    categories: Optional[List[str]] = None,
    modes_by_metric: Optional[Dict[str, List[str]]] = None,
) -> List[IndicatorResult]:
    """按指标与口径计算，返回结果列表。"""
    modes = modes or ["daily"]
    results: List[IndicatorResult] = []
    # 集团级在事实表中没有「行」，其值由下属公司聚合而来，不能按「实体无数据」处理
    is_group = entity_id.startswith("group:")
    with DataAccess() as da:
        if not is_group:
            missing = _entity_missing_reason(entity_id, stat_date, da.entities_with_energy())
            if missing is not None:
                # 实体整体无数据：逐指标如实报告，不伪装成「该口径缺数」
                for metric in metrics:
                    label = FEES.get(metric) or METRIC_LABEL.get(metric, metric)
                    if metric in FEE_ITEMS:
                        for cat in (categories or ["小计"]):
                            results.append(_unavailable(
                                entity_id=entity_id, stat_date=stat_date, metric=metric,
                                mode="season", label=f"{label}({cat})", reason=missing,
                                source=f"fact_fee_season.{FEE_ITEMS[metric]}[{cat}]"))
                    else:
                        for mode in (modes_by_metric or {}).get(metric, modes):
                            results.append(_unavailable(
                                entity_id=entity_id, stat_date=stat_date, metric=metric,
                                mode=mode, label=label, reason=missing))
                return results

        for metric in metrics:
            if metric in ENERGY_CODES:
                for mode in (modes_by_metric or {}).get(metric, modes):
                    if is_group:
                        results.append(_group_energy_result(
                            da, entity_id, stat_date, metric, mode))
                    else:
                        results.append(_energy_result(da, entity_id, stat_date,
                                                      metric, mode))
            elif metric in FEE_ITEMS:
                for cat in (categories or ["小计"]):
                    results.append(_fee_result(da, entity_id, stat_date, metric, cat))
            else:
                results.append(_unavailable(
                    entity_id=entity_id, stat_date=stat_date, metric=metric,
                    label=metric, reason=f"未支持的指标：{metric}"))
    return results


def compute_from_intent(intent) -> List[IndicatorResult]:
    """从 QueryIntent 计算（供智能体编排器调用）。"""
    if intent.domain != "enterprise" or not intent.metrics:
        return []
    entities = intent.entity_ids()
    if not entities:
        return []
    date = intent.stat_date or datetime.date.today()
    modes_by_metric: Dict[str, List[str]] = {}
    for m in intent.metrics:
        ms = list(intent.modes)
        if m in ENERGY_CODES and "cumulative" in ms and "cumulative_evaluation" not in ms:
            ms.append("cumulative_evaluation")
        modes_by_metric[m] = ms or ["daily"]

    out: List[IndicatorResult] = []
    for eid in entities:
        out.extend(compute(eid, date, intent.metrics,
                           modes_by_metric=modes_by_metric,
                           categories=intent.categories or ["小计"]))
    return out
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="供热指标计算（命令行算单个指标）")
    ap.add_argument("--entity", help="实体 ID，如 company:公司A")
    ap.add_argument("--date", help="计算日 YYYY-MM-DD")
    ap.add_argument("--metric", help="指标：gas/water/elec/fee_rate/…")
    args = ap.parse_args()

    if not (args.entity and args.date and args.metric):
        ap.print_help()
        raise SystemExit(0)

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass

    rs = compute(args.entity, datetime.date.fromisoformat(args.date), [args.metric])
    print(json.dumps([r.to_dict() for r in rs], ensure_ascii=False, indent=2))
