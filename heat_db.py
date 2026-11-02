"""供热运营数据层：MySQL 表结构定义与 ETL 入库。

四个数据域各自独立建模，不强行互相映射（数据来源之间本无对应关系）：

  1. 企业运营域  01_business_data                -> dim_entity / fact_energy_daily / fact_fee_season / fact_daily_note
  2. 气象域      02_heapo_heatpump/weather_data  -> dim_weather_station / fact_weather_hourly / fact_weather_daily
  3. 换热站域    03_xai4heat_scada               -> dim_substation / fact_scada_hourly
  4. 住户域      02_heapo_heatpump/{meta,reports,smart_meter_data}
                                                 -> dim_household / fact_household_protocol / fact_household_meter

注意事项：
  * 气象站点为匿名化编号（如 8jB），无地理位置信息，只能作为「演示/替代站点」用作
    度日数折标的温度输入，已在 dim_weather_station.is_demo 标记。
  * 住户 15min 时序约 4.8GB / 1.8 亿行，不入库，由 read_household_15min() 按需读取文件。
  * 收费日报模板注释（注 [3]）写的是「2021～2022 年度采暖季」，与统计日期（2023-12-16）
    不一致，属模板陈旧文本；原注释原文已存入 fact_daily_note，season 取 config.HEATING_SEASON。

用法：
    python heat_db.py --init                    # 建库 + 建表
    python heat_db.py --load all                # 全部数据域入库
    python heat_db.py --load business           # 只入库企业运营数据
"""

import argparse
import datetime
import logging
import os
from typing import Any, Dict, List, Optional

import openpyxl
import pandas as pd
from sqlalchemy import (
    Boolean,
    Column,
    Date,
    DateTime,
    Double,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Session

import config

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("heat_db")

EXCEL_EPOCH = datetime.date(1899, 12, 30)


# ============================================================================
# 表结构
# ============================================================================


class Base(DeclarativeBase):
    pass


class Entity(Base):
    """实体维度：集团 / 公司 / 项目。"""

    __tablename__ = "dim_entity"

    entity_id = Column(String(64), primary_key=True)
    entity_level = Column(String(16), nullable=False)  # group | company | project
    group_name = Column(String(64))
    company = Column(String(64))
    project = Column(String(64))
    display_name = Column(String(128))


class EnergyDaily(Base):
    """运营能耗日报事实（表内原值，含日值/累计值与单平米指标）。"""

    __tablename__ = "fact_energy_daily"
    __table_args__ = (
        UniqueConstraint("entity_id", "stat_date", "metric_code", name="uq_energy"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    entity_id = Column(String(64), nullable=False, index=True)
    stat_date = Column(Date, nullable=False, index=True)
    metric_code = Column(String(48), nullable=False)
    value = Column(Double)
    unit = Column(String(16))


class FeeSeason(Base):
    """热费收缴事实（保留 小计/居民/非居民 三级分类）。"""

    __tablename__ = "fact_fee_season"
    __table_args__ = (
        UniqueConstraint(
            "entity_id", "stat_date", "item_code", "category", name="uq_fee"
        ),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    entity_id = Column(String(64), nullable=False, index=True)
    stat_date = Column(Date, nullable=False, index=True)
    season = Column(String(16))
    item_code = Column(String(48), nullable=False)
    category = Column(String(16), nullable=False)  # 小计 | 居民 | 非居民
    value = Column(Double)
    unit = Column(String(16))


class DailyNote(Base):
    """日报文字内容（标题 / 表头 / 动态评价 / 注释），原文保留。"""

    __tablename__ = "fact_daily_note"

    id = Column(Integer, primary_key=True, autoincrement=True)
    company = Column(String(64), index=True)
    stat_date = Column(Date, index=True)
    report_type = Column(String(16))  # energy | fee
    section_name = Column(String(16))  # title | header | narrative | note
    seq = Column(Integer, default=0)
    content = Column(Text)


class WeatherStation(Base):
    """气象站点维度。is_demo 标记该站点非本项目实测区域（仅作演示/替代）。"""

    __tablename__ = "dim_weather_station"

    weather_id = Column(String(16), primary_key=True)
    label = Column(String(64))
    is_demo = Column(Boolean, default=True)
    remark = Column(String(255))
    company = Column(String(64))  # 公司级温度映射（演示）


class WeatherHourly(Base):
    __tablename__ = "fact_weather_hourly"
    __table_args__ = (
        UniqueConstraint("weather_id", "ts", name="uq_weather_h"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    weather_id = Column(String(16), nullable=False, index=True)
    ts = Column(DateTime, nullable=False, index=True)
    temperature_avg = Column(Double)
    dew_point = Column(Double)
    humidity_avg = Column(Double)
    precipitation_total = Column(Double)
    pressure_qfe = Column(Double)
    pressure_qnh = Column(Double)
    pressure_qff = Column(Double)
    sunshine_duration = Column(Double)
    wind_speed = Column(Double)


class WeatherDaily(Base):
    """气象逐日。含 SIA / US 两种供暖度日数，供度日数折标与趋势分析直接使用。"""

    __tablename__ = "fact_weather_daily"
    __table_args__ = (
        UniqueConstraint("weather_id", "ts", name="uq_weather_d"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    weather_id = Column(String(16), nullable=False, index=True)
    ts = Column(Date, nullable=False, index=True)
    temperature_max = Column(Double)
    temperature_min = Column(Double)
    temperature_avg = Column(Double)
    hdd_sia = Column(Double)
    hdd_us = Column(Double)
    cdd_us = Column(Double)
    humidity_avg = Column(Double)
    precipitation_total = Column(Double)
    sunshine_duration = Column(Double)


class Substation(Base):
    """换热站维度。"""

    __tablename__ = "dim_substation"

    substation_id = Column(String(16), primary_key=True)
    heating_area_m2 = Column(Double)


class ScadaHourly(Base):
    """换热站 SCADA 逐小时运行数据（一次/二次侧供回水温度、室外温度、累计热量增量）。"""

    __tablename__ = "fact_scada_hourly"
    __table_args__ = (
        UniqueConstraint("substation_id", "ts", name="uq_scada"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    substation_id = Column(String(16), nullable=False, index=True)
    ts = Column(DateTime, nullable=False, index=True)
    t_amb = Column(Double)
    t_ref = Column(Double)
    t_sup_prim = Column(Double)
    t_ret_prim = Column(Double)
    t_sup_sec = Column(Double)
    t_ret_sec = Column(Double)
    delta_e = Column(Double)


class Household(Base):
    """住户维度：问卷元数据 + 采集可用性 + 气候站点归属。"""

    __tablename__ = "dim_household"

    household_id = Column(String(16), primary_key=True)
    group_label = Column(String(16))
    weather_id = Column(String(16), index=True)
    has_pv = Column(Boolean)
    protocols_available = Column(Boolean)
    protocols_multiple_visits = Column(Boolean)
    report_ids = Column(String(255))
    meta_available = Column(Boolean)
    smd_15min_available = Column(Boolean)
    smd_daily_available = Column(Boolean)
    smd_monthly_available = Column(Boolean)
    building_type = Column(String(32))
    living_area = Column(Double)
    residents = Column(Double)
    heatpump_type = Column(String(32))
    floor_heating = Column(Boolean)
    radiator = Column(Boolean)
    dhw_by_heatpump = Column(Boolean)
    dhw_by_electric = Column(Boolean)
    dhw_by_solar = Column(Boolean)
    has_dryer = Column(Boolean)
    has_freezer = Column(Boolean)
    has_ev = Column(Boolean)


class HouseholdProtocol(Base):
    """上门勘查报告（自 106 列中选取分析所需字段）。"""

    __tablename__ = "fact_household_protocol"

    report_id = Column(String(16), primary_key=True)
    household_id = Column(String(16), index=True)
    visit_year = Column(Integer)
    visit_date = Column(Date)
    building_type = Column(String(32))
    housing_units = Column(Double)
    construction_year = Column(Double)
    construction_year_interval = Column(String(24))
    renovated_windows = Column(Boolean)
    renovated_roof = Column(Boolean)
    renovated_walls = Column(Boolean)
    renovated_floor = Column(Boolean)
    floor_area_heated_total = Column(Double)
    building_residents = Column(Double)
    heatpump_type = Column(String(32))
    heatpump_installation_year = Column(Double)
    heatpump_heating_capacity = Column(Double)
    heatpump_normpoint_cop = Column(Double)
    heat_distribution_radiators = Column(Boolean)
    heat_distribution_floor_heating = Column(Boolean)


class HouseholdMeter(Base):
    """住户电表时序（daily / monthly / cumulative 三种分辨率合并存表）。"""

    __tablename__ = "fact_household_meter"
    __table_args__ = (
        UniqueConstraint("household_id", "resolution", "ts", name="uq_hh_meter"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    household_id = Column(String(16), nullable=False, index=True)
    resolution = Column(String(16), nullable=False)  # daily | monthly | cumulative
    group_label = Column(String(16))
    affects_time_point = Column(String(16))
    ts = Column(DateTime, nullable=False, index=True)
    kwh_received_total = Column(Double)
    kwh_received_heatpump = Column(Double)
    kwh_received_other = Column(Double)
    kwh_returned_total = Column(Double)
    kvarh_cap_total = Column(Double)
    kvarh_cap_heatpump = Column(Double)
    kvarh_cap_other = Column(Double)
    kvarh_ind_total = Column(Double)
    kvarh_ind_heatpump = Column(Double)
    kvarh_ind_other = Column(Double)


# ============================================================================
# 运营能耗日报表的指标定义（行号自第 4 行起，与报表模板一一对应）
# ============================================================================

ENERGY_METRICS: List[tuple] = [
    ("total_heating_area", "m2", "总供热面积"),
    ("residential_heating_area", "m2", "居民供热面积"),
    ("non_residential_heating_area", "m2", "非居民供热面积"),
    ("daily_gas", "Nm3", "日气耗量"),
    ("daily_gas_index", "Nm3/m2", "单平米日气耗指标"),
    ("cumulative_gas", "Nm3", "累计气耗量"),
    ("cumulative_gas_index", "Nm3/m2", "单平米累计气耗指标"),
    ("daily_elec", "kWh", "日电耗量"),
    ("daily_elec_index", "kWh/m2", "单平米日电耗指标"),
    ("cumulative_elec", "kWh", "累计电耗量"),
    ("cumulative_elec_index", "kWh/m2", "单平米累计电耗指标"),
    ("daily_water", "t", "日水耗量"),
    ("daily_water_index", "kg/m2", "单平米日水耗指标"),
    ("cumulative_water", "t", "累计水耗量"),
    ("cumulative_water_index", "kg/m2", "单平米累计水耗指标"),
]
ENERGY_FIRST_ROW = 4
ENERGY_NOTE_ROW = 20  # 注 [1] 起始行

# 热费收缴日报表：每项占 3 行（小计 / 居民 / 非居民）
FEE_ITEMS: List[tuple] = [
    ("season_fee_area", "万平米", "本季收费面积"),
    ("season_receivable_households", "户", "本季应收户数"),
    ("season_receivable_amount", "万元", "本季应收金额"),
    ("arrears_receivable_amount", "万元", "旧欠应收金额"),
    ("daily_season_received_households", "户", "日 本季实收户数"),
    ("daily_season_received_amount", "万元", "日 本季实收金额"),
    ("daily_arrears_received_amount", "万元", "日 旧欠实收金额"),
    ("cumulative_season_received_households", "户", "累计 本季实收户数"),
    ("cumulative_season_received_amount", "万元", "累计 本季实收金额"),
    ("cumulative_arrears_received_amount", "万元", "累计 旧欠实收金额"),
    ("household_payment_rate", "%", "户缴率"),
    ("season_fee_rate", "%", "本季收费率"),
    ("arrears_fee_rate", "%", "旧欠收费率"),
]
FEE_FIRST_ROW = 4
FEE_CATEGORIES = ("小计", "居民", "非居民")
FEE_NOTE_ROW = 44  # 注 [1] 起始行

# 项目列：E..L 对应 项目1..项目8；D 列为「合计」
ENERGY_VALUE_COLS = list(range(4, 13))  # D..L


# ============================================================================
# 解析辅助
# ============================================================================


def _isna(v: Any) -> bool:
    """判断缺失值：涵盖 None、pandas NaN/NaT 与 float('nan')。"""
    if v is None:
        return True
    try:
        return bool(pd.isna(v))
    except (TypeError, ValueError):
        return False


def _cell(ws, row: int, col: int) -> Optional[Any]:
    v = ws.cell(row=row, column=col).value
    if isinstance(v, str):
        s = v.strip()
        return s or None
    return v


def _num(v: Any) -> Optional[float]:
    if _isna(v):
        return None
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip().replace(",", "")
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _to_bool(v: Any) -> Optional[bool]:
    if _isna(v) or v == "":
        return None
    s = str(v).strip().lower()
    if s in ("true", "1", "yes"):
        return True
    if s in ("false", "0", "no"):
        return False
    return None


def _excel_serial_to_date(v: Any) -> Optional[datetime.date]:
    """报表「统计日期」为 Excel 序列号（如 45276 = 2023-12-16）。"""
    if v is None:
        return None
    if isinstance(v, datetime.datetime):
        return v.date()
    if isinstance(v, datetime.date):
        return v
    n = _num(v)
    if n is None:
        return None
    try:
        return EXCEL_EPOCH + datetime.timedelta(days=int(n))
    except (OverflowError, ValueError):
        return None


def _parse_ts(series: pd.Series, as_date: bool = False) -> pd.Series:
    """统一时间戳：带 UTC 偏移（如 +00:00）先归一到 UTC，再去掉时区。"""
    dt = pd.to_datetime(series, utc=True, errors="coerce").dt.tz_localize(None)
    return dt.dt.date if as_date else dt


# ============================================================================
# 建库建表
# ============================================================================


def get_engine(echo: bool = False):
    return create_engine(config.DB_URL, echo=echo, pool_pre_ping=True, future=True)


def init_db() -> None:
    """创建数据库（若不存在）并建全部表。"""
    server = create_engine(config.DB_SERVER_URL, future=True, pool_pre_ping=True)
    db_name = config.DB_NAME.replace("`", "")
    with server.connect() as conn:
        conn.execute(
            text(
                f"CREATE DATABASE IF NOT EXISTS `{db_name}` "
                f"CHARACTER SET {config.DB_CHARSET} COLLATE {config.DB_CHARSET}_general_ci"
            )
        )
        conn.commit()
    server.dispose()
    logger.info("数据库就绪：%s", config.DB_NAME)

    engine = get_engine()
    Base.metadata.create_all(engine)
    engine.dispose()
    logger.info("已建表：%s", ", ".join(sorted(Base.metadata.tables)))


def reset_db() -> None:
    """删除全部表并重建（仅影响本项目库 heating_ops）。"""
    engine = get_engine()
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    engine.dispose()
    logger.info("已重建全部表")


def _truncate(engine, *tables: str) -> None:
    with engine.begin() as conn:
        for t in tables:
            conn.execute(text(f"TRUNCATE TABLE {t}"))


# ============================================================================
# 域 1：企业运营数据（两个日报表）
# ============================================================================


def _parse_report_header(ws, path: str) -> tuple:
    """两张日报表共有的表头：单位、统计日期、报表日期。"""
    company = _cell(ws, 2, 3) or "未知单位"
    stat_date = _excel_serial_to_date(_cell(ws, 2, 7))
    report_date = _excel_serial_to_date(_cell(ws, 2, 11))
    if stat_date is None:
        raise ValueError(f"未能解析统计日期：{path}")
    return company, stat_date, report_date


def _parse_projects(ws) -> List[tuple]:
    """项目列名取自表头（E..L），返回 [(列号, 项目名)]。"""
    out: List[tuple] = []
    for col in range(5, 13):
        name = _cell(ws, 3, col)
        if name:
            out.append((col, str(name)))
    return out


def _collect_notes(ws, company: str, stat_date: Any, report_type: str,
                   narrative: Any, note_row: int) -> List[Dict[str, Any]]:
    """报表的标题 / 动态说明 / 逐条注释。

    两张报表的这一段结构完全一致，只有 `report_type`、动态说明的取值位置
    与注释起始行不同，故收在一处。`narrative` 由调用方按各自规则取好传入。
    """
    def note(section: str, seq: int, content: str) -> Dict[str, Any]:
        return {"company": company, "stat_date": stat_date, "report_type": report_type,
                "section_name": section, "seq": seq, "content": content}

    notes = [note("title", 0, str(_cell(ws, 1, 1) or ""))]
    if narrative:
        notes.append(note("narrative", 0, str(narrative)))
    for i, r in enumerate(range(note_row, ws.max_row + 1), start=1):
        c = _cell(ws, r, 2) or _cell(ws, r, 1)
        if c:
            notes.append(note("note", i, str(c)))
    return notes


def parse_energy_report(path: str) -> Dict[str, Any]:
    """解析《集团运营能耗日报表》。"""
    ws = openpyxl.load_workbook(path, data_only=True).worksheets[0]
    company, stat_date, report_date = _parse_report_header(ws, path)
    projects = _parse_projects(ws)

    rows: List[Dict[str, Any]] = []
    for offset, (metric_code, unit, label) in enumerate(ENERGY_METRICS):
        r = ENERGY_FIRST_ROW + offset
        actual_unit = _cell(ws, r, 3)
        if actual_unit and str(actual_unit) != unit:
            logger.warning(
                "能耗报表第 %d 行单位与模板不符：期望 %s，实际 %s（%s）",
                r, unit, actual_unit, label,
            )
        total = _num(_cell(ws, r, 4))
        if total is not None:
            rows.append({"entity_id": f"company:{company}", "stat_date": stat_date,
                         "metric_code": metric_code, "value": total, "unit": unit})
        for col, pname in projects:
            v = _num(_cell(ws, r, col))
            if v is not None:
                rows.append({"entity_id": f"project:{company}:{pname}", "stat_date": stat_date,
                             "metric_code": metric_code, "value": v, "unit": unit})

    narrative = _cell(ws, 16 + ENERGY_FIRST_ROW - 1, 4) or _cell(ws, 19, 4)
    return {
        "company": company,
        "stat_date": stat_date,
        "report_date": report_date,
        "projects": [p for _, p in projects],
        "energy_rows": rows,
        "notes": _collect_notes(ws, company, stat_date, "energy", narrative, ENERGY_NOTE_ROW),
    }


def parse_fee_report(path: str) -> Dict[str, Any]:
    """解析《集团热费收缴日报表》。"""
    ws = openpyxl.load_workbook(path, data_only=True).worksheets[0]
    company, stat_date, report_date = _parse_report_header(ws, path)
    projects = _parse_projects(ws)

    rows: List[Dict[str, Any]] = []
    for idx, (item_code, unit, label) in enumerate(FEE_ITEMS):
        base = FEE_FIRST_ROW + idx * len(FEE_CATEGORIES)
        for k, category in enumerate(FEE_CATEGORIES):
            r = base + k
            total = _num(_cell(ws, r, 4))
            if total is not None:
                rows.append({"entity_id": f"company:{company}", "stat_date": stat_date,
                             "season": config.HEATING_SEASON, "item_code": item_code,
                             "category": category, "value": total, "unit": unit})
            for col, pname in projects:
                v = _num(_cell(ws, r, col))
                if v is not None:
                    rows.append({"entity_id": f"project:{company}:{pname}", "stat_date": stat_date,
                                 "season": config.HEATING_SEASON, "item_code": item_code,
                                 "category": category, "value": v, "unit": unit})

    narrative = _cell(ws, FEE_FIRST_ROW + len(FEE_ITEMS) * 3, 4)
    return {
        "company": company,
        "stat_date": stat_date,
        "report_date": report_date,
        "projects": [p for _, p in projects],
        "fee_rows": rows,
        "notes": _collect_notes(ws, company, stat_date, "fee", narrative, FEE_NOTE_ROW),
    }


def load_business(engine, business_dir: Optional[str] = None) -> None:
    business_dir = business_dir or config.BUSINESS_DATA_DIR
    energy_path = os.path.join(business_dir, "运营能耗-公司A.xlsx")
    fee_path = os.path.join(business_dir, "热费收缴-公司A.xlsx")
    for p in (energy_path, fee_path):
        if not os.path.exists(p):
            raise FileNotFoundError(f"未找到业务数据文件：{p}")

    energy = parse_energy_report(energy_path)
    fee = parse_fee_report(fee_path)
    logger.info(
        "解析完成：能耗 %d 条 / 收费 %d 条，项目列 %s",
        len(energy["energy_rows"]), len(fee["fee_rows"]), energy["projects"],
    )

    _truncate(
        engine, "fact_energy_daily", "fact_fee_season", "fact_daily_note", "dim_entity"
    )

    company = energy["company"]
    entities: List[Dict[str, Any]] = [
        {
            "entity_id": f"group:{group_name(company)}",
            "entity_level": "group",
            "group_name": group_name(company),
            "company": None,
            "project": None,
            "display_name": group_name(company),
        },
        {
            "entity_id": f"company:{company}",
            "entity_level": "company",
            "group_name": group_name(company),
            "company": company,
            "project": None,
            "display_name": company,
        },
    ]
    seen = {e["entity_id"] for e in entities}
    for pname in energy["projects"]:
        eid = f"project:{company}:{pname}"
        if eid in seen:
            continue
        seen.add(eid)
        entities.append(
            {
                "entity_id": eid,
                "entity_level": "project",
                "group_name": group_name(company),
                "company": company,
                "project": pname,
                "display_name": f"{company}{pname}",
            }
        )

    with Session(engine) as s:
        s.bulk_insert_mappings(Entity, entities)
        s.bulk_insert_mappings(EnergyDaily, energy["energy_rows"])
        s.bulk_insert_mappings(FeeSeason, fee["fee_rows"])
        s.bulk_insert_mappings(DailyNote, energy["notes"] + fee["notes"])
        s.commit()

    logger.info(
        "企业运营数据入库完成：实体 %d / 能耗 %d / 收费 %d / 文字 %d",
        len(entities), len(energy["energy_rows"]), len(fee["fee_rows"]),
        len(energy["notes"]) + len(fee["notes"]),
    )


def group_name(company: str) -> str:
    """集团名：当前仅公司A，集团即该单位本身；接入多家公司后按配置汇总。"""
    return "集团"


# ============================================================================
# 域 2：气象数据
# ============================================================================


def load_weather(engine, weather_dir: Optional[str] = None) -> None:
    base = os.path.join(weather_dir or config.HEATPUMP_DATA_DIR, "weather_data")
    if not os.path.isdir(base):
        raise FileNotFoundError(f"未找到气象数据目录：{base}")

    _truncate(engine, "fact_weather_hourly", "fact_weather_daily", "dim_weather_station")

    hourly_map = {
        "Temperature_avg_hourly": "temperature_avg",
        "DewPoint_hourly": "dew_point",
        "Humidity_avg_hourly": "humidity_avg",
        "Precipitation_total_hourly": "precipitation_total",
        "Pressure_BarometricHeight_avg_hourly": "pressure_qfe",
        "Pressure_SeaLevelStandardAtmosphere_avg_hourly": "pressure_qnh",
        "Pressure_SeaLevel_avg_hourly": "pressure_qff",
        "Sunshine_duration_hourly": "sunshine_duration",
        "WindSpeed_hourly": "wind_speed",
    }
    daily_map = {
        "Temperature_max_daily": "temperature_max",
        "Temperature_min_daily": "temperature_min",
        "Temperature_avg_daily": "temperature_avg",
        "HeatingDegree_SIA_daily": "hdd_sia",
        "HeatingDegree_US_daily": "hdd_us",
        "CoolingDegree_US_daily": "cdd_us",
        "Humidity_avg_daily": "humidity_avg",
        "Precipitation_total_daily": "precipitation_total",
        "Sunshine_duration_daily": "sunshine_duration",
    }

    def _load_dir(sub: str, mapping: dict, as_date: bool) -> pd.DataFrame:
        d = os.path.join(base, sub)
        if not os.path.isdir(d):
            return pd.DataFrame()
        frames = []
        for fn in sorted(os.listdir(d)):
            if not fn.lower().endswith(".csv"):
                continue
            df = pd.read_csv(os.path.join(d, fn), sep=";", encoding="utf-8-sig")
            df.columns = [c.strip() for c in df.columns]
            df = df.rename(columns=mapping)
            df["weather_id"] = df["Weather_ID"].astype(str).str.strip()
            df["ts"] = _parse_ts(df["Timestamp"], as_date=as_date)
            cols = ["weather_id", "ts"] + [c for c in mapping.values() if c in df.columns]
            frames.append(df[cols].dropna(subset=["weather_id", "ts"]))
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    h = _load_dir("hourly", hourly_map, as_date=False)
    d = _load_dir("daily", daily_map, as_date=True)

    stations: List[Dict[str, Any]] = []
    ids = sorted(set(h.get("weather_id", pd.Series(dtype=str)).tolist())
                 | set(d.get("weather_id", pd.Series(dtype=str)).tolist()))
    for wid in ids:
        stations.append(
            {
                "weather_id": wid,
                "label": f"气象站 {wid}",
                "is_demo": True,
                "remark": "匿名化站点编号，无地理位置；仅作演示/替代温度源，非本项目实测区域",
                "company": None,
            }
        )

    with Session(engine) as s:
        if stations:
            s.bulk_insert_mappings(WeatherStation, stations)
        s.commit()

    if not h.empty:
        h.to_sql("fact_weather_hourly", engine, if_exists="append", index=False,
                 chunksize=2000, method="multi")
    if not d.empty:
        d.to_sql("fact_weather_daily", engine, if_exists="append", index=False,
                 chunksize=2000, method="multi")

    logger.info("气象数据入库完成：站点 %d / 逐时 %d / 逐日 %d",
                len(stations), len(h), len(d))


# ============================================================================
# 域 3：换热站 SCADA
# ============================================================================


def load_scada(engine, scada_dir: Optional[str] = None) -> None:
    base = scada_dir or config.SCADA_DATA_DIR
    if not os.path.isdir(base):
        raise FileNotFoundError(f"未找到 SCADA 数据目录：{base}")

    _truncate(engine, "fact_scada_hourly", "dim_substation")

    area_path = os.path.join(base, "xai4heat_heating_area.csv")
    stations: List[Dict[str, Any]] = []
    if os.path.exists(area_path):
        adf = pd.read_csv(area_path, encoding="utf-8-sig")
        adf.columns = [c.strip() for c in adf.columns]
        for _, row in adf.iterrows():
            stations.append(
                {
                    "substation_id": str(row["substation_id"]).strip(),
                    "heating_area_m2": _num(row.get("heating_area_m2")),
                }
            )

    frames = []
    for fn in sorted(os.listdir(base)):
        if not fn.lower().endswith(".csv") or "heating_area" in fn:
            continue
        df = pd.read_csv(os.path.join(base, fn), encoding="utf-8-sig")
        df.columns = [c.strip() for c in df.columns]
        df["substation_id"] = df["datetime"].notna().map(lambda _: fn.split("_")[2])
        df["ts"] = pd.to_datetime(df["datetime"], errors="coerce")
        cols = ["substation_id", "ts", "t_amb", "t_ref", "t_sup_prim",
                "t_ret_prim", "t_sup_sec", "t_ret_sec", "delta_e"]
        frames.append(df[[c for c in cols if c in df.columns]].dropna(subset=["ts"]))
    scada = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    if not stations and not scada.empty:
        for sid in scada["substation_id"].unique():
            stations.append({"substation_id": sid, "heating_area_m2": None})

    with Session(engine) as s:
        if stations:
            s.bulk_insert_mappings(Substation, stations)
        s.commit()

    if not scada.empty:
        scada.to_sql("fact_scada_hourly", engine, if_exists="append", index=False,
                     chunksize=2000, method="multi")

    logger.info("SCADA 入库完成：换热站 %d / 逐时 %d", len(stations), len(scada))


# ============================================================================
# 域 4：住户数据（daily / monthly / cumulative 入库；15min 按需读取）
# ============================================================================

METER_RESOLUTIONS = {
    "daily": "daily",
    "monthly": "monthly",
    "cumulative_counters": "cumulative",
}

METER_COL_MAP = {
    "kWh_received_Total": "kwh_received_total",
    "kWh_received_HeatPump": "kwh_received_heatpump",
    "kWh_received_Other": "kwh_received_other",
    "kWh_returned_Total": "kwh_returned_total",
    "kvarh_received_capacitive_Total": "kvarh_cap_total",
    "kvarh_received_capacitive_HeatPump": "kvarh_cap_heatpump",
    "kvarh_received_capacitive_Other": "kvarh_cap_other",
    "kvarh_received_inductive_Total": "kvarh_ind_total",
    "kvarh_received_inductive_HeatPump": "kvarh_ind_heatpump",
    "kvarh_received_inductive_Other": "kvarh_ind_other",
}

PROTOCOL_COL_MAP = {
    "Report_ID": "report_id",
    "Household_ID": "household_id",
    "Visit_Year": "visit_year",
    "Visit_Date": "visit_date",
    "Building_Type": "building_type",
    "Building_HousingUnits": "housing_units",
    "Building_ConstructionYear": "construction_year",
    "Building_ConstructionYear_Interval": "construction_year_interval",
    "Building_Renovated_Windows": "renovated_windows",
    "Building_Renovated_Roof": "renovated_roof",
    "Building_Renovated_Walls": "renovated_walls",
    "Building_Renovated_Floor": "renovated_floor",
    "Building_FloorAreaHeated_Total": "floor_area_heated_total",
    "Building_Residents": "building_residents",
    "HeatPump_Installation_Type": "heatpump_type",
    "HeatPump_Installation_Year": "heatpump_installation_year",
    "HeatPump_Installation_HeatingCapacity": "heatpump_heating_capacity",
    "HeatPump_Installation_Normpoint_COP": "heatpump_normpoint_cop",
    "HeatDistribution_System_Radiators": "heat_distribution_radiators",
    "HeatDistribution_System_FloorHeating": "heat_distribution_floor_heating",
}


def load_household(engine, heatpump_dir: Optional[str] = None) -> None:
    base = heatpump_dir or config.HEATPUMP_DATA_DIR
    meta_dir = os.path.join(base, "meta_data")
    reports_dir = os.path.join(base, "reports")
    smd_dir = os.path.join(base, "smart_meter_data")

    _truncate(engine, "fact_household_meter", "fact_household_protocol", "dim_household")

    # ---- 住户维度：households.csv + meta_data.csv ----
    hh_path = os.path.join(meta_dir, "households.csv")
    md_path = os.path.join(meta_dir, "meta_data.csv")
    hh = pd.read_csv(hh_path, sep=";", encoding="utf-8-sig") if os.path.exists(hh_path) else pd.DataFrame()
    md = pd.read_csv(md_path, sep=";", encoding="utf-8-sig") if os.path.exists(md_path) else pd.DataFrame()
    if not hh.empty:
        hh.columns = [c.strip() for c in hh.columns]
        hh["household_id"] = hh["Household_ID"].astype(str).str.strip()
    if not md.empty:
        md.columns = [c.strip() for c in md.columns]
        md["household_id"] = md["Household_ID"].astype(str).str.strip()

    merged = hh
    if not md.empty:
        merged = hh.merge(md, on="household_id", how="outer") if not hh.empty else md

    households: List[Dict[str, Any]] = []
    for _, r in merged.iterrows():
        households.append(
            {
                "household_id": str(r["household_id"]),
                "group_label": _s(r.get("Group")),
                "weather_id": _s(r.get("Weather_ID")),
                "has_pv": _to_bool(r.get("Installation_HasPVSystem")),
                "protocols_available": _to_bool(r.get("Protocols_Available")),
                "protocols_multiple_visits": _to_bool(r.get("Protocols_HasMultipleVisits")),
                "report_ids": _s(r.get("Protocols_ReportIDs")),
                "meta_available": _to_bool(r.get("MetaData_Available")),
                "smd_15min_available": _to_bool(r.get("SmartMeterData_Available_15min")),
                "smd_daily_available": _to_bool(r.get("SmartMeterData_Available_Daily")),
                "smd_monthly_available": _to_bool(r.get("SmartMeterData_Available_Monthly")),
                "building_type": _s(r.get("Survey_Building_Type")),
                "living_area": _num(r.get("Survey_Building_LivingArea")),
                "residents": _num(r.get("Survey_Building_Residents")),
                "heatpump_type": _s(r.get("Survey_HeatPump_Installation_Type")),
                "floor_heating": _to_bool(r.get("Survey_HeatDistribution_System_FloorHeating")),
                "radiator": _to_bool(r.get("Survey_HeatDistribution_System_Radiator")),
                "dhw_by_heatpump": _to_bool(r.get("Survey_DHW_Production_ByHeatPump")),
                "dhw_by_electric": _to_bool(r.get("Survey_DHW_Production_ByElectricWaterHeater")),
                "dhw_by_solar": _to_bool(r.get("Survey_DHW_Production_BySolar")),
                "has_dryer": _to_bool(r.get("Survey_Installation_HasDryer")),
                "has_freezer": _to_bool(r.get("Survey_Installation_HasFreezer")),
                "has_ev": _to_bool(r.get("Survey_Installation_HasElectricVehicle")),
            }
        )

    # ---- 勘查报告 ----
    pf_path = os.path.join(reports_dir, "protocols.csv")
    protocols: List[Dict[str, Any]] = []
    if os.path.exists(pf_path):
        pdf = pd.read_csv(pf_path, sep=";", encoding="utf-8-sig")
        pdf.columns = [c.strip() for c in pdf.columns]
        keep = {k: v for k, v in PROTOCOL_COL_MAP.items() if k in pdf.columns}
        for _, r in pdf.iterrows():
            rec = {}
            for src, dst in keep.items():
                v = r.get(src)
                if dst in ("renovated_windows", "renovated_roof", "renovated_walls",
                           "renovated_floor", "heat_distribution_radiators",
                           "heat_distribution_floor_heating"):
                    rec[dst] = _to_bool(v)
                elif dst in ("report_id", "household_id", "building_type",
                             "construction_year_interval", "heatpump_type"):
                    rec[dst] = _s(v)
                elif dst == "visit_date":
                    d = pd.to_datetime(v, errors="coerce")
                    rec[dst] = None if pd.isna(d) else d.date()
                elif dst == "visit_year":
                    rec[dst] = int(_num(v)) if _num(v) is not None else None
                else:
                    rec[dst] = _num(v)
            if rec.get("report_id"):
                protocols.append(rec)

    with Session(engine) as s:
        if households:
            s.bulk_insert_mappings(Household, households)
        if protocols:
            s.bulk_insert_mappings(HouseholdProtocol, protocols)
        s.commit()

    # ---- 电表时序（daily / monthly / cumulative）----
    total_rows = 0
    for sub, resolution in METER_RESOLUTIONS.items():
        d = os.path.join(smd_dir, sub)
        if not os.path.isdir(d):
            logger.warning("跳过缺失目录：%s", d)
            continue
        frames = []
        for fn in sorted(os.listdir(d)):
            if not fn.lower().endswith(".csv"):
                continue
            df = pd.read_csv(os.path.join(d, fn), sep=";", encoding="utf-8-sig")
            df.columns = [c.strip() for c in df.columns]
            df = df.rename(columns=METER_COL_MAP)
            df["household_id"] = df["Household_ID"].astype(str).str.strip()
            df["resolution"] = resolution
            df["group_label"] = df.get("Group").astype("string").str.strip() if "Group" in df else None
            df["affects_time_point"] = (
                df.get("AffectsTimePoint").astype("string").str.strip()
                if "AffectsTimePoint" in df else None
            )
            df["ts"] = _parse_ts(df["Timestamp"], as_date=False)
            cols = ["household_id", "resolution", "group_label", "affects_time_point", "ts"] + list(METER_COL_MAP.values())
            frames.append(df[[c for c in cols if c in df.columns]].dropna(subset=["ts"]))
        if not frames:
            continue
        meter = pd.concat(frames, ignore_index=True)
        meter.to_sql("fact_household_meter", engine, if_exists="append",
                     index=False, chunksize=2000, method="multi")
        total_rows += len(meter)
        logger.info("  电表 %s 入库 %d 行", resolution, len(meter))

    logger.info("住户数据入库完成：住户 %d / 勘查 %d / 电表 %d",
                len(households), len(protocols), total_rows)


def _s(v: Any) -> Optional[str]:
    if _isna(v):
        return None
    s = str(v).strip()
    return s or None


def read_household_15min(
    household_id: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """按需读取单个住户的 15min 时序（不入库；单户约 13 万行）。"""
    path = os.path.join(
        config.HEATPUMP_DATA_DIR, "smart_meter_data", "15min", f"{household_id}.csv"
    )
    if not os.path.exists(path):
        raise FileNotFoundError(f"未找到住户 15min 数据：{path}")
    df = pd.read_csv(path, sep=";", encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    df["ts"] = _parse_ts(df["Timestamp"], as_date=False)
    if start:
        df = df[df["ts"] >= pd.Timestamp(start)]
    if end:
        df = df[df["ts"] <= pd.Timestamp(end)]
    return df.to_dict("records")


# ============================================================================
# 校验
# ============================================================================

TABLES = [
    "dim_entity", "fact_energy_daily", "fact_fee_season", "fact_daily_note",
    "dim_weather_station", "fact_weather_hourly", "fact_weather_daily",
    "dim_substation", "fact_scada_hourly",
    "dim_household", "fact_household_protocol", "fact_household_meter",
]
def main() -> None:
    ap = argparse.ArgumentParser(description="供热运营数据层：建库建表与 ETL 入库")
    ap.add_argument("--init", action="store_true", help="创建数据库与全部表")
    ap.add_argument("--reset", action="store_true", help="删除并重建全部表（清空数据）")
    ap.add_argument(
        "--load",
        choices=["business", "weather", "scada", "household", "all"],
        help="执行 ETL 入库",
    )
    args = ap.parse_args()

    if args.init:
        init_db()

    if args.reset:
        reset_db()

    if args.load:
        engine = get_engine()
        targets = (
            ["business", "weather", "scada", "household"]
            if args.load == "all"
            else [args.load]
        )
        for t in targets:
            logger.info("=== 入库数据域：%s ===", t)
            if t == "business":
                load_business(engine)
            elif t == "weather":
                load_weather(engine)
            elif t == "scada":
                load_scada(engine)
            elif t == "household":
                load_household(engine)
        engine.dispose()

    if not any([args.init, args.reset, args.load]):
        ap.print_help()


if __name__ == "__main__":
    main()
