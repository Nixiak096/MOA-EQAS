"""测试公共设施：环境探测、跳过判定、样例数据坐标、确定性模式。"""

import datetime
import functools
import unittest

import config

# 关闭大模型：测试要离线、快、可复现。仍然走完整四智能体协作与全部确定性校验。
config.AGENT_USE_LLM = False

# ---- 样例数据坐标（来自 documents/ 下的业务报表，见 heat_db.py 入库结果）----
# 有数据的实体：公司A、项目 1/4/5/6/7；dim_entity 里另有项目 2/3/8 但无任何事实数据
COMPANY = "company:公司A"
PROJECT_WITH_DATA = "project:公司A:项目5"      # 有完整能耗与收费数据
PROJECT_NO_DATA = "project:公司A:项目3"        # 维度表登记，但事实表无数据
SUBSTATION = "L4"
WEATHER_ID = "8jB"
REFERENCE_DATE = datetime.date(2023, 12, 16)   # 库中唯一统计日


def project_root_date():
    """库中最新统计日期；库不可用时返回 None。"""
    try:
        import heat_db
        from sqlalchemy import text

        eng = heat_db.get_engine()
        with eng.connect() as c:
            d = c.execute(text("SELECT MAX(stat_date) FROM fact_energy_daily")).scalar()
        eng.dispose()
        return d
    except Exception:  # noqa: BLE001
        return None


@functools.lru_cache(maxsize=1)
def db_available() -> bool:
    return project_root_date() is not None


@functools.lru_cache(maxsize=1)
def kb_state():
    """(可用性, 说明)。知识库被其他进程（如正在运行的 api_server）占用时不可用。"""
    try:
        import api_server

        st = api_server._knowledge.status()
        return bool(st["available"]), st.get("error") or ""
    except Exception as e:  # noqa: BLE001
        return False, str(e)


def require_db() -> None:
    if not db_available():
        raise unittest.SkipTest(
            "关系库不可用（未启动 MySQL 或未执行 python heat_db.py --load all），跳过")


def require_kb() -> None:
    ok, why = kb_state()
    if not ok:
        raise unittest.SkipTest(f"知识库不可用，跳过：{why[:120]}")


def by_item(review, name):
    """按检查项名称取审核结果，避免依赖检查项顺序。"""
    for c in review.get("checks", []):
        if c["item"] == name:
            return c
    raise AssertionError(f"审核结果中没有检查项「{name}」，实际有："
                         f"{[c['item'] for c in review.get('checks', [])]}")


def indicator(label, **over):
    """构造一个 IndicatorResult，用于不依赖数据库的单元测试。"""
    from indicators import IndicatorResult

    base = dict(entity_id=COMPANY, stat_date=REFERENCE_DATE, metric="gas",
                mode="daily", label=label, value=1.0, unit="Nm3/m2",
                formula="日耗指标值 = 日耗量 / 面积",
                source="fact_energy_daily.daily_gas", available=True)
    base.update(over)
    return IndicatorResult(**base)
