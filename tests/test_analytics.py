"""统计分析单元测试 / 集成测试。

异常检测部分是纯函数，用构造序列做单元测试（能精确控制"应当检出/不应检出"）；
取数部分是集成测试，需要 MySQL。
"""

import datetime
import unittest

import analytics
from tests.support import (COMPANY, REFERENCE_DATE, SUBSTATION, WEATHER_ID, require_db)


def series(values, start=datetime.date(2023, 11, 1)):
    return [(start + datetime.timedelta(days=i), v) for i, v in enumerate(values)]


class TestTrend(unittest.TestCase):
    def test_too_few_points_unavailable(self):
        r = analytics.trend_from_points("d", "m", "标签", series([1.0, 2.0]))
        self.assertFalse(r.available)
        self.assertIn("不足以判断趋势", r.reason)

    def test_empty_unavailable(self):
        r = analytics.trend_from_points("d", "m", "标签", [])
        self.assertFalse(r.available)
        self.assertEqual(r.reason, "无数据点")

    def test_rising_and_falling(self):
        up = analytics.trend_from_points("d", "m", "标签", series([1, 2, 3, 4, 5]))
        self.assertEqual(up.direction, "上升")
        self.assertGreater(up.slope_per_unit, 0)

        down = analytics.trend_from_points("d", "m", "标签", series([5, 4, 3, 2, 1]))
        self.assertEqual(down.direction, "下降")

    def test_flat(self):
        r = analytics.trend_from_points("d", "m", "标签", series([10, 10, 10, 10]))
        self.assertEqual(r.direction, "平稳")

    def test_stats(self):
        r = analytics.trend_from_points("d", "m", "标签", series([2, 4, 4, 4, 5, 5, 7, 9]))
        self.assertEqual(r.stats.n, 8)
        self.assertAlmostEqual(r.stats.mean, 5.0)
        self.assertAlmostEqual(r.stats.median, 4.5)


class TestAnomalyDetection(unittest.TestCase):
    def test_threshold_finds_injected_spike(self):
        pts = series([10.0] * 9 + [100.0])
        found = analytics.detect_threshold(pts, label="用气量")
        self.assertTrue(found)
        self.assertEqual(found[-1].ts, str(pts[-1][0]))
        self.assertEqual(found[-1].kind, "threshold")
        self.assertIn("σ", found[-1].detail)

    def test_threshold_needs_enough_points(self):
        self.assertEqual(analytics.detect_threshold(series([1, 2, 100])), [])

    def test_threshold_no_false_positive_on_flat_series(self):
        self.assertEqual(analytics.detect_threshold(series([5.0] * 20)), [])

    def test_jump_percent_mode(self):
        found = analytics.detect_jump(series([100.0, 100.0, 200.0]), pct=30.0)
        self.assertEqual(len(found), 1)
        self.assertIn("+100.00%", found[0].detail)

    def test_jump_ignores_pct_when_abs_delta_given(self):
        """气温用绝对阈值：日间自然波动常超 30%，用相对率会大量误报。"""
        pts = series([5.0, 15.0])
        self.assertTrue(analytics.detect_jump(pts, pct=30.0))
        self.assertEqual(analytics.detect_jump(pts, abs_delta=15.0), [])
        self.assertTrue(analytics.detect_jump(pts, abs_delta=6.0, unit="℃"))

    def test_jump_skips_zero_baseline(self):
        """相对变化率以 0 为分母无意义，代码应跳过而不是抛异常或报 ——。"""
        self.assertEqual(analytics.detect_jump(series([0.0, 10.0]), pct=30.0), [])

    def test_missing_detects_gap(self):
        pts = [(datetime.date(2023, 11, 1), 1.0),
               (datetime.date(2023, 11, 2), 1.0),
               (datetime.date(2023, 11, 6), 1.0)]
        found = analytics.detect_missing(pts, expected_gap_days=1)
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0].kind, "missing")


class TestDataBackedAnalysis(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        require_db()

    def test_compare_projects_sorted_and_consistent(self):
        res = analytics.compare_projects("daily_gas_index", company="公司A",
                                         stat_date=REFERENCE_DATE)
        self.assertTrue(res.available, res.reason)
        self.assertGreaterEqual(len(res.entries), 2)
        vals = [e["value"] for e in res.entries]
        self.assertEqual(vals, sorted(vals, reverse=True), "应按数值降序排列")
        self.assertAlmostEqual(res.max, max(vals))
        self.assertAlmostEqual(res.min, min(vals))
        self.assertAlmostEqual(res.mean, sum(vals) / len(vals))
        self.assertEqual(res.max_entity, res.entries[0]["label"])

    def test_scada_analysis_structure(self):
        res = analytics.scada_analysis(SUBSTATION)
        self.assertTrue(res.get("available"), res.get("reason"))
        self.assertGreater(res["n_points"], 0)
        self.assertEqual(res["delta_t"]["unit"], "℃")
        self.assertEqual(len(res["delta_t"]["reasonable_range"]), 2)
        self.assertIn("count", res["deviation_anomalies"])
        stats = res["delta_t"]["stats"]
        self.assertLessEqual(stats["minimum"], stats["mean"])
        self.assertLessEqual(stats["mean"], stats["maximum"])

    def test_scada_unknown_substation(self):
        res = analytics.scada_analysis("L999")
        self.assertFalse(res["available"])
        self.assertIn("无 SCADA 数据", res["reason"])

    def test_weather_trend_structure(self):
        res = analytics.weather_trend(WEATHER_ID)
        if not res.get("available"):
            self.skipTest(f"气象站 {WEATHER_ID} 无数据：{res.get('reason')}")
        self.assertIsInstance(res["hdd_sia_sum"], float)
        self.assertIn(res["trend"]["direction"], ("上升", "下降", "平稳"))
        for key in ("threshold", "jump", "missing"):
            self.assertIn(key, res["anomalies"])

    def test_household_top_is_a_reader_method(self):
        """回归：household_top 是 DataReader 的方法；曾按模块级函数调用导致 AttributeError。"""
        self.assertFalse(hasattr(analytics, "household_top"),
                         "household_top 只作为 DataReader 方法存在")
        with analytics.DataReader() as dr:
            rows = dr.household_top(limit=5)
        self.assertEqual(len(rows), 5)
        self.assertIn("household_id", rows[0])
        self.assertIn("avg_kwh", rows[0])
        vals = [r["avg_kwh"] for r in rows]
        self.assertEqual(vals, sorted(vals, reverse=True))

    def test_enterprise_anomalies(self):
        res = analytics.enterprise_anomalies(COMPANY, "daily_gas_index")
        d = res.to_dict()
        self.assertIn("available", d)
        self.assertIn("anomalies", d)

    def test_data_reader_lists_dimensions(self):
        with analytics.DataReader() as dr:
            self.assertIn(SUBSTATION, dr.list_substations())
            self.assertIn(WEATHER_ID, dr.list_stations())
            self.assertEqual(dr.latest_stat_date(), REFERENCE_DATE)
            ids = [e[0] for e in dr.list_entities("project", "公司A")]
        self.assertIn("project:公司A:项目1", ids)

    def test_dimension_lists_entity_without_data(self):
        """维度表登记 ≠ 有数据：项目3 在 dim_entity 里但事实表无行。

        list_entities 读的是维度表，因此会列出它；取数/对比环节必须跳过它，
        不能让「查不到值」被当成 0 或静默丢弃。
        """
        with analytics.DataReader() as dr:
            self.assertIn("project:公司A:项目3", [e[0] for e in dr.list_entities("project", "公司A")])
            self.assertIsNone(dr.energy_at("project:公司A:项目3", REFERENCE_DATE,
                                           "daily_gas_index"))

        res = analytics.compare_projects("daily_gas_index", company="公司A",
                                        stat_date=REFERENCE_DATE)
        self.assertNotIn("project:公司A:项目3", [e["entity_id"] for e in res.entries])


if __name__ == "__main__":
    unittest.main()
