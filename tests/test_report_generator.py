"""报告生成单元测试。

报告是交付物本体，硬约束有三条（课题要求「含数据、结论、来源」）：
  * 可用指标进「结论」，不可用指标进「口径与数据说明」并写明原因 —— 不猜测；
  * 每份报告必须标注数据来源（审核员据此校验引用）；
  * 图表以「规格」形式产出（kind/x/series），前端才能直接渲染。
"""

import os
import unittest

import report_generator as rg
from tests.support import (COMPANY, SUBSTATION, WEATHER_ID, indicator, require_db)


class TestIndicatorReport(unittest.TestCase):
    def test_unavailable_items_go_to_caveats_not_conclusions(self):
        ok = indicator("日气耗指标值", value=0.0685)
        bad = indicator("日气耗指标值环比", available=False, value=None, unit="",
                        reason="缺少上一日数据（当前库中仅单日快照）")
        rep = rg.build_indicator_report([ok, bad])
        self.assertEqual(len(rep.sections["结论"]), 1)
        self.assertIn("日气耗指标值", rep.sections["结论"][0])
        self.assertNotIn("环比", " ".join(rep.sections["结论"]))
        self.assertTrue(any("缺少上一日数据" in c for c in rep.caveats))

    def test_sources_always_present(self):
        rep = rg.build_indicator_report([indicator("日气耗指标值")])
        self.assertTrue(rep.sources)

    def test_table_and_formula_present(self):
        rep = rg.build_indicator_report([indicator("日气耗指标值")])
        rows = rep.sections["指标数据"]
        self.assertTrue(rows[0].startswith("| 指标"))
        self.assertIn("日耗指标值 = 日耗量 / 面积", " ".join(rows))

    def test_all_fields_present_for_failure_case(self):
        rep = rg.build_indicator_report([
            indicator("日气耗指标值", available=False, value=None, unit="",
                      reason="库中不存在该实体"),
        ])
        self.assertNotIn("结论", rep.sections)
        self.assertEqual(len(rep.caveats), 1)

    def test_chart_spec_shape(self):
        rep = rg.build_indicator_report([
            indicator("日气耗指标值", mode="daily", value=0.0685),
            indicator("累计气耗指标值", mode="cumulative", value=1.742),
        ])
        self.assertEqual(len(rep.charts), 1)
        spec = rep.charts[0].to_dict()
        self.assertEqual(spec["kind"], "bar")
        self.assertEqual(spec["x"], ["日气耗指标值", "累计气耗指标值"])
        self.assertEqual(len(spec["series"]), 1)
        self.assertEqual(len(spec["series"][0]["data"]), 2)
        for key in ("kind", "title", "x", "series", "y_unit", "note"):
            self.assertIn(key, spec)

    def test_no_chart_for_fee_only_results(self):
        """收费率是季口径，不进「日/累计/评价值」对比图。"""
        rep = rg.build_indicator_report([
            indicator("本季收费率(居民)", mode="season", value=82.6, unit="%"),
        ])
        self.assertEqual(rep.charts, [])

    def test_to_dict_and_markdown(self):
        rep = rg.build_indicator_report([indicator("日气耗指标值")])
        d = rep.to_dict()
        for key in ("title", "subtitle", "generated_at", "sections",
                    "charts", "sources", "caveats"):
            self.assertIn(key, d)
        md = rep.to_markdown()
        self.assertIn("## 结论", md)
        self.assertIn("## 数据来源", md)
        self.assertIn("日气耗指标值", md)


class TestScadaReport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        require_db()

    def test_scada_report_structure(self):
        import analytics

        sa = analytics.scada_analysis(SUBSTATION)
        rep = rg.build_scada_report(sa)
        self.assertIn("结论", rep.sections)
        self.assertIn("指标数据", rep.sections)
        self.assertTrue(rep.charts, "温差调节特性图应作为规格产出")
        self.assertEqual(rep.charts[0].kind, "line")
        self.assertEqual(rep.charts[0].y_unit, "℃")
        self.assertGreaterEqual(len(rep.sources), 1)

    def test_anomaly_sample_heading_states_it_is_a_sample(self):
        """回归：异常提示只列 8 条却不说抽样，会被误读成"只有 8 条异常"。"""
        import analytics

        sa = analytics.scada_analysis(SUBSTATION)
        total = sa["deviation_anomalies"]["count"]
        if total <= 8:
            self.skipTest("越界样本不足 8 条，无需抽样标注")
        rep = rg.build_scada_report(sa)
        heading = [h for h in rep.sections if h.startswith("异常提示")]
        self.assertTrue(heading)
        self.assertIn(f"共 {total} 条", heading[0])
        self.assertIn("抽样示意", heading[0])

    def test_unavailable_scada_reports_reason(self):
        rep = rg.build_scada_report({"available": False, "reason": "换热站 L999 无 SCADA 数据"})
        self.assertIn("L999", " ".join(rep.caveats))
        self.assertFalse(rep.sections)


class TestSmartMeterDictionary(unittest.TestCase):
    """智能电表字段字典必须覆盖数据里真实存在的每一列。

    该字典是本项目自行整理的（数据源未提供），一旦新增分辨率或列名变化就会漏字段，
    而字段释义类问题只能靠它回答。这里以真实表头为准做覆盖性校验。
    """

    HP = "documents/02_heapo_heatpump/smart_meter_data"

    def _real_columns(self):
        import io
        cols = {}
        for sub in ("overview", "cumulative_counters", "daily", "monthly", "15min"):
            d = os.path.join(self.HP, sub)
            if not os.path.isdir(d):
                continue
            files = sorted(os.listdir(d))
            # overview 必须全读；时序目录各取前几个足以确认表头
            for fn in (files if sub == "overview" else files[:3]):
                with io.open(os.path.join(d, fn), encoding="utf-8", errors="replace") as f:
                    for c in f.readline().strip().split(";"):
                        if c.strip():
                            cols.setdefault(c.strip(), set()).add(sub)
        return cols

    def test_dictionary_covers_every_column(self):
        import csv
        real = self._real_columns()
        path = os.path.join(self.HP, "smart_meter_variables.csv")
        with open(path, encoding="utf-8") as f:
            documented = {r["VariableName"].strip() for r in csv.DictReader(f, delimiter=";")}
        missing = sorted(set(real) - documented)
        self.assertFalse(missing, f"以下数据列没有字段说明：{missing}")

    def test_dictionary_has_no_phantom_columns(self):
        import csv
        real = self._real_columns()
        path = os.path.join(self.HP, "smart_meter_variables.csv")
        with open(path, encoding="utf-8") as f:
            documented = {r["VariableName"].strip() for r in csv.DictReader(f, delimiter=";")}
        phantom = sorted(documented - set(real))
        self.assertFalse(phantom, f"字典里有数据中不存在的列：{phantom}")

    def test_key_field_documents_sparsity(self):
        """kWh_received_HeatPump 的关键事实：本数据集中绝大多数行为空。"""
        with open(os.path.join(self.HP, "smart_meter_variables.csv"), encoding="utf-8") as f:
            text = f.read()
        row = [l for l in text.splitlines() if l.startswith("kWh_received_HeatPump")]
        self.assertTrue(row)
        self.assertIn("稀疏", row[0])
        self.assertIn("3.8", row[0])
        self.assertIn("kWh_received_Total", row[0], "应给出无值时的替代口径")


class TestOtherReports(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        require_db()

    def test_household_report(self):
        import analytics

        with analytics.DataReader() as dr:
            top = dr.household_top(limit=1)
        if not top:
            self.skipTest("库中无住户数据")
        ha = analytics.household_analysis(top[0]["household_id"])
        rep = rg.build_household_report(ha)
        if not ha.get("available"):
            self.assertTrue(rep.caveats)
            return
        self.assertIn("结论", rep.sections)
        self.assertTrue(rep.sources)
        self.assertTrue(rep.charts)
        self.assertEqual(rep.charts[0].kind, "bar")

    def test_weather_report(self):
        import analytics

        wt = analytics.weather_trend(WEATHER_ID)
        rep = rg.build_weather_report(wt)
        if not wt.get("available"):
            self.assertTrue(rep.caveats)
            return
        self.assertIn("结论", rep.sections)
        self.assertTrue(rep.sources)
        self.assertEqual(rep.charts[0].kind, "line")
        self.assertEqual(rep.charts[0].y_unit, "℃")

    def test_weather_chart_survives_downed_points(self):
        """回归：趋势对象不带全量点，报告曾据此判空，导致气温趋势图永远是空的。"""
        import analytics

        wt = analytics.weather_trend(WEATHER_ID)
        if not wt.get("available"):
            self.skipTest("气象站不可用")
        self.assertNotIn("points", wt["trend"], "趋势字典刻意不带全量点")
        self.assertTrue(wt.get("chart_points"), "必须另外给出抽稀绘图序列")
        self.assertLessEqual(len(wt["chart_points"]), 130)
        self.assertGreaterEqual(wt.get("chart_step"), 1)

        chart = rg.build_weather_report(wt).charts[0]
        self.assertEqual(len(chart.x), len(wt["chart_points"]))
        self.assertEqual(len(chart.series[0]["data"]), len(wt["chart_points"]))


if __name__ == "__main__":
    unittest.main()
