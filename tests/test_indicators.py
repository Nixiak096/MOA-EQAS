"""指标核算单元测试（需要 MySQL）。

侧重点不在于「算出某个固定数字」，而在于把口径约束钉住：
  * 公式确实按《数据及计算说明》执行（用库中原值复算对照，不写死魔数）；
  * 单位必须标注且具体（取自库中原生单位，而不是"按源量/面积"这类含糊说法）；
  * 算不出来的项必须给出原因，绝不静默缺失；
  * 实体在库中不存在时要点名说明（曾经只报"报表中无该分类数据"，误导性很强）。
"""

import datetime
import unittest

import config
import indicators
from tests.support import (COMPANY, PROJECT_NO_DATA, PROJECT_WITH_DATA, REFERENCE_DATE,
                           require_db)


class TestEnergyIndicators(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        require_db()

    def test_daily_index_equals_quantity_over_area(self):
        with indicators.DataAccess() as da:
            qty = da.energy(COMPANY, REFERENCE_DATE, indicators.ENERGY_CODES["gas"]["daily"])
            area = da.energy(COMPANY, REFERENCE_DATE, indicators.AREA_CODE)
        self.assertIsNotNone(qty)
        self.assertIsNotNone(area)

        res = indicators.compute(COMPANY, REFERENCE_DATE, ["gas"], modes=["daily"])[0]
        self.assertTrue(res.available, res.reason)
        self.assertAlmostEqual(res.value, qty / area, places=9)
        self.assertIn("日气耗量", res.formula)
        self.assertIn("面积", res.formula)

    def test_unit_is_concrete_not_vague(self):
        """回归：指标单位曾经为空，被审核员判为不合格。"""
        res = indicators.compute(COMPANY, REFERENCE_DATE, ["gas"], modes=["daily"])[0]
        self.assertTrue(res.unit)
        self.assertNotIn("按源", res.unit)
        # 气耗单平米指标的原生单位是 Nm3/m2（电耗 kWh/m2、水耗 kg/m2，三者不同）
        with indicators.DataAccess() as da:
            self.assertEqual(res.unit, da.unit(COMPANY, REFERENCE_DATE, "daily_gas_index"))

    def test_all_modes_have_unit_and_formula(self):
        modes = ["daily", "cumulative", "evaluation", "degree_day"]
        for r in indicators.compute(COMPANY, REFERENCE_DATE, ["gas"], modes=modes):
            if not r.available:
                continue
            self.assertTrue(r.unit, f"{r.label} 缺少单位")
            self.assertTrue(r.formula, f"{r.label} 缺少公式说明")

    def test_mom_unavailable_with_single_snapshot(self):
        """库中只有单日快照，环比应当不可用并说明原因。"""
        res = indicators.compute(COMPANY, REFERENCE_DATE, ["gas"], modes=["mom"])[0]
        self.assertFalse(res.available)
        self.assertIn("上一日", res.reason)

    def test_degree_day_requires_temperature_station(self):
        original = config.DEGREE_DAY_STATION
        config.DEGREE_DAY_STATION = ""
        try:
            res = indicators.compute(COMPANY, REFERENCE_DATE, ["gas"], modes=["degree_day"])[0]
            self.assertFalse(res.available)
            self.assertIn("DEGREE_DAY_STATION", res.reason)
        finally:
            config.DEGREE_DAY_STATION = original

    def test_degree_day_formula(self):
        res = indicators.compute(COMPANY, REFERENCE_DATE, ["gas"], modes=["degree_day"])[0]
        if not res.available:
            self.skipTest(f"度日数折标不可用：{res.reason}")
        with indicators.DataAccess() as da:
            qty = da.energy(COMPANY, REFERENCE_DATE, "daily_gas")
            area = da.energy(COMPANY, REFERENCE_DATE, indicators.AREA_CODE)
            temp = da.outdoor_avg_temp(config.DEGREE_DAY_STATION, REFERENCE_DATE)
        expect = qty * ((config.DEGREE_DAY_BASE_TEMP + config.DEGREE_DAY_INDOOR_OFFSET)
                        / (config.DEGREE_DAY_BASE_TEMP - temp)) / area
        self.assertAlmostEqual(res.value, expect, places=9)


class TestFeeIndicators(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        require_db()

    def test_fee_rate_cross_check_consistent(self):
        res = indicators.compute(PROJECT_WITH_DATA, REFERENCE_DATE, ["fee_rate"],
                                 categories=["居民"])[0]
        self.assertTrue(res.available, res.reason)
        cc = res.cross_check
        self.assertIsNotNone(cc)
        self.assertTrue(cc.get("consistent"),
                        f"重算 {cc.get('recomputed')} vs 报表 {cc.get('reported')}")
        self.assertLessEqual(cc["delta_pp"], indicators.RATE_TOLERANCE)
        self.assertAlmostEqual(cc["recomputed"],
                               cc["inputs"]["累计本季实收金额"]
                               / cc["inputs"]["本季应收金额"] * 100, places=6)

    def test_fee_rate_by_category_differs(self):
        house = indicators.compute(PROJECT_WITH_DATA, REFERENCE_DATE, ["fee_rate"],
                                   categories=["居民"])[0]
        nonres = indicators.compute(PROJECT_WITH_DATA, REFERENCE_DATE, ["fee_rate"],
                                    categories=["非居民"])[0]
        self.assertTrue(house.available and nonres.available)
        self.assertEqual(house.unit, "%")
        self.assertNotEqual(house.value, nonres.value,
                            "居民与非居民收费应分别取数，不能取成同一个值")

    def test_company_fee_rate_uses_household_weighting(self):
        """回归：公司级收费率曾被「金额口径」复算，得出 1.79pp 的假不一致。

        源报表两级口径不同（已逐项核对到 0.0000pp）：
          项目级 = 累计本季实收金额 ÷ 本季应收金额
          公司级 = Σ(项目收费率 × 项目本季应收户数) ÷ Σ(项目本季应收户数)
        """
        res = indicators.compute(COMPANY, REFERENCE_DATE, ["fee_rate"],
                                 categories=["非居民"])[0]
        self.assertTrue(res.available, res.reason)
        cc = res.cross_check
        self.assertIn("户数", cc["method"])
        self.assertTrue(cc["consistent"],
                        f"公司级应按户数口径复算，偏差 {cc['delta_pp']:.4f}pp")
        self.assertLessEqual(cc["delta_pp"], indicators.RATE_TOLERANCE)
        self.assertIn("项目", cc["inputs"]["加权构成"])
        self.assertIn("应收户数", res.formula)

    def test_project_fee_rate_uses_amount_formula(self):
        res = indicators.compute(PROJECT_WITH_DATA, REFERENCE_DATE, ["fee_rate"],
                                 categories=["非居民"])[0]
        cc = res.cross_check
        self.assertIn("金额口径", cc["method"])
        self.assertTrue(cc["consistent"])
        self.assertIn("本季应收金额", res.formula)

    def test_company_and_project_rates_use_different_formulas(self):
        """同一分类下，公司级与项目级的「所用公式」必须不同，否则读者会以为能直接平均。"""
        co = indicators.compute(COMPANY, REFERENCE_DATE, ["fee_rate"], categories=["居民"])[0]
        pr = indicators.compute(PROJECT_WITH_DATA, REFERENCE_DATE, ["fee_rate"],
                                categories=["居民"])[0]
        self.assertNotEqual(co.formula, pr.formula)


class TestGroupLevelAggregation(unittest.TestCase):
    """集团级：按《数据及计算说明》「集团 = 各公司之和」聚合，不是查一行现成数据。

    库中当前只有「公司A」一家公司，因此集团级各项应与公司级完全相同 ——
    这正好可以逐项验证聚合是否按同一口径实现（而不是对单平米指标求平均）。
    """

    GROUP = "group:集团"

    @classmethod
    def setUpClass(cls):
        require_db()

    def test_group_child_companies(self):
        with indicators.DataAccess() as da:
            kids = da.child_companies("集团")
        self.assertIn(COMPANY, kids)

    def test_group_equals_company_for_every_energy_mode(self):
        modes = ["daily", "cumulative", "evaluation", "cumulative_evaluation",
                 "degree_day", "season_cumulative",
                 "season_cumulative_evaluation", "mom"]
        for mode in modes:
            g = indicators.compute(self.GROUP, REFERENCE_DATE, ["gas"], modes=[mode])[0]
            c = indicators.compute(COMPANY, REFERENCE_DATE, ["gas"], modes=[mode])[0]
            self.assertEqual(g.available, c.available, f"{mode} 可用性不一致")
            if g.available:
                self.assertAlmostEqual(g.value, c.value, places=9,
                                       msg=f"{mode} 集团级与公司级不一致")
            else:
                self.assertTrue(g.reason, f"{mode} 不可用时必须说明原因")

    def test_group_index_uses_area_sum_not_average(self):
        """集团级单平米指标 = Σ耗量 ÷ Σ面积，而不是各公司指标值的平均。"""
        res = indicators.compute(self.GROUP, REFERENCE_DATE, ["gas"], modes=["daily"])[0]
        self.assertTrue(res.available, res.reason)
        with indicators.DataAccess() as da:
            kids = da.child_companies("集团")
            qty = sum(da.energy(k, REFERENCE_DATE, "daily_gas") or 0 for k in kids)
            area = sum(da.energy(k, REFERENCE_DATE, indicators.AREA_CODE) or 0 for k in kids)
        self.assertAlmostEqual(res.value, qty / area, places=9)
        self.assertIn("面积之和", res.formula)
        self.assertIn("参与聚合的公司", res.inputs)

    def test_group_fee_rate_equals_company_when_single_company(self):
        for cat in ("小计", "居民", "非居民"):
            g = indicators.compute(self.GROUP, REFERENCE_DATE, ["fee_rate"],
                                   categories=[cat])[0]
            c = indicators.compute(COMPANY, REFERENCE_DATE, ["fee_rate"],
                                   categories=[cat])[0]
            self.assertTrue(g.available, g.reason)
            self.assertAlmostEqual(g.value, c.value, places=9, msg=cat)
            self.assertIn("集团级", g.label)
            self.assertIn("各公司之和", g.formula)

    def test_group_without_member_companies_is_honest(self):
        res = indicators.compute("group:不存在的集团", REFERENCE_DATE, ["gas"],
                                 modes=["daily"])[0]
        self.assertFalse(res.available)
        self.assertIn("没有归属", res.reason)

    def test_group_agg_field_not_exposed_in_api_dict(self):
        res = indicators.compute(self.GROUP, REFERENCE_DATE, ["gas"], modes=["daily"])[0]
        self.assertNotIn("agg", res.to_dict(), "内部分量不应出现在对外结构里")


class TestFormulaNamingAndInputUnits(unittest.TestCase):
    """报告可核对性：公式名要与指标名一致，计算输入要带单位。"""

    @classmethod
    def setUpClass(cls):
        require_db()

    def test_formula_uses_metric_specific_name(self):
        for metric, word in (("gas", "气耗"), ("water", "水耗"), ("elec", "电耗")):
            res = indicators.compute(COMPANY, REFERENCE_DATE, [metric], modes=["daily"])[0]
            if not res.available:
                continue
            self.assertIn(word, res.formula,
                          f"{metric} 的公式应写具体能源名，而不是通用「日耗指标值」")
            self.assertIn(word, res.label)

    def test_inputs_are_unit_annotated(self):
        res = indicators.compute(COMPANY, REFERENCE_DATE, ["gas"], modes=["daily"])[0]
        units = res.input_units
        self.assertTrue(units, "计算输入必须逐项标注单位")
        for key in res.inputs:
            self.assertIn(key, units, f"输入项 {key} 缺少单位")
        self.assertEqual(units["面积"], "m2")
        self.assertEqual(units["日气耗量"], "Nm3")

    def test_degree_day_inputs_units(self):
        res = indicators.compute(COMPANY, REFERENCE_DATE, ["gas"], modes=["degree_day"])[0]
        if not res.available:
            self.skipTest(f"度日数折标不可用：{res.reason}")
        self.assertEqual(res.input_units["当天室外平均温度"], "℃")
        self.assertEqual(res.input_units["折标系数"], "无量纲")

    def test_fee_inputs_units(self):
        res = indicators.compute(PROJECT_WITH_DATA, REFERENCE_DATE, ["fee_rate"],
                                 categories=["居民"])[0]
        units = res.cross_check["input_units"]
        self.assertEqual(units["本季应收金额"], "万元")

    def test_to_dict_carries_input_units(self):
        res = indicators.compute(COMPANY, REFERENCE_DATE, ["gas"], modes=["daily"])[0]
        self.assertIn("input_units", res.to_dict())


class TestAvailabilityDiscipline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        require_db()

    def test_missing_entity_names_the_entity_and_lists_available(self):
        """回归：实体不存在时曾只报"报表中无该分类数据"，看不出是实体本身没有数据。"""
        res = indicators.compute(PROJECT_NO_DATA, REFERENCE_DATE, ["gas"], modes=["daily"])[0]
        self.assertFalse(res.available)
        self.assertIn(PROJECT_NO_DATA, res.reason)
        self.assertIn(COMPANY, res.reason)          # 列出现有实体，便于纠错

    def test_missing_entity_covers_every_metric_and_category(self):
        res = indicators.compute(PROJECT_NO_DATA, REFERENCE_DATE,
                                 ["gas", "fee_rate"], modes=["daily"],
                                 categories=["居民", "非居民"])
        self.assertEqual(len(res), 3)               # 1 个能耗口径 + 2 个收费分类
        for r in res:
            self.assertFalse(r.available)
            self.assertIn(PROJECT_NO_DATA, r.reason)

    def test_every_unavailable_result_states_a_reason(self):
        res = indicators.compute(COMPANY, REFERENCE_DATE, ["gas"],
                                 modes=["daily", "mom", "cumulative", "degree_day"])
        for r in res:
            if not r.available:
                self.assertTrue(r.reason, f"{r.label} 不可用但未说明原因")

    def test_unsupported_metric_is_reported(self):
        res = indicators.compute(COMPANY, REFERENCE_DATE, ["no_such_metric"])[0]
        self.assertFalse(res.available)
        self.assertIn("未支持", res.reason)


class TestComputeFromIntent(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        require_db()

    def test_intent_driven_compute(self):
        import intent_parser as ip
        it = ip.parse_rules("公司A今天的气耗指标是多少？", reference_date=REFERENCE_DATE)
        res = indicators.compute_from_intent(it)
        self.assertTrue(res)
        self.assertTrue(all(r.entity_id == COMPANY for r in res))
        self.assertTrue(any(r.available for r in res))

    def test_non_enterprise_intent_yields_nothing(self):
        import intent_parser as ip
        it = ip.parse_rules("国家标准对室内温度的要求是什么？",
                            reference_date=REFERENCE_DATE)
        self.assertEqual(indicators.compute_from_intent(it), [])

    def test_season_question_does_not_fall_back_to_today(self):
        """回归：命中「本季」时 stat_date 曾为 None → 落到 today() → 查无数据。"""
        import intent_parser as ip
        it = ip.parse_rules("项目5的本季收费率是多少？", reference_date=REFERENCE_DATE)
        res = indicators.compute_from_intent(it)
        self.assertTrue(any(r.available for r in res),
                        f"本季口径应当能取到数据：{[(r.label, r.reason) for r in res]}")
        self.assertNotEqual(res[0].stat_date, datetime.date.today())


if __name__ == "__main__":
    unittest.main()
