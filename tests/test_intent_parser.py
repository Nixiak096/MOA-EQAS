"""意图解析单元测试（纯逻辑，不依赖数据库与大模型）。

重点覆盖两处曾经出错的地方，做成回归用例：
  * 命中「本季」这类季节口径时，stat_date 仍须落到具体统计日（否则下游退化成 today() 查无数据）；
  * 大模型不得推翻规则已经识别出的数据域（曾把「气耗/水耗/电耗」误判成换热站域，
    导致分析师空转、报告只剩检索条文）。
"""

import datetime
import unittest

import intent_parser as ip

from tests.support import REFERENCE_DATE


class TestDateResolution(unittest.TestCase):
    def test_relative_today(self):
        it = ip.parse_rules("公司A今天的气耗指标是多少？", reference_date=REFERENCE_DATE)
        self.assertEqual(it.stat_date, REFERENCE_DATE)

    def test_relative_yesterday(self):
        it = ip.parse_rules("公司A昨天的气耗指标是多少？", reference_date=REFERENCE_DATE)
        self.assertEqual(it.stat_date, REFERENCE_DATE - datetime.timedelta(days=1))

    def test_explicit_date_wins(self):
        it = ip.parse_rules("公司A2023年12月1日的气耗指标是多少？",
                            reference_date=REFERENCE_DATE)
        self.assertEqual(it.stat_date, datetime.date(2023, 12, 1))

    def test_season_word_still_resolves_stat_date(self):
        """回归：「本季」是口径标签，取数仍需要一个具体 stat_date。"""
        it = ip.parse_rules("项目5的本季收费率是多少？", reference_date=REFERENCE_DATE)
        self.assertEqual(it.season, "2023-2024")
        self.assertEqual(it.stat_date, REFERENCE_DATE,
                         "命中季节口径时 stat_date 不能留空，否则会退化成 today() 而查无数据")

    def test_unspecified_date_defaults_to_reference(self):
        it = ip.parse_rules("公司A的气耗指标是多少？", reference_date=REFERENCE_DATE)
        self.assertEqual(it.stat_date, REFERENCE_DATE)


class TestDomainRouting(unittest.TestCase):
    def test_energy_is_enterprise(self):
        it = ip.parse_rules("公司A今天的气耗指标是多少？", reference_date=REFERENCE_DATE)
        self.assertEqual(it.domain, "enterprise")
        self.assertEqual(it.metrics, ["gas"])

    def test_substation_is_scada(self):
        it = ip.parse_rules("L4换热站昨天的二次侧供回水温差正常吗？",
                            reference_date=REFERENCE_DATE)
        self.assertEqual(it.domain, "scada")
        self.assertEqual(it.substations, ["L4"])

    def test_household_domain(self):
        it = ip.parse_rules("住户的电表用电量排名前几名是谁？", reference_date=REFERENCE_DATE)
        self.assertEqual(it.domain, "household")

    def test_weather_domain(self):
        it = ip.parse_rules("最近的气象温度趋势怎么样？", reference_date=REFERENCE_DATE)
        self.assertEqual(it.domain, "weather")

    def test_standard_question_is_knowledge(self):
        it = ip.parse_rules("国家标准对室内温度的要求是什么？", reference_date=REFERENCE_DATE)
        self.assertEqual(it.domain, "knowledge")


class TestMetricsAndModes(unittest.TestCase):
    def test_metric_aliases(self):
        it = ip.parse_rules("公司A的燃气用气量是多少？", reference_date=REFERENCE_DATE)
        self.assertIn("gas", it.metrics)

    def test_multiple_metrics(self):
        it = ip.parse_rules("公司A今天的气耗、水耗和电耗指标分别是多少？",
                            reference_date=REFERENCE_DATE)
        self.assertEqual(set(it.metrics), {"gas", "water", "elec"})

    def test_cumulative_mode(self):
        it = ip.parse_rules("公司A今天的累计气耗指标值是多少？", reference_date=REFERENCE_DATE)
        self.assertIn("cumulative", it.modes)

    def test_compare_mom(self):
        it = ip.parse_rules("公司A今天的气耗和昨天比有什么变化？", reference_date=REFERENCE_DATE)
        self.assertEqual(it.compare, "mom")
        self.assertIn("mom", it.modes)

    def test_categories_resident_split(self):
        it = ip.parse_rules("项目5的本季收费率居民和非居民分别是多少？",
                            reference_date=REFERENCE_DATE)
        self.assertEqual(set(it.categories), {"居民", "非居民"})

    def test_all_projects_marks_project_level_without_list(self):
        """「各项目」只标明层级，具体项目列表由分析师按库中实体展开。

        entity_ids() 因此返回空列表 —— 这是刻意的：编排器据此走多项目横向对比分支。
        """
        it = ip.parse_rules("各项目今天的气耗指标是多少？", reference_date=REFERENCE_DATE)
        self.assertEqual(it.level, "project")
        self.assertEqual(it.projects, [])
        self.assertEqual(it.entity_ids(), [])
        self.assertTrue(any("待补全" in n for n in it.notes))

    def test_named_projects_expand_to_entity_ids(self):
        it = ip.parse_rules("项目1和项目5今天的气耗指标是多少？", reference_date=REFERENCE_DATE)
        self.assertEqual(it.entity_ids(),
                         ["project:公司A:项目1", "project:公司A:项目5"])


class TestLlmCannotOverrideRules(unittest.TestCase):
    """大模型只做「补齐」，不得推翻规则已识别的数据域。"""

    def setUp(self):
        self._orig = ip.parse_with_llm

    def tearDown(self):
        ip.parse_with_llm = self._orig

    def _inject(self, payload):
        ip.parse_with_llm = lambda q: payload

    def test_scada_hallucination_rejected(self):
        self._inject({"domain": "scada", "metrics": [], "modes": [], "substations": []})
        it = ip.parse("公司A今天的气耗、水耗和电耗指标分别是多少？",
                      reference_date=REFERENCE_DATE, use_llm=True)
        self.assertEqual(it.domain, "enterprise", "无站号的换热站判定不应覆盖数据域")
        self.assertEqual(set(it.metrics), {"gas", "water", "elec"})

    def test_household_hallucination_rejected(self):
        self._inject({"domain": "household", "metrics": [], "modes": [], "substations": []})
        it = ip.parse("公司A今天的气耗指标是多少？",
                      reference_date=REFERENCE_DATE, use_llm=True)
        self.assertEqual(it.domain, "enterprise")
        self.assertTrue(any("保留规则判定" in n for n in it.notes),
                        "拒绝大模型判定时应在 notes 里说明原因")

    def test_scada_accepted_when_substation_given(self):
        self._inject({"domain": "scada", "metrics": [], "modes": [],
                      "substations": ["L4"]})
        it = ip.parse("换热站运行工况怎么样？", reference_date=REFERENCE_DATE, use_llm=True)
        self.assertEqual(it.domain, "scada")
        self.assertEqual(it.substations, ["L4"])

    def test_knowledge_accepted_only_without_metrics(self):
        self._inject({"domain": "knowledge", "metrics": [], "modes": [], "substations": []})
        it = ip.parse("供热系统由哪些部分组成？", reference_date=REFERENCE_DATE, use_llm=True)
        self.assertEqual(it.domain, "knowledge")

        it2 = ip.parse("公司A今天的气耗指标是多少？",
                       reference_date=REFERENCE_DATE, use_llm=True)
        self.assertEqual(it2.domain, "enterprise",
                         "规则已识别出指标时，不应被大模型改判为知识类")


class TestIntentSerialization(unittest.TestCase):
    def test_to_dict_fields(self):
        it = ip.parse_rules("项目5的本季收费率是多少？", reference_date=REFERENCE_DATE)
        d = it.to_dict()
        for key in ("raw", "domain", "metrics", "modes", "level", "company",
                    "projects", "categories", "stat_date", "season", "notes"):
            self.assertIn(key, d)
        self.assertEqual(d["stat_date"], REFERENCE_DATE.isoformat())
        self.assertEqual(d["projects"], ["项目5"])

    def test_to_dict_stat_date_none_safe(self):
        it = ip.parse_rules("国家标准对室内温度的要求是什么？")
        self.assertIsNone(it.stat_date)
        self.assertIsNone(it.to_dict()["stat_date"])


if __name__ == "__main__":
    unittest.main()
