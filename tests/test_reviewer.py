"""审核员单元测试（构造上下文，不依赖数据库与大模型）。

审核员是这套系统「结论可信」的兜底：它必须以确定性规则逐项校验，
不通过就带意见退回重算，且**不可以**放行静默缺失或空转的结果。
这里把每一项校验的正反两面都钉住。
"""

import unittest

import intent_parser as ip
import report_generator as rg
from multi_agents import AgentContext, ReviewerAgent
from tests.support import REFERENCE_DATE, by_item, indicator


def make_intent(domain="knowledge", metrics=None, categories=None):
    it = ip.QueryIntent(raw="测试")
    it.domain = domain
    it.metrics = metrics or []
    it.categories = categories or []
    it.stat_date = REFERENCE_DATE
    it.company = "公司A"
    return it


def make_report(sections=True, sources=True):
    rep = rg.Report(title="测试报告", generated_at="2026-01-01 00:00:00")
    if sections:
        rep.add("结论", ["日气耗指标值 = **0.0685 Nm3/m2**"])
    if sources:
        rep.add_source("fact_energy_daily.daily_gas")
    return rep


def review_of(indicators=None, intent=None, report=None, analytics=None):
    ctx = AgentContext(question="测试")
    ctx.intent = intent if intent is not None else make_intent()
    ctx.indicators = indicators or []
    ctx.report = report if report is not None else make_report()
    ctx.analytics = analytics or {}
    return ReviewerAgent(use_llm=False).run(ctx)


class TestAnalysisExecuted(unittest.TestCase):
    """回归：分析师领域选错且无实体时会静默空转，报告只剩检索条文，曾经照样判合格。"""

    def test_silent_noop_is_rejected(self):
        rev = review_of(indicators=[], intent=make_intent(domain="enterprise", metrics=["gas"]),
                        report=make_report(sections=False, sources=False))
        self.assertEqual(by_item(rev, "分析执行")["status"], "fail")
        self.assertEqual(rev["review_status"], "rejected")

    def test_declared_block_passes_with_reason(self):
        rev = review_of(indicators=[], intent=make_intent(domain="scada"),
                        report=make_report(sections=False, sources=False),
                        analytics={"blocked": "未解析出换热站编号（形如 L4）"})
        check = by_item(rev, "分析执行")
        self.assertEqual(check["status"], "pass")
        self.assertIn("未解析出换热站编号", check["comment"])

    def test_produced_results_pass(self):
        rev = review_of(indicators=[indicator("日气耗指标值")])
        self.assertEqual(by_item(rev, "分析执行")["status"], "pass")

    def test_knowledge_domain_passes_without_results(self):
        rev = review_of(indicators=[], intent=make_intent(domain="knowledge"),
                        report=make_report(sections=False, sources=False),
                        analytics={"note": "规范/方法类问题，数据核算不适用"})
        self.assertEqual(by_item(rev, "分析执行")["status"], "pass")


class TestFormulaAndUnit(unittest.TestCase):
    def test_missing_formula_fails(self):
        rev = review_of(indicators=[indicator("日气耗指标值", formula="")])
        self.assertEqual(by_item(rev, "公式依据")["status"], "fail")

    def test_missing_unit_fails(self):
        rev = review_of(indicators=[indicator("日气耗指标值", unit="")])
        self.assertEqual(by_item(rev, "单位标注")["status"], "fail")

    def test_season_mode_exempt_from_unit(self):
        rev = review_of(indicators=[indicator("本季收费率(小计)", unit="", mode="season")])
        self.assertEqual(by_item(rev, "单位标注")["status"], "pass")

    def test_available_with_unit_and_formula_passes(self):
        rev = review_of(indicators=[indicator("日气耗指标值")])
        self.assertEqual(by_item(rev, "公式依据")["status"], "pass")
        self.assertEqual(by_item(rev, "单位标注")["status"], "pass")


class TestUnavailableDiscipline(unittest.TestCase):
    def test_unavailable_without_reason_fails(self):
        bad = indicator("环比", available=False, value=None, reason="", unit="")
        rev = review_of(indicators=[bad], analytics={"blocked": "x"})
        self.assertEqual(by_item(rev, "不可用项说明")["status"], "fail")

    def test_unavailable_with_reason_passes(self):
        bad = indicator("环比", available=False, value=None, reason="缺少上一日数据", unit="")
        rev = review_of(indicators=[bad], analytics={"blocked": "x"})
        check = by_item(rev, "不可用项说明")
        self.assertEqual(check["status"], "pass")
        self.assertIn("1 项不可用", check["comment"])


class TestCitation(unittest.TestCase):
    def test_report_with_body_must_cite_sources(self):
        rev = review_of(indicators=[indicator("日气耗指标值")],
                        report=make_report(sections=True, sources=False))
        self.assertEqual(by_item(rev, "引用来源")["status"], "fail")

    def test_empty_report_needs_no_sources(self):
        """回归：整篇只有「口径与数据说明」时，没有正文也就没有来源可标，不应判不合格。"""
        rev = review_of(indicators=[], intent=make_intent(domain="scada"),
                        report=make_report(sections=False, sources=False),
                        analytics={"blocked": "未解析出换热站编号"})
        check = by_item(rev, "引用来源")
        self.assertEqual(check["status"], "pass")
        self.assertIn("无来源需要标注", check["comment"])

    def test_report_with_sources_passes(self):
        rev = review_of(indicators=[indicator("日气耗指标值")],
                        report=make_report(sections=True, sources=True))
        self.assertEqual(by_item(rev, "引用来源")["status"], "pass")


class TestFeeAndLevelChecks(unittest.TestCase):
    def test_fee_without_category_fails(self):
        rev = review_of(indicators=[indicator("本季收费率", mode="season", unit="%")],
                        intent=make_intent(domain="enterprise", metrics=["fee_rate"],
                                           categories=[]),
                        report=make_report(sections=False))
        self.assertEqual(by_item(rev, "居民/非居民口径")["status"], "fail")

    def test_fee_with_category_passes(self):
        rev = review_of(indicators=[indicator("本季收费率(居民)", mode="season", unit="%")],
                        intent=make_intent(domain="enterprise", metrics=["fee_rate"],
                                           categories=["居民"]),
                        report=make_report(sections=False))
        check = by_item(rev, "居民/非居民口径")
        self.assertEqual(check["status"], "pass")
        self.assertIn("居民", check["comment"])

    def test_level_check_not_applicable_outside_enterprise(self):
        rev = review_of(indicators=[indicator("x")], intent=make_intent(domain="knowledge"))
        self.assertEqual(by_item(rev, "层级归属")["status"], "pass")


class TestScoring(unittest.TestCase):
    def test_score_drops_with_issues(self):
        clean = review_of(indicators=[indicator("日气耗指标值")])
        dirty = review_of(indicators=[indicator("日气耗指标值", formula="", unit="")],
                          analytics={"blocked": "x"})
        self.assertEqual(clean["quality_score"], 100)
        self.assertLess(dirty["quality_score"], clean["quality_score"])
        self.assertEqual(dirty["review_status"], "rejected")
        self.assertTrue(dirty["issues"])

    def test_no_llm_note_when_llm_disabled(self):
        rev = review_of(indicators=[indicator("日气耗指标值")])
        self.assertNotIn("llm_note", rev)


if __name__ == "__main__":
    unittest.main()
