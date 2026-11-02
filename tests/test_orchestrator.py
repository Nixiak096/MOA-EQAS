"""多智能体编排集成测试。

用确定性链路（use_llm=False）跑完整协作：检索员 → 分析师 → 撰稿人 → 审核员，
校验「四步都跑到、结论可回溯、算不出来要说清为什么」。
知识库相关的用例在没有向量库时自动跳过，因此本文件在只有 MySQL 的环境下也能跑。
"""

import json
import unittest
from unittest import mock

import llm_client
from multi_agents import MAX_REVIEW_ROUNDS, AgentOrchestrator
from tests.support import REFERENCE_DATE, require_db

AGENT_ORDER = ["检索员", "分析师", "撰稿人", "审核员"]


def run(question, with_kb=False):
    factory = None
    if with_kb:
        from tests.support import kb_state
        ok, _ = kb_state()
        if ok:
            import api_server
            factory = api_server._knowledge.factory()
    return AgentOrchestrator(factory, use_llm=False).run_task(question)


class TestEnterpriseFlow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        require_db()

    def test_four_agents_run_in_order(self):
        res = run("公司A今天的气耗指标是多少？")
        self.assertEqual([s["agent"] for s in res["steps"]], AGENT_ORDER)
        self.assertEqual(res["domain"], "enterprise")

    def test_result_shape(self):
        res = run("公司A今天的气耗指标是多少？")
        for key in ("question", "domain", "answer", "report", "indicators",
                    "analytics", "review", "steps", "review_rounds"):
            self.assertIn(key, res)

    def test_report_is_auditable(self):
        res = run("公司A今天的气耗指标是多少？")
        rep = res["report"]
        self.assertTrue(rep["sections"].get("结论"))
        self.assertTrue(rep["sources"], "报告必须标注数据来源")
        self.assertEqual(res["review"]["review_status"], "approved")

    def test_indicators_carry_formula_and_unit(self):
        res = run("公司A今天的日气耗指标值和累计气耗指标值分别是多少？")
        self.assertTrue(res["indicators"])
        for r in res["indicators"]:
            if r["available"]:
                self.assertTrue(r["unit"])
                self.assertTrue(r["formula"])
                self.assertIsNotNone(r["value"])

    def test_missing_entity_explains_itself(self):
        res = run("项目3的本季收费率是多少？")
        caveats = " ".join(res["report"]["caveats"])
        self.assertIn("项目3", caveats)
        self.assertEqual(res["review"]["review_status"], "approved",
                         "如实说明「库中无此实体」应当判合格，而不是伪装成有数据")

    def test_unparsable_entity_is_not_silently_empty(self):
        """回归：领域选定但缺实体时，报告只剩检索条文却仍判合格。"""
        res = run("换热站的二次侧供回水温差正常吗？")
        caveats = " ".join(res["report"]["caveats"])
        self.assertIn("未做数据核算", caveats)
        check = [c for c in res["review"]["checks"] if c["item"] == "分析执行"][0]
        self.assertEqual(check["status"], "pass")
        self.assertIn("未解析出换热站编号", check["comment"])

    def test_review_rounds_within_limit(self):
        res = run("公司A今天的气耗指标是多少？")
        self.assertLessEqual(res["review_rounds"], MAX_REVIEW_ROUNDS)


class TestHouseholdFlow(unittest.TestCase):
    """回归：household_top 是 DataReader 的方法，曾按模块级函数调用而抛 AttributeError。"""

    @classmethod
    def setUpClass(cls):
        require_db()

    def test_household_question_runs(self):
        res = run("住户的用电量排名前几名是谁？")
        self.assertEqual(res["domain"], "household")
        self.assertEqual([s["agent"] for s in res["steps"]], AGENT_ORDER)
        self.assertIn("household_ranking", res["analytics"])
        self.assertTrue(res["analytics"]["household_ranking"])


class TestKnowledgeFlow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        require_db()
        from tests.support import require_kb
        require_kb()

    def test_knowledge_question_cites_retrieved_documents(self):
        res = run("度日数折标是怎么规定的？", with_kb=True)
        self.assertEqual(res["domain"], "knowledge")
        rep = res["report"]
        self.assertTrue(rep["sections"].get("依据条文"))
        self.assertTrue(rep["sources"], "检索到的条文必须作为来源标注出来")
        self.assertEqual(res["review"]["review_status"], "approved")

    def test_standard_question_retrieves_the_standard(self):
        """例A：室内温度应命中 GB 50736 的条文，而不是给出一堆无关文档。"""
        res = run("国家标准规定居民室内供暖温度是多少？", with_kb=True)
        self.assertEqual(res["domain"], "knowledge")
        names = " ".join(res["report"]["sources"])
        self.assertIn("GB50736", names)
        text = " ".join(res["report"]["sections"].get("依据条文", []))
        self.assertIn("18℃～24℃", text)

    def test_pressure_question_retrieves_forced_standard(self):
        """例B：回水压力应命中 GB 55010 全文强制规范的 50kPa 条文。"""
        res = run("供热管网运行时回水压力有什么安全要求？", with_kb=True)
        names = " ".join(res["report"]["sources"])
        text = " ".join(res["report"]["sections"].get("依据条文", []))
        self.assertIn("GB55010", names)
        self.assertIn("50kPa", text)


class TestComplianceQuestion(unittest.TestCase):
    """例C：既要算数、又要对照规范的混合型问题。

    域分流是单值的，纯数据域会丢掉"算不算达标"这半问；这里验证补齐了规范依据。
    """

    @classmethod
    def setUpClass(cls):
        require_db()
        from tests.support import require_kb
        require_kb()

    def test_compliance_words_detected(self):
        from multi_agents import _asks_compliance
        self.assertTrue(_asks_compliance("公司A本季收费率75%，算不算达标？"))
        self.assertTrue(_asks_compliance("这样做符合标准吗？"))
        self.assertFalse(_asks_compliance("公司A今天的气耗指标是多少？"))

    def test_hybrid_question_adds_standard_section(self):
        res = run("公司A本季收费率75%，算不算达标？", with_kb=True)
        rep = res["report"]
        # 数据侧仍然要算出数值
        self.assertTrue(rep["sections"].get("结论"))
        # 并且补上规范依据
        section = rep["sections"].get("依据条文与合规性说明")
        self.assertTrue(section, "达标类问题必须补规范依据，否则只答了半问")
        text = " ".join(section)
        self.assertIn("GBT33833", " ".join(rep["sources"]))
        # 关键：不得凭空给出收费率达标阈值
        self.assertTrue(
            "未规定" in text or "不是国家标准" in text or "无判定依据" in text,
            f"不得臆造达标阈值，应说明标准未规定；实际：{text[:200]}",
        )

    def test_plain_metric_question_has_no_compliance_section(self):
        res = run("公司A今天的气耗指标是多少？", with_kb=True)
        self.assertNotIn("依据条文与合规性说明", res["report"]["sections"])


class TestStreamingCallbacks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        require_db()

    def test_events_are_well_formed(self):
        events = []
        AgentOrchestrator(None, use_llm=False).run_task(
            "公司A今天的气耗指标是多少？",
            stream_callback=lambda s: events.append(json.loads(s)))

        self.assertTrue(events)
        types = [e["type"] for e in events]
        self.assertIn("step", types)
        self.assertIn("final", types)
        self.assertEqual(types[-1], "final", "最后一个事件应是 final 汇总")

    def test_each_agent_emits_running_then_done(self):
        events = []
        AgentOrchestrator(None, use_llm=False).run_task(
            "公司A今天的气耗指标是多少？",
            stream_callback=lambda s: events.append(json.loads(s)))

        seen = {}
        for e in events:
            if e["type"] == "step":
                seen.setdefault(e["agent"], []).append(e["status"])
        for agent in AGENT_ORDER:
            self.assertEqual(seen.get(agent), ["running", "done"], f"{agent} 事件不成对")

    def test_done_step_carries_detail(self):
        events = []
        AgentOrchestrator(None, use_llm=False).run_task(
            "公司A今天的气耗指标是多少？",
            stream_callback=lambda s: events.append(json.loads(s)))
        ana = [e for e in events if e["type"] == "step"
               and e["agent"] == "分析师" and e["status"] == "done"][0]
        self.assertIn("n_indicators", ana)
        self.assertIn("analytics_keys", ana)
        rev = [e for e in events if e["type"] == "step"
               and e["agent"] == "审核员" and e["status"] == "done"][0]
        self.assertIn("status_text", rev)
        self.assertIn("score", rev)


class TestThinkingTrace(unittest.TestCase):
    """「思考过程」流：trace 事件覆盖四个角色，模型文字以 stream 增量推出。

    这是前端右侧大画面所依赖的数据源，故单列回归。
    """

    @classmethod
    def setUpClass(cls):
        require_db()

    def _run(self, use_llm=False, patch_llm=False):
        events = []
        patchers = []
        if patch_llm:
            def fake(_self, prompt, system=None, on_delta=None, **kw):
                for piece in ("甲", "乙", "丙"):
                    if on_delta:
                        on_delta(piece)
                return "甲乙丙"

            patchers = [
                mock.patch.object(llm_client.LLMClient, "chat", autospec=True,
                                  side_effect=lambda _s, m, **k: "甲乙丙"),
                mock.patch.object(llm_client.LLMClient, "complete_stream", autospec=True,
                                  side_effect=fake),
            ]
            for p in patchers:
                p.start()
        try:
            AgentOrchestrator(None, use_llm=use_llm).run_task(
                "公司A今天的气耗指标是多少？",
                stream_callback=lambda s: events.append(json.loads(s)))
        finally:
            for p in patchers:
                p.stop()
        return events

    def test_trace_events_cover_all_four_agents(self):
        events = self._run()
        traces = [e for e in events if e["type"] == "trace"]
        self.assertTrue(traces, "应至少产生一条过程事件")
        self.assertEqual({e["agent"] for e in traces}, set(AGENT_ORDER))
        # 流式事件都在 final 之前，最后一帧仍是 final
        self.assertEqual(events[-1]["type"], "final")

    def test_trace_reports_real_intermediate_results(self):
        """过程里应能看到意图解析结果、指标公式与逐项审核结论，而不只是「正在运行」。"""
        traces = [e for e in self._run() if e["type"] == "trace"]
        joined = "\n".join(e["text"] for e in traces)
        self.assertIn("意图解析", joined)
        self.assertIn("日气耗指标值", joined)
        self.assertIn("←", joined)                 # 指标行会带出所用公式
        self.assertIn("层级归属", joined)          # 审核员的校验项
        self.assertTrue(any(e.get("level") in ("ok", "warn") for e in traces),
                        "过程事件应区分「通过/提示」等级")

    def test_llm_text_streams_as_deltas(self):
        events = self._run(use_llm=True, patch_llm=True)
        streams = [e for e in events if e["type"] == "stream"]
        self.assertTrue(streams, "有订阅者时模型文字应以增量推出")
        self.assertTrue({e["agent"] for e in streams} & {"分析师", "撰稿人", "审核员"})
        writer = "".join(e["text"] for e in streams if e["agent"] == "撰稿人")
        self.assertIn("甲乙丙", writer)

    def test_non_streaming_path_still_returns_report(self):
        """不传 stream_callback 时（/chat 非流式路径）仍应正常返回完整结果。"""
        result = AgentOrchestrator(None, use_llm=False).run_task(
            "公司A今天的气耗指标是多少？", stream_callback=None)
        self.assertIn("report", result)
        self.assertTrue(result["indicators"])


if __name__ == "__main__":
    unittest.main()
