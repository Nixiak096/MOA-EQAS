"""角色 → 模型 的解析与生效验证。

覆盖四件事：
  1. 配置解析与回退语义（含 ollama 分支、缓存重置）；
  2. 角色中文名与 multi_agents 中各 Agent.name 一致（防止两处走偏）；
  3. **四个角色在同一次问答里确实调用了各自配置的模型**（不联网，用打桩记录 self.model）；
  4. `use_llm=False` 时一个模型都不调，且不产生模型生成的分节。
"""

import unittest
from unittest import mock

import config
import llm_client
from tests import support


class RoleModelConfigTest(unittest.TestCase):
    """配置解析：覆盖、回退、ollama 分支、缓存与重置。"""

    def setUp(self):
        self._saved = {k: getattr(config, k) for k in
                       ("LLM_PROVIDER", "LLM_MODEL", "OLLAMA_MODEL",
                        "LLM_MODEL_RETRIEVER", "LLM_MODEL_ANALYST",
                        "LLM_MODEL_WRITER", "LLM_MODEL_REVIEWER")}
        # 从「无任何角色覆盖」的干净状态起步：本机 .env 里可能已按角色配了模型，
        # 不清空会让这些用例依赖环境，断言随之漂移。
        for role in ("retriever", "analyst", "writer", "reviewer"):
            setattr(config, f"LLM_MODEL_{role.upper()}", "")
        llm_client.reset_llm_cache()

    def tearDown(self):
        for k, v in self._saved.items():
            setattr(config, k, v)
        llm_client.reset_llm_cache()

    def test_no_override_falls_back_to_global_model(self):
        config.LLM_MODEL = "global-model"
        for role in ("retriever", "analyst", "writer", "reviewer"):
            setattr(config, f"LLM_MODEL_{role.upper()}", "")
        self.assertEqual(llm_client.effective_model("writer"), "global-model")
        self.assertEqual(llm_client.get_llm("writer").model_name(), "global-model")

    def test_unknown_role_falls_back_to_global_model(self):
        config.LLM_MODEL = "global-model"
        self.assertEqual(llm_client.effective_model("nobody"), "global-model")
        self.assertEqual(llm_client.get_llm("nobody").model_name(), "global-model")

    def test_override_wins_and_others_keep_default(self):
        config.LLM_MODEL = "global-model"
        config.LLM_MODEL_REVIEWER = "strong-model"
        self.assertEqual(llm_client.get_llm("reviewer").model_name(), "strong-model")
        self.assertEqual(llm_client.get_llm("writer").model_name(), "global-model")

    def test_role_models_shape_and_overridden_flag(self):
        config.LLM_MODEL = "global-model"
        config.LLM_MODEL_WRITER = "cheap-model"
        rows = llm_client.role_models()
        self.assertEqual([r["role"] for r in rows],
                         ["retriever", "analyst", "writer", "reviewer"])
        self.assertEqual([r["label"] for r in rows],
                         ["检索员", "分析师", "撰稿人", "审核员"])
        by_role = {r["role"]: r for r in rows}
        self.assertEqual(by_role["writer"]["model"], "cheap-model")
        self.assertTrue(by_role["writer"]["overridden"])
        self.assertFalse(by_role["retriever"]["overridden"])

    def test_ollama_provider_falls_back_to_ollama_model(self):
        config.LLM_PROVIDER = "ollama"
        config.OLLAMA_MODEL = "qwen2.5:7b"
        config.LLM_MODEL = "should-not-be-used"
        for role in ("retriever", "analyst", "writer", "reviewer"):
            setattr(config, f"LLM_MODEL_{role.upper()}", "")
        self.assertEqual(llm_client.default_model_name(), "qwen2.5:7b")
        self.assertEqual(llm_client.effective_model("retriever"), "qwen2.5:7b")

    def test_override_applies_under_ollama_too(self):
        config.LLM_PROVIDER = "ollama"
        config.LLM_MODEL_ANALYST = "llama3.1:8b"
        self.assertEqual(llm_client.get_llm("analyst").model_name(), "llama3.1:8b")

    def test_cache_returns_stale_value_until_reset(self):
        config.LLM_MODEL = "first-model"
        self.assertEqual(llm_client.get_llm().model_name(), "first-model")
        config.LLM_MODEL = "second-model"
        self.assertEqual(llm_client.get_llm().model_name(), "first-model",
                         "未 reset 时应仍是缓存的旧客户端")
        llm_client.reset_llm_cache()
        self.assertEqual(llm_client.get_llm().model_name(), "second-model")

    def test_reset_clears_role_entries_too(self):
        config.LLM_MODEL_REVIEWER = "old-strong"
        self.assertEqual(llm_client.get_llm("reviewer").model_name(), "old-strong")
        config.LLM_MODEL_REVIEWER = "new-strong"
        llm_client.reset_llm_cache()
        self.assertEqual(llm_client.get_llm("reviewer").model_name(), "new-strong")

    def test_role_models_does_not_construct_clients(self):
        """展示用接口只读配置，不应因为被 /health 调用而构造客户端。"""
        llm_client.reset_llm_cache()
        llm_client.role_models()
        self.assertEqual(llm_client._llm_cache, {},
                         "role_models() 不应产生任何客户端缓存")


class RoleSpecDriftTest(unittest.TestCase):
    """角色中文名两处必须一致：llm_client.ROLE_SPECS 与各 Agent.name。"""

    def test_labels_match_agent_names(self):
        import multi_agents

        agents = {a.name: a for a in (multi_agents.RetrieverAgent,
                                      multi_agents.AnalystAgent,
                                      multi_agents.WriterAgent,
                                      multi_agents.ReviewerAgent)}
        for role, label in llm_client.ROLE_SPECS:
            self.assertIn(label, agents, f"ROLE_SPECS 里的「{label}」没有对应的 Agent")
            self.assertEqual(label, agents[label].name)


class RolesHitDifferentModelsTest(unittest.TestCase):
    """核心：一次问答里，四个角色各自走到了自己配置的模型。"""

    def setUp(self):
        support.require_db()          # 指标计算需要真实 MySQL
        self._saved = {k: getattr(config, k) for k in
                       ("LLM_MODEL_RETRIEVER", "LLM_MODEL_ANALYST",
                        "LLM_MODEL_WRITER", "LLM_MODEL_REVIEWER")}
        for role in ("retriever", "analyst", "writer", "reviewer"):
            setattr(config, f"LLM_MODEL_{role.upper()}", "")
        self.prompts = []             # 记录 (model, prompt) 便于断言

        def rec(_self, messages, **kwargs):
            text = messages[-1]["content"] if messages else ""
            self.prompts.append((_self.model, text))
            return "（打桩输出：本条由测试桩生成，不含任何数字）"

        def rec_stream(_self, prompt, system=None, on_delta=None, **kwargs):
            """流式桩：把整段文字当作一个增量推出去，并同样记录所用模型。"""
            self.prompts.append((_self.model, prompt))
            text = "（打桩输出：本条由测试桩生成，不含任何数字）"
            if on_delta:
                on_delta(text)
            return text

        # chat（非流式链路）与 complete_stream（有订阅者时的流式链路）都要打桩，
        # 否则带 stream_callback 的用例会真的去请求模型服务。
        self.patchers = [
            mock.patch.object(llm_client.LLMClient, "chat", autospec=True, side_effect=rec),
            mock.patch.object(llm_client.LLMClient, "complete_stream",
                              autospec=True, side_effect=rec_stream),
        ]
        for p in self.patchers:
            p.start()

    def tearDown(self):
        for p in self.patchers:
            p.stop()
        for k, v in self._saved.items():
            setattr(config, k, v)
        llm_client.reset_llm_cache()

    def test_four_roles_use_four_distinct_models(self):
        import multi_agents

        config.LLM_MODEL_RETRIEVER = "model-retriever"
        config.LLM_MODEL_ANALYST = "model-analyst"
        config.LLM_MODEL_WRITER = "model-writer"
        config.LLM_MODEL_REVIEWER = "model-reviewer"
        llm_client.reset_llm_cache()

        orch = multi_agents.AgentOrchestrator(storage_factory=None, use_llm=True)
        result = orch.run_task("公司A今天的气耗指标是多少？")

        used = [m for m, _ in self.prompts]
        self.assertEqual(set(used),
                         {"model-retriever", "model-analyst",
                          "model-writer", "model-reviewer"},
                         f"四个角色没有各走各的模型，实际调用模型为：{used}")

        # 分析师新增的解读确实落到了它自己的模型上
        analyst_prompts = [p for m, p in self.prompts if m == "model-analyst"]
        self.assertTrue(any("已核算结果" in p for p in analyst_prompts),
                        "分析师应基于已核算结果生成解读")

        # 解读单独成节并标注；数值分节不受影响
        sections = result["report"]["sections"]
        self.assertIn("数据解读（模型生成，仅供参考）", sections)
        self.assertIn("大模型综述（模型生成，仅供参考）", sections)
        self.assertNotIn("打桩输出", " ".join(sections.get("结论", [])))

    def test_model_name_rides_along_on_step_events(self):
        import multi_agents

        config.LLM_MODEL_WRITER = "model-writer"
        config.LLM_MODEL_REVIEWER = "model-reviewer"
        llm_client.reset_llm_cache()

        events = []
        orch = multi_agents.AgentOrchestrator(storage_factory=None, use_llm=True)
        orch.run_task("公司A今天的气耗指标是多少？",
                      stream_callback=lambda s: events.append(__import__("json").loads(s)))

        steps = [e for e in events if e.get("type") == "step"]
        by_agent = {e["agent"]: e.get("model") for e in steps}
        self.assertEqual(by_agent.get("撰稿人"), "model-writer")
        self.assertEqual(by_agent.get("审核员"), "model-reviewer")

    def test_reviewer_approves_labelled_model_sections(self):
        """模型生成内容已标注 → 「模型生成内容标注」校验通过。"""
        import multi_agents

        config.LLM_MODEL_ANALYST = "model-analyst"
        llm_client.reset_llm_cache()

        orch = multi_agents.AgentOrchestrator(storage_factory=None, use_llm=True)
        result = orch.run_task("公司A今天的气耗指标是多少？")
        item = support.by_item(result["review"], "模型生成内容标注")
        self.assertEqual(item["status"], "pass")
        self.assertIn("标注", item["comment"])


class LlmDisabledTest(unittest.TestCase):
    """use_llm=False：一个模型都不调，也不产生模型生成的分节。"""

    def setUp(self):
        support.require_db()
        llm_client.reset_llm_cache()
        self.prompts = []

        def rec(_self, messages, **kwargs):
            self.prompts.append(_self.model)
            return ""

        self.patcher = mock.patch.object(llm_client.LLMClient, "chat",
                                         autospec=True, side_effect=rec)
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()
        llm_client.reset_llm_cache()

    def test_no_model_is_called_when_disabled(self):
        import multi_agents

        orch = multi_agents.AgentOrchestrator(storage_factory=None, use_llm=False)
        result = orch.run_task("公司A今天的气耗指标是多少？")

        self.assertEqual(self.prompts, [], "关闭大模型后不应有任何调用")
        self.assertNotIn("interpretation", result["analytics"])
        sections = result["report"]["sections"]
        self.assertFalse([k for k in sections if "模型生成" in k],
                         "关闭大模型后不应出现模型生成的分节")
        self.assertNotIn("llm_note", result["review"])

    def test_role_models_still_readable_when_disabled(self):
        """/health 的模型列表与开关无关，仍要能读出来。"""
        rows = llm_client.role_models()
        self.assertEqual(len(rows), 4)
        self.assertTrue(all(r["model"] for r in rows))


if __name__ == "__main__":
    unittest.main()
