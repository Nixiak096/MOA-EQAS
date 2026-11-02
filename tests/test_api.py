"""后端接口集成测试（Flask 测试客户端，真连 MySQL / Milvus，不调大模型）。

覆盖三层：
  * 通用与健康检查；
  * 多智能体问答（同步 /chat 与 SSE /chat/stream）——前端就是照这两个契约写的；
  * 供热业务端点与知识库端点。
知识库相关用例在没有向量库时自动跳过。
"""

import json
import unittest

import api_server
from tests.support import (COMPANY, PROJECT_WITH_DATA, SUBSTATION, WEATHER_ID,
                           require_db, require_kb)


class ApiTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        require_db()
        api_server.app.config["TESTING"] = True
        cls.client = api_server.app.test_client()

    def get_json(self, path, **kw):
        r = self.client.get(path, **kw)
        return r, r.get_json()


class TestGeneralEndpoints(ApiTestCase):
    def test_health(self):
        r, d = self.get_json("/health")
        self.assertEqual(r.status_code, 200)
        self.assertIn(d["status"], ("healthy", "degraded"))
        self.assertTrue(d["database"]["available"])
        self.assertEqual(d["database"]["detail"]["latest_stat_date"], "2023-12-16")
        kb = d["knowledge_base"]
        self.assertIn("available", kb)
        self.assertIn("chunk_count", kb)
        self.assertNotIn("error", d["database"].get("detail", {}))

    def test_api_index_lists_endpoints(self):
        r, d = self.get_json("/api")
        self.assertEqual(r.status_code, 200)
        for path in ("POST /chat", "POST /chat/stream", "GET /indicators",
                     "POST /knowledge/upload"):
            self.assertIn(path, d["endpoints"])

    def test_frontend_is_served_at_root(self):
        r = self.client.get("/")
        self.assertEqual(r.status_code, 200)
        self.assertIn("text/html", r.headers["Content-Type"])
        self.assertIn("多智能体", r.get_data(as_text=True))

    def test_frontend_assets_served(self):
        for asset, ctype in (("/app.js", "javascript"), ("/charts.js", "javascript"),
                             ("/style.css", "css")):
            r = self.client.get(asset)
            self.assertEqual(r.status_code, 200, asset)
            self.assertIn(ctype, r.headers["Content-Type"], asset)


class TestIndicatorEndpoints(ApiTestCase):
    def test_indicators_defaults(self):
        r, d = self.get_json("/indicators")
        self.assertEqual(r.status_code, 200)
        self.assertTrue(d["success"])
        self.assertTrue(d["indicators"])
        self.assertEqual(d["entity_id"], COMPANY)
        self.assertIn(d["stat_date"], d["available_dates"])

    def test_indicators_explicit_params(self):
        r, d = self.get_json(
            f"/indicators?entity={PROJECT_WITH_DATA}&metrics=gas,water&modes=daily")
        self.assertEqual(r.status_code, 200)
        labels = [x["label"] for x in d["indicators"]]
        self.assertEqual(len(labels), 2)
        self.assertTrue(any("气耗" in x for x in labels))
        self.assertTrue(any("水耗" in x for x in labels))

    def test_indicators_fee_with_categories(self):
        r, d = self.get_json(f"/indicators?entity={PROJECT_WITH_DATA}"
                             "&metrics=fee_rate&categories=居民,非居民")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(len(d["indicators"]), 2)
        self.assertTrue(all(x["available"] for x in d["indicators"]))

    def test_compare(self):
        r, d = self.get_json("/indicators/compare?metric=daily_gas_index")
        self.assertEqual(r.status_code, 200)
        cmp_ = d["compare"]
        self.assertTrue(cmp_["available"], cmp_.get("reason"))
        self.assertGreaterEqual(len(cmp_["entries"]), 2)

    def test_alerts_enterprise(self):
        r, d = self.get_json("/alerts?entity=company:公司A&metric=daily_gas_index")
        self.assertEqual(r.status_code, 200)
        self.assertIn("anomalies", d)

    def test_alerts_substation(self):
        r, d = self.get_json(f"/alerts?substation={SUBSTATION}")
        self.assertEqual(r.status_code, 200)
        self.assertTrue(d["analysis"]["available"])

    def test_report_json_and_markdown(self):
        r, d = self.get_json("/report?metrics=gas")
        self.assertEqual(r.status_code, 200)
        self.assertTrue(d["report"]["sources"])
        self.assertTrue(d["report"]["sections"]["结论"])

        r2 = self.client.get("/report?metrics=gas&format=markdown")
        self.assertEqual(r2.status_code, 200)
        self.assertIn("## 结论", r2.get_data(as_text=True))


class TestScadaHouseholdWeatherEndpoints(ApiTestCase):
    def test_scada_analysis(self):
        r, d = self.get_json(f"/scada/{SUBSTATION}")
        self.assertEqual(r.status_code, 200)
        self.assertTrue(d["analysis"]["available"])
        self.assertEqual(d["analysis"]["substation_id"], SUBSTATION)

    def test_scada_markdown(self):
        r = self.client.get(f"/scada/{SUBSTATION}?format=markdown")
        self.assertEqual(r.status_code, 200)
        self.assertIn("换热站", r.get_data(as_text=True))

    def test_scada_unknown_is_404(self):
        r, d = self.get_json("/scada/L999")
        self.assertEqual(r.status_code, 200)
        self.assertFalse(d["analysis"]["available"])
        r2, _ = self.get_json("/scada/L999?format=markdown")
        self.assertEqual(r2.status_code, 404)

    def test_temperature_curve_is_downsampled(self):
        r, d = self.get_json(f"/scada/{SUBSTATION}/temperature-curve")
        self.assertEqual(r.status_code, 200)
        self.assertGreater(d["count"], d["returned"])
        self.assertLessEqual(d["returned"], 1000)
        self.assertEqual(len(d["series"]["timestamp"]), d["returned"])
        self.assertEqual(len(d["series"]["delta_t_sec"]), d["returned"])
        self.assertGreaterEqual(d["stride"], 1)

    def test_temperature_curve_custom_limit(self):
        r, d = self.get_json(f"/scada/{SUBSTATION}/temperature-curve?max_points=100")
        self.assertEqual(r.status_code, 200)
        self.assertLessEqual(d["returned"], 100)

    def test_households_top(self):
        r, d = self.get_json("/households/top?limit=3")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(len(d["households"]), 3)
        self.assertIn("household_id", d["households"][0])

    def test_household_detail(self):
        _, top = self.get_json("/households/top?limit=1")
        hid = top["households"][0]["household_id"]
        r, d = self.get_json(f"/households/{hid}")
        self.assertEqual(r.status_code, 200)
        self.assertTrue(d["analysis"].get("available"), d["analysis"].get("reason"))

    def test_weather(self):
        r, d = self.get_json(f"/weather/{WEATHER_ID}")
        self.assertEqual(r.status_code, 200)
        self.assertTrue(d["analysis"].get("available"), d["analysis"].get("reason"))

    def test_weather_markdown_has_trend_chart_payload(self):
        _, d = self.get_json(f"/weather/{WEATHER_ID}")
        if not d["analysis"].get("available"):
            self.skipTest("气象站不可用")
        self.assertTrue(d["analysis"]["chart_points"], "绘图序列不能为空")


class TestChatEndpoints(ApiTestCase):
    def test_chat_returns_report_and_indicators(self):
        r = self.client.post("/chat", json={"question": "公司A今天的气耗指标是多少？"})
        self.assertEqual(r.status_code, 200)
        d = r.get_json()
        self.assertTrue(d["success"])
        self.assertEqual(d["domain"], "enterprise")
        self.assertTrue(d["answer"])
        self.assertTrue(d["report"]["sources"])
        self.assertTrue(d["indicators"])
        self.assertEqual([s["agent"] for s in d["steps"]],
                         ["检索员", "分析师", "撰稿人", "审核员"])
        self.assertEqual(d["review"]["review_status"], "approved")

    def test_chat_requires_question(self):
        r = self.client.post("/chat", json={})
        self.assertEqual(r.status_code, 400)
        self.assertFalse(r.get_json()["success"])

    def test_report_section_order_preserved(self):
        """回归：Flask 默认按字母序输出 JSON 键，会把报告分节顺序打乱。

        报告的叙事顺序（结论→指标数据→计算输入→…）本身是信息，中文键的字母序毫无意义。
        """
        r = self.client.post("/chat", json={"question": "公司A今天的气耗指标是多少？"})
        sections = list(r.get_json()["report"]["sections"].keys())
        self.assertEqual(sections[0], "结论")
        self.assertLess(sections.index("结论"), sections.index("指标数据"))
        if "计算输入" in sections:
            self.assertLess(sections.index("指标数据"), sections.index("计算输入"))

    def test_chat_stream_sends_steps_then_complete(self):
        events = self._read_sse("/chat/stream", "公司A今天的气耗指标是多少？")
        types = [e["type"] for e in events]
        self.assertIn("step", types)
        self.assertIn("final", types)
        self.assertEqual(types[-1], "complete", "complete 必须是最后一个事件")

        steps = [e for e in events if e["type"] == "step"]
        agents = {e["agent"] for e in steps}
        self.assertEqual(agents, {"检索员", "分析师", "撰稿人", "审核员"})
        for e in steps:
            self.assertIn(e["status"], ("running", "done"))

        complete = events[-1]
        for key in ("domain", "answer", "report", "indicators", "review", "steps"):
            self.assertIn(key, complete)
        self.assertTrue(complete["report"]["sources"])

    def test_chat_stream_requires_question(self):
        r = self.client.post("/chat/stream", json={})
        self.assertEqual(r.status_code, 400)

    def test_chat_stream_error_event_on_failure(self):
        """编排器抛错时必须以 error 事件收尾，而不是让连接静默断掉。"""
        original = api_server._build_orchestrator

        def boom():
            raise RuntimeError("模拟编排失败")

        api_server._build_orchestrator = boom
        try:
            events = self._read_sse("/chat/stream", "任意问题")
        finally:
            api_server._build_orchestrator = original
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["type"], "error")
        self.assertIn("模拟编排失败", events[0]["error"])

    def _read_sse(self, path, question):
        """按 SSE 帧解析流式响应（chunk 边界可能切开一帧，需缓冲）。"""
        resp = self.client.post(path, json={"question": question})
        self.assertEqual(resp.status_code, 200)
        self.assertIn("text/event-stream", resp.headers["Content-Type"])
        self.assertEqual(resp.headers.get("X-Accel-Buffering"), "no",
                         "需关闭代理缓冲，否则流式会被攒成一整包")

        events, buf = [], ""
        for chunk in resp.response:
            buf += chunk.decode("utf-8") if isinstance(chunk, bytes) else chunk
            while "\n\n" in buf:
                frame, buf = buf.split("\n\n", 1)
                data = "".join(l[5:].lstrip() for l in frame.split("\n")
                               if l.startswith("data:"))
                if data:
                    events.append(json.loads(data))
        return events


class TestKnowledgeEndpoints(ApiTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        require_kb()

    SAMPLE = ("# 供热规范示例\n\n1.0.1 民用建筑主要房间室内温度应保持 18℃~24℃。\n"
              "1.0.2 二次侧回水压力不应低于 50kPa，以保证系统不出现倒空与汽化。\n")

    def test_list(self):
        r, d = self.get_json("/knowledge/list")
        self.assertEqual(r.status_code, 200)
        self.assertGreater(d["count"], 0)
        self.assertIn("chunk_count", d["documents"][0])

    def test_search(self):
        r = self.client.post("/knowledge/search",
                             json={"query": "度日数折标公式", "top_k": 3})
        self.assertEqual(r.status_code, 200)
        d = r.get_json()
        self.assertTrue(d["hits"])
        hit = d["hits"][0]
        for key in ("filename", "content", "similarity", "chunk_index"):
            self.assertIn(key, hit)

    def test_search_requires_query(self):
        r = self.client.post("/knowledge/search", json={})
        self.assertEqual(r.status_code, 400)

    def test_upload_get_delete_roundtrip(self):
        before = self.get_json("/knowledge/list")[1]["count"]

        r = self.client.post("/knowledge/upload", data={
            "file": (open_upload(self.SAMPLE), "规范示例.md"),
        }, content_type="multipart/form-data")
        self.assertEqual(r.status_code, 201, r.get_data(as_text=True))
        d = r.get_json()
        self.assertTrue(d["success"])
        self.assertGreaterEqual(d["chunks"], 1)
        self.assertEqual(d["filename"], "规范示例.md", "应保留原始文件名")
        doc_id = d["document_id"]

        try:
            after = self.get_json("/knowledge/list")[1]
            self.assertEqual(after["count"], before + 1)
            self.assertIn(doc_id, [x["document_id"] for x in after["documents"]])

            g = self.client.get(f"/knowledge/{doc_id}")
            self.assertEqual(g.status_code, 200)
            self.assertEqual(g.get_json()["document"]["chunk_count"], d["chunks"])

            s = self.client.post("/knowledge/search",
                                 json={"query": "室内温度应该保持多少度", "top_k": 3})
            names = [h["filename"] for h in s.get_json()["hits"]]
            self.assertIn("规范示例.md", names, "新入库的条文应可被检索到")
        finally:
            rm = self.client.delete(f"/knowledge/{doc_id}")
            self.assertEqual(rm.status_code, 200)
            self.assertTrue(rm.get_json()["staged_file_removed"],
                            "上传的暂存文件应随删除一并清理")

        self.assertEqual(self.get_json("/knowledge/list")[1]["count"], before)

    def test_upload_rejects_unsupported_format(self):
        r = self.client.post("/knowledge/upload", data={
            "file": (open_upload(b"MZ\x90\x00"), "evil.exe"),
        }, content_type="multipart/form-data")
        self.assertEqual(r.status_code, 400)
        self.assertIn("不支持的格式", r.get_json()["error"])

    def test_upload_requires_file_field(self):
        r = self.client.post("/knowledge/upload", data={},
                             content_type="multipart/form-data")
        self.assertEqual(r.status_code, 400)

    def test_get_missing_document_404(self):
        r = self.client.get("/knowledge/不存在的文档")
        self.assertEqual(r.status_code, 404)

    def test_delete_missing_document_404(self):
        r = self.client.delete("/knowledge/不存在的文档")
        self.assertEqual(r.status_code, 404)


def open_upload(payload):
    """把字节或字符串包成 Flask 测试客户端可用的文件对象。"""
    import io

    if isinstance(payload, str):
        payload = payload.encode("utf-8")
    return io.BytesIO(payload)


if __name__ == "__main__":
    unittest.main()
