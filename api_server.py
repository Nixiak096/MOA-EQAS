"""供热运营数据查询与分析系统 —— 后端 API 服务（Flask）。

装配关系：
    POST /chat、/chat/stream   -> AgentOrchestrator（检索员→分析师→撰稿人→审核员）
    /knowledge/*               -> DocumentProcessor + MilvusStorage
    供热业务端点               -> indicators / analytics / report_generator（见文件末段）

知识库连接一律走**短连接**（`with MilvusStorage() as s`）：pymilvus 的空闲心跳会被
Milvus 以 GOAWAY(too_many_pings) 限流，而多智能体流程中有大段大模型调用，
连接必须用完即关，不能跨越大模型阶段。

知识库不可用时服务不崩溃，降级为「纯数据链路」：指标核算/统计分析/报表全部照常，
只是检索员取不到知识库条文。

启动：
    python api_server.py
"""

import json
import logging
import os
import queue
import threading
import uuid
from datetime import datetime

from flask import Flask, Response, jsonify, request, send_from_directory, stream_with_context
from flask_cors import CORS
from werkzeug.utils import secure_filename

import config
import document_processor as dp
import llm_client
from multi_agents import AgentOrchestrator

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("api_server")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = os.path.join(BASE_DIR, "web")
UPLOAD_FOLDER = os.path.abspath(os.path.join(BASE_DIR, config.UPLOAD_FOLDER))

# SSE 心跳间隔（秒）：大模型阶段可能静默数十秒，需定期发注释帧防止代理断连
SSE_HEARTBEAT = 15


def _parse_size(spec: str, default: int = 50 * 1024 * 1024) -> int:
    """把 "50MB" 这类配置解析为字节数。"""
    s = str(spec).strip().upper()
    units = {"KB": 1024, "MB": 1024 ** 2, "GB": 1024 ** 3}
    for unit, factor in units.items():
        if s.endswith(unit):
            try:
                return int(float(s[: -len(unit)]) * factor)
            except ValueError:
                return default
    try:
        return int(s)
    except ValueError:
        return default


# 前端静态资源挂在根路径，使 web/ 下可写相对引用（./app.js）。API 路由均为精确路径，
# 优先级高于 /<path:filename>，不会互相遮蔽。
app = Flask(
    __name__,
    static_folder=WEB_DIR if os.path.isdir(WEB_DIR) else None,
    static_url_path="",
)
app.config["MAX_CONTENT_LENGTH"] = _parse_size(config.MAX_UPLOAD_SIZE)

# Flask 默认按字母序输出 JSON 键，会把报告的「结论→指标数据→计算输入→口径校验→综述」
# 顺序打乱（中文键的字母序毫无意义）。分节顺序本身是信息，保持插入顺序。
app.json.sort_keys = False

CORS(app)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ============================================================================
# 知识库装配（探测一次并缓存结果）
# ============================================================================


class _KnowledgeState:
    """探测知识库可用性；不可用时整体降级，不影响数据链路。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._probed = False
        self._factory = None
        self._error = ""

    def factory(self):
        """返回 MilvusStorage 类（可用作工厂）或 None。"""
        if not self._probed:
            with self._lock:
                if not self._probed:
                    self._probe()
        return self._factory

    def _probe(self) -> None:
        try:
            with dp.MilvusStorage() as s:
                s.create_collection()
            self._factory = dp.MilvusStorage
            self._error = ""
            logger.info("知识库可用：%s / 集合 %s", config.MILVUS_DB_URI, config.MILVUS_COLLECTION)
        except Exception as e:  # noqa: BLE001
            self._factory = None
            self._error = str(e)
            logger.warning("知识库不可用，降级为纯数据链路：%s", e)
        finally:
            self._probed = True

    def status(self) -> dict:
        self.factory()
        return {"available": self._factory is not None, "error": self._error}


_knowledge = _KnowledgeState()
_processor = dp.DocumentProcessor()


def _storage():
    """获取一个短连接存储句柄；知识库不可用时抛 RuntimeError。"""
    factory = _knowledge.factory()
    if factory is None:
        raise RuntimeError(f"知识库不可用：{_knowledge._error}")
    return factory()


# ============================================================================
# 健康检查 / 服务信息
# ============================================================================


@app.route("/health", methods=["GET"])
def health_check():
    ks = _knowledge.status()
    doc_count = None
    if ks["available"]:
        try:
            with _storage() as s:
                doc_count = s.count()
        except Exception as e:  # noqa: BLE001
            ks = {"available": False, "error": str(e)}
    db_ok, db_detail = _db_status()
    return jsonify({
        "status": "healthy" if db_ok else "degraded",
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.0",
        "knowledge_base": {
            "available": ks["available"],
            "collection": config.MILVUS_COLLECTION,
            "uri": config.MILVUS_DB_URI,
            "chunk_count": doc_count,
            "error": ks["error"],
        },
        "database": {"available": db_ok, "detail": db_detail},
        # 各角色实际生效的模型（按角色可分别覆盖 LLM_MODEL_*），供前端展示与配置核对
        "use_llm": config.AGENT_USE_LLM,
        "models": llm_client.role_models(),
    })


def _db_status():
    """返回 (可用性, 详情)。详情在成功时是库状态快照，失败时是错误信息。"""
    try:
        import heat_db
        from sqlalchemy import text

        eng = heat_db.get_engine()
        with eng.connect() as c:
            latest = c.execute(text("SELECT MAX(stat_date) FROM fact_energy_daily")).scalar()
        eng.dispose()
        return True, {"latest_stat_date": str(latest) if latest else None}
    except Exception as e:  # noqa: BLE001
        return False, {"error": str(e)}


@app.route("/api", methods=["GET"])
def api_index():
    return jsonify({
        "service": "基于多智能体协作的供热运营数据查询与分析系统",
        "version": "2.0.0",
        "endpoints": {
            "GET /health": "健康检查（关系库 / 知识库）",
            "POST /chat": "多智能体问答（同步，返回完整报告）",
            "POST /chat/stream": "多智能体问答（SSE 流式，逐步推送四智能体过程）",
            "POST /knowledge/upload": "上传文档入库",
            "GET /knowledge/list": "列出知识库文档",
            "GET /knowledge/<id>": "获取文档分片全文",
            "DELETE /knowledge/<id>": "删除文档",
            "POST /knowledge/search": "知识库检索（调试用）",
            "GET /indicators": "指标核算",
            "GET /indicators/compare": "多项目横向对比",
            "GET /alerts": "异常检测",
            "GET /report": "报告生成",
            "GET /scada/<substation_id>": "换热站工况分析",
            "GET /scada/<substation_id>/temperature-curve": "换热站供回水温度曲线",
            "GET /households/<household_id>": "住户用电分析",
            "GET /households/top": "住户用电排名",
            "GET /weather/<weather_id>": "气象趋势分析",
        },
        "timestamp": datetime.now().isoformat(),
    })


@app.route("/", methods=["GET"])
def index():
    if os.path.exists(os.path.join(WEB_DIR, "index.html")):
        return send_from_directory(WEB_DIR, "index.html")
    return api_index()


# ============================================================================
# 知识库
# ============================================================================


def _error(msg: str, code: int = 400):
    return jsonify({"success": False, "error": msg}), code


@app.route("/knowledge/upload", methods=["POST"])
def upload_document():
    if "file" not in request.files:
        return _error("缺少 file 字段（multipart/form-data）")
    file = request.files["file"]
    original = file.filename or ""
    if not original:
        return _error("文件名为空")

    ext = os.path.splitext(original)[1].lower()
    if ext not in dp.PARSABLE_EXTS:
        return _error(f"不支持的格式 {ext or '（无扩展名）'}，"
                      f"可接受：{sorted(dp.PARSABLE_EXTS)}")

    # 磁盘名做安全化 + 随机后缀避免覆盖；展示名保留原始文件名
    stem = secure_filename(os.path.splitext(original)[0]) or "doc"
    stored = os.path.join(UPLOAD_FOLDER, f"{stem}-{uuid.uuid4().hex[:8]}{ext}")

    try:
        file.save(stored)
        doc = _processor.process(stored)
        doc.filename = original
        doc.metadata["original_filename"] = original
        with _storage() as s:
            s.create_collection()
            n_chunks = s.insert(doc)
            s.flush()             # 服务端 Milvus 不自动落盘，不 flush 会查不到刚上传的内容
        return jsonify({
            "success": True,
            "message": "文档已解析入库",
            "document_id": doc.document_id,
            "filename": original,
            "chunks": n_chunks,
            "timestamp": datetime.now().isoformat(),
        }), 201
    except Exception as e:  # noqa: BLE001
        logger.error("文档入库失败：%s", e)
        if os.path.exists(stored):
            os.remove(stored)
        return _error(str(e), 500)


@app.route("/knowledge/list", methods=["GET"])
def list_documents():
    try:
        with _storage() as s:
            documents = s.list_documents()
    except Exception as e:  # noqa: BLE001
        return _error(str(e), 503)
    return jsonify({
        "documents": documents,
        "count": len(documents),
        "timestamp": datetime.now().isoformat(),
    })


@app.route("/knowledge/search", methods=["POST"])
def search_knowledge():
    data = request.get_json(silent=True) or {}
    query = (data.get("query") or "").strip()
    if not query:
        return _error("query 不能为空")
    top_k = int(data.get("top_k") or 5)
    try:
        with _storage() as s:
            hits = s.search(query, top_k=top_k)
    except Exception as e:  # noqa: BLE001
        return _error(str(e), 503)
    return jsonify({"success": True, "query": query, "hits": hits, "count": len(hits)})


@app.route("/knowledge/<path:document_id>", methods=["GET"])
def get_document(document_id):
    try:
        with _storage() as s:
            doc = s.get(document_id)
    except Exception as e:  # noqa: BLE001
        return _error(str(e), 503)
    if not doc:
        return _error("文档不存在", 404)
    return jsonify({"success": True, "document": doc,
                    "timestamp": datetime.now().isoformat()})


@app.route("/knowledge/<path:document_id>", methods=["DELETE"])
def delete_document(document_id):
    try:
        with _storage() as s:
            doc = s.get(document_id)
            if not doc:
                return _error("文档不存在", 404)
            s.delete(document_id)
            staged = doc.get("path")
    except Exception as e:  # noqa: BLE001
        return _error(str(e), 503)
    # 上传入库的文档，其暂存副本一并清掉，避免 uploads/ 越积越多
    removed_staged = _remove_staged_file(staged)
    return jsonify({
        "success": True,
        "message": f"已删除文档 {document_id}",
        "staged_file_removed": removed_staged,
        "timestamp": datetime.now().isoformat(),
    })


def _remove_staged_file(rel_path) -> bool:
    """删除上传时落到 UPLOAD_FOLDER 的暂存副本。

    只允许删该目录内的文件 —— documents/ 下的知识库原始文档属用户资料，一律不动。
    """
    if not rel_path:
        return False
    root = os.path.abspath(UPLOAD_FOLDER) + os.sep
    target = os.path.abspath(os.path.join(BASE_DIR, str(rel_path)))
    if not target.startswith(root) or not os.path.isfile(target):
        return False
    try:
        os.remove(target)
        return True
    except OSError as e:  # noqa: BLE001
        logger.warning("暂存文件删除失败 %s：%s", target, e)
        return False


# ============================================================================
# 多智能体问答
# ============================================================================


def _build_orchestrator() -> AgentOrchestrator:
    return AgentOrchestrator(_knowledge.factory(), use_llm=config.AGENT_USE_LLM)


def _question_from_request():
    data = request.get_json(silent=True) or {}
    return (data.get("question") or "").strip()


@app.route("/chat", methods=["POST"])
def chat():
    question = _question_from_request()
    if not question:
        return _error("question 不能为空")

    try:
        result = _build_orchestrator().run_task(question)
    except Exception as e:  # noqa: BLE001
        logger.exception("问答失败")
        return _error(str(e), 500)

    return jsonify({
        "success": True,
        "question": question,
        "domain": result["domain"],
        "answer": result["answer"],
        "report": result["report"],
        "indicators": result["indicators"],
        "analytics": result["analytics"],
        "compare": result.get("compare"),
        "anomalies": result.get("anomalies"),
        "review": result["review"],
        "steps": result["steps"],
        "review_rounds": result["review_rounds"],
        "timestamp": datetime.now().isoformat(),
    })


@app.route("/chat/stream", methods=["POST"])
def chat_stream():
    question = _question_from_request()
    if not question:
        return _error("question 不能为空")

    events: "queue.Queue" = queue.Queue()

    def worker():
        def emit(payload: str) -> None:
            events.put(("data", payload))

        try:
            result = _build_orchestrator().run_task(question, stream_callback=emit)
            emit(json.dumps({
                "type": "complete",
                "domain": result["domain"],
                "answer": result["answer"],
                "report": result["report"],
                "indicators": result["indicators"],
                "analytics": result["analytics"],
                "compare": result.get("compare"),
                "anomalies": result.get("anomalies"),
                "review": result["review"],
                "steps": result["steps"],
                "review_rounds": result["review_rounds"],
            }, ensure_ascii=False, default=str))
        except Exception as e:  # noqa: BLE001
            logger.exception("流式问答失败")
            emit(json.dumps({"type": "error", "error": str(e)}, ensure_ascii=False))
        finally:
            events.put(("end", None))

    threading.Thread(target=worker, daemon=True).start()

    def generate():
        while True:
            try:
                kind, payload = events.get(timeout=SSE_HEARTBEAT)
            except queue.Empty:
                yield ": ping\n\n"  # 心跳注释帧：维持代理连接
                continue
            if kind == "end":
                break
            yield f"data: {payload}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # 关闭 nginx 缓冲，否则流式会被攒成一整包
        },
    )


# ============================================================================
# 供热业务端点
# ============================================================================


def _arg_date(name: str, default=None):
    """解析 YYYY-MM-DD 查询参数；缺省返回 default。"""
    import datetime as _dt

    raw = request.args.get(name)
    if not raw:
        return default
    return _dt.date.fromisoformat(raw)


def _arg_list(name: str):
    raw = request.args.get(name)
    if not raw:
        return []
    return [x.strip() for x in raw.split(",") if x.strip()]


@app.route("/indicators", methods=["GET"])
def get_indicators():
    """指标核算：/indicators?entity=company:公司A&metrics=gas,water&modes=daily&date=2023-12-16

    参数缺省时按「公司A / 气耗 / 日值 / 库中最新日期」给出一份可直接演示的结果。
    """
    import datetime as _dt

    import indicators

    entity = request.args.get("entity") or "company:公司A"
    metrics = _arg_list("metrics") or ["gas"]
    modes = _arg_list("modes") or ["daily"]
    categories = _arg_list("categories") or ["小计"]

    try:
        with indicators.DataAccess() as da:
            dates = da.available_dates(entity)
        stat_date = _arg_date("date") or (dates[-1] if dates else _dt.date.today())
        results = indicators.compute(entity, stat_date, metrics,
                                     modes=modes, categories=categories)
    except Exception as e:  # noqa: BLE001
        logger.exception("指标核算失败")
        return _error(str(e), 500)

    return jsonify({
        "success": True,
        "entity_id": entity,
        "stat_date": stat_date.isoformat(),
        "available_dates": [str(d) for d in (dates or [])],
        "indicators": [r.to_dict() for r in results],
        "count": len(results),
    })


@app.route("/indicators/compare", methods=["GET"])
def compare_indicators():
    """多项目横向对比：/indicators/compare?metric=daily_gas_index&company=公司A&date=..."""
    import analytics

    metric_code = request.args.get("metric") or "daily_gas_index"
    company = request.args.get("company") or "公司A"
    try:
        res = analytics.compare_projects(metric_code, company=company,
                                         stat_date=_arg_date("date"))
    except Exception as e:  # noqa: BLE001
        logger.exception("横向对比失败")
        return _error(str(e), 500)
    return jsonify({"success": True, "compare": res.to_dict()})


@app.route("/alerts", methods=["GET"])
def get_alerts():
    """异常检测：/alerts?entity=company:公司A&metric=daily_gas_index

    亦可 /alerts?substation=L4 查看换热站工况异常。
    """
    import analytics

    substation = request.args.get("substation")
    try:
        if substation:
            result = analytics.scada_analysis(substation, start=_arg_date("start"),
                                              end=_arg_date("end"))
            return jsonify({"success": True, "target": f"substation:{substation}",
                            "analysis": result})

        entity = request.args.get("entity") or "company:公司A"
        metric_code = request.args.get("metric") or "daily_gas_index"
        res = analytics.enterprise_anomalies(entity, metric_code)
        return jsonify({"success": True, "target": entity, "anomalies": res.to_dict()})
    except Exception as e:  # noqa: BLE001
        logger.exception("异常检测失败")
        return _error(str(e), 500)


@app.route("/report", methods=["GET"])
def get_report():
    """报告生成：/report?entity=company:公司A&metrics=gas&format=json|markdown"""
    import datetime as _dt

    import indicators
    import report_generator as rg

    entity = request.args.get("entity") or "company:公司A"
    metrics = _arg_list("metrics") or ["gas"]
    modes = _arg_list("modes") or ["daily"]

    try:
        with indicators.DataAccess() as da:
            dates = da.available_dates(entity)
        stat_date = _arg_date("date") or (dates[-1] if dates else _dt.date.today())
        results = indicators.compute(entity, stat_date, metrics, modes=modes)
        rep = rg.build_indicator_report(
            results, title="供热运营指标核算报告",
            scope=f"实体 {entity}，计算日 {stat_date}",
        )
    except Exception as e:  # noqa: BLE001
        logger.exception("报告生成失败")
        return _error(str(e), 500)

    if (request.args.get("format") or "json").lower() == "markdown":
        return Response(rep.to_markdown(), mimetype="text/plain; charset=utf-8")
    return jsonify({"success": True, "report": rep.to_dict()})


@app.route("/scada/<substation_id>", methods=["GET"])
def get_scada(substation_id):
    """换热站工况分析：/scada/L4?start=...&end=...&format=json|markdown"""
    import analytics
    import report_generator as rg

    try:
        result = analytics.scada_analysis(substation_id, start=_arg_date("start"),
                                          end=_arg_date("end"))
    except Exception as e:  # noqa: BLE001
        logger.exception("换热站分析失败")
        return _error(str(e), 500)

    if (request.args.get("format") or "").lower() == "markdown":
        if not result.get("available", True):
            return _error(result.get("reason", "无数据"), 404)
        return Response(rg.build_scada_report(result).to_markdown(),
                        mimetype="text/plain; charset=utf-8")
    return jsonify({"success": True, "substation_id": substation_id, "analysis": result})


@app.route("/scada/<substation_id>/temperature-curve", methods=["GET"])
def get_scada_temperature_curve(substation_id):
    """换热站供回水温度曲线（供前端图表直接渲染）。

    原始数据为小时级、单站上万点，直接返回体积过大，故按等步长抽稀到
    max_points（默认 1000）个点；原始点数在 count 中如实给出。
    """
    import analytics

    max_points = max(10, int(request.args.get("max_points") or 1000))
    try:
        with analytics.DataReader() as dr:
            rows = dr.scada_series(substation_id, start=_arg_date("start"),
                                   end=_arg_date("end"))
    except Exception as e:  # noqa: BLE001
        logger.exception("温度曲线查询失败")
        return _error(str(e), 500)

    if not rows:
        return _error(f"换热站 {substation_id} 无数据", 404)

    total = len(rows)
    # 向上取整，保证抽稀后的点数不超过 max_points（用整除会多出最多约 stride-1 个点）
    stride = max(1, -(-total // max_points))
    sampled = rows[::stride]

    def _delta(r):
        if r["t_sup_sec"] is None or r["t_ret_sec"] is None:
            return None
        return round(r["t_sup_sec"] - r["t_ret_sec"], 3)

    series = {
        "timestamp": [str(r["ts"]) for r in sampled],
        "t_amb": [r["t_amb"] for r in sampled],
        "t_sup_prim": [r["t_sup_prim"] for r in sampled],
        "t_ret_prim": [r["t_ret_prim"] for r in sampled],
        "t_sup_sec": [r["t_sup_sec"] for r in sampled],
        "t_ret_sec": [r["t_ret_sec"] for r in sampled],
        "delta_t_sec": [_delta(r) for r in sampled],
    }
    return jsonify({
        "success": True,
        "substation_id": substation_id,
        "count": total,
        "returned": len(sampled),
        "stride": stride,
        "reasonable_delta_t": [config.SCADA_DELTA_T_MIN, config.SCADA_DELTA_T_MAX],
        "series": series,
    })


@app.route("/households/top", methods=["GET"])
def get_households_top():
    """住户用电排名：/households/top?limit=10&resolution=daily|15min"""
    import analytics

    limit = int(request.args.get("limit") or 10)
    resolution = request.args.get("resolution") or "daily"
    try:
        with analytics.DataReader() as dr:
            rows = dr.household_top(limit=limit, resolution=resolution)
    except Exception as e:  # noqa: BLE001
        logger.exception("住户排名失败")
        return _error(str(e), 500)
    return jsonify({"success": True, "households": rows, "count": len(rows)})


@app.route("/households/<household_id>", methods=["GET"])
def get_household(household_id):
    """住户用电分析：/households/123?resolution=daily&format=json|markdown"""
    import analytics
    import report_generator as rg

    resolution = request.args.get("resolution") or "daily"
    try:
        result = analytics.household_analysis(household_id, resolution=resolution)
    except Exception as e:  # noqa: BLE001
        logger.exception("住户分析失败")
        return _error(str(e), 500)

    if (request.args.get("format") or "").lower() == "markdown":
        if not result.get("available", True):
            return _error(result.get("reason", "无数据"), 404)
        return Response(rg.build_household_report(result).to_markdown(),
                        mimetype="text/plain; charset=utf-8")
    return jsonify({"success": True, "household_id": household_id, "analysis": result})


@app.route("/weather/<weather_id>", methods=["GET"])
def get_weather(weather_id):
    """气象趋势分析：/weather/8jB?start=...&end=...&format=json|markdown"""
    import analytics
    import report_generator as rg

    try:
        result = analytics.weather_trend(weather_id, start=_arg_date("start"),
                                         end=_arg_date("end"))
    except Exception as e:  # noqa: BLE001
        logger.exception("气象分析失败")
        return _error(str(e), 500)

    if (request.args.get("format") or "").lower() == "markdown":
        if not result.get("available"):
            return _error(result.get("reason", "无数据"), 404)
        return Response(rg.build_weather_report(result).to_markdown(),
                        mimetype="text/plain; charset=utf-8")
    return jsonify({"success": True, "weather_id": weather_id, "analysis": result})


# ============================================================================
# 启动
# ============================================================================


def bootstrap() -> None:
    """启动自检：探测关系库与知识库。二者任一不可用都不阻止服务启动。"""
    db_ok, db_err = _db_status()
    logger.info("关系数据库：%s", "可用" if db_ok else f"不可用（{db_err}）")
    ks = _knowledge.status()
    if ks["available"]:
        with _storage() as s:
            logger.info("知识库分片数：%d", s.count())
    else:
        logger.warning("知识库不可用：%s", ks["error"])


if __name__ == "__main__":
    bootstrap()
    logger.info("API 服务启动于 %s:%s", config.AGENT_APP_HOST, config.AGENT_APP_PORT)
    app.run(host=config.AGENT_APP_HOST, port=config.AGENT_APP_PORT,
            threaded=True, debug=False)
