"""文档预处理与向量检索。

流水线：解析(parse) → 清洗(clean) → 分片(split) → 向量化(embed) → 入 Milvus

支持格式：.pdf .docx .xlsx .csv .txt .md .json
目录遍历：递归（知识库分散在 documents/ 各子目录）
向量模型：真实文本向量模型（见 llm_client.EmbeddingClient）
向量库：  Milvus（pymilvus 真实客户端）

用法：
    python document_processor.py --ingest              # 构建知识库（递归解析 documents/）
    python document_processor.py --ingest --paths A B  # 只入库指定路径
    python document_processor.py --list                # 列出已入库文档
    python document_processor.py --query "日气耗指标怎么算"   # 检索测试
    python document_processor.py --drop                # 删除集合
"""

import argparse
import logging
import os
import re
import time
from typing import Any, Dict, Iterable, List, Optional

import openpyxl
from pymilvus import (
    CollectionSchema,
    DataType,
    FieldSchema,
    MilvusClient,
)

import config
from llm_client import EmbeddingClient, get_embedding

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("document_processor")

PARSABLE_EXTS = {".pdf", ".docx", ".xlsx", ".xlsm", ".csv", ".txt", ".md", ".json"}

# 单个文件解析出的文本上限（防止超大数据文件撑爆内存）
MAX_CHARS_PER_FILE = 2_000_000


# ============================================================================
# 数据结构
# ============================================================================


class Document:
    def __init__(
        self,
        document_id: str,
        filename: str,
        content: str,
        chunks: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.document_id = document_id
        self.filename = filename
        self.content = content
        self.chunks = chunks or []
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_id": self.document_id,
            "filename": self.filename,
            "content_length": len(self.content),
            "chunk_count": len(self.chunks),
            "metadata": self.metadata,
        }


# ============================================================================
# 解析器
# ============================================================================


def _parse_pdf(path: str) -> str:
    from pypdf import PdfReader

    reader = PdfReader(path)
    parts = []
    for i, page in enumerate(reader.pages, 1):
        try:
            t = page.extract_text() or ""
        except Exception as e:  # noqa: BLE001
            logger.warning("%s 第 %d 页解析失败：%s", os.path.basename(path), i, e)
            t = ""
        if t.strip():
            parts.append(t)
    return "\n".join(parts)


def _parse_docx(path: str) -> str:
    import docx  # python-docx

    d = docx.Document(path)
    parts: List[str] = [p.text for p in d.paragraphs if p.text and p.text.strip()]
    for ti, table in enumerate(d.tables, 1):
        parts.append(f"[表 {ti}]")
        for row in table.rows:
            cells = [c.text.strip().replace("\n", " ") for c in row.cells]
            if any(cells):
                parts.append(" | ".join(cells))
    return "\n".join(parts)


def _parse_xlsx(path: str) -> str:
    wb = openpyxl.load_workbook(path, data_only=True, read_only=True)
    parts: List[str] = []
    for ws in wb.worksheets:
        if ws.max_row == 1 and ws.max_column == 1 and ws.cell(1, 1).value is None:
            continue  # 空工作表
        parts.append(f"[工作表] {ws.title}")
        for row in ws.iter_rows(values_only=True):
            cells = ["" if v is None else str(v).strip().replace("\n", " ") for v in row]
            if any(cells):
                parts.append(" | ".join(cells))
    wb.close()
    return "\n".join(parts)


def _parse_csv(path: str) -> str:
    """CSV 转文本：保留表头并给出前若干行样本（数据字典类文件通常较短）。"""
    import csv

    with open(path, "r", encoding="utf-8-sig", errors="replace", newline="") as f:
        sample = f.read(8192)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=";,")
        except csv.Error:
            dialect = csv.excel
        reader = csv.reader(f, dialect)
        rows = []
        for i, row in enumerate(reader):
            if i > 5000:
                rows.append("...（后续行略）")
                break
            rows.append(" | ".join(c.strip() for c in row))
    return "\n".join(rows)


def _parse_text(path: str) -> str:
    with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
        return f.read()


PARSERS = {
    ".pdf": _parse_pdf,
    ".docx": _parse_docx,
    ".xlsx": _parse_xlsx,
    ".xlsm": _parse_xlsx,
    ".csv": _parse_csv,
    ".txt": _parse_text,
    ".md": _parse_text,
    ".json": _parse_text,
}


# ============================================================================
# 预处理流水线
# ============================================================================


class DocumentProcessor:
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        embedding_client: Optional[EmbeddingClient] = None,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._embedding = embedding_client

    @property
    def embedding(self) -> EmbeddingClient:
        if self._embedding is None:
            self._embedding = get_embedding()
        return self._embedding

    # ---- 解析 ----

    def parse(self, file_path: str) -> str:
        ext = os.path.splitext(file_path)[1].lower()
        parser = PARSERS.get(ext)
        if parser is None:
            raise ValueError(f"不支持的格式：{ext}（{os.path.basename(file_path)}）")
        text = parser(file_path) or ""
        if len(text) > MAX_CHARS_PER_FILE:
            logger.warning(
                "%s 文本过长（%d 字符），截断至 %d",
                os.path.basename(file_path), len(text), MAX_CHARS_PER_FILE,
            )
            text = text[:MAX_CHARS_PER_FILE]
        return text

    # ---- 清洗 ----

    def clean(self, text: str) -> str:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f]", "", text)
        text = re.sub(r"[ \t　]+", " ", text)
        text = re.sub(r" *\n *", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    # ---- 分片 ----

    # 断句优先级：换行 > 句末标点/表格分隔 > 空白
    _BREAK_GROUPS = ("\n", "。！？；.!?;|", " \t")

    def split(self, text: str) -> List[str]:
        """定长分片，优先在行与句边界断开，片间保留重叠。"""
        if not text:
            return []
        chunks: List[str] = []
        start = 0
        n = len(text)
        hard = self.chunk_size
        soft = int(self.chunk_size * 0.6)  # 允许回退到的最小长度

        while start < n:
            end = min(start + hard, n)
            if end < n:
                window = text[start + soft:end]
                for group in self._BREAK_GROUPS:
                    cut = -1
                    for ch in group:
                        idx = window.rfind(ch)
                        if idx > cut:
                            cut = idx
                    if cut != -1:
                        end = start + soft + cut + 1
                        break
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end >= n:
                break
            # 重叠尽量对齐到行/词边界，避免从词中间开始
            new_start = max(end - self.chunk_overlap, start + 1)
            nl = text.rfind("\n", new_start, end)
            if nl != -1:
                new_start = nl + 1
            else:
                sp = text.rfind(" ", new_start, end)
                if sp != -1:
                    new_start = sp + 1
            start = new_start
        return chunks

    # ---- 向量化 ----

    def embed(self, texts: List[str]) -> List[List[float]]:
        return self.embedding.embed(texts)

    # ---- 单文件处理 ----

    def process(self, file_path: str) -> Document:
        filename = os.path.basename(file_path)
        raw = self.parse(file_path)
        cleaned = self.clean(raw)
        chunks = self.split(cleaned)
        doc_id = _make_doc_id(file_path)
        return Document(
            document_id=doc_id,
            filename=filename,
            content=cleaned,
            chunks=chunks,
            metadata={
                "path": os.path.relpath(file_path).replace("\\", "/"),
                "ext": os.path.splitext(file_path)[1].lower(),
                "size_bytes": os.path.getsize(file_path),
                "chunk_count": len(chunks),
            },
        )

    # ---- 目录处理（递归）----

    def process_directory(
        self,
        directory: str,
        recursive: bool = True,
    ) -> List[Document]:
        docs: List[Document] = []
        if not os.path.isdir(directory):
            raise FileNotFoundError(f"目录不存在：{directory}")
        for root, _dirs, files in os.walk(directory):
            for fn in sorted(files):
                if os.path.splitext(fn)[1].lower() not in PARSABLE_EXTS:
                    continue
                path = os.path.join(root, fn)
                try:
                    docs.append(self.process(path))
                except Exception as e:  # noqa: BLE001
                    logger.warning("解析失败，跳过 %s：%s", path, e)
            if not recursive:
                break
        logger.info("共解析 %d 个文档", len(docs))
        return docs


def _make_doc_id(file_path: str) -> str:
    """文档 ID：取相对路径，去掉扩展名并做安全化。"""
    rel = os.path.relpath(file_path).replace("\\", "/")
    rel = os.path.splitext(rel)[0]
    return re.sub(r"[^0-9A-Za-z_./一-鿿-]", "_", rel)


# ============================================================================
# Milvus 向量库
# ============================================================================


class MilvusError(RuntimeError):
    pass


class MilvusStorage:
    """Milvus 向量库封装（MilvusClient 统一接口）。

    同一套代码适配两种后端，由 config.MILVUS_URI 决定：
      * 本地文件路径  -> Milvus Lite（开发环境，无需 Docker）
      * http://主机:端口 -> Milvus 服务端（部署环境）
    """

    def __init__(
        self,
        uri: Optional[str] = None,
        collection_name: Optional[str] = None,
        dim: Optional[int] = None,
    ):
        self.uri = uri or config.MILVUS_DB_URI
        self.collection_name = collection_name or config.MILVUS_COLLECTION
        self.dim = dim or config.EMBEDDING_DIM
        self.metric = config.MILVUS_METRIC
        self._client: Optional[MilvusClient] = None

    @property
    def is_lite(self) -> bool:
        return not str(self.uri).lower().startswith(("http://", "https://"))

    # ---- 连接 ----

    # 连接重试次数：
    #   Lite —— 嵌入式服务端在最后一个 client 关闭后会停机，下次 connect 需把它重新拉起，
    #           这个重启偶发来不及（报 "Fail connecting to server on 127.0.0.1:<port>"），
    #           失败得很快，所以多试几次很便宜；
    #   服务端 —— 真连不上时每次都要等满超时，多试几次代价高，只重试一次即可。
    LITE_CONNECT_RETRIES = 3
    SERVER_CONNECT_RETRIES = 2

    def connect(self) -> None:
        if self.is_lite:
            parent = os.path.dirname(os.path.abspath(self.uri))
            if parent:
                os.makedirs(parent, exist_ok=True)

        attempts = self.LITE_CONNECT_RETRIES if self.is_lite else self.SERVER_CONNECT_RETRIES
        last_exc: Optional[Exception] = None
        for attempt in range(attempts):
            try:
                self._client = MilvusClient(uri=self.uri, timeout=config.MILVUS_TIMEOUT)
                if attempt:
                    logger.info("Milvus 第 %d 次连接成功：%s", attempt + 1, self.uri)
                else:
                    logger.info(
                        "已连接 Milvus（%s）：%s",
                        "Lite 本地文件" if self.is_lite else "服务端", self.uri,
                    )
                return
            except Exception as e:  # noqa: BLE001
                last_exc = e
                if attempt < attempts - 1:
                    time.sleep(1.5 * (attempt + 1))

        hint = (
            "本地 Lite 模式无需 Docker，请检查 MILVUS_URI 路径是否可写；"
            "也可安装依赖：pip install milvus-lite"
            if self.is_lite
            else "请先启动 Milvus：docker-compose up -d etcd minio milvus-standalone"
        )
        raise MilvusError(f"无法初始化 Milvus（{self.uri}）：{last_exc}\n{hint}") from last_exc

    @property
    def client(self) -> MilvusClient:
        if self._client is None:
            self.connect()
        return self._client

    def close(self) -> None:
        """关闭连接。

        重要：pymilvus 硬编码了 grpc.keepalive_permit_without_calls=True 且心跳 10s，
        空闲连接会被 Milvus 服务端以 GOAWAY(too_many_pings) 限流。
        因此**不要让连接跨越大模型调用等长耗时阶段**，用完即关。
        """
        if self._client is not None:
            try:
                self._client.close()
            except Exception as e:  # noqa: BLE001
                logger.debug("关闭 Milvus 连接时异常（忽略）：%s", e)
            self._client = None

    def __enter__(self) -> "MilvusStorage":
        self.connect()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def _fields(self) -> List[FieldSchema]:
        return [
            FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="filename", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="path", dtype=DataType.VARCHAR, max_length=1024),
            FieldSchema(name="chunk_index", dtype=DataType.INT64),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dim),
        ]

    def _index_params(self):
        """向量字段的索引参数（AUTOINDEX + 配置的度量方式）。"""
        params = self.client.prepare_index_params()
        params.add_index(field_name="embedding", index_type="AUTOINDEX", metric_type=self.metric)
        return params

    @staticmethod
    def _is_lite() -> bool:
        """是否在用 Milvus Lite（本地文件而非服务端）。

        Lite 会自行建索引并载入，**不需要也不应该手动建索引**：
        它的索引落盘走 faiss 的 C++ 文件 I/O，遇到含中文的项目路径会报
        「Illegal byte sequence」而失败，并把集合置为 released，反而弄坏本来可用的库。
        """
        uri = str(config.MILVUS_DB_URI or "")
        return not uri.startswith(("http://", "https://"))

    def ensure_index(self) -> None:
        """确保向量字段已建索引（幂等）。

        **服务端 Milvus 必须先有索引才能载入集合**：没有索引时 load 报 `index not found`，
        随后 query/search 全报 `collection not loaded`——表现为「统计说有条数、但检索与列表都是空」。
        Milvus Lite 自动建索引，此处直接跳过（见 _is_lite 的说明）。
        """
        if self._is_lite():
            return
        try:
            if self.client.list_indexes(self.collection_name):
                return
        except Exception as e:  # noqa: BLE001
            logger.warning("读取索引列表失败，仍尝试建索引：%s", e)
        self.client.create_index(
            collection_name=self.collection_name, index_params=self._index_params())
        logger.info("已为集合 %s 建立向量索引（metric=%s）", self.collection_name, self.metric)

    def load(self) -> None:
        """把集合载入内存。

        **服务端 Milvus 必须显式载入**（Milvus Lite 会自动完成），否则 query/search
        一律报 `collection not loaded`。对已载入的集合是幂等的。
        """
        try:
            self.client.load_collection(self.collection_name)
        except Exception as e:  # noqa: BLE001
            logger.warning("集合载入失败（若已载入可忽略）：%s", e)

    @staticmethod
    def _not_loaded(e: Exception) -> bool:
        """失败是否属于「集合不可用，补齐索引/载入后可重试」。

        服务端报 `collection not loaded` / `index not found`，
        Lite 报 `Collection ... is in state 'released'`，三种都该触发自愈。
        """
        msg = str(e).lower()
        return "not loaded" in msg or "index not found" in msg or "released" in msg

    def _retry_on_unloaded(self, call):
        """执行一次；若因「集合未载入」失败，则载入后重试一次。

        Milvus 重启后集合的载入状态会丢失，有了这层重试，应用无需人工干预即可自愈。
        """
        try:
            return call()
        except Exception as e:  # noqa: BLE001
            if not self._not_loaded(e):
                raise
            logger.info("集合未载入或缺索引，自动补齐后重试：%s", str(e)[:80])
            self.ensure_index()      # 缺索引会导致载入失败，两者一起补（都幂等）
            self.load()
            return call()

    def create_collection(self, drop_existing: bool = False) -> None:
        if drop_existing and self.client.has_collection(self.collection_name):
            self.client.drop_collection(self.collection_name)
            logger.info("已删除旧集合 %s", self.collection_name)

        if not self.client.has_collection(self.collection_name):
            schema = CollectionSchema(
                fields=self._fields(),
                description="供热运营知识库（计算说明/规范/数据字典）",
            )
            if self._is_lite():
                # Lite：保持原有行为（自己建索引），不传 index_params
                self.client.create_collection(
                    collection_name=self.collection_name,
                    schema=schema,
                    metric_type=self.metric,
                )
            else:
                # 服务端：建集合时一并建索引，否则之后无法载入
                self.client.create_collection(
                    collection_name=self.collection_name,
                    schema=schema,
                    index_params=self._index_params(),
                )
            logger.info(
                "已创建集合 %s（dim=%d, metric=%s）",
                self.collection_name, self.dim, self.metric,
            )

        # 服务端 Milvus：必须有索引才能载入，载入后才能 query/search。
        # 两步对 Lite 都是幂等的无副作用操作；对已存在但缺索引的集合也在此补齐。
        self.ensure_index()
        self.load()

    # ---- 写入 ----

    def insert(self, doc: Document) -> int:
        if not doc.chunks:
            logger.warning("%s 无有效分片，跳过", doc.filename)
            return 0
        vectors = self.embed(doc.chunks)
        if len(vectors) != len(doc.chunks):
            raise MilvusError(
                f"向量数量({len(vectors)})与分片数量({len(doc.chunks)})不一致：{doc.filename}"
            )
        dim = len(vectors[0])
        if dim != self.dim:
            raise MilvusError(
                f"向量维度 {dim} 与集合维度 {self.dim} 不一致，"
                "请修改 .env 中 EMBEDDING_DIM 后重建集合（--drop）。"
            )

        path = doc.metadata.get("path", "")
        rows = [
            {
                "doc_id": doc.document_id,
                "filename": doc.filename,
                "path": path,
                "chunk_index": i,
                "content": chunk,
                "embedding": vec,
            }
            for i, (chunk, vec) in enumerate(zip(doc.chunks, vectors))
        ]
        self.client.insert(collection_name=self.collection_name, data=rows)
        logger.info("已写入 %s：%d 个分片", doc.filename, len(doc.chunks))
        return len(doc.chunks)

    def insert_many(self, docs: Iterable[Document]) -> int:
        total = 0
        for d in docs:
            try:
                total += self.insert(d)
            except Exception as e:  # noqa: BLE001
                logger.error("写入失败 %s：%s", d.filename, e)
        if total:
            self.flush()          # 整批写完统一落盘一次（避免每个文档都 flush）
        return total

    def flush(self) -> None:
        """把已插入的数据落盘。

        **服务端 Milvus 不像 Milvus Lite 那样自动落盘**：不 flush 的话集合统计
        （row_count）会一直是 0，进程退出后这批数据也可能不落库。

        实现上有个坑：pymilvus 2.3.x 的 `MilvusClient` **没有 flush 方法**
        （3.x 才加），因此只能走 ORM 的 `Collection.flush()`，并为此单独建一个
        ORM 连接（`MilvusClient` 建的连接不注册到 ORM 的别名上）。
        """
        if self._is_lite():
            return          # Lite 自动落盘，不必也不能走这条路径
        try:
            from pymilvus import Collection, connections

            alias = "flush_helper"
            if not connections.has_connection(alias):
                connections.connect(alias=alias, uri=config.MILVUS_DB_URI)
            Collection(name=self.collection_name, using=alias).flush()
        except Exception as e:  # noqa: BLE001
            logger.warning("flush 失败（数据可能稍后才落盘）：%s", e)

    def embed(self, texts: List[str]) -> List[List[float]]:
        return get_embedding().embed(texts)

    # ---- 检索 ----

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        vec = get_embedding().embed_one(query)
        res = self._retry_on_unloaded(lambda: self.client.search(
            collection_name=self.collection_name,
            data=[vec],
            limit=top_k,
            output_fields=["doc_id", "filename", "path", "chunk_index", "content"],
            search_params={"metric_type": self.metric},
        ))
        out: List[Dict[str, Any]] = []
        for hits in res:
            for h in hits:
                ent = h.get("entity", {}) if isinstance(h, dict) else getattr(h, "entity", {})
                out.append(
                    {
                        "document_id": ent.get("doc_id"),
                        "filename": ent.get("filename"),
                        "path": ent.get("path"),
                        "chunk_index": ent.get("chunk_index"),
                        "content": ent.get("content"),
                        "similarity": float(
                            h.get("distance") if isinstance(h, dict) else h.score
                        ),
                    }
                )
        return out

    # ---- 文档管理 ----

    def count(self) -> int:
        try:
            stats = self.client.get_collection_stats(self.collection_name)
            return int(stats.get("row_count", 0))
        except Exception:  # noqa: BLE001
            return 0

    def list_documents(self) -> List[Dict[str, Any]]:
        rows = self._query('pk >= 0', ["doc_id", "filename", "path"])
        agg: Dict[str, Dict[str, Any]] = {}
        for r in rows:
            did = r.get("doc_id")
            if did is None:
                continue
            item = agg.setdefault(
                did,
                {
                    "document_id": did,
                    "filename": r.get("filename"),
                    "path": r.get("path"),
                    "chunk_count": 0,
                },
            )
            item["chunk_count"] += 1
        return sorted(agg.values(), key=lambda x: x["document_id"])

    def get(self, document_id: str) -> Optional[Dict[str, Any]]:
        rows = self._query(
            f'doc_id == "{_escape(document_id)}"',
            ["doc_id", "filename", "path", "chunk_index", "content"],
        )
        if not rows:
            return None
        rows.sort(key=lambda r: r.get("chunk_index") or 0)
        return {
            "document_id": document_id,
            "filename": rows[0].get("filename"),
            "path": rows[0].get("path"),
            "chunk_count": len(rows),
            "chunks": [r.get("content") for r in rows],
        }

    def delete(self, document_id: str) -> None:
        self.client.delete(
            collection_name=self.collection_name,
            filter=f'doc_id == "{_escape(document_id)}"',
        )
        logger.info("已删除文档 %s", document_id)

    def drop(self) -> None:
        if self.client.has_collection(self.collection_name):
            self.client.drop_collection(self.collection_name)
            logger.info("已删除集合 %s", self.collection_name)

    # ---- 内部 ----

    def _query(self, expr: str, output_fields: List[str], limit: int = 16384) -> List[Dict]:
        res = self._retry_on_unloaded(lambda: self.client.query(
            collection_name=self.collection_name,
            filter=expr,
            output_fields=output_fields,
            limit=limit,
        ))
        # MilvusClient.query 在不同版本下返回 list / ExtraList，统一成 list[dict]
        return [dict(r) for r in list(res)]


def _escape(s: str) -> str:
    return str(s).replace("\\", "\\\\").replace('"', '\\"')


# ============================================================================
# CLI
# ============================================================================


def _default_knowledge_paths() -> List[str]:
    """知识库来源：只含知识类文档，不含已入关系库的时序数据。

    住户电表（5,281 个 CSV / 约 4.8GB）、气象与 SCADA 时序数据均已在 MySQL 中，
    由指标计算与统计分析直接取数，不走向量库，避免把原始时序灌进向量检索。

    国家标准（GB/CJJ 等）正文需另行提供，放入 documents/05_standards/ 即自动纳入；
    未提供时，规范类问题的回答能力受限。
    """
    candidates = [
        # 指标定义与计算公式（知识库核心）+ 两张业务报表（含口径注释）
        config.BUSINESS_DATA_DIR,
        # 数据字典：变量含义，使系统理解各字段业务语义
        os.path.join(config.HEATPUMP_DATA_DIR, "meta_data", "meta_data_variables.csv"),
        os.path.join(config.HEATPUMP_DATA_DIR, "reports", "protocols_variables.csv"),
        os.path.join(config.HEATPUMP_DATA_DIR, "weather_data", "overview"),
        # 智能电表字段字典（数据源未附带，由本项目按实际列整理；
        # 含 kWh_received_HeatPump 等字段释义与稀疏性说明）
        os.path.join(config.HEATPUMP_DATA_DIR, "smart_meter_data",
                     "smart_meter_variables.csv"),
        # 国家规范条文摘要（用户提供后自动纳入）
        os.path.join(config.DOCUMENTS_DIR, "05_standards"),
    ]
    return [p for p in candidates if os.path.exists(p)]


def build_processor() -> DocumentProcessor:
    return DocumentProcessor()


def ingest(paths: Optional[List[str]] = None, drop_existing: bool = True) -> None:
    processor = build_processor()
    targets = paths or _default_knowledge_paths()

    docs: List[Document] = []
    for t in targets:
        if os.path.isfile(t):
            try:
                docs.append(processor.process(t))
            except Exception as e:  # noqa: BLE001
                logger.warning("解析失败，跳过 %s：%s", t, e)
        elif os.path.isdir(t):
            try:
                docs.extend(processor.process_directory(t))
            except Exception as e:  # noqa: BLE001
                logger.warning("目录处理失败，跳过 %s：%s", t, e)
        else:
            logger.warning("路径不存在，跳过：%s", t)

    # 去重（同一文件可能被目录与显式路径重复覆盖）
    uniq: Dict[str, Document] = {d.document_id: d for d in docs}
    docs = list(uniq.values())
    logger.info("待入库文档 %d 个，总分片 %d", len(docs), sum(len(d.chunks) for d in docs))
    if not docs:
        raise SystemExit("没有可入库的文档，请检查 documents/ 与路径配置。")

    storage = MilvusStorage()
    storage.connect()
    storage.create_collection(drop_existing=drop_existing)
    n = storage.insert_many(docs)
    print(f"\n入库完成：文档 {len(docs)} 个 / 分片 {n} 个 / 集合 {storage.collection_name}")


def main() -> None:
    ap = argparse.ArgumentParser(description="文档预处理与向量检索")
    ap.add_argument("--ingest", action="store_true", help="构建知识库")
    ap.add_argument("--paths", nargs="*", help="指定入库路径（默认使用内置知识源列表）")
    ap.add_argument("--no-drop", action="store_true", help="保留已有集合，追加写入")
    ap.add_argument("--list", action="store_true", help="列出已入库文档")
    ap.add_argument("--query", help="检索测试")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--drop", action="store_true", help="删除集合")
    ap.add_argument("--parse-only", nargs="*", help="只解析不入库，打印清洗分片结果（调试用）")
    args = ap.parse_args()

    if args.parse_only is not None:
        p = build_processor()
        targets = args.parse_only or _default_knowledge_paths()
        for t in targets:
            if os.path.isfile(t):
                d = p.process(t)
                print(f"\n=== {d.filename}  分片 {len(d.chunks)} 个 ===")
                for i, c in enumerate(d.chunks[:3]):
                    print(f"  [{i}] {c[:160]}")
        return

    if args.ingest:
        ingest(args.paths, drop_existing=not args.no_drop)
        return

    if args.drop:
        s = MilvusStorage()
        s.connect()
        s.drop()
        return

    if args.list or args.query:
        s = MilvusStorage()
        s.connect()
        s.create_collection()
        if args.list:
            docs = s.list_documents()
            print(f"\n集合 {s.collection_name}：{len(docs)} 个文档，{s.count()} 个分片")
            for d in docs:
                print(f"  {d['document_id']:<60}{d['chunk_count']:>5} 片")
        if args.query:
            print(f"\n检索：{args.query}")
            for r in s.search(args.query, top_k=args.top_k):
                print(f"  {r['similarity']:.4f}  {r['filename']}[{r['chunk_index']}]  {r['content'][:90]}")
        return

    ap.print_help()


if __name__ == "__main__":
    main()
