# 供热运营数据查询与分析系统 —— 应用镜像（多阶段构建）
#
# 与开发环境对齐到 Python 3.10（开发机为 3.10.9），避免解释器版本差异带来的隐性风险。
# 依赖装在独立的 /app/deps 目录，运行时通过 PYTHONPATH 引入 —— 不写死
# /usr/local/lib/python3.x/site-packages 这种带版本号的路径，换 Python 版本也不会失效。

FROM python:3.10-slim AS builder

WORKDIR /app

# torch 装 CPU 版（本地向量模型 EMBEDDING_PROVIDER=local 需要它）：
#   * 用 --extra-index-url 而不是 --index-url —— 后者会**整个替换**默认源，
#     torch 的依赖（typing-extensions 等）就取不到了；PyTorch 索引里那份
#     typing-extensions 元数据还是下划线命名，pip 会判定 Name 不一致而丢弃。
#   * 用带本地版本标签的 torch==2.5.1+cpu 精确锁定：PyPI 上的 2.5.1 是 CUDA 版
#     （2GB+），加上 +cpu 后只匹配 PyTorch 索引里的 CPU 轮子（约 175MB）。
RUN pip install --no-cache-dir --target /app/deps \
      --extra-index-url https://download.pytorch.org/whl/cpu \
      "torch==2.5.1+cpu"

# 其余依赖（含 transformers）从 PyPI 安装。
# 过滤掉 milvus-lite：部署连 Milvus 服务端，不需要嵌入式本地向量库，
# 代码里也没有直接 import 它（业务代码只用 pymilvus.MilvusClient）。
COPY requirements.txt .
RUN grep -v '^[[:space:]]*milvus-lite' requirements.txt > /tmp/req-deploy.txt \
    && pip install --no-cache-dir --target /app/deps -r /tmp/req-deploy.txt


FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app:/app/deps \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

WORKDIR /app

COPY --from=builder /app/deps /app/deps

# 只复制运行所需文件。.env / milvus_data / models / uploads / documents 等
# 由 .dockerignore 排除（其中 documents 与 models 通过 volume 挂载进来）。
COPY *.py requirements.txt ./
COPY web/ ./web/

# uploads 目录在镜像里就建好并交给 appuser：命名卷首次挂载时会继承镜像中该路径的
# 属主，这样非 root 运行也能写入（否则卷是 root 属主，上传会失败）。
# 只 chown uploads 而不是整个 /app —— /app/deps 有 1GB+ 只读依赖，
# 递归 chown 会让构建多花好几分钟且毫无必要（其余路径世界可读即可）。
RUN mkdir -p uploads \
    && apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app/uploads
USER appuser

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -fsS http://localhost:5000/health || exit 1

CMD ["python", "api_server.py"]
