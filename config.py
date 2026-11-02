"""集中配置：所有可调参数统一从环境变量读取，供各模块引用。

密钥类参数（LLM_API_KEY / EMBEDDING_API_KEY / DB_PASSWORD）在 .env 中配置，
.env.example 仅作模板，不含真实密钥。
"""

import os

from dotenv import load_dotenv

load_dotenv()


def _str(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


def _int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


# ---------------- 关系数据库（MySQL）----------------
DB_HOST = _str("DB_HOST", "127.0.0.1")
DB_PORT = _int("DB_PORT", 3306)
DB_USER = _str("DB_USER", "root")
DB_PASSWORD = _str("DB_PASSWORD", "")
DB_NAME = _str("DB_NAME", "heating_ops")
DB_CHARSET = _str("DB_CHARSET", "utf8mb4")

DB_URL = (
    f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    f"?charset={DB_CHARSET}"
)
# 建库阶段使用（此时 DB_NAME 可能尚不存在）
DB_SERVER_URL = (
    f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/?charset={DB_CHARSET}"
)

# ---------------- 向量数据库（Milvus）----------------
# 注意：环境变量名不能叫 MILVUS_URI —— pymilvus 在 import 时会自行调用 load_dotenv()
# 并读取 MILVUS_URI 作为服务端地址（要求 http(s):// 形式），本地 Lite 路径会导致其导入报错。
# MILVUS_DB_URI 两种形态：
#   * 本地文件路径（如 milvus_data/heating_knowledge.db）—— Milvus Lite，开发环境，无需 Docker
#   * http://主机:端口（如 http://milvus-standalone:19530）—— Milvus 服务端，部署环境
MILVUS_DB_URI = _str("MILVUS_DB_URI", "milvus_data/heating_knowledge.db")
MILVUS_COLLECTION = _str("MILVUS_COLLECTION", "heating_knowledge")
MILVUS_METRIC = _str("MILVUS_METRIC", "IP")
# 连接/请求超时（秒）：避免网络或鉴权异常时无限等待
MILVUS_TIMEOUT = _int("MILVUS_TIMEOUT", 10)

# ---------------- 大语言模型 ----------------
LLM_PROVIDER = _str("LLM_PROVIDER", "openai").strip().lower()
LLM_BASE_URL = _str("LLM_BASE_URL", "https://api.openai.com/v1")
LLM_API_KEY = _str("LLM_API_KEY", "")
LLM_MODEL = _str("LLM_MODEL", "gpt-4o-mini")
LLM_TEMPERATURE = _float("LLM_TEMPERATURE", 0.2)
LLM_TIMEOUT = _int("LLM_TIMEOUT", 60)

# 各角色模型名覆盖：留空则回退上方的 LLM_MODEL。
# 只覆盖「模型名」——provider / base_url / 密钥 / 温度 / 超时仍走全局统一值。
# 角色标识与取值规则见 llm_client.ROLE_SPECS 与 llm_client.get_llm()。
LLM_MODEL_RETRIEVER = _str("LLM_MODEL_RETRIEVER", "")
LLM_MODEL_ANALYST = _str("LLM_MODEL_ANALYST", "")
LLM_MODEL_WRITER = _str("LLM_MODEL_WRITER", "")
LLM_MODEL_REVIEWER = _str("LLM_MODEL_REVIEWER", "")

OLLAMA_BASE_URL = _str("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = _str("OLLAMA_MODEL", "qwen2.5:7b")

# ---------------- 文本向量化 ----------------
# EMBEDDING_PROVIDER 取值：openai（OpenAI 兼容接口）| ollama（本地 Ollama）| local（本地 transformers 模型）
EMBEDDING_PROVIDER = _str("EMBEDDING_PROVIDER", "openai").strip().lower()
EMBEDDING_BASE_URL = _str("EMBEDDING_BASE_URL", "https://api.openai.com/v1")
EMBEDDING_API_KEY = _str("EMBEDDING_API_KEY", LLM_API_KEY)
EMBEDDING_MODEL = _str("EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_DIM = _int("EMBEDDING_DIM", 1536)
OLLAMA_EMBEDDING_MODEL = _str("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
# local 分支：本地已下载的 transformers 模型目录（离线，无需密钥）
EMBEDDING_LOCAL_PATH = _str("EMBEDDING_LOCAL_PATH", "")
EMBEDDING_BATCH_SIZE = _int("EMBEDDING_BATCH_SIZE", 16)

# ---------------- 供热业务参数（依据《数据及计算说明》）----------------
EVALUATION_FACTOR = _float("EVALUATION_FACTOR", 121)
DEGREE_DAY_BASE_TEMP = _float("DEGREE_DAY_BASE_TEMP", 18)
DEGREE_DAY_INDOOR_OFFSET = _float("DEGREE_DAY_INDOOR_OFFSET", 0.7)
LEGAL_HEATING_START_MONTH = _int("LEGAL_HEATING_START_MONTH", 11)
LEGAL_HEATING_START_DAY = _int("LEGAL_HEATING_START_DAY", 15)
DEGREE_DAY_BASELINE_MONTH = _int("DEGREE_DAY_BASELINE_MONTH", 11)
DEGREE_DAY_BASELINE_DAY = _int("DEGREE_DAY_BASELINE_DAY", 14)
HEATING_SEASON = _str("HEATING_SEASON", "2023-2024")
# 供热开始日期（设定口）：用于「已供暖天数 = 计算日 - 供热开始日期」。
# 留空则该口径返回不可用，系统不猜测。
HEATING_START_DATE = _str("HEATING_START_DATE", "")
# 公司级度日数折标所用气象站（HEAPO 站点为匿名化编号，仅作演示/替代温度源）。
# 留空则度日数折标因缺室外平均温度而返回不可用。
DEGREE_DAY_STATION = _str("DEGREE_DAY_STATION", "")

# ---------------- 异常检测与工况分析阈值 ----------------
# 统计越限阈值：超出「均值 ± k × 标准差」判为异常
ANOMALY_SIGMA = _float("ANOMALY_SIGMA", 2.0)
# 突变检测阈值：相邻时点变化率超过该百分比判为突变（能耗类）
ANOMALY_JUMP_PCT = _float("ANOMALY_JUMP_PCT", 30.0)
# 气温突变阈值（℃）：气温日间自然波动大，用绝对变化而非百分比
ANOMALY_TEMP_JUMP_C = _float("ANOMALY_TEMP_JUMP_C", 6.0)
# 度日数归一化的最小分母（℃·日）：HDD 过小时「用电量/HDD」会被放大成假异常
HDD_MIN_FOR_NORMALIZATION = _float("HDD_MIN_FOR_NORMALIZATION", 5.0)
# 换热站二次侧供回水温差合理区间（℃，工程经验值；过低疑水力失调，过高疑流量不足）
SCADA_DELTA_T_MIN = _float("SCADA_DELTA_T_MIN", 3.0)
SCADA_DELTA_T_MAX = _float("SCADA_DELTA_T_MAX", 25.0)

# ---------------- 服务 ----------------
AGENT_APP_HOST = _str("AGENT_APP_HOST", "0.0.0.0")
AGENT_APP_PORT = _int("AGENT_APP_PORT", 5000)
# 是否在问答链路里调用大模型（意图解析补全 / 综述润色 / 审核员定性审查）。
# 关掉后仍走完整四智能体协作与全部确定性校验，只是不调模型：便于离线演示与自动化测试。
AGENT_USE_LLM = _str("AGENT_USE_LLM", "true").strip().lower() not in ("0", "false", "no")
UPLOAD_FOLDER = _str("UPLOAD_FOLDER", "uploads")
MAX_UPLOAD_SIZE = _str("MAX_UPLOAD_SIZE", "50MB")
DOCUMENTS_DIR = _str("DOCUMENTS_DIR", "documents")
LOG_LEVEL = _str("LOG_LEVEL", "INFO")

# ---------------- 数据资产目录（documents/ 下四类）----------------
BUSINESS_DATA_DIR = os.path.join(DOCUMENTS_DIR, "01_business_data")
HEATPUMP_DATA_DIR = os.path.join(DOCUMENTS_DIR, "02_heapo_heatpump")
SCADA_DATA_DIR = os.path.join(DOCUMENTS_DIR, "03_xai4heat_scada")
REFERENCE_DATA_DIR = os.path.join(DOCUMENTS_DIR, "04_reference")
