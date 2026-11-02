"""大语言模型与文本向量化接入层。

统一从 config 读取配置，支持两种 provider（由 LLM_PROVIDER / EMBEDDING_PROVIDER 切换）：

  * openai —— 任何 OpenAI 兼容接口（OpenAI / 豆包 / DeepSeek / 本地 vLLM 等）
              对话：POST {base_url}/chat/completions
              向量：POST {base_url}/embeddings
  * ollama —— 本地 Ollama
              对话：POST {base_url}/api/chat
              向量：POST {base_url}/api/embed
  * local  —— 本地 transformers 模型（仅向量化，链路完全离线、无需密钥）
              默认适配 bert-base-chinese（hidden_size=768），均值池化 + L2 归一化

密钥来自 .env 的 LLM_API_KEY / EMBEDDING_API_KEY。未配置时抛出
LLMNotConfiguredError 并给出填写指引，不影响其余模块导入与运行。

自检（填好密钥后执行）：
    python llm_client.py --selftest
"""

import argparse
import json
import logging
import os
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import requests

import config

logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("llm_client")

RETRY_STATUS = {429, 500, 502, 503, 504}


class LLMNotConfiguredError(RuntimeError):
    """模型服务未配置（通常是密钥为空）。"""


def _require_key(key: str, env_name: str, provider: str) -> None:
    if provider == "openai" and not key:
        raise LLMNotConfiguredError(
            f"未配置 {env_name}。请在项目根目录的 .env 中填写后重试，例如：\n"
            f"    {env_name}=sk-xxxxxxxx\n"
            f"若使用本地模型，可将 LLM_PROVIDER / EMBEDDING_PROVIDER 改为 ollama。"
        )


def _post(url: str, headers: Dict[str, str], payload: Dict[str, Any],
          timeout: int, retries: int = 2) -> Dict[str, Any]:
    """带简单重试的 POST，返回 JSON。"""
    last_exc: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
            if resp.status_code in RETRY_STATUS and attempt < retries:
                wait = 2 ** attempt
                logger.warning("请求 %s 返回 %s，%.0fs 后重试", url, resp.status_code, wait)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:  # noqa: PERF203
            last_exc = e
            if attempt < retries:
                time.sleep(2 ** attempt)
                continue
            raise
    raise RuntimeError(f"请求失败：{last_exc}")


# ============================================================================
# 对话模型
# ============================================================================


class LLMClient:
    def __init__(
        self,
        provider: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        timeout: Optional[int] = None,
    ):
        self.provider = (provider or config.LLM_PROVIDER).lower()
        self.temperature = config.LLM_TEMPERATURE if temperature is None else temperature
        self.timeout = timeout or config.LLM_TIMEOUT
        if self.provider == "ollama":
            self.base_url = (base_url or config.OLLAMA_BASE_URL).rstrip("/")
            self.model = model or config.OLLAMA_MODEL
            self.api_key = ""
        else:
            self.base_url = (base_url or config.LLM_BASE_URL).rstrip("/")
            self.model = model or config.LLM_MODEL
            self.api_key = api_key if api_key is not None else config.LLM_API_KEY

    # ---- 对外接口 ----

    def chat(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        """单轮/多轮对话，返回助手文本。"""
        if self.provider == "ollama":
            return self._chat_ollama(messages, **kwargs)
        return self._chat_openai(messages, **kwargs)

    def complete(self, prompt: str, system: Optional[str] = None, **kwargs: Any) -> str:
        """便捷方法：单条 prompt。"""
        messages: List[Dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return self.chat(messages, **kwargs)

    def chat_json(self, prompt: str, system: Optional[str] = None, **kwargs: Any) -> Any:
        """要求模型输出 JSON，并解析为 Python 对象（容错去除 ```json 包裹）。"""
        text = self.complete(prompt, system=system, **kwargs)
        return parse_json_block(text)

    def chat_stream(self, messages: List[Dict[str, str]],
                    on_delta: Optional[Callable[[str], None]] = None,
                    **kwargs: Any) -> str:
        """流式对话：每收到一段增量就回调 on_delta(文本)，返回完整文本。

        用于把大模型的生成过程实时呈现给前端。**不做重试** —— 流式下一旦重发，
        前面已经推给前端的增量就会重复；失败时由调用方按「无模型输出」降级处理。
        """
        if self.provider == "ollama":
            return self._chat_ollama_stream(messages, on_delta, **kwargs)
        return self._chat_openai_stream(messages, on_delta, **kwargs)

    def complete_stream(self, prompt: str, system: Optional[str] = None,
                        on_delta: Optional[Callable[[str], None]] = None,
                        **kwargs: Any) -> str:
        """便捷方法：单条 prompt 的流式版本。"""
        messages: List[Dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return self.chat_stream(messages, on_delta=on_delta, **kwargs)

    def model_name(self) -> str:
        return self.model

    # ---- provider 实现 ----

    def _chat_openai(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        _require_key(self.api_key, "LLM_API_KEY", self.provider)
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": kwargs.pop("model", self.model),
            "messages": messages,
            "temperature": kwargs.pop("temperature", self.temperature),
            **kwargs,
        }
        data = _post(url, headers, payload, self.timeout)
        return (data.get("choices") or [{}])[0].get("message", {}).get("content", "") or ""

    def _chat_ollama(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": kwargs.pop("model", self.model),
            "messages": messages,
            "stream": False,
            "options": {"temperature": kwargs.pop("temperature", self.temperature)},
            **kwargs,
        }
        data = _post(url, {}, payload, self.timeout)
        return (data.get("message") or {}).get("content", "") or ""

    # ---- 流式实现 ----

    @staticmethod
    def _forward(on_delta: Optional[Callable[[str], None]], piece: str) -> None:
        """把一段增量交给回调；回调自身异常不能影响生成。"""
        if piece and on_delta:
            try:
                on_delta(piece)
            except Exception:  # noqa: BLE001
                pass

    def _chat_openai_stream(self, messages: List[Dict[str, str]],
                            on_delta: Optional[Callable[[str], None]],
                            **kwargs: Any) -> str:
        """OpenAI 兼容接口的流式：逐行读 SSE，取 choices[0].delta.content。"""
        _require_key(self.api_key, "LLM_API_KEY", self.provider)
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": kwargs.pop("model", self.model),
            "messages": messages,
            "temperature": kwargs.pop("temperature", self.temperature),
            "stream": True,
            **kwargs,
        }
        parts: List[str] = []
        with requests.post(url, headers=headers, json=payload,
                           timeout=self.timeout, stream=True) as resp:
            resp.raise_for_status()
            # 必须按 UTF-8 解码：text/event-stream 不带 charset 时 requests 会默认
            # 按 ISO-8859-1 解，中文会变成乱码（非流式路径走 resp.json() 无此问题）
            for raw in resp.iter_lines(decode_unicode=False):
                if not raw:
                    continue
                line = raw.decode("utf-8", errors="replace")
                if not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue
                choices = obj.get("choices") or [{}]
                piece = (choices[0].get("delta") or {}).get("content") or ""
                if piece:
                    parts.append(piece)
                    self._forward(on_delta, piece)
        return "".join(parts)

    def _chat_ollama_stream(self, messages: List[Dict[str, str]],
                            on_delta: Optional[Callable[[str], None]],
                            **kwargs: Any) -> str:
        """Ollama 的流式：逐行读 JSON，取 message.content。"""
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": kwargs.pop("model", self.model),
            "messages": messages,
            "stream": True,
            "options": {"temperature": kwargs.pop("temperature", self.temperature)},
            **kwargs,
        }
        parts: List[str] = []
        with requests.post(url, headers={}, json=payload,
                           timeout=self.timeout, stream=True) as resp:
            resp.raise_for_status()
            # 同上：按 UTF-8 解码，避免中文乱码
            for raw in resp.iter_lines(decode_unicode=False):
                if not raw:
                    continue
                try:
                    obj = json.loads(raw.decode("utf-8", errors="replace"))
                except json.JSONDecodeError:
                    continue
                piece = (obj.get("message") or {}).get("content") or ""
                if piece:
                    parts.append(piece)
                    self._forward(on_delta, piece)
        return "".join(parts)


# ============================================================================
# 文本向量化
# ============================================================================


class LocalBertEmbedder:
    """本地 transformers 模型向量化（均值池化 + L2 归一化）。

    用于 EMBEDDING_PROVIDER=local：完全离线、无需密钥。
    默认适配本机已下载的 bert-base-chinese（hidden_size=768）。
    输出经 L2 归一化，故 Milvus 用内积（IP）即等价于余弦相似度。
    """

    _instances: Dict[str, "LocalBertEmbedder"] = {}

    def __init__(self, model_path: str, batch_size: Optional[int] = None):
        self.model_path = model_path
        self.batch_size = batch_size or config.EMBEDDING_BATCH_SIZE
        self._tok = None
        self._model = None
        self._torch = None

    @classmethod
    def get(cls, model_path: str) -> "LocalBertEmbedder":
        """按路径缓存实例，避免重复加载模型。"""
        if model_path not in cls._instances:
            cls._instances[model_path] = cls(model_path)
        return cls._instances[model_path]

    def load(self) -> None:
        if self._model is not None:
            return
        if not os.path.isdir(self.model_path):
            raise LLMNotConfiguredError(
                f"本地向量模型目录不存在：{self.model_path}\n"
                "请在 .env 中把 EMBEDDING_LOCAL_PATH 指向已下载的 transformers 模型目录。"
            )
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as e:
            raise LLMNotConfiguredError(
                f"本地向量化需要 transformers 与 torch：{e}\n"
                "安装：pip install transformers torch"
            ) from e
        logger.info("正在加载本地向量模型：%s", self.model_path)
        self._tok = AutoTokenizer.from_pretrained(self.model_path)
        self._model = AutoModel.from_pretrained(self.model_path)
        self._model.eval()
        self._torch = torch
        hidden = int(getattr(self._model.config, "hidden_size", 0))
        logger.info("本地向量模型加载完成，hidden_size=%d", hidden)
        if hidden and config.EMBEDDING_DIM and hidden != config.EMBEDDING_DIM:
            raise LLMNotConfiguredError(
                f"模型 hidden_size={hidden} 与 .env 中 EMBEDDING_DIM={config.EMBEDDING_DIM} 不一致。\n"
                f"请把 EMBEDDING_DIM 改为 {hidden}，并重建向量集合"
                "（python document_processor.py --ingest 会自动重建）。"
            )

    @property
    def hidden_size(self) -> int:
        self.load()
        return int(getattr(self._model.config, "hidden_size", config.EMBEDDING_DIM))

    def embed(self, texts: List[str]) -> List[List[float]]:
        self.load()
        torch = self._torch
        out: List[List[float]] = []
        with torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i:i + self.batch_size]
                enc = self._tok(
                    batch, padding=True, truncation=True,
                    max_length=512, return_tensors="pt",
                )
                hidden = self._model(**enc).last_hidden_state          # [B, T, H]
                mask = enc["attention_mask"].unsqueeze(-1).float()      # [B, T, 1]
                summed = (hidden * mask).sum(dim=1)                    # 均值池化（排除 padding）
                counts = mask.sum(dim=1).clamp(min=1e-9)
                vecs = torch.nn.functional.normalize(summed / counts, p=2, dim=1)
                out.extend(v.tolist() for v in vecs)
        return out


class EmbeddingClient:
    def __init__(
        self,
        provider: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ):
        self.provider = (provider or config.EMBEDDING_PROVIDER).lower()
        self._local: Optional[LocalBertEmbedder] = None
        if self.provider == "ollama":
            self.base_url = (base_url or config.OLLAMA_BASE_URL).rstrip("/")
            self.model = model or config.OLLAMA_EMBEDDING_MODEL
            self.api_key = ""
        elif self.provider == "local":
            self.model = model or config.EMBEDDING_LOCAL_PATH
            self._local = LocalBertEmbedder.get(self.model)
            self.base_url = ""
            self.api_key = ""
        else:
            self.base_url = (base_url or config.EMBEDDING_BASE_URL).rstrip("/")
            self.model = model or config.EMBEDDING_MODEL
            self.api_key = api_key if api_key is not None else config.EMBEDDING_API_KEY

    @property
    def dim(self) -> int:
        if self.provider == "local" and self._local is not None:
            return self._local.hidden_size
        return config.EMBEDDING_DIM

    def embed(self, texts: List[str]) -> List[List[float]]:
        """批量向量化。"""
        if not texts:
            return []
        if self.provider == "ollama":
            return self._embed_ollama(texts)
        if self.provider == "local":
            assert self._local is not None
            return self._local.embed(texts)
        return self._embed_openai(texts)

    def embed_one(self, text: str) -> List[float]:
        return self.embed([text])[0]

    def _embed_openai(self, texts: List[str]) -> List[List[float]]:
        _require_key(self.api_key, "EMBEDDING_API_KEY", self.provider)
        url = f"{self.base_url}/embeddings"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {"model": self.model, "input": texts}
        data = _post(url, headers, payload, config.LLM_TIMEOUT)
        items = sorted(data.get("data", []), key=lambda d: d.get("index", 0))
        return [it["embedding"] for it in items]

    def _embed_ollama(self, texts: List[str]) -> List[List[float]]:
        url = f"{self.base_url}/api/embed"
        try:
            data = _post(url, {}, {"model": self.model, "input": texts}, config.LLM_TIMEOUT)
            if "embeddings" in data:
                return data["embeddings"]
        except requests.HTTPError:
            # 兼容旧版 Ollama：逐条调用 /api/embeddings
            logger.info("Ollama /api/embed 不可用，回退到 /api/embeddings")
        out: List[List[float]] = []
        for t in texts:
            data = _post(
                f"{self.base_url}/api/embeddings",
                {}, {"model": self.model, "prompt": t}, config.LLM_TIMEOUT,
            )
            out.append(data.get("embedding", []))
        return out


# ============================================================================
# 工具
# ============================================================================


def parse_json_block(text: str) -> Any:
    """从模型输出中提取 JSON（容忍 ```json 代码块与前后说明文字）。"""
    if text is None:
        return None
    s = text.strip()
    if s.startswith("```"):
        s = s.split("```")[1] if len(s.split("```")) > 1 else s
        if s.lstrip().lower().startswith("json"):
            s = s.lstrip()[4:]
    s = s.strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    # 退而求其次：截取首个 { 或 [ 到最后一个 } 或 ]
    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        i, j = s.find(open_ch), s.rfind(close_ch)
        if i != -1 and j > i:
            try:
                return json.loads(s[i:j + 1])
            except json.JSONDecodeError:
                continue
    logger.warning("未能从模型输出中解析 JSON：%s", text[:200])
    return None


# 角色标识 → 中文名。中文名必须与 multi_agents 中各 Agent 的 name 属性一致
# （tests/test_llm_roles.py 有一致性断言，防止两处走偏）；顺序即前端展示顺序。
ROLE_SPECS: List[Tuple[str, str]] = [
    ("retriever", "检索员"),
    ("analyst", "分析师"),
    ("writer", "撰稿人"),
    ("reviewer", "审核员"),
]

_ROLE_MODEL_KEYS = {
    "retriever": "LLM_MODEL_RETRIEVER",
    "analyst": "LLM_MODEL_ANALYST",
    "writer": "LLM_MODEL_WRITER",
    "reviewer": "LLM_MODEL_REVIEWER",
}

# 按角色缓存客户端（键为角色标识；None 为全局默认项）。首次并发会重复构造，无害。
_llm_cache: Dict[Optional[str], LLMClient] = {}
_embedding_singleton: Optional[EmbeddingClient] = None


def role_model_override(role: Optional[str]) -> str:
    """该角色配置的模型名覆盖值；未配置（或角色未知）返回空串。"""
    key = _ROLE_MODEL_KEYS.get(role or "")
    return (getattr(config, key, "") or "").strip() if key else ""


def default_model_name() -> str:
    """全局默认模型名，按 provider 解析（openai 取 LLM_MODEL，ollama 取 OLLAMA_MODEL）。"""
    return config.LLM_MODEL if config.LLM_PROVIDER == "openai" else config.OLLAMA_MODEL


def effective_model(role: Optional[str]) -> str:
    """该角色最终生效的模型名：配了覆盖就用覆盖，否则回退全局默认。"""
    return role_model_override(role) or default_model_name()


def get_llm(role: Optional[str] = None) -> LLMClient:
    """取大模型客户端。

    role 为空（默认）时返回全局默认客户端，行为与引入角色前完全一致。
    指定角色且该角色配了模型名覆盖时，返回用该模型构造的客户端；
    没配覆盖则与默认客户端等价（复用同一实例，不重复构造）。
    """
    if role not in (None, "") and role_model_override(role) == "":
        role = None
    if role not in _llm_cache:
        override = role_model_override(role)
        _llm_cache[role] = LLMClient(model=override) if override else LLMClient()
    return _llm_cache[role]


def reset_llm_cache() -> None:
    """清空客户端缓存（含默认项）。改过 config.LLM_MODEL* 之后必须调用。"""
    _llm_cache.clear()


def role_models() -> List[Dict[str, Any]]:
    """角色 → 生效模型名，供接口层 /health 与前端展示。

    只读配置、**不构造客户端**（避免副作用，也便于被流式步骤事件高频调用）。
    """
    return [
        {
            "role": role,
            "label": label,
            "model": effective_model(role),
            "overridden": role_model_override(role) != "",
        }
        for role, label in ROLE_SPECS
    ]


def get_embedding() -> EmbeddingClient:
    global _embedding_singleton
    if _embedding_singleton is None:
        _embedding_singleton = EmbeddingClient()
    return _embedding_singleton


def is_configured() -> Dict[str, Any]:
    """返回各接入项是否已就绪，供接口层 /health 展示与启动自检使用。"""
    llm_ok = bool(config.LLM_API_KEY) if config.LLM_PROVIDER == "openai" else True
    llm_model = default_model_name()

    if config.EMBEDDING_PROVIDER == "openai":
        emb_ok = bool(config.EMBEDDING_API_KEY)
        emb_model = config.EMBEDDING_MODEL
    elif config.EMBEDDING_PROVIDER == "local":
        emb_ok = bool(config.EMBEDDING_LOCAL_PATH) and os.path.isdir(
            config.EMBEDDING_LOCAL_PATH
        )
        emb_model = config.EMBEDDING_LOCAL_PATH or "(未配置 EMBEDDING_LOCAL_PATH)"
    else:
        emb_ok = True
        emb_model = config.OLLAMA_EMBEDDING_MODEL

    return {
        "llm_provider": config.LLM_PROVIDER,
        "llm_model": llm_model,
        "llm_ready": llm_ok,
        "embedding_provider": config.EMBEDDING_PROVIDER,
        "embedding_model": emb_model,
        "embedding_dim": config.EMBEDDING_DIM,
        "embedding_ready": emb_ok,
    }


def selftest() -> int:
    """连通性自检：填好密钥后执行 python llm_client.py --selftest。"""
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    state = is_configured()
    print("接入配置：")
    for k, v in state.items():
        print(f"  {k:<20}{v}")
    rc = 0

    print("\n[1/2] 对话模型 …", end=" ")
    try:
        llm = get_llm()
        out = llm.complete("用一句话说明什么是供热系统的日气耗指标。")
        print("OK")
        print("     模型：", llm.model_name())
        print("     回答：", out[:120].replace("\n", " "))
    except LLMNotConfiguredError as e:
        print("未配置")
        print("     " + str(e).replace("\n", "\n     "))
        rc = 1
    except Exception as e:  # noqa: BLE001
        print("失败：", e)
        rc = 1

    print("\n[2/2] 向量模型 …", end=" ")
    try:
        emb = get_embedding()
        vec = emb.embed_one("供热运营数据查询")
        print("OK")
        print(f"     维度：{len(vec)}（配置期望 {emb.dim}）")
        if emb.dim and len(vec) != emb.dim:
            print("     提示：实际维度与 EMBEDDING_DIM 不一致，请同步修改 .env")
    except LLMNotConfiguredError as e:
        print("未配置")
        print("     " + str(e).replace("\n", "\n     "))
        rc = 1
    except Exception as e:  # noqa: BLE001
        print("失败：", e)
        rc = 1

    return rc


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="模型接入层自检")
    ap.add_argument("--selftest", action="store_true", help="测试对话与向量模型连通性")
    args = ap.parse_args()
    if args.selftest:
        raise SystemExit(selftest())
    ap.print_help()
