"""Ollama 嵌入后端（默认 bge-m3，1024 维，多语言）。

只做 HTTP 适配，不做任何模型相关逻辑 —— 模型能力全部复用 Ollama。

作者: 晨星
"""

from __future__ import annotations

import logging

import httpx
import numpy as np

from stellarnx.core.config import Config
from stellarnx.core.errors import DependencyUnavailable, UpstreamTimeout
from stellarnx.core.linalg import normalize

logger = logging.getLogger(__name__)


class OllamaEmbedder:
    """满足 Embedder 协议的 Ollama 实现。"""

    def __init__(self, cfg: Config, model: str | None = None, host: str | None = None) -> None:
        self.cfg = cfg
        self.model = model or cfg.embed_model
        self.host = (host or cfg.ollama_host).rstrip("/")
        # trust_env=False：本机有 SOCKS5 代理，默认信任环境变量会把 127.0.0.1 请求
        # 也送进代理，代理在远端连 localhost 必然失败（502 / 10054）。
        self._client = httpx.Client(timeout=cfg.ollama_timeout_s, trust_env=False)
        self._dim: int | None = None

    @property
    def name(self) -> str:
        return f"ollama:{self.model}"

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._dim = len(self.embed(["探测"])[0])
        return self._dim

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1), dtype="float32")
        vectors: list[list[float]] = []
        for text in texts:
            vectors.append(self._embed_single(text))
        width = max(len(v) for v in vectors)
        matrix = np.zeros((len(vectors), width), dtype="float64")
        for row, vec in enumerate(vectors):
            matrix[row, : len(vec)] = vec
        return normalize(matrix)

    def _embed_single(self, text: str) -> list[float]:
        try:
            resp = self._client.post(
                f"{self.host}/api/embed",
                json={"model": self.model, "input": text},
            )
        except httpx.TimeoutException as exc:
            raise UpstreamTimeout(f"Ollama 嵌入超时: {self.model}") from exc
        except httpx.HTTPError as exc:
            raise DependencyUnavailable(f"无法连接 Ollama: {exc}") from exc
        if resp.status_code != 200:
            raise DependencyUnavailable(
                f"Ollama 嵌入返回 {resp.status_code}: {resp.text[:200]}")
        payload = resp.json()
        embeddings = payload.get("embeddings")
        if not embeddings:
            raise DependencyUnavailable(f"Ollama 嵌入返回空: {self.model}")
        return list(embeddings[0])

    def is_available(self) -> bool:
        try:
            self.embed(["探针"])
            return True
        except Exception:
            return False
