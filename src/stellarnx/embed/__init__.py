"""嵌入层：文本 -> 向量。

对外只暴露 build_embedder()，返回满足 Embedder 协议的对象。
hash 实现零依赖且跨进程确定性（用 blake2b，不用 Python 内置 hash —— 后者按进程加盐）。

作者: 晨星
"""

from __future__ import annotations

from stellarnx.core.config import Config
from stellarnx.core.contracts import Embedder
from stellarnx.core.errors import DependencyUnavailable
from stellarnx.core.linalg import normalize
from stellarnx.embed.hash_embed import HashEmbedder
from stellarnx.embed.ollama_embed import OllamaEmbedder

__all__ = ["Embedder", "HashEmbedder", "OllamaEmbedder", "build_embedder", "normalize"]


def build_embedder(cfg: Config) -> Embedder:
    """按配置构造嵌入器。真实后端不可用时自动降级到 hash，不抛异常。"""
    backend = (cfg.embed_backend or "hash").lower()
    if backend == "hash":
        return HashEmbedder()
    if backend == "ollama":
        try:
            emb = OllamaEmbedder(cfg)
            emb.embed(["连通性探针"])
            return emb
        except Exception as exc:
            import logging

            logging.getLogger(__name__).warning(
                "Ollama 嵌入不可用，降级到 hash 嵌入: %s", exc)
            return HashEmbedder()
    raise DependencyUnavailable(f"未知嵌入后端: {backend}")
