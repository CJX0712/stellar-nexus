"""零依赖哈希嵌入：离线与 CI 的支点。

设计要点：
1. 跨进程确定性 —— 用 blake2b 派生哈希，绝不用 Python 内置 hash()（按进程加盐）。
2. 中英混排友好 —— CJK 走单字 + 相邻二字组，ASCII 走词，因此中文短文本也有语义重叠信号。
3. 带符号 hashing trick —— 随机符号抵消桶碰撞带来的系统性偏差。

作者: 晨星
"""

from __future__ import annotations

import hashlib
from collections import Counter

import numpy as np

from stellarnx.core.linalg import normalize
from stellarnx.core.text import tokenize

_DEFAULT_DIM = 512

__all__ = ["HashEmbedder", "tokenize"]


def _bucket(token: str, dim: int) -> tuple[int, int]:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    value = int.from_bytes(digest, "big")
    return value % dim, (value >> 17) & 1


class HashEmbedder:
    """满足 Embedder 协议的零依赖实现。"""

    def __init__(self, dim: int = _DEFAULT_DIM) -> None:
        self.name = "hash"
        self.dim = dim

    def embed_one(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype="float64")
        counts = Counter(tokenize(text))
        for token, count in counts.items():
            idx, sign = _bucket(token, self.dim)
            vec[idx] += (1.0 if sign else -1.0) * float(count)
        return vec

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype="float32")
        matrix = np.vstack([self.embed_one(t) for t in texts])
        return normalize(matrix)
