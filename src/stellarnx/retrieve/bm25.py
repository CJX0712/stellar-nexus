"""BM25 稀疏检索。零依赖自实现（不引入 rank-bm25），与稠密路共享切词器。

作者: 晨星
"""

from __future__ import annotations

import math
from collections import Counter

from stellarnx.core.text import tokenize
from stellarnx.core.types import Chunk, Hit


class BM25Retriever:
    """经典 BM25。唯一职责：基于词项统计做稀疏召回。"""

    name = "bm25"

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._chunks: list[Chunk] = []
        self._tf: list[Counter[str]] = []
        self._df: Counter[str] = Counter()
        self._lengths: list[int] = []
        self._avg_len = 0.0

    def index(self, chunks: list[Chunk]) -> None:
        """重建索引。片段顺序决定 chunk_id 顺序，与稠密路保持一致。"""
        self._chunks = list(chunks)
        self._tf = [Counter(tokenize(c.text)) for c in self._chunks]
        self._lengths = [sum(tf.values()) for tf in self._tf]
        self._df = Counter()
        for tf in self._tf:
            self._df.update(tf.keys())
        total = sum(self._lengths)
        self._avg_len = total / len(self._lengths) if self._lengths else 0.0

    @property
    def size(self) -> int:
        return len(self._chunks)

    def all_ids(self) -> list[str]:
        """当前索引覆盖的片段 id。用于「索引是否与片段库一致」的自检。"""
        return [c.id for c in self._chunks]

    def scores(self, query: str) -> list[float]:
        """返回与 _chunks 等长的分数数组。"""
        n = len(self._chunks)
        if n == 0:
            return []
        query_tokens = Counter(tokenize(query))
        out = [0.0] * n
        for token, q_count in query_tokens.items():
            df = self._df.get(token, 0)
            if df == 0:
                continue
            idf = math.log(1.0 + (n - df + 0.5) / (df + 0.5))
            for i in range(n):
                freq = self._tf[i].get(token, 0)
                if freq == 0:
                    continue
                length = self._lengths[i]
                denom = freq + self.k1 * (
                    1.0 - self.b + self.b * (length / self._avg_len if self._avg_len else 1.0))
                out[i] += idf * (freq * (self.k1 + 1.0)) / denom * q_count
        return out

    def search(self, query: str, k: int) -> list[Hit]:
        scores = self.scores(query)
        order = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        return [
            Hit(
                chunk_id=self._chunks[i].id,
                doc_id=self._chunks[i].doc_id,
                text=self._chunks[i].text,
                score=float(scores[i]),
                source="bm25",
                meta=dict(self._chunks[i].meta),
            )
            for i in order
            if scores[i] > 0.0
        ]
