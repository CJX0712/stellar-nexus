"""零依赖重排实现。模型不可用时的保底方案，也是离线单测的默认实现。

作者: 晨星
"""

from __future__ import annotations

from stellarnx.core.text import tokenize
from stellarnx.core.types import Hit


class IdentityReranker:
    """不做重排，只截断。用于关闭重排的对照实验。"""

    name = "identity"

    def rerank(self, query: str, hits: list[Hit], k: int) -> list[Hit]:
        return sorted(hits, key=lambda h: -h.score)[:k]


class LexicalReranker:
    """词汇覆盖重排：查询词在片段中的覆盖率越高越靠前。

    比 Identity 强，且完全零依赖、零模型，是 ONNX 不可用时的真实降级档。
    用 IDF 加权，避免"的/了/是"这类高频词主导排序。
    """

    name = "lexical"

    def __init__(self) -> None:
        self._df: dict[str, int] = {}
        self._n = 0

    def fit(self, corpus: list[str]) -> LexicalReranker:
        self._n = len(corpus)
        counts: dict[str, int] = {}
        for text in corpus:
            for token in set(tokenize(text)):
                counts[token] = counts.get(token, 0) + 1
        self._df = counts
        return self

    def _idf(self, token: str) -> float:
        df = self._df.get(token, 0)
        if df == 0:
            return 1.0
        return 1.0 + (self._n / (1.0 + df)) ** 0.5

    def rerank(self, query: str, hits: list[Hit], k: int) -> list[Hit]:
        if not hits:
            return []
        query_tokens = set(tokenize(query))
        scored: list[Hit] = []
        for hit in hits:
            doc_tokens = set(tokenize(hit.text))
            overlap = query_tokens & doc_tokens
            coverage = sum(self._idf(t) for t in overlap) / max(
                1.0, sum(self._idf(t) for t in query_tokens))
            # 位置先验：查询词出现在片段开头略加权
            head = hit.text[: max(60, len(hit.text) // 3)]
            head_bonus = 0.05 if (query_tokens & set(tokenize(head))) else 0.0
            new_score = hit.score * 0.35 + coverage * 0.65 + head_bonus
            scored.append(
                Hit(
                    chunk_id=hit.chunk_id,
                    doc_id=hit.doc_id,
                    text=hit.text,
                    score=float(new_score),
                    source=f"{hit.source}+lexical",
                    meta=dict(hit.meta),
                )
            )
        return sorted(scored, key=lambda h: -h.score)[:k]
