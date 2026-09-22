"""混合检索融合：RRF 多路合并 + MMR 去冗余。

RRF 对分数尺度不敏感（稀疏 BM25 与稠密余弦量纲不同），比加权求和稳；
MMR 保证进入上下文的片段彼此不重复，直接提升答案的信息密度。

作者: 晨星
"""

from __future__ import annotations

import numpy as np

from stellarnx.core.contracts import Embedder, Retriever
from stellarnx.core.types import Hit


def rrf_fuse(ranked_lists: list[list[Hit]], k: int = 60) -> list[Hit]:
    """倒数排名融合。同一片段在多路出现会累积得分。"""
    fused: dict[str, float] = {}
    payload: dict[str, Hit] = {}
    for hits in ranked_lists:
        for rank, hit in enumerate(hits):
            fused[hit.chunk_id] = fused.get(hit.chunk_id, 0.0) + 1.0 / (k + rank + 1)
            if hit.chunk_id not in payload:
                payload[hit.chunk_id] = hit
    ordered = sorted(fused.items(), key=lambda kv: -kv[1])
    return [
        Hit(
            chunk_id=cid,
            doc_id=payload[cid].doc_id,
            text=payload[cid].text,
            score=score,
            source="hybrid",
            meta=dict(payload[cid].meta),
        )
        for cid, score in ordered
    ]


def mmr_select(
    query_vec: np.ndarray,
    candidates: list[Hit],
    embedder: Embedder,
    k: int,
    lambda_: float = 0.7,
) -> list[Hit]:
    """最大边际相关性选择：相关性与多样性折中。"""
    if not candidates or k <= 0:
        return []
    if len(candidates) <= k:
        return list(candidates)

    matrix = embedder.embed([c.text for c in candidates])
    query_vec = np.asarray(query_vec, dtype="float32").reshape(-1)
    if query_vec.shape[0] != matrix.shape[1]:
        return candidates[:k]
    relevance = matrix @ query_vec

    selected: list[int] = []
    remaining = list(range(len(candidates)))
    while remaining and len(selected) < k:
        if not selected:
            best = int(np.argmax(relevance[remaining]))
            selected.append(remaining.pop(best))
            continue
        selected_matrix = matrix[selected]
        diversity = matrix[remaining] @ selected_matrix.T
        max_sim = diversity.max(axis=1)
        mmr_scores = lambda_ * relevance[remaining] - (1.0 - lambda_) * max_sim
        best = int(np.argmax(mmr_scores))
        selected.append(remaining.pop(best))
    return [candidates[i] for i in selected]


class HybridRetriever:
    """融合检索器。唯一职责：调度子检索器并合并结果。"""

    name = "hybrid"

    def __init__(
        self,
        retrievers: list[Retriever],
        embedder: Embedder | None = None,
        rrf_k: int = 60,
        mmr_lambda: float = 0.7,
        use_mmr: bool = True,
    ) -> None:
        self.retrievers = retrievers
        self.embedder = embedder
        self.rrf_k = rrf_k
        self.mmr_lambda = mmr_lambda
        self.use_mmr = use_mmr

    def search(self, query: str, k: int) -> list[Hit]:
        ranked = [r.search(query, k * 2) for r in self.retrievers]
        fused = rrf_fuse(ranked, self.rrf_k)
        if self.use_mmr and self.embedder is not None and len(fused) > k:
            query_vec = self.embedder.embed([query])[0]
            return mmr_select(query_vec, fused, self.embedder, k, self.mmr_lambda)
        return fused[:k]
