"""稠密检索：嵌入查询 -> 向量索引 -> 回表取文本。

作者: 晨星
"""

from __future__ import annotations

from stellarnx.core.contracts import Embedder, VectorIndex
from stellarnx.core.types import Hit
from stellarnx.store.document_store import DocumentStore


class DenseRetriever:
    """向量召回。唯一职责：把 query 变成一个向量并取回最相似片段。"""

    name = "dense"

    def __init__(self, embedder: Embedder, index: VectorIndex, store: DocumentStore) -> None:
        self.embedder = embedder
        self.index = index
        self.store = store

    def search(self, query: str, k: int) -> list[Hit]:
        if self.index.count() == 0:
            return []
        vector = self.embedder.embed([query])
        pairs = self.index.search(vector, k)[0]
        hits: list[Hit] = []
        for chunk_id, score in pairs:
            chunk = self.store.get_chunk(chunk_id)
            if chunk is None:
                continue
            hits.append(
                Hit(
                    chunk_id=chunk.id,
                    doc_id=chunk.doc_id,
                    text=chunk.text,
                    score=float(score),
                    source="dense",
                    meta=dict(chunk.meta),
                )
            )
        return hits
