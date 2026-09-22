"""存储层：文档与向量的持久化。

DocumentStore 存文本（sqlite，标准库），VectorIndex 存向量（FAISS / numpy 回退）。
两者通过 chunk_id 关联，互不侵入。

作者: 晨星
"""

from stellarnx.store.document_store import DocumentStore
from stellarnx.store.vector_index import InMemoryIndex, build_index

__all__ = ["DocumentStore", "InMemoryIndex", "build_index"]
