"""存储层单测。核心不变量：真实向量后端生效、删改后计数正确、片段可回表。作者: 晨星"""

from __future__ import annotations

import numpy as np

from stellarnx.core.types import Chunk
from stellarnx.store import DocumentStore, build_index


def test_build_index_prefers_faiss_and_records_fallback():
    index = build_index(8)
    assert index.name == "faiss"
    assert getattr(index, "_fallback_reason", None) is None


def test_index_add_search_delete():
    index = build_index(4)
    vectors = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], dtype="float32")
    index.add(["a", "b", "c"], vectors)
    assert index.count() == 3
    top = index.search(np.array([[1, 0, 0, 0]], dtype="float32"), 1)[0]
    assert top[0][0] == "a"
    index.delete(["a"])
    assert index.count() == 2
    assert index.search(np.array([[1, 0, 0, 0]], dtype="float32"), 2)[0][0][0] != "a"


def test_document_store_roundtrip():
    store = DocumentStore(":memory:")
    store.add_document("d1", title="标题", source="test")
    store.add_chunks([Chunk(id="d1:0", doc_id="d1", text="片段内容", ordinal=0)])
    assert store.count_documents() == 1
    assert store.count_chunks() == 1
    assert store.get_chunk("d1:0").text == "片段内容"
    assert store.delete_document("d1") == 1
    assert store.count_chunks() == 0


def test_get_chunks_preserves_order_and_skips_missing():
    store = DocumentStore(":memory:")
    store.add_chunks([
        Chunk(id="a", doc_id="d", text="A", ordinal=0),
        Chunk(id="b", doc_id="d", text="B", ordinal=1),
    ])
    assert [c.id for c in store.get_chunks(["b", "missing", "a"])] == ["b", "a"]
