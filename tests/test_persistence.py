"""持久化往返测试。

覆盖的是一类特别阴的故障：文档落到了 sqlite，向量索引和 BM25 却只在内存里。
进程重启后「库里有数据、检索全落空」。因此这里不看「跑没跑过 restore」，
只看重启后**检索结果是否与重启前逐位一致**。

作者: 晨星
"""

from __future__ import annotations

import json

import pytest

from stellarnx.core.config import Config
from stellarnx.pipeline.orchestrator import NexusSystem, build_system

CORPUS = [
    {"doc_id": "a", "text": "向量索引使用 FAISS 实现，同时保留 numpy 回退实现。" * 3},
    {"doc_id": "b", "text": "检索阶段由 BM25 稀疏召回与稠密向量召回两路并行组成。" * 3},
]


@pytest.fixture()
def cfg(tmp_path):
    return Config(
        embed_backend="hash",
        llm_backend="mock",
        rerank_backend="lexical",
        data_dir=str(tmp_path),
    )


def test_restore_makes_persisted_chunks_searchable(cfg, tmp_path):
    db = tmp_path / "stellarnx.db"

    first = NexusSystem(cfg, str(db))
    first.ingest_many(CORPUS)
    before = [h.chunk_id for h in first.search("向量索引 FAISS", 5)]
    assert before, "首次入库后必须能检索到"
    assert first.index_is_consistent()
    first.store.close()

    # 模拟进程重启：全新对象，只靠磁盘上的库
    second = build_system(cfg, str(db))
    assert second.store.count_chunks() == first.index.count() or second.store.count_chunks() > 0
    assert second.index_is_consistent(), "重启后索引必须与片段库一致"

    after = [h.chunk_id for h in second.search("向量索引 FAISS", 5)]
    assert after == before, "重启前后检索结果必须逐位一致"
    second.store.close()


def test_build_system_restores_from_default_data_dir(cfg, tmp_path):
    """store_path 省略时按 cfg.data_dir 落库，并自动恢复。"""
    built = build_system(cfg)
    built.ingest_many(CORPUS)
    assert (tmp_path / "stellarnx.db").exists(), "应按 data_dir 建库"
    built.store.close()

    again = build_system(cfg)
    assert again.store.count_chunks() > 0
    assert again.index_is_consistent()
    hits = again.search("BM25 稀疏召回", 5)
    assert hits and hits[0].doc_id == "b"
    again.store.close()


def test_restore_on_empty_store_is_noop(cfg, tmp_path):
    system = NexusSystem(cfg, str(tmp_path / "empty.db"))
    assert system.restore() == 0
    assert system.index.count() == 0
    assert system.bm25.size == 0
    assert system.index_is_consistent()
    system.store.close()


def test_deleting_document_keeps_index_consistent(cfg, tmp_path):
    system = NexusSystem(cfg, str(tmp_path / "del.db"))
    system.ingest_many(CORPUS)
    system.store.delete_document("a")
    system.restore()
    assert system.index_is_consistent()
    assert all(h.doc_id != "a" for h in system.search("向量索引", 5))
    system.store.close()


def test_reingest_same_doc_is_idempotent(cfg, tmp_path):
    """同 doc_id 重复入库必须幂等：既不留重复向量，也不留陈旧片段。

    这是实测抓到的缺陷：`add_chunks` 是 INSERT OR REPLACE，旧片段被替换；
    但向量是 `index.add` 追加，同一个 chunk_id 在索引里变成了两份 ——
    现象就是 `chunks=16` 而 `vectors=32`。
    """
    system = NexusSystem(cfg, str(tmp_path / "idem.db"))
    # 必须超过 chunk_size（默认 480），否则只产生 1 个片段，多片段路径测不到
    long_text = "星枢系统的向量索引基于 FAISS，并保留 numpy 回退实现。" * 30
    system.ingest_many([{"doc_id": "d1", "text": long_text}])
    first_chunks = system.store.count_chunks()
    first_vectors = system.index.count()
    assert first_chunks == first_vectors and first_chunks > 1

    system.ingest_many([{"doc_id": "d1", "text": long_text}])
    assert system.store.count_chunks() == first_chunks, "重复入库不得产生新片段"
    assert system.index.count() == first_vectors, "重复入库不得产生重复向量"
    assert system.index_is_consistent()
    system.store.close()


def test_shorter_revision_removes_stale_chunks(cfg, tmp_path):
    """新版本比旧版本短时，超出的旧片段必须被清掉。

    否则「文档里已经删掉的内容仍能被搜出来」—— 这类问题在检索结果里
    看起来完全正常，只有对着原文才发现多了一段。
    """
    system = NexusSystem(cfg, str(tmp_path / "rev.db"))
    long_text = "旧版本内容：星枢使用 FAISS 向量索引。" * 30
    short_text = "新版本内容：星枢改用 numpy 内积索引。"
    system.ingest_many([{"doc_id": "d1", "text": long_text}])
    assert system.store.count_chunks() > 1

    system.ingest_many([{"doc_id": "d1", "text": short_text}])
    remaining = system.store.chunks_of_doc("d1")
    assert len(remaining) == 1, f"旧片段未清干净，剩余 {len(remaining)} 条"
    assert system.index.count() == 1
    assert system.index_is_consistent()
    assert all("旧版本内容" not in h.text for h in system.search("FAISS 向量索引", 5))
    system.store.close()


def test_duplicate_doc_ids_in_one_batch_keep_last(cfg, tmp_path):
    """同一批里出现重复 doc_id，只保留最后一条。"""
    system = NexusSystem(cfg, str(tmp_path / "batch.db"))
    system.ingest_many([
        {"doc_id": "d1", "text": "第一版内容，关于苹果。" * 20},
        {"doc_id": "d1", "text": "第二版内容，关于香蕉。" * 20},
    ])
    assert system.store.count_documents() == 1
    assert system.index_is_consistent()
    assert all("苹果" not in h.text for h in system.search("苹果 香蕉", 10))
    system.store.close()


def test_document_listing_counts_chunks(cfg, tmp_path):
    system = NexusSystem(cfg, str(tmp_path / "list.db"))
    system.ingest_many(CORPUS)
    docs = {d["doc_id"]: d for d in system.store.list_documents()}
    assert docs["a"]["chunks"] == len(system.store.chunks_of_doc("a"))
    assert docs["a"]["chunks"] > 0
    system.store.close()


def test_pipeline_answer_survives_restart(cfg, tmp_path):
    """端到端：重启后仍能给出带引用的答案，而不是空手而回。"""
    db = str(tmp_path / "e2e.db")
    first = NexusSystem(cfg, db)
    first.ingest_many(CORPUS)
    first.store.close()

    second = build_system(cfg, db)
    answer = second.answer("向量索引用什么实现？")
    assert answer.text.strip()
    assert answer.citations, "重启后答案仍必须带引用"
    payload = json.loads(json.dumps(answer.to_dict(), ensure_ascii=False))
    assert payload["citations"][0]["chunk_id"]
    assert "trace" in payload["metadata"]
    second.store.close()
