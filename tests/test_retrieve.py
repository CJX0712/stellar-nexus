"""检索层单测。核心不变量：稀疏/稠密各自命中，融合后正例仍在前列。作者: 晨星"""

from __future__ import annotations

from stellarnx.core.types import Hit
from stellarnx.embed import HashEmbedder
from stellarnx.retrieve import BM25Retriever, rrf_fuse
from stellarnx.retrieve.hybrid import mmr_select


def test_bm25_ranks_relevant_first(sample_chunks):
    retriever = BM25Retriever()
    retriever.index(sample_chunks)
    hits = retriever.search("向量数据库 相似度检索", 3)
    assert hits, "BM25 必须召回至少一条"
    assert hits[0].chunk_id == "d1:0"
    assert hits[0].source == "bm25"


def test_bm25_returns_empty_for_unknown_terms(sample_chunks):
    retriever = BM25Retriever()
    retriever.index(sample_chunks)
    # 查询词与语料无任何共用字，BM25 必须零召回
    assert retriever.search("珊瑚礁生态系统", 3) == []


def test_rrf_boosts_item_present_in_both_lists(sample_hits):
    # b 在两路分别排第 2 与第 1，a 只在第一路第 1，c 只在第二路第 2
    list_a = [sample_hits[0], sample_hits[1]]
    list_b = [sample_hits[1], sample_hits[2]]
    fused = rrf_fuse([list_a, list_b], k=60)
    assert fused[0].chunk_id == sample_hits[1].chunk_id, "两路都出现的片段应排第一"


def test_rrf_marks_source_as_hybrid(sample_hits):
    assert all(h.source == "hybrid" for h in rrf_fuse([sample_hits]))


def test_mmr_avoids_near_duplicates():
    embedder = HashEmbedder()
    candidates = [
        Hit(chunk_id=f"c{i}", doc_id="d", text=text, score=1.0)
        for i, text in enumerate([
            "苹果是一种水果，富含维生素。",
            "苹果是一种水果，富含维生素和纤维。",
            "香蕉是热带水果，钾含量高。",
        ])
    ]
    picked = mmr_select(embedder.embed(["水果"])[0], candidates, embedder, 2, 0.2)
    assert {p.chunk_id for p in picked} != {"c0", "c1"}


def test_dense_retriever_self_hit(system):
    system.ingest_text("doc", "星枢系统的嵌入模块把文本变成向量。" * 20)
    hits = system.dense.search("嵌入模块", 3)
    assert hits and hits[0].doc_id == "doc"
