"""编排层单测。核心不变量：端到端产出带引用答案，路由正确，trace 完整。作者: 晨星"""

from __future__ import annotations

from stellarnx.core.types import RoutePath

CORPUS = (
    "向量数据库用于保存嵌入向量。星枢系统使用 FAISS 作为向量索引，"
    "同时保留 numpy 回退实现。检索阶段采用 BM25 与稠密向量双路召回。"
) * 20


def test_ingest_returns_chunk_count(system):
    count = system.ingest_text("handbook", CORPUS)
    assert count > 1
    assert system.store.count_chunks() == count
    assert system.index.count() == count


def test_answer_has_text_and_citations(system):
    system.ingest_text("handbook", CORPUS)
    answer = system.answer("星枢系统使用什么作为向量索引？")
    assert answer.text
    assert answer.citations
    assert all(c.chunk_id for c in answer.citations)


def test_answer_route_is_rag(system):
    system.ingest_text("handbook", CORPUS)
    assert system.answer("星枢系统使用什么作为向量索引？").route.path is RoutePath.RAG


def test_direct_path_skips_verification(system):
    answer = system.answer("你好")
    assert answer.route.path is RoutePath.DIRECT
    assert answer.verification is None


def test_trace_contains_expected_spans(system):
    system.ingest_text("handbook", CORPUS)
    answer = system.answer("星枢系统使用什么作为向量索引？")
    children = system.answer("再问一次").metadata["trace"].get("children", [])
    names = {c["name"] for c in children}
    assert "route" in names
    assert all(c["duration_ms"] >= 0 for c in children)
    assert answer.trace_id


def test_search_returns_hits_sorted(system):
    system.ingest_text("handbook", CORPUS)
    hits = system.search("向量索引", 5)
    assert hits
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True) or len(hits) == 1


def test_clear_resets_state(system):
    system.ingest_text("handbook", CORPUS)
    system.clear()
    assert system.store.count_chunks() == 0
    assert system.index.count() == 0
    assert system.search("任意", 5) == []


def test_stats_reports_backends(system):
    stats = system.stats()
    assert stats["vector_backend"] == "faiss"
    assert stats["llm"] == "mock"
    assert stats["dim"] > 0
