"""API 层单测。核心不变量：进程内可拉起、各端点契约稳定、错误码正确。作者: 晨星"""

from __future__ import annotations

CORPUS = "星枢系统的检索层包含 BM25 与稠密向量两路召回，并用 RRF 融合。" * 30


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["author"] == "晨星"
    assert body["backends"]["vector"] == "faiss"


def test_ingest_then_chat(client):
    assert client.post("/ingest", json={"docs": [
        {"doc_id": "d1", "text": CORPUS}]}).json()["chunks"] > 1
    body = client.post("/chat", json={"query": "检索层包含哪两路召回？"}).json()
    assert body["text"]
    assert body["route"]["path"] in {"rag", "multihop", "agent", "direct"}
    assert "groundedness" in body["verification"]


def test_ingest_rejects_malformed_docs(client):
    assert client.post("/ingest", json={"docs": [{"doc_id": "x"}]}).status_code == 400


def test_search_endpoint(client):
    client.post("/ingest", json={"docs": [{"doc_id": "d1", "text": CORPUS}]})
    hits = client.post("/search", json={"query": "两路召回", "k": 3}).json()
    assert hits
    assert "chunk_id" in hits[0] and "score" in hits[0]


def test_documents_lifecycle(client):
    client.post("/ingest", json={"docs": [{"doc_id": "d1", "text": CORPUS}]})
    assert any(d["doc_id"] == "d1" for d in client.get("/documents").json())
    assert client.delete("/documents/d1").json()["removed_chunks"] > 0


def test_trace_roundtrip(client):
    client.post("/ingest", json={"docs": [{"doc_id": "d1", "text": CORPUS}]})
    body = client.post("/chat", json={"query": "两路召回是什么？", "with_trace": True}).json()
    trace_id = body["trace_id"]
    assert client.get(f"/trace/{trace_id}").status_code == 200
    assert client.get("/trace/does-not-exist").status_code == 404


def test_stats_endpoint(client):
    assert client.get("/stats").json()["vector_backend"] == "faiss"


def test_evaluate_endpoint(client):
    body = client.post("/evaluate").json()
    assert "metrics" in body
    assert body["metrics"]["recall@5"] >= 0.75
    assert body["mode"] == "configured"


def test_evaluate_offline_flag_pins_backends(client):
    """/evaluate?offline=true 必须锁死零依赖后端。

    这条断言守的是一个真实的可用性问题：真实模型下 8 条用例要几分钟，
    而门禁需要「快且确定」。如果 offline 开关被忽略，控制台点一次评测
    就要等好几分钟，且指标随采样波动 —— 门禁就失去意义了。
    """
    body = client.post("/evaluate?offline=true").json()
    assert body["mode"] == "offline"
    assert body["backends"] == {"embedder": "hash", "llm": "mock", "reranker": "lexical"}
    assert body["metrics"]["recall@5"] >= 0.75


def test_evaluate_missing_data_returns_404(client):
    assert client.post("/evaluate?corpus=data/nope.jsonl").status_code == 404


def test_documents_include_chunk_counts(client):
    client.post("/ingest", json={"docs": [{"doc_id": "d1", "text": CORPUS}]})
    docs = client.get("/documents").json()
    assert docs and all(isinstance(d["chunks"], int) and d["chunks"] > 0 for d in docs)


def test_root_serves_console(client):
    response = client.get("/")
    assert response.status_code == 200


def test_selftest_all_green():
    from stellarnx.selftest import run_selftest

    results = run_selftest()
    failed = [r for r in results if not r["ok"]]
    assert not failed, [f"{r['name']}: {r['detail']}" for r in failed]
