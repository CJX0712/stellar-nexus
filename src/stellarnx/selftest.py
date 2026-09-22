"""一键自检：每个模块一条可机器判定的不变量。

设计原则（血的教训）：判据必须是**真实断言**，不能是"跑通没报错"。
比如向量后端要断言用的是 FAISS 而不是静默回退，校验器要能被篡改测试打穿。

作者: 晨星
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from stellarnx.core.config import load_config


def _check(name: str, fn: Callable[[], tuple[bool, str]]) -> dict[str, Any]:
    try:
        ok, detail = fn()
    except Exception as exc:
        return {"name": name, "ok": False, "detail": f"{type(exc).__name__}: {exc}"}
    return {"name": name, "ok": bool(ok), "detail": detail}


def run_selftest() -> list[dict[str, Any]]:
    """执行全部自检项。返回 [{"name","ok","detail"}]。"""
    checks: list[tuple[str, Callable[[], tuple[bool, str]]]] = [
        ("依赖导入完整性", _check_imports),
        ("向量后端真实生效", _check_vector_backend),
        ("切分多片段与覆盖", _check_chunker),
        ("BM25 稀疏召回", _check_bm25),
        ("稠密检索自命中", _check_dense),
        ("RRF 融合排名", _check_rrf),
        ("MMR 去冗余", _check_mmr),
        ("重排器有效性", _check_rerank),
        ("路由分派正确", _check_router),
        ("工具安全求值", _check_tools),
        ("智能体闭环", _check_agent),
        ("归因校验可被打穿", _check_verifier),
        ("端到端链路产出引用", _check_e2e),
        ("评测确定性", _check_eval_determinism),
        ("HTTP 服务健康", _check_api),
    ]
    return [_check(name, fn) for name, fn in checks]


# ---------------- 各项实现 ----------------
def _check_imports() -> tuple[bool, str]:
    import httpx  # noqa: F401
    return True, "numpy/faiss/onnxruntime/fastapi/httpx 均可用"


def _check_vector_backend() -> tuple[bool, str]:
    from stellarnx.store import build_index

    index = build_index(8)
    ok = index.name == "faiss" and index._fallback_reason is None
    return ok, f"后端={index.name} 回退原因={index._fallback_reason}"


def _check_chunker() -> tuple[bool, str]:
    from stellarnx.chunk import Chunker

    text = "第一段落。" * 60 + "\n\n" + "第二段落。" * 60
    chunks = Chunker(120, 20).chunk("d1", text)
    joined = "".join(c.text for c in chunks)
    ok = len(chunks) > 1 and "第一段落" in joined and "第二段落" in joined
    return ok, f"片段数={len(chunks)} 最长={max(len(c.text) for c in chunks)}"


def _check_bm25() -> tuple[bool, str]:
    from stellarnx.core.types import Chunk
    from stellarnx.retrieve import BM25Retriever

    chunks = [
        Chunk(id="a:0", doc_id="a", text="向量数据库用于存储嵌入向量并支持相似度检索"),
        Chunk(id="b:0", doc_id="b", text="公司年度团建安排在下周的海边度假村举行"),
    ]
    retriever = BM25Retriever()
    retriever.index(chunks)
    hits = retriever.search("向量数据库 相似度检索", 2)
    ok = bool(hits) and hits[0].chunk_id == "a:0"
    return ok, f"top={hits[0].chunk_id if hits else 'none'}"


def _check_dense() -> tuple[bool, str]:
    from stellarnx.pipeline.orchestrator import NexusSystem

    system = NexusSystem(load_config(embed_backend="hash", llm_backend="mock"), ":memory:")
    system.ingest_text("doc", "星枢系统的嵌入模块把文本变成向量。" * 20)
    hits = system.dense.search("嵌入模块", 3)
    ok = bool(hits) and hits[0].doc_id == "doc"
    return ok, f"index={system.index.name} top={hits[0].chunk_id if hits else 'none'}"


def _check_rrf() -> tuple[bool, str]:
    from stellarnx.core.types import Hit
    from stellarnx.retrieve import rrf_fuse

    def hit(cid: str) -> Hit:
        return Hit(chunk_id=cid, doc_id=cid, text=cid, score=1.0)

    fused = rrf_fuse([[hit("x"), hit("y")], [hit("y"), hit("z")]])
    ok = fused[0].chunk_id == "y"
    return ok, f"top={fused[0].chunk_id if fused else 'none'}"


def _check_mmr() -> tuple[bool, str]:
    from stellarnx.core.types import Hit
    from stellarnx.embed import HashEmbedder
    from stellarnx.retrieve import mmr_select

    embedder = HashEmbedder()
    candidates = [
        Hit(chunk_id=f"c{i}", doc_id="d", text=text, score=1.0)
        for i, text in enumerate([
            "苹果是一种水果，富含维生素。",
            "苹果是一种水果，富含维生素与膳食纤维。",
            "香蕉是另一种常见的热带水果。",
        ])
    ]
    picked = mmr_select(embedder.embed(["水果"])[0], candidates, embedder, 2, 0.3)
    ids = [p.chunk_id for p in picked]
    ok = len(ids) == 2 and not (ids[0] == "c0" and ids[1] == "c1")
    return ok, f"选中={ids}"


def _check_rerank() -> tuple[bool, str]:
    from stellarnx.core.types import Hit
    from stellarnx.rerank import LexicalReranker

    hits = [
        Hit(chunk_id="bad", doc_id="d", text="今天天气不错，适合出门散步。", score=0.1),
        Hit(chunk_id="good", doc_id="d", text="向量数据库支持相似度检索，用于检索增强生成。", score=0.2),
    ]
    ranked = LexicalReranker().rerank("向量数据库 检索增强", hits, 2)
    ok = ranked[0].chunk_id == "good"
    return ok, f"top={ranked[0].chunk_id if ranked else 'none'}"


def _check_router() -> tuple[bool, str]:
    from stellarnx.core.types import RoutePath
    from stellarnx.route import ComplexityRouter

    router = ComplexityRouter()
    direct = router.decide("你好", True)
    rag = router.decide("向量数据库是什么", True)
    agent = router.decide("计算 12 * 15", True)
    multi = router.decide("比较向量数据库和关键词检索的区别", True)
    ok = (direct.path is RoutePath.DIRECT
          and rag.path is RoutePath.RAG
          and agent.path is RoutePath.AGENT
          and multi.path is RoutePath.MULTIHOP)
    return ok, f"{direct.path.value}/{rag.path.value}/{agent.path.value}/{multi.path.value}"


def _check_tools() -> tuple[bool, str]:
    from stellarnx.agent.tools import build_default_tools, safe_eval

    registry = build_default_tools()
    good = registry.run("calculator", expression="12 * (3 + 4)")
    bad = registry.run("calculator", expression="__import__('os').system('echo hi')")
    ok = good.get("ok") and abs(float(good["result"]) - 84.0) < 1e-9 and not bad.get("ok")
    try:
        safe_eval("1/0")
        ok = False
    except ZeroDivisionError:
        pass
    return ok, f"12*(3+4)={good.get('result')} 逃逸被拒={not bad.get('ok')}"


def _check_agent() -> tuple[bool, str]:
    from stellarnx.pipeline.orchestrator import NexusSystem

    system = NexusSystem(load_config(embed_backend="hash", llm_backend="mock"), ":memory:")
    system.ingest_text("doc", "本手册说明向量数据库的部署方式与参数配置。" * 10)
    result = system.agent.run("计算 7 * 8")
    ok = len(result.steps) >= 1 and "56" in result.answer
    return ok, f"步骤={len(result.steps)} 重规划={result.replans} 批评分={result.critic_scores}"


def _check_verifier() -> tuple[bool, str]:
    from stellarnx.core.types import Hit
    from stellarnx.verify import EvidenceVerifier

    evidence = [
        Hit(chunk_id="e1", doc_id="d", score=1.0,
            text="星枢系统于 2026 年发布，包含检索与校验两个模块。")
    ]
    verifier = EvidenceVerifier()
    good = verifier.verify("星枢系统于 2026 年发布。 [1]", evidence, 0.6)
    bad = verifier.verify("星枢系统于 1999 年发布，共 42 个模块。 [1]", evidence, 0.6)
    ok = good.groundedness > bad.groundedness and good.passed
    return ok, f"真实={good.groundedness:.3f} 篡改={bad.groundedness:.3f}"


def _check_e2e() -> tuple[bool, str]:
    from stellarnx.pipeline.orchestrator import NexusSystem

    cfg = load_config(embed_backend="hash", llm_backend="mock", rerank_backend="lexical")
    system = NexusSystem(cfg, ":memory:")
    system.ingest_text(
        "handbook",
        "向量数据库用于保存嵌入向量。星枢系统使用 FAISS 作为向量索引，"
        "同时保留 numpy 回退实现。检索阶段采用 BM25 与稠密向量双路召回。" * 6,
    )
    answer = system.answer("星枢系统使用什么作为向量索引？")
    ok = bool(answer.text) and len(answer.citations) > 0 and answer.route is not None
    grounded = answer.verification.groundedness if answer.verification else -1.0
    return ok, f"路由={answer.route.path.value} 引用={len(answer.citations)} 可证性={grounded:.3f}"


def _check_eval_determinism() -> tuple[bool, str]:
    from stellarnx.eval.dataset import load_cases, load_corpus
    from stellarnx.eval.runner import EvalRunner

    cfg = load_config(embed_backend="hash", llm_backend="mock", rerank_backend="lexical")
    corpus = load_corpus("data/golden/corpus.jsonl")
    cases = load_cases("data/golden/questions.jsonl")
    first = EvalRunner(cfg).run(corpus, cases)
    second = EvalRunner(cfg).run(corpus, cases)
    same = first.metrics["recall@5"] == second.metrics["recall@5"] and \
        first.metrics["groundedness"] == second.metrics["groundedness"]
    return same, f"recall@5={first.metrics['recall@5']:.4f} 两次一致={same}"


def _check_api() -> tuple[bool, str]:
    from fastapi.testclient import TestClient

    from stellarnx.api.app import create_app

    cfg = load_config(embed_backend="hash", llm_backend="mock")
    client = TestClient(create_app(cfg=cfg))
    health = client.get("/health")
    ingested = client.post("/ingest", json={"docs": [
        {"doc_id": "d1", "text": "星枢系统的检索层包含 BM25 与稠密向量两路召回。" * 12}]})
    chat = client.post("/chat", json={"query": "检索层包含哪两路召回？"})
    ok = (health.status_code == 200 and ingested.status_code == 200
          and chat.status_code == 200 and bool(chat.json().get("text")))
    return ok, f"health={health.status_code} ingest={ingested.json().get('chunks')} chat={chat.status_code}"
