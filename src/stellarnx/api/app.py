"""FastAPI 应用装配。入口只做装配，零业务逻辑。

作者: 晨星
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from dataclasses import replace
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from stellarnx import __author__, __version__
from stellarnx.api.schemas import ChatRequest, HealthResponse, IngestRequest, SearchRequest
from stellarnx.core.config import Config, load_config
from stellarnx.core.errors import StellarNexusError
from stellarnx.core.paths import CORPUS_PATH, QUESTIONS_PATH, WEB_DIR, resolve
from stellarnx.eval.dataset import load_cases, load_corpus
from stellarnx.eval.runner import EvalRunner

logger = logging.getLogger(__name__)
_WEB_DIR = WEB_DIR
_TRACE_CAP = 200


def create_app(system: Any = None, cfg: Config | None = None) -> FastAPI:
    """构造应用。system 可注入，便于测试用 fake 管道。"""
    from stellarnx.pipeline.orchestrator import NexusSystem

    cfg = cfg or load_config()
    app = FastAPI(
        title="星枢 StellarNexus",
        version=__version__,
        description="可自证的模块化复合智能系统",
        contact={"name": __author__},
    )
    app.state.cfg = cfg
    app.state.system = system or NexusSystem(cfg, ":memory:")
    app.state.traces: dict[str, dict[str, Any]] = {}

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        sys_obj = app.state.system
        return HealthResponse(
            backends={
                "embedder": sys_obj.embedder.name,
                "llm": sys_obj.llm.name,
                "reranker": sys_obj.reranker.name,
                "vector": sys_obj.index.name,
            }
        )

    @app.get("/stats")
    def stats() -> dict[str, Any]:
        return app.state.system.stats()

    @app.post("/ingest")
    def ingest(req: IngestRequest) -> dict[str, Any]:
        if not req.docs:
            raise HTTPException(status_code=400, detail="docs 不能为空")
        for doc in req.docs:
            if "doc_id" not in doc or "text" not in doc:
                raise HTTPException(status_code=400, detail="每个文档需含 doc_id 与 text")
        count = app.state.system.ingest_many(req.docs)
        return {"chunks": count, "documents": app.state.system.store.count_documents()}

    @app.get("/documents")
    def list_documents() -> list[dict[str, Any]]:
        return app.state.system.store.list_documents()

    @app.delete("/documents/{doc_id}")
    def delete_document(doc_id: str) -> dict[str, Any]:
        removed = app.state.system.store.delete_document(doc_id)
        app.state.system._rebuild_bm25()
        return {"removed_chunks": removed, "doc_id": doc_id}

    @app.post("/search")
    def search(req: SearchRequest) -> list[dict[str, Any]]:
        hits = app.state.system.search(req.query, req.k)
        return [
            {"chunk_id": h.chunk_id, "doc_id": h.doc_id, "score": round(h.score, 6),
             "source": h.source, "text": h.text[:600]}
            for h in hits
        ]

    @app.post("/chat")
    def chat(req: ChatRequest) -> dict[str, Any]:
        answer = app.state.system.answer(req.query)
        app.state.traces[answer.trace_id] = answer.metadata.get("trace", {})
        if len(app.state.traces) > _TRACE_CAP:
            app.state.traces.pop(next(iter(app.state.traces)))
        payload = answer.to_dict()
        if not req.with_trace:
            payload["metadata"].pop("trace", None)
        return payload

    @app.get("/chat/stream")
    def chat_stream(query: str) -> Any:
        """分阶段事件流：路由 -> 检索 -> 生成 -> 校验 -> 完成。"""
        from sse_starlette.sse import EventSourceResponse

        system = app.state.system

        def events() -> Iterator[dict[str, str]]:
            try:
                yield {"event": "stage", "data": json.dumps({"stage": "start", "query": query})}
                answer = system.answer(query)
                app.state.traces[answer.trace_id] = answer.metadata.get("trace", {})
                for span in answer.metadata.get("trace", {}).get("children", []):
                    yield {
                        "event": "stage",
                        "data": json.dumps({
                            "stage": span.get("name"),
                            "duration_ms": span.get("duration_ms"),
                            "attrs": span.get("attrs"),
                        }),
                    }
                yield {"event": "answer", "data": json.dumps(
                    {"text": answer.text, "trace_id": answer.trace_id,
                     "citations": [c.chunk_id for c in answer.citations]},
                    ensure_ascii=False)}
                yield {"event": "done", "data": json.dumps(
                    {"groundedness": answer.verification.groundedness
                     if answer.verification else None})}
            except StellarNexusError as exc:
                yield {"event": "error", "data": json.dumps({"code": exc.code, "detail": str(exc)})}

        return EventSourceResponse(events())

    @app.get("/trace/{trace_id}")
    def get_trace(trace_id: str) -> dict[str, Any]:
        trace = app.state.traces.get(trace_id)
        if trace is None:
            raise HTTPException(status_code=404, detail="trace 不存在或已被淘汰")
        return trace

    @app.post("/evaluate")
    def evaluate(corpus: str = str(CORPUS_PATH),
                 questions: str = str(QUESTIONS_PATH),
                 offline: bool = False) -> dict[str, Any]:
        """跑黄金集并返回门禁结论。

        ``offline=True`` 时强制换成零依赖后端。为什么需要这个开关：
        真实模型下每条用例要 20 秒以上，8 条就要好几分钟，而门禁的价值在于
        **快、确定、能反复跑**。用真实模型评出来的数字是「质量参考」，
        不是门禁基线 —— 两者混在一个接口里，只会让人以为系统坏了。
        """
        # 路径必须相对仓库根解析，否则服务换个启动目录就 404
        corpus_path = resolve(corpus)
        question_path = resolve(questions)
        if not corpus_path.exists() or not question_path.exists():
            raise HTTPException(status_code=404, detail="评测数据文件不存在")
        cfg = app.state.cfg
        if offline:
            cfg = replace(cfg, embed_backend="hash", llm_backend="mock",
                          rerank_backend="lexical")
        report = EvalRunner(cfg).run(load_corpus(corpus_path), load_cases(question_path))
        payload = report.to_dict()
        payload["mode"] = "offline" if offline else "configured"
        payload["backends"] = {
            "embedder": cfg.embed_backend,
            "llm": cfg.llm_backend,
            "reranker": cfg.rerank_backend,
        }
        return payload

    @app.get("/", include_in_schema=False)
    def index() -> Any:
        index_file = _WEB_DIR / "index.html"
        if index_file.exists():
            return FileResponse(index_file)
        return JSONResponse({"service": "stellar-nexus", "author": __author__})

    if _WEB_DIR.exists():
        app.mount("/web", StaticFiles(directory=str(_WEB_DIR)), name="web")

    return app
