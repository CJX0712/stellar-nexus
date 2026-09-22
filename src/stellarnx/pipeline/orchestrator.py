"""星枢系统编排器。

职责边界：本文件只做"把模块接起来"和"路由之后走哪条路"，
任何具体能力（嵌入、检索、重排、生成、校验）都不在这里实现。

作者: 晨星
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from stellarnx.agent.loop import AgentLoop
from stellarnx.agent.tools import build_default_tools
from stellarnx.chunk import Chunker
from stellarnx.core.config import Config
from stellarnx.core.paths import PROJECT_ROOT as _PROJECT_ROOT
from stellarnx.core.types import Answer, Chunk, Citation, Hit, RoutePath
from stellarnx.embed import build_embedder
from stellarnx.llm import build_llm, build_rag_messages
from stellarnx.rerank import build_reranker
from stellarnx.retrieve import BM25Retriever, DenseRetriever, HybridRetriever, rrf_fuse
from stellarnx.retrieve.hybrid import mmr_select
from stellarnx.route import ComplexityRouter
from stellarnx.store import DocumentStore, build_index
from stellarnx.trace import Tracer
from stellarnx.verify import EvidenceVerifier, extract_citations

logger = logging.getLogger(__name__)

_SPLIT = re.compile(r"和|与|以及|对比|比较|分别|、|；")


class NexusSystem:
    """星枢系统。对外只有四个动作：ingest / search / answer / stats。"""

    def __init__(self, cfg: Config | None = None, store_path: str = ":memory:") -> None:
        self.cfg = cfg or Config()
        self.embedder = build_embedder(self.cfg)
        self.store = DocumentStore(store_path)
        self.index = build_index(self.embedder.dim)
        self.chunker = Chunker(self.cfg.chunk_size, self.cfg.chunk_overlap)
        self.bm25 = BM25Retriever()
        self.dense = DenseRetriever(self.embedder, self.index, self.store)
        self.hybrid = HybridRetriever(
            [self.bm25, self.dense],
            embedder=self.embedder,
            rrf_k=self.cfg.rrf_k,
            mmr_lambda=self.cfg.mmr_lambda,
        )
        self.llm = build_llm(self.cfg)
        # 重排器依赖 LLM，因此必须在 LLM 之后构造
        self.reranker = build_reranker(self.cfg, llm=self.llm)
        self.verifier = EvidenceVerifier()
        self.router = ComplexityRouter()
        self.agent = AgentLoop(
            build_default_tools(self.hybrid),
            max_steps=self.cfg.agent_max_steps,
            max_replan=self.cfg.agent_max_replan,
        )

    # ---------- 写入 ----------
    def ingest_text(self, doc_id: str, text: str, meta: dict[str, Any] | None = None) -> int:
        """写入一篇文档，返回产生的片段数。同 doc_id 重复写入是覆盖。"""
        meta = dict(meta or {})
        self._drop_document(doc_id)
        self.store.add_document(doc_id, title=meta.get("title", doc_id),
                                source=meta.get("source", "inline"), meta=meta)
        chunks = self.chunker.chunk(doc_id, text, meta)
        if not chunks:
            self._rebuild_bm25()
            return 0
        self.store.add_chunks(chunks)
        self._index_chunks(chunks)
        self._rebuild_bm25()
        return len(chunks)

    def ingest_many(self, docs: list[dict[str, Any]]) -> int:
        """批量写入。docs 元素需含 doc_id 与 text，可选 meta。同 doc_id 重复写入是覆盖。"""
        # 同一批里出现重复 doc_id 时只保留最后一条，与「覆盖」语义一致。
        # 否则先写的那份会被后写的覆盖计数、却把片段全部留在库里。
        latest: dict[str, dict[str, Any]] = {}
        for doc in docs:
            latest[str(doc["doc_id"])] = doc

        new_chunks: list[Chunk] = []
        for doc in latest.values():
            doc_id = str(doc["doc_id"])
            meta = dict(doc.get("meta") or {})
            self._drop_document(doc_id)
            self.store.add_document(doc_id, title=meta.get("title", doc_id),
                                    source=meta.get("source", "batch"), meta=meta)
            new_chunks.extend(self.chunker.chunk(doc_id, str(doc["text"]), meta))
        if not new_chunks:
            self._rebuild_bm25()
            return 0
        self.store.add_chunks(new_chunks)
        self._index_chunks(new_chunks)
        self._rebuild_bm25()
        return len(new_chunks)

    def _drop_document(self, doc_id: str) -> None:
        """清掉某文档的旧片段与旧向量，使重复入库成为幂等覆盖。

        踩过的坑：`store.add_chunks` 用的是 INSERT OR REPLACE，旧片段被替换掉了，
        但向量是 `index.add` 追加写入的 —— 同一个 chunk_id 会在索引里留下**两份**。
        症状是 `chunks=16 而 vectors=32`，检索还能返回重复 id。

        更隐蔽的是「新版本比旧版本短」的情况：ordinal 变小的片段被覆盖，
        ordinal 超出新长度的旧片段**从来没被删掉**，会一直留在库里被检索到，
        表现为「文档里已经删掉的内容仍然能被搜出来」。
        所以覆盖必须显式清空，而不是指望 REPLACE。
        """
        old_ids = [c.id for c in self.store.chunks_of_doc(doc_id)]
        if not old_ids:
            return
        self.index.delete(old_ids)
        self.store.delete_document(doc_id)

    def _index_chunks(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        ids = [c.id for c in chunks]
        # 兜底去重：任何调用方重复喂同一个 id，索引里都不应出现两份向量
        self.index.delete(ids)
        vectors = self.embedder.embed([c.text for c in chunks])
        self.index.add(ids, vectors)

    def _rebuild_bm25(self) -> None:
        self.bm25.index(self.store.all_chunks())

    def clear(self) -> None:
        self.store.clear()
        self.index.clear()
        self.bm25.index([])

    def restore(self) -> int:
        """从持久化存储重建内存索引。

        踩过的坑：文档存在 sqlite 里，向量索引与 BM25 却只在内存中。
        进程重启后库里明明有片段，检索却全部落空 —— 这种"看起来有数据、
        实际搜不到"的故障最难排查。所以启动时必须显式重建，且要以
        「索引条数 == 片段条数」为完成判据，而不是「跑过一遍就算完」。
        """
        chunks = self.store.all_chunks()
        self.index.clear()
        if not chunks:
            self.bm25.index([])
            return 0
        self._index_chunks(chunks)
        self._rebuild_bm25()
        if self.index.count() != len(chunks):
            raise RuntimeError(
                f"索引恢复不完整: 片段 {len(chunks)} 条，向量 {self.index.count()} 条")
        return len(chunks)

    def index_is_consistent(self) -> bool:
        """索引与片段库是否一致。部署前自检用。"""
        return (self.store.count_chunks() == self.index.count()
                == len(self.bm25.all_ids()))

    # ---------- 检索 ----------
    def search(self, query: str, k: int | None = None) -> list[Hit]:
        top_k = k or self.cfg.top_k_retrieve
        if self.store.count_chunks() == 0:
            return []
        candidates = self.hybrid.search(query, top_k)
        return self.reranker.rerank(query, candidates, min(top_k, len(candidates)))

    def _multihop_search(self, query: str, k: int) -> list[Hit]:
        """把复合问题拆成子查询分别召回，再融合。"""
        parts = [p.strip() for p in _SPLIT.split(query) if len(p.strip()) >= 2]
        sub_queries = parts[:3] or [query]
        lists = [self.hybrid.search(q, k) for q in sub_queries]
        lists.append(self.hybrid.search(query, k))
        fused = rrf_fuse(lists, self.cfg.rrf_k)
        if len(fused) > k:
            fused = mmr_select(self.embedder.embed([query])[0], fused, self.embedder, k,
                               self.cfg.mmr_lambda)
        return fused[:k]

    # ---------- 问答 ----------
    def answer(self, query: str) -> Answer:
        tracer = Tracer()
        corpus_ready = self.store.count_chunks() > 0

        with tracer.span("route", query=query[:120]) as span:
            decision = self.router.decide(query, corpus_ready)
            span.attrs["path"] = decision.path.value
            span.attrs["reason"] = decision.reason

        if decision.path is RoutePath.DIRECT:
            with tracer.span("generate", mode="direct"):
                text = self.llm.complete([{"role": "user", "content": query}])
            return self._finish(text, [], decision, tracer,
                                verification=None, note="direct_no_claim")

        if decision.path is RoutePath.AGENT:
            with tracer.span("agent"):
                agent_result = self.agent.run(query)
            evidence = agent_result.evidence or self.search(query, self.cfg.top_k_rerank)
            # 工具产出本身也是证据：计算器给出的 512 不在任何文档里，
            # 若不并入证据，归因校验会把正确的工具答案一律判为无据。
            if agent_result.observations:
                evidence = [
                    *evidence,
                    Hit(chunk_id="agent:observation", doc_id="agent",
                        text="；".join(agent_result.observations), score=1.0, source="agent"),
                ]
            with tracer.span("generate", mode="agent"):
                text = self.llm.complete(build_rag_messages(query, evidence[:5]))
            # 智能体链路同样必须过校验，否则工具返回的偏差会原样输出
            with tracer.span("verify", mode="agent"):
                report = self.verifier.verify(text, evidence, self.cfg.groundedness_threshold)
            answer = self._finish(text, evidence, decision, tracer,
                                  verification=report, note="agent")
            answer.metadata["agent"] = agent_result.to_dict()
            return answer

        if decision.path is RoutePath.MULTIHOP:
            with tracer.span("retrieve", mode="multihop"):
                evidence = self._multihop_search(query, self.cfg.top_k_retrieve)
        else:
            with tracer.span("retrieve", mode="rag"):
                evidence = self.hybrid.search(query, self.cfg.top_k_retrieve)

        with tracer.span("rerank", candidates=len(evidence)):
            evidence = self.reranker.rerank(query, evidence, self.cfg.top_k_rerank)

        with tracer.span("generate", evidence=len(evidence)):
            text = self.llm.complete(build_rag_messages(query, evidence))

        with tracer.span("verify"):
            report = self.verifier.verify(text, evidence, self.cfg.groundedness_threshold)
            for attempt in range(self.cfg.max_regenerate):
                if report.passed:
                    break
                # 未通过则收紧提示，要求只复述资料中原句
                strict = build_rag_messages(
                    f"{query}\n（上一次回答缺乏依据，请只复述资料中的原句，不要推断。）",
                    evidence,
                )
                text = self.llm.complete(strict)
                report = self.verifier.verify(text, evidence, self.cfg.groundedness_threshold)
                logger.info("重生成第 %d 轮，groundedness=%.3f", attempt + 1, report.groundedness)

        return self._finish(text, evidence, decision, tracer, verification=report)

    # ---------- 辅助 ----------
    def _finish(
        self,
        text: str,
        evidence: list[Hit],
        decision: Any,
        tracer: Tracer,
        verification: Any = None,
        note: str = "",
    ) -> Answer:
        root = tracer.finish()
        citations = self._build_citations(text, evidence)
        return Answer(
            text=text,
            citations=citations,
            route=decision,
            verification=verification,
            trace_id=tracer.trace_id,
            latency_ms=root.duration_ms,
            metadata={
                "note": note,
                "embedder": self.embedder.name,
                "llm": self.llm.name,
                "reranker": self.reranker.name,
                "vector_backend": self.index.name,
                "trace": root.to_dict(),
            },
        )

    @staticmethod
    def _build_citations(text: str, evidence: list[Hit]) -> list[Citation]:
        """按答案中的 [n] 标记挑出对应片段；无标记时退回前三。"""
        if not evidence:
            return []
        referenced = extract_citations(text)
        picked: list[Hit] = []
        for number in referenced:
            if 1 <= number <= len(evidence) and evidence[number - 1] not in picked:
                picked.append(evidence[number - 1])
        if not picked:
            picked = evidence[:3]
        return [
            Citation(chunk_id=h.chunk_id, doc_id=h.doc_id, text=h.text[:400],
                     score=round(h.score, 6))
            for h in picked
        ]

    def stats(self) -> dict[str, Any]:
        return {
            "documents": self.store.count_documents(),
            "chunks": self.store.count_chunks(),
            "vectors": self.index.count(),
            "embedder": self.embedder.name,
            "dim": self.embedder.dim,
            "llm": self.llm.name,
            "reranker": self.reranker.name,
            "vector_backend": self.index.name,
            "config": self.cfg.to_dict(),
        }


def build_system(cfg: Config | None = None, store_path: str | None = None) -> NexusSystem:
    """构造系统。

    store_path 为 None 时按配置落到 ``<data_dir>/stellarnx.db``（持久化），
    显式传 ":memory:" 则用内存库。持久化模式下必须重建内存索引，
    否则重启后「库里有、搜不到」。
    """
    cfg = cfg or Config()
    if store_path is None:
        if cfg.data_dir:
            directory = Path(cfg.data_dir)
            # 相对路径按仓库根解析，避免服务换个启动目录就把库建到别处
            if not directory.is_absolute():
                directory = Path(_PROJECT_ROOT) / directory
            directory.mkdir(parents=True, exist_ok=True)
            store_path = str(directory / "stellarnx.db")
        else:
            store_path = ":memory:"
    system = NexusSystem(cfg, store_path)
    if system.store.count_chunks() > 0:
        system.restore()
    return system
