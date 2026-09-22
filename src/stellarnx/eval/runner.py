"""评测执行器：灌语料 -> 跑用例 -> 算指标 -> 对阈值。

关键设计（踩过坑才写下的）：
- 每次 run() 都新建内存管道，指标因此**确定**，重复运行逐位不变；
- 语料长度必须超过切分阈值，否则多片段路径永远测不到；
- 报告同时给出聚合指标与逐条明细，失败用例可定位。

作者: 晨星
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from stellarnx.core.config import Config
from stellarnx.eval.dataset import EvalCase
from stellarnx.eval.metrics import mean, mrr_at_k, ndcg_at_k, percentile, recall_at_k
from stellarnx.pipeline.orchestrator import NexusSystem

DEFAULT_THRESHOLDS = {
    "recall@1": 0.30,
    "recall@3": 0.60,
    "recall@5": 0.75,
    "mrr@10": 0.45,
    "ndcg@5": 0.45,
    "groundedness": 0.55,
    "latency_p95_ms": 8000.0,
}


@dataclass
class CaseResult:
    """单条用例结果。"""

    query: str
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    mrr_at_10: float
    ndcg_at_5: float
    groundedness: float
    latency_ms: float
    route: str
    top_chunk: str
    passed: bool
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalReport:
    """评测报告。"""

    metrics: dict[str, float]
    thresholds: dict[str, float]
    cases: list[CaseResult]
    passed: bool
    failures: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "metrics": {k: round(v, 6) for k, v in self.metrics.items()},
            "thresholds": self.thresholds,
            "passed": self.passed,
            "failures": self.failures,
            "cases": [
                {
                    "query": c.query,
                    "recall@1": round(c.recall_at_1, 4),
                    "recall@3": round(c.recall_at_3, 4),
                    "recall@5": round(c.recall_at_5, 4),
                    "mrr@10": round(c.mrr_at_10, 4),
                    "ndcg@5": round(c.ndcg_at_5, 4),
                    "groundedness": round(c.groundedness, 4),
                    "latency_ms": round(c.latency_ms, 2),
                    "route": c.route,
                    "top_chunk": c.top_chunk,
                    "passed": c.passed,
                }
                for c in self.cases
            ],
        }


class EvalRunner:
    """评测执行器。唯一职责：在全新管道上跑完整个数据集并给出报告。"""

    def __init__(self, cfg: Config | None = None, thresholds: dict[str, float] | None = None) -> None:
        self.cfg = cfg or Config()
        # 自定义阈值与默认合并且以自定义为准，避免只传一项时其它项 KeyError
        self.thresholds = {**DEFAULT_THRESHOLDS, **(thresholds or {})}

    def run(
        self,
        corpus: list[dict[str, Any]],
        cases: list[EvalCase],
        k: int = 10,
    ) -> EvalReport:
        system = NexusSystem(self.cfg, ":memory:")
        system.ingest_many(corpus)
        if system.store.count_chunks() == 0:
            raise ValueError("语料为空，无法评测")

        results: list[CaseResult] = []
        for case in cases:
            results.append(self._run_case(system, case, k))

        metrics = {
            "recall@1": mean([r.recall_at_1 for r in results]),
            "recall@3": mean([r.recall_at_3 for r in results]),
            "recall@5": mean([r.recall_at_5 for r in results]),
            "mrr@10": mean([r.mrr_at_10 for r in results]),
            "ndcg@5": mean([r.ndcg_at_5 for r in results]),
            "groundedness": mean([r.groundedness for r in results]),
            "latency_p50_ms": percentile([r.latency_ms for r in results], 50),
            "latency_p95_ms": percentile([r.latency_ms for r in results], 95),
            "cases": float(len(results)),
        }

        failures: list[str] = []
        for name, threshold in self.thresholds.items():
            value = metrics.get(name)
            if value is None:
                continue
            if name.startswith("latency"):
                ok = value <= threshold
            else:
                ok = value >= threshold
            if not ok:
                failures.append(f"{name}={value:.4f} 未达阈值 {threshold}")

        return EvalReport(
            metrics=metrics,
            thresholds=self.thresholds,
            cases=results,
            passed=not failures,
            failures=failures,
        )

    def _run_case(self, system: NexusSystem, case: EvalCase, k: int) -> CaseResult:
        gold_ids = self._gold_ids(system, case)

        started = time.perf_counter()
        hits = system.search(case.query, k)
        answer = system.answer(case.query)
        latency = (time.perf_counter() - started) * 1000.0

        retrieved = [h.chunk_id for h in hits]
        groundedness = (
            answer.verification.groundedness if answer.verification is not None else 0.0
        )
        passed = (
            recall_at_k(retrieved, gold_ids, 3) >= 0.99
            and groundedness >= self.thresholds["groundedness"]
        )
        return CaseResult(
            query=case.query,
            recall_at_1=recall_at_k(retrieved, gold_ids, 1),
            recall_at_3=recall_at_k(retrieved, gold_ids, 3),
            recall_at_5=recall_at_k(retrieved, gold_ids, 5),
            mrr_at_10=mrr_at_k(retrieved, gold_ids, 10),
            ndcg_at_5=ndcg_at_k(retrieved, gold_ids, 5),
            groundedness=groundedness,
            latency_ms=latency,
            route=answer.route.path.value if answer.route else "none",
            top_chunk=retrieved[0] if retrieved else "",
            passed=passed,
            detail={"gold_ids": gold_ids[:5], "answer": answer.text[:200]},
        )

    @staticmethod
    def _gold_ids(system: NexusSystem, case: EvalCase) -> list[str]:
        """按 gold_doc + gold_text 子串定位真实片段 id。"""
        matches = [
            c.id
            for c in system.store.chunks_of_doc(case.gold_doc)
            if case.gold_text in c.text
        ]
        if matches:
            return matches
        return [c.id for c in system.store.chunks_of_doc(case.gold_doc)]
