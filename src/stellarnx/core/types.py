"""全局数据类型。所有模块只依赖这些类型，不依赖彼此的具体实现。

作者: 晨星
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class RoutePath(str, Enum):
    """推理链路枚举。路由器的唯一输出空间。"""

    DIRECT = "direct"
    RAG = "rag"
    MULTIHOP = "multihop"
    AGENT = "agent"


@dataclass(slots=True)
class Chunk:
    """文档片段。切分模块的产物，向量库的最小存储单位。"""

    id: str
    doc_id: str
    text: str
    ordinal: int = 0
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Hit:
    """检索命中项。检索与重排模块之间传递的统一形态。"""

    chunk_id: str
    doc_id: str
    text: str
    score: float
    source: str = "dense"
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Citation:
    """答案中的一条引用。必须指向真实存在的 Chunk。"""

    chunk_id: str
    doc_id: str
    text: str
    score: float


@dataclass(slots=True)
class Claim:
    """答案拆出的一条断言及其可证性判定。"""

    text: str
    supported: bool
    score: float
    evidence_id: str | None = None


@dataclass(slots=True)
class VerifyReport:
    """归因校验报告。

    groundedness = 被证据支持的断言数 / 断言总数，取值 [0, 1]。
    断言数为 0 时 groundedness 为 0.0，而不是 1.0 —— 空答案不算"可证"。
    """

    groundedness: float
    claims: list[Claim]
    threshold: float
    passed: bool
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RouteDecision:
    """路由决策。路由器唯一对外产物。"""

    path: RoutePath
    reason: str
    confidence: float
    features: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Answer:
    """系统最终答案。含文本、引用、路由与校验结果，可完整溯源。"""

    text: str
    citations: list[Citation] = field(default_factory=list)
    route: RouteDecision | None = None
    verification: VerifyReport | None = None
    trace_id: str = ""
    latency_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "citations": [
                {"chunk_id": c.chunk_id, "doc_id": c.doc_id, "text": c.text, "score": round(c.score, 6)}
                for c in self.citations
            ],
            "route": None if self.route is None else {
                "path": self.route.path.value,
                "reason": self.route.reason,
                "confidence": round(self.route.confidence, 4),
                "features": self.route.features,
            },
            "verification": None if self.verification is None else {
                "groundedness": round(self.verification.groundedness, 4),
                "threshold": self.verification.threshold,
                "passed": self.verification.passed,
                "claims": [
                    {"text": c.text, "supported": c.supported,
                     "score": round(c.score, 4), "evidence_id": c.evidence_id}
                    for c in self.verification.claims
                ],
            },
            "trace_id": self.trace_id,
            "latency_ms": round(self.latency_ms, 2),
            "metadata": self.metadata,
        }
