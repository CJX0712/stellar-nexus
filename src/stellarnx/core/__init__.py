"""核心层：只定义协议、类型、配置与错误，不含任何业务实现。"""

from stellarnx.core.config import Config, load_config
from stellarnx.core.contracts import (
    LLM,
    Embedder,
    Reranker,
    Retriever,
    VectorIndex,
    Verifier,
)
from stellarnx.core.errors import StellarNexusError
from stellarnx.core.types import (
    Answer,
    Chunk,
    Citation,
    Claim,
    Hit,
    RouteDecision,
    RoutePath,
    VerifyReport,
)

__all__ = [
    "LLM",
    "Answer",
    "Chunk",
    "Citation",
    "Claim",
    "Config",
    "Embedder",
    "Hit",
    "Reranker",
    "Retriever",
    "RouteDecision",
    "RoutePath",
    "StellarNexusError",
    "VectorIndex",
    "Verifier",
    "VerifyReport",
    "load_config",
]
