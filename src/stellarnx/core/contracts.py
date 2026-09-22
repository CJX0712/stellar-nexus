"""模块间协议（Protocol）。所有能力模块面向协议编程，实现可运行时替换。

这是"每个模块可独立验证"的技术支点：单测注入零依赖实现，生产注入真实实现，
业务代码一行不动。

作者: 晨星
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np

from stellarnx.core.types import Hit, VerifyReport


@runtime_checkable
class Embedder(Protocol):
    """文本向量化。唯一职责：文本列表 -> 二维浮点矩阵。"""

    name: str
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray:
        """返回 shape=(len(texts), dim) 的 float32 矩阵，行向量已 L2 归一化。"""
        ...


class VectorIndex(Protocol):
    """向量索引。唯一职责：按向量相似度检索 id。不做文本存储。"""

    name: str

    def add(self, ids: list[str], vectors: np.ndarray) -> None:
        ...

    def search(self, vectors: np.ndarray, k: int) -> list[list[tuple[str, float]]]:
        """每个查询向量返回最多 k 个 (id, score)，score 越大越相似。"""
        ...

    def delete(self, ids: list[str]) -> None:
        ...

    def clear(self) -> None:
        ...

    def count(self) -> int:
        ...


class Retriever(Protocol):
    """检索器。唯一职责：query -> 候选 Hit 列表。"""

    name: str

    def search(self, query: str, k: int) -> list[Hit]:
        ...


class Reranker(Protocol):
    """重排器。唯一职责：对候选 Hit 精排并截断。"""

    name: str

    def rerank(self, query: str, hits: list[Hit], k: int) -> list[Hit]:
        ...


class LLM(Protocol):
    """大语言模型适配。唯一职责：消息 -> 文本。"""

    name: str

    def complete(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        ...

    def is_available(self) -> bool:
        ...


class Verifier(Protocol):
    """可证性校验器。唯一职责：判定答案断言是否被给定证据支持。"""

    name: str

    def verify(self, answer: str, evidence: list[Hit], threshold: float) -> VerifyReport:
        ...


class Tool(Protocol):
    """智能体可用工具。唯一职责：参数字典 -> 结果字典。"""

    name: str
    description: str

    def run(self, **kwargs: Any) -> dict[str, Any]:
        ...
