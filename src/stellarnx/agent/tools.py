"""工具注册表。每个工具只做一件事，输入输出都是普通 dict（可 JSON 序列化）。

作者: 晨星
"""

from __future__ import annotations

import ast
import operator
from collections.abc import Callable
from typing import Any

from stellarnx.core.contracts import Retriever
from stellarnx.core.types import Hit

_BIN_OPS: dict[type, Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}
_UNARY_OPS: dict[type, Callable[[Any], Any]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def safe_eval(expression: str) -> float:
    """受限算术求值。只允许数字与四则运算，杜绝 __import__ 等逃逸。"""
    tree = ast.parse(expression.strip(), mode="eval")

    def walk(node: ast.AST) -> Any:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
            return _BIN_OPS[type(node.op)](walk(node.left), walk(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
            return _UNARY_OPS[type(node.op)](walk(node.operand))
        raise ValueError(f"不允许的表达式元素: {type(node).__name__}")

    return float(walk(tree.body))


class ToolRegistry:
    """工具注册表。唯一职责：按名字注册与执行工具。"""

    def __init__(self) -> None:
        self._tools: dict[str, Callable[..., dict[str, Any]]] = {}
        self._descriptions: dict[str, str] = {}

    def register(self, name: str, description: str, func: Callable[..., dict[str, Any]]) -> None:
        self._tools[name] = func
        self._descriptions[name] = description

    def names(self) -> list[str]:
        return sorted(self._tools)

    def describe(self) -> list[dict[str, str]]:
        return [{"name": n, "description": self._descriptions[n]} for n in self.names()]

    def run(self, name: str, **kwargs: Any) -> dict[str, Any]:
        if name not in self._tools:
            return {"ok": False, "error": f"未注册的工具: {name}"}
        try:
            result = self._tools[name](**kwargs)
        except Exception as exc:
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        result.setdefault("ok", True)
        return result


def build_default_tools(retriever: Retriever | None = None) -> ToolRegistry:
    """默认工具集：计算器 + 文本统计 + 可选知识库检索。"""
    registry = ToolRegistry()

    def calculator(expression: str = "") -> dict[str, Any]:
        return {"ok": True, "expression": expression, "result": safe_eval(expression)}

    def text_stats(text: str = "") -> dict[str, Any]:
        return {
            "ok": True,
            "chars": len(text),
            "cjk_chars": sum(1 for c in text if "\u4e00" <= c <= "\u9fff"),
            "lines": len([line for line in text.splitlines() if line.strip()]),
        }

    registry.register("calculator", "计算算术表达式，参数 expression", calculator)
    registry.register("text_stats", "统计文本长度与行数，参数 text", text_stats)

    if retriever is not None:

        def kb_search(query: str = "", top_k: int = 3) -> dict[str, Any]:
            hits: list[Hit] = retriever.search(query, int(top_k))
            return {
                "ok": True,
                "query": query,
                "hits": [
                    {"chunk_id": h.chunk_id, "doc_id": h.doc_id,
                     "score": round(h.score, 6), "text": h.text[:300]}
                    for h in hits
                ],
            }

        registry.register("kb_search", "检索知识库，参数 query / top_k", kb_search)

    return registry
