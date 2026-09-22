"""轻量 Span 追踪。零依赖，输出可直接 JSON 序列化供前端渲染。

作者: 晨星
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Span:
    """一个追踪节点。children 构成树，duration_ms 在结束时填充。"""

    name: str
    trace_id: str
    start_ms: float
    end_ms: float | None = None
    attrs: dict[str, Any] = field(default_factory=dict)
    children: list[Span] = field(default_factory=list)

    @property
    def duration_ms(self) -> float:
        if self.end_ms is None:
            return 0.0
        return round(self.end_ms - self.start_ms, 3)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "trace_id": self.trace_id,
            "duration_ms": self.duration_ms,
            "attrs": self.attrs,
            "children": [c.to_dict() for c in self.children],
        }


class Tracer:
    """追踪器。用 with tracer.span("x") 包裹任意代码块。"""

    def __init__(self, trace_id: str | None = None) -> None:
        self.trace_id = trace_id or uuid.uuid4().hex[:12]
        self.root = Span(name="root", trace_id=self.trace_id, start_ms=self._now())
        self._stack: list[Span] = [self.root]

    @staticmethod
    def _now() -> float:
        return time.perf_counter() * 1000.0

    @contextmanager
    def span(self, name: str, **attrs: Any) -> Iterator[Span]:
        node = Span(name=name, trace_id=self.trace_id, start_ms=self._now(), attrs=dict(attrs))
        self._stack[-1].children.append(node)
        self._stack.append(node)
        try:
            yield node
        finally:
            node.end_ms = self._now()
            self._stack.pop()

    def finish(self) -> Span:
        self.root.end_ms = self._now()
        return self.root

    def to_dict(self) -> dict[str, Any]:
        self.finish()
        return self.root.to_dict()

    def flat(self) -> list[Span]:
        """深度优先展开成列表，便于按名字聚合耗时。"""
        out: list[Span] = []

        def walk(node: Span) -> None:
            out.append(node)
            for child in node.children:
                walk(child)

        walk(self.root)
        return out
