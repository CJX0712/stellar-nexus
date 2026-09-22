"""黄金数据集定义与加载。

作者: 晨星
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class EvalCase:
    """一条评测用例。gold 用"文档 + 片段子串"定位，不依赖切分后生成的具体 id。"""

    query: str
    gold_doc: str
    gold_text: str
    must_include: list[str] = field(default_factory=list)
    route_expect: str | None = None


def _read_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        yield json.loads(line)


def load_corpus(path: str | Path) -> list[dict[str, Any]]:
    """加载语料。每行 {doc_id, text, meta?}。"""
    return [dict(row) for row in _read_jsonl(path)]


def load_cases(path: str | Path) -> list[EvalCase]:
    """加载评测用例。每行 {query, gold_doc, gold_text, must_include?, route_expect?}。"""
    cases: list[EvalCase] = []
    for row in _read_jsonl(path):
        cases.append(
            EvalCase(
                query=str(row["query"]),
                gold_doc=str(row["gold_doc"]),
                gold_text=str(row["gold_text"]),
                must_include=list(row.get("must_include") or []),
                route_expect=row.get("route_expect"),
            )
        )
    return cases
