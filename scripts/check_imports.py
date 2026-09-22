"""分层依赖检查器。

为什么需要它：本项目真实踩过「循环导入」的坑 ——
``embed/__init__.py`` 导出 ``normalize``，``hash_embed.py`` 又 ``from stellarnx.embed import normalize``。
症状是「单独 import 子模块能过、从入口 import 就炸」，排查成本很高。

根治办法是把共用能力下沉到依赖图下层（``normalize`` 移到 ``core/linalg.py``），
而不是调 import 顺序。这个脚本把「下沉」这件事变成可自动校验的规则：

1. **无环**：包级依赖图里不允许出现强连通分量（大小 > 1 的环）。
2. **单向**：只允许从高层依赖低层，同层之间可以互相依赖。

作者: 晨星
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "stellarnx"

# 层级越小越底层。规则：importer.layer >= imported.layer
LAYERS: dict[str, int] = {
    "__root__": 0,
    "core": 0,
    "trace": 1,
    "embed": 1,
    "chunk": 1,
    "store": 1,
    "retrieve": 2,
    "rerank": 2,
    "llm": 2,
    "verify": 2,
    "route": 2,
    "agent": 2,
    "eval": 3,
    "pipeline": 3,
    "api": 4,
    "cli": 4,
    "selftest": 4,
}


def package_of(path: Path) -> str:
    """文件所属的子包名。src/stellarnx/__init__.py 视为包根，归入 core 层。"""
    rel = path.relative_to(SRC)
    parts = rel.parts
    if len(parts) == 1:
        return "__root__" if parts[0] == "__init__.py" else parts[0].removesuffix(".py")
    return parts[0]


def internal_imports(path: Path) -> set[str]:
    """抽取本文件 import 的 stellarnx 子包集合。"""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module == "stellarnx" or node.module.startswith("stellarnx."):
                tail = node.module.split(".")
                if len(tail) > 1:
                    found.add(tail[1])
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("stellarnx."):
                    found.add(alias.name.split(".")[1])
    return found


def build_graph() -> dict[str, set[str]]:
    graph: dict[str, set[str]] = {}
    for path in sorted(SRC.rglob("*.py")):
        pkg = package_of(path)
        deps = internal_imports(path)
        deps.discard(pkg)
        graph.setdefault(pkg, set()).update(deps)
    return graph


def find_cycles(graph: dict[str, set[str]]) -> list[list[str]]:
    """Tarjan 强连通分量。大小 > 1 即为环。"""
    index: dict[str, int] = {}
    low: dict[str, int] = {}
    on_stack: set[str] = set()
    stack: list[str] = []
    counter = [0]
    cycles: list[list[str]] = []

    def strongconnect(node: str) -> None:
        index[node] = low[node] = counter[0]
        counter[0] += 1
        stack.append(node)
        on_stack.add(node)
        for nxt in sorted(graph.get(node, ())):
            if nxt not in index:
                strongconnect(nxt)
                low[node] = min(low[node], low[nxt])
            elif nxt in on_stack:
                low[node] = min(low[node], index[nxt])
        if low[node] == index[node]:
            component: list[str] = []
            while True:
                member = stack.pop()
                on_stack.discard(member)
                component.append(member)
                if member == node:
                    break
            if len(component) > 1:
                cycles.append(sorted(component))

    sys.setrecursionlimit(10_000)
    for node in sorted(graph):
        if node not in index:
            strongconnect(node)
    return cycles


def main() -> int:
    graph = build_graph()
    unknown = sorted(set(graph) - set(LAYERS))
    failures: list[str] = []

    lines = ["包级依赖图（importer -> imported）:"]
    for pkg in sorted(graph):
        deps = ", ".join(sorted(graph[pkg])) or "-"
        lines.append(f"  {pkg:<10} L{LAYERS.get(pkg, '?')}  ->  {deps}")
    print("\n".join(lines))

    if unknown:
        failures.append(f"LAYERS 未声明这些包: {unknown}")

    for src, deps in sorted(graph.items()):
        for dep in sorted(deps):
            if src not in LAYERS or dep not in LAYERS:
                continue
            if LAYERS[src] < LAYERS[dep]:
                failures.append(
                    f"分层倒置: {src}(L{LAYERS[src]}) 依赖了上层的 {dep}(L{LAYERS[dep]})")

    cycles = find_cycles(graph)
    for cycle in cycles:
        failures.append(f"循环依赖: {' <-> '.join(cycle)}")

    print()
    if failures:
        for item in failures:
            print(f"[FAIL] {item}")
        print(f"\n依赖检查失败，{len(failures)} 项问题")
        return 1

    print(f"[PASS] 无循环依赖，共 {len(graph)} 个包")
    print(f"[PASS] 分层单向，最深 {max(LAYERS[p] for p in graph)} 层")
    print("\n通过 2/2")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
