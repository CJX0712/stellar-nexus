"""向量索引。FAISS 为主，numpy 暴力为回退。

回退不是静默的：`backend_name` 与 `_fallback_reason` 可被断言，
自检里必须显式检查真实后端生效，避免"降级了却以为在用 FAISS"。

作者: 晨星
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

try:  # pragma: no cover - 导入成功与否取决于环境
    import faiss

    _FAISS_IMPORT_ERROR: str | None = None
except Exception as exc:
    faiss = None  # type: ignore[assignment]
    _FAISS_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"


class InMemoryIndex:
    """numpy 暴力余弦索引。零依赖，结果精确，作为 FAISS 的等价回退。"""

    def __init__(self, dim: int) -> None:
        self.name = "numpy"
        self.dim = dim
        self._ids: list[str] = []
        self._matrix = np.zeros((0, dim), dtype="float32")
        self._fallback_reason: str | None = None

    def add(self, ids: list[str], vectors: np.ndarray) -> None:
        if not ids:
            return
        matrix = np.asarray(vectors, dtype="float32")
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        if matrix.shape[1] != self.dim:
            raise ValueError(f"向量维度 {matrix.shape[1]} 与索引维度 {self.dim} 不一致")
        if self._matrix.shape[0] == 0:
            self._matrix = matrix
        else:
            self._matrix = np.vstack([self._matrix, matrix])
        self._ids.extend(ids)

    def search(self, vectors: np.ndarray, k: int) -> list[list[tuple[str, float]]]:
        if not self._ids:
            return [[] for _ in range(len(vectors))]
        matrix = np.asarray(vectors, dtype="float32")
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        scores = matrix @ self._matrix.T
        return [self._row_to_pairs(row, k) for row in scores]

    def _row_to_pairs(self, row: np.ndarray, k: int) -> list[tuple[str, float]]:
        top = min(k, len(self._ids))
        order = np.argsort(-row)[:top]
        return [(self._ids[i], float(row[i])) for i in order]

    def delete(self, ids: list[str]) -> None:
        keep = [i for i, cid in enumerate(self._ids) if cid not in set(ids)]
        self._ids = [self._ids[i] for i in keep]
        self._matrix = self._matrix[keep] if keep else np.zeros((0, self.dim), "float32")

    def clear(self) -> None:
        self._ids = []
        self._matrix = np.zeros((0, self.dim), dtype="float32")

    def count(self) -> int:
        return len(self._ids)


class FaissIndex:
    """FAISS 内积索引（向量已归一化，内积即余弦）。"""

    def __init__(self, dim: int) -> None:
        if faiss is None:
            raise RuntimeError(f"faiss 不可用: {_FAISS_IMPORT_ERROR}")
        self.name = "faiss"
        self.dim = dim
        self._index = faiss.IndexFlatIP(dim)
        self._ids: list[str] = []
        self._fallback_reason: str | None = None

    def add(self, ids: list[str], vectors: np.ndarray) -> None:
        if not ids:
            return
        matrix = np.ascontiguousarray(np.asarray(vectors, dtype="float32"))
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        if matrix.shape[1] != self.dim:
            raise ValueError(f"向量维度 {matrix.shape[1]} 与索引维度 {self.dim} 不一致")
        self._index.add(matrix)
        self._ids.extend(ids)

    def search(self, vectors: np.ndarray, k: int) -> list[list[tuple[str, float]]]:
        if not self._ids:
            return [[] for _ in range(len(vectors))]
        matrix = np.ascontiguousarray(np.asarray(vectors, dtype="float32"))
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        scores, indices = self._index.search(matrix, min(k, len(self._ids)))
        out: list[list[tuple[str, float]]] = []
        for row_idx, row_ids in enumerate(indices):
            pairs: list[tuple[str, float]] = []
            for col, idx in enumerate(row_ids):
                if idx < 0 or idx >= len(self._ids):
                    continue
                pairs.append((self._ids[idx], float(scores[row_idx][col])))
            out.append(pairs)
        return out

    def delete(self, ids: list[str]) -> None:
        """FAISS 的 IndexFlatIP 不支持原地删除，按需重建。"""
        target = set(ids)
        if not target:
            return
        keep = [i for i, cid in enumerate(self._ids) if cid not in target]
        if len(keep) == len(self._ids):
            return
        # 必须在 reset 之前把保留向量取出来，否则位置已失效
        kept = np.vstack([self._reconstruct(i) for i in keep]).astype("float32") if keep \
            else np.zeros((0, self.dim), dtype="float32")
        self._ids = [self._ids[i] for i in keep]
        self._index.reset()
        if keep:
            self._index.add(np.ascontiguousarray(kept))

    def _reconstruct(self, position: int) -> np.ndarray:
        return self._index.reconstruct(position)

    def clear(self) -> None:
        self._index.reset()
        self._ids = []

    def count(self) -> int:
        return int(self._index.ntotal)


def build_index(dim: int, prefer: str = "faiss") -> InMemoryIndex | FaissIndex:
    """构造向量索引。FAISS 不可用时回退 numpy，并把原因记录在对象上。"""
    if prefer == "faiss" and faiss is not None:
        try:
            return FaissIndex(dim)
        except Exception as exc:
            logger.warning("FAISS 初始化失败，回退 numpy: %s", exc)
    index = InMemoryIndex(dim)
    index._fallback_reason = _FAISS_IMPORT_ERROR or f"prefer={prefer}"
    return index
