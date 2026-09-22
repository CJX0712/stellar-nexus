"""共享线性代数工具。放在 core 层，避免 embed 子模块反向依赖包 __init__ 造成循环导入。

作者: 晨星
"""

from __future__ import annotations

import numpy as np


def normalize(matrix: np.ndarray) -> np.ndarray:
    """行向量 L2 归一化。全零行保持全零，不产生 NaN。"""
    raw = np.asarray(matrix, dtype="float64")
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return (raw / norms).astype("float32")


def cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """两组向量的余弦相似度矩阵，shape=(len(a), len(b))。"""
    return normalize(a) @ normalize(b).T
