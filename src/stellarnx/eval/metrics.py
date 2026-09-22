"""检索与生成指标。纯函数，无副作用，全部单测覆盖。

作者: 晨星
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def recall_at_k(retrieved: Sequence[str], gold: Sequence[str], k: int) -> float:
    """前 k 个结果命中任一 gold 的比例（gold 非空时）。"""
    if not gold:
        return 0.0
    top = set(list(retrieved)[:k])
    hits = len(top & set(gold))
    return hits / len(set(gold))


def mrr_at_k(retrieved: Sequence[str], gold: Sequence[str], k: int) -> float:
    """首个命中结果的排名倒数。未命中为 0。"""
    gold_set = set(gold)
    for rank, item in enumerate(list(retrieved)[:k], start=1):
        if item in gold_set:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: Sequence[str], gold: Sequence[str], k: int) -> float:
    """二值相关性 nDCG@k。"""
    gold_set = set(gold)
    dcg = 0.0
    for rank, item in enumerate(list(retrieved)[:k], start=1):
        if item in gold_set:
            dcg += 1.0 / math.log2(rank + 1)
    ideal_hits = min(len(gold_set), k)
    idcg = sum(1.0 / math.log2(r + 1) for r in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def percentile(values: Sequence[float], p: float) -> float:
    """线性插值分位数。空输入返回 0。"""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * (p / 100.0)
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return float(ordered[low])
    return float(ordered[low] + (ordered[high] - ordered[low]) * (position - low))


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0
