"""评测层：黄金集 + 指标 + 门禁。

铁律：评测永远在**全新管道**上运行（新建内存库、重新灌语料），
绝不复用运行期索引 —— 否则残留文档会让指标静默失真。

作者: 晨星
"""

from stellarnx.eval.dataset import EvalCase, load_cases, load_corpus
from stellarnx.eval.metrics import mrr_at_k, ndcg_at_k, percentile, recall_at_k
from stellarnx.eval.runner import EvalReport, EvalRunner

__all__ = [
    "EvalCase", "EvalReport", "EvalRunner", "load_cases", "load_corpus",
    "mrr_at_k", "ndcg_at_k", "percentile", "recall_at_k",
]
