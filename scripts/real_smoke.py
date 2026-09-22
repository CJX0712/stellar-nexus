"""真实模型冒烟：Ollama bge-m3 嵌入 + qwen2.5:1.5b 生成 + LLM 列表式重排。

与离线冒烟的区别是它走真实推理，因此只在人工验证或本地执行时运行，
CI 门禁不依赖它（CI 跑离线路径）。

作者: 晨星
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stellarnx.core.config import load_config  # noqa: E402
from stellarnx.eval.dataset import load_cases, load_corpus  # noqa: E402
from stellarnx.eval.runner import EvalRunner  # noqa: E402
from stellarnx.pipeline.orchestrator import NexusSystem  # noqa: E402

QUESTIONS = [
    "星枢系统默认采用什么向量索引？",
    "为什么混合检索要用 RRF 而不是直接加权求和？",
    "索引库装不上时系统还能用吗？",
    "计算 128 * 4 等于多少",
    "你好",
]


def main() -> int:
    cfg = load_config(embed_backend="ollama", llm_backend="ollama", rerank_backend="llm")
    system = NexusSystem(cfg, ":memory:")
    print(f"后端: 嵌入={system.embedder.name} 生成={system.llm.name} "
          f"重排={system.reranker.name} 向量={system.index.name}")

    corpus = load_corpus(ROOT / "data/golden/corpus.jsonl")
    started = time.perf_counter()
    chunks = system.ingest_many(corpus)
    print(f"入库: {chunks} 片段, {(time.perf_counter() - started) * 1000:.0f} ms")
    print(f"向量维度: {system.embedder.dim}")

    for question in QUESTIONS:
        started = time.perf_counter()
        answer = system.answer(question)
        elapsed = (time.perf_counter() - started) * 1000
        grounded = answer.verification.groundedness if answer.verification else None
        print(f"\n问: {question}")
        print(f"  路由: {answer.route.path.value} ({answer.route.reason}) "
              f"耗时: {elapsed:.0f} ms")
        print(f"  答: {answer.text[:220]}")
        print(f"  引用: {len(answer.citations)} 条, 可证性: {grounded}")

    print("\n== 真实模型评测（子集） ==")
    cases = load_cases(ROOT / "data/golden/questions.jsonl")[:8]
    # 真实模型在 CPU 上单请求数十秒，延迟阈值不适用于此场景，只卡质量指标
    thresholds = {"recall@5": 0.75, "groundedness": 0.55, "latency_p95_ms": 600000.0}
    report = EvalRunner(cfg, thresholds).run(corpus, cases)
    print({k: round(v, 4) for k, v in report.metrics.items()})
    print("达标:", report.passed, report.failures)
    return 0 if report.metrics["recall@5"] >= 0.75 else 1


if __name__ == "__main__":
    raise SystemExit(main())
