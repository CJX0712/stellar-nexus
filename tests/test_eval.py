"""评测层单测。核心不变量：指标确定性、语料足够长、失败可被检出。作者: 晨星"""

from __future__ import annotations

from pathlib import Path

from stellarnx.eval.dataset import load_cases, load_corpus
from stellarnx.eval.metrics import mean, mrr_at_k, ndcg_at_k, percentile, recall_at_k
from stellarnx.eval.runner import EvalRunner

ROOT = Path(__file__).resolve().parents[1]
CORPUS_PATH = ROOT / "data" / "golden" / "corpus.jsonl"
QUESTIONS_PATH = ROOT / "data" / "golden" / "questions.jsonl"


def test_metrics_basics():
    retrieved = ["b", "a", "c"]
    gold = ["a"]
    assert recall_at_k(retrieved, gold, 1) == 0.0
    assert recall_at_k(retrieved, gold, 3) == 1.0
    assert mrr_at_k(retrieved, gold, 3) == 0.5
    assert 0.0 < ndcg_at_k(retrieved, gold, 3) <= 1.0


def test_metrics_handle_empty():
    assert recall_at_k([], [], 5) == 0.0
    assert mrr_at_k(["a"], [], 5) == 0.0
    assert ndcg_at_k(["a"], [], 5) == 0.0
    assert mean([]) == 0.0


def test_percentile():
    assert percentile([1.0, 2.0, 3.0], 50) == 2.0
    assert percentile([5.0], 95) == 5.0
    assert percentile([], 95) == 0.0


def test_golden_corpus_is_long_enough():
    for doc in load_corpus(CORPUS_PATH):
        assert len(doc["text"]) > 480, f"{doc['doc_id']} 太短，覆盖不到多片段路径"


def test_golden_cases_reference_existing_docs():
    docs = {d["doc_id"] for d in load_corpus(CORPUS_PATH)}
    for case in load_cases(QUESTIONS_PATH):
        assert case.gold_doc in docs


def test_evaluation_is_deterministic(offline_config):
    corpus = load_corpus(CORPUS_PATH)
    cases = load_cases(QUESTIONS_PATH)
    first = EvalRunner(offline_config).run(corpus, cases)
    second = EvalRunner(offline_config).run(corpus, cases)
    assert first.metrics["recall@5"] == second.metrics["recall@5"]
    assert first.metrics["groundedness"] == second.metrics["groundedness"]


def test_evaluation_meets_thresholds(offline_config):
    report = EvalRunner(offline_config).run(
        load_corpus(CORPUS_PATH), load_cases(QUESTIONS_PATH))
    assert report.metrics["recall@5"] >= 0.75, report.failures
    assert report.metrics["groundedness"] >= 0.55, report.failures


def test_thresholds_can_fail_run(offline_config):
    strict = {"recall@5": 1.1}
    report = EvalRunner(offline_config, strict).run(
        load_corpus(CORPUS_PATH), load_cases(QUESTIONS_PATH))
    assert not report.passed
    assert report.failures
