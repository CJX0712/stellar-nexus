"""归因校验单测。关键不变量：校验器必须能被篡改测试打穿。作者: 晨星"""

from __future__ import annotations

from stellarnx.core.types import Hit
from stellarnx.verify import EvidenceVerifier, extract_citations, split_claims

EVIDENCE = [
    Hit(chunk_id="e1", doc_id="d", score=1.0,
        text="星枢系统于 2026 年发布，包含检索与校验两个模块。"),
]


def test_split_claims_strips_citations():
    claims = split_claims("星枢系统于 2026 年发布。 [1] 校验模块独立可测。 [2]")
    assert len(claims) == 2
    assert "[" not in claims[0].text


def test_extract_citations_order_and_dedup():
    assert extract_citations("见 [2] 与 [1] 以及 [2]") == [2, 1]


def test_grounded_answer_passes():
    report = EvidenceVerifier().verify("星枢系统于 2026 年发布。 [1]", EVIDENCE, 0.6)
    assert report.passed
    assert report.groundedness >= 0.99


def test_fabricated_numbers_are_rejected():
    report = EvidenceVerifier().verify(
        "星枢系统于 1999 年发布，共 42 个模块。 [1]", EVIDENCE, 0.6)
    assert report.groundedness < 0.5
    assert not report.passed


def test_empty_answer_scores_zero():
    report = EvidenceVerifier().verify("", EVIDENCE, 0.6)
    assert report.groundedness == 0.0
    assert not report.passed


def test_unrelated_answer_scores_low():
    report = EvidenceVerifier().verify(
        "今天天气不错，适合出门散步看海。 [1]", EVIDENCE, 0.6)
    assert report.groundedness < 0.5


def test_evidence_id_is_recorded():
    report = EvidenceVerifier().verify("星枢系统于 2026 年发布。 [1]", EVIDENCE, 0.6)
    assert report.claims[0].evidence_id == "e1"
