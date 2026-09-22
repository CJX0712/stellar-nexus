"""路由层单测。核心不变量：四类查询落到四条不同链路。作者: 晨星"""

from __future__ import annotations

from stellarnx.core.types import RoutePath
from stellarnx.route import ComplexityRouter, extract_features


def test_greeting_goes_direct():
    assert ComplexityRouter().decide("你好", True).path is RoutePath.DIRECT


def test_empty_corpus_goes_direct():
    assert ComplexityRouter().decide("向量数据库是什么", False).path is RoutePath.DIRECT


def test_standard_question_goes_rag():
    assert ComplexityRouter().decide("向量数据库是什么", True).path is RoutePath.RAG


def test_calculation_goes_agent():
    assert ComplexityRouter().decide("计算 12 * 15 等于多少", True).path is RoutePath.AGENT


def test_comparison_goes_multihop():
    decision = ComplexityRouter().decide("比较向量数据库和关键词检索的区别", True)
    assert decision.path is RoutePath.MULTIHOP


def test_decision_carries_reason_and_features():
    decision = ComplexityRouter().decide("向量数据库是什么？", True)
    assert decision.reason
    assert decision.features["corpus_ready"] is True
    assert decision.features["has_question_mark"] is True
    assert extract_features("向量数据库是什么", True)["has_question_mark"] is False


def test_extract_features_is_pure():
    a = extract_features("你好", True)
    b = extract_features("你好", True)
    assert a == b
    assert a["greeting"] is True
