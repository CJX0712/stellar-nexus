"""自适应推理路由。

动机：所有查询都走"检索+重排+生成"是浪费 —— 打招呼也要建向量、算重排。
路由器用可解释的词面特征做分派，输出 RouteDecision（含理由与置信度），
全程可审计，不靠黑盒打分。

作者: 晨星
"""

from __future__ import annotations

import re
from typing import Any

from stellarnx.core.types import RouteDecision, RoutePath

_GREETING = ("你好", "您好", "hi", "hello", "谢谢", "再见", "在吗", "是谁", "你能做什么")
# 算式：数字 运算符 数字。只有这种形态才说明用户真的想算
_CALC_EXPR = re.compile(r"\d+\s*[\+\-\*\/×÷]\s*\d+")
# 计算类关键词需搭配数字才成立，否则「计算的复杂度」「加权求和」这类概念题会被误判
_CALC_KEYWORD = re.compile(r"计算|求和|平均|总计|换算|等于|汇率")
# 概念类疑问词：出现即说明是问原理，不是要算结果
_CONCEPT_HINT = re.compile(r"为什么|为何|如何|怎样|原因|是什么|什么区别|负责什么")
_COMPARE_HINT = ("比较", "对比", "区别", "差异", "分别", "哪个", "谁更", "优劣")
_MULTIHOP_HINT = ("并且", "同时", "以及", "然后", "再", "两者", "各自")
_AGGREGATE_HINT = ("总结", "概括", "归纳", "梳理", "列出", "有哪些")


def extract_features(query: str, corpus_ready: bool = True) -> dict[str, Any]:
    """提取可解释的路由特征。纯函数，便于单测逐项断言。"""
    text = (query or "").strip()
    lowered = text.lower()
    return {
        "length": len(text),
        "has_question_mark": "?" in text or "？" in text,
        "corpus_ready": corpus_ready,
        "greeting": any(g in lowered for g in _GREETING) and len(text) <= 12,
        "calc_hint": _is_calculation(text),
        "compare_hint": any(k in text for k in _COMPARE_HINT),
        "multihop_hint": any(k in text for k in _MULTIHOP_HINT),
        "aggregate_hint": any(k in text for k in _AGGREGATE_HINT),
    }


def _is_calculation(text: str) -> bool:
    """判定是否为真实计算意图。

    踩过的坑：早期版本只要出现数字或「计算/求和」就判为计算，
    结果「BM25 负责什么任务」「加权求和为什么更好」全被丢给智能体，
    导致这些用例走不上检索链路。现在要求算式形态或关键词+数字，
    且显式排除概念类疑问词。
    """
    if _CALC_EXPR.search(text):
        return True
    if _CONCEPT_HINT.search(text):
        return False
    return bool(_CALC_KEYWORD.search(text)) and bool(re.search(r"\d", text))


class ComplexityRouter:
    """规则路由器。唯一职责：query -> RouteDecision。"""

    name = "complexity-router"

    def decide(self, query: str, corpus_ready: bool = True) -> RouteDecision:
        f = extract_features(query, corpus_ready)

        if not f["corpus_ready"]:
            return RouteDecision(RoutePath.DIRECT, "知识库为空，直接生成", 0.9, f)
        if f["greeting"]:
            return RouteDecision(RoutePath.DIRECT, "寒暄类查询，无需检索", 0.95, f)
        if f["calc_hint"] and not f["aggregate_hint"]:
            return RouteDecision(
                RoutePath.AGENT, "含计算/换算意图，交智能体调用工具", 0.8, f)
        if f["compare_hint"] or (f["multihop_hint"] and f["length"] > 12):
            return RouteDecision(
                RoutePath.MULTIHOP, "含比较/多实体意图，走多跳检索", 0.75, f)
        if f["aggregate_hint"]:
            return RouteDecision(RoutePath.MULTIHOP, "汇总类问题需更宽召回", 0.7, f)
        return RouteDecision(RoutePath.RAG, "标准知识问答，走单跳检索增强", 0.85, f)
