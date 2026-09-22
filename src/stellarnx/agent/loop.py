"""Planner - Executor - Critic 闭环。

Planner 产出步骤（不执行），Executor 执行（不决策），Critic 判定（不执行）。
三者互不越界，因此可分别单测；Critic 不通过时触发重规划，轮次有硬上限。

作者: 晨星
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from stellarnx.agent.tools import ToolRegistry
from stellarnx.core.text import tokenize
from stellarnx.core.types import Hit

_EXPR = re.compile(r"[\d\.\s\+\-\*/×÷（）()]+")


@dataclass
class Step:
    """计划中的一步。"""

    tool: str
    args: dict[str, Any]
    note: str = ""


@dataclass
class AgentResult:
    """智能体最终产物。"""

    goal: str
    answer: str
    steps: list[dict[str, Any]] = field(default_factory=list)
    observations: list[str] = field(default_factory=list)
    critic_scores: list[float] = field(default_factory=list)
    replans: int = 0
    evidence: list[Hit] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal": self.goal,
            "answer": self.answer,
            "steps": self.steps,
            "observations": self.observations,
            "critic_scores": [round(s, 4) for s in self.critic_scores],
            "replans": self.replans,
        }


class HeuristicPlanner:
    """启发式规划器。离线可用；规则可解释、可单测。

    真实场景可替换为 LLMPlanner —— 只要产出的 Step 列表结构一致即可。
    """

    name = "heuristic-planner"

    def plan(self, goal: str, history: list[str], budget: int) -> list[Step]:
        steps: list[Step] = []
        if _EXPR.search(goal) and any(op in goal for op in "+-*/×÷"):
            expression = self._extract_expression(goal)
            if expression:
                steps.append(Step("calculator", {"expression": expression}, "解析算式"))
        if not history or "kb_search" not in " ".join(history):
            steps.append(Step("kb_search", {"query": goal, "top_k": 3}, "检索相关知识"))
        if len(history) >= 1 and not steps:
            steps.append(Step("text_stats", {"text": goal}, "补充统计信息"))
        return steps[:budget]

    @staticmethod
    def _extract_expression(goal: str) -> str:
        matches = [m.group().strip() for m in _EXPR.finditer(goal)]
        candidates = [m for m in matches if any(op in m for op in "+-*/×÷")]
        if not candidates:
            return ""
        expr = max(candidates, key=len)
        return expr.replace("×", "*").replace("÷", "/").replace("（", "(").replace("）", ")")


class Executor:
    """执行器。只执行，不判断要不要执行。"""

    name = "executor"

    def __init__(self, registry: ToolRegistry) -> None:
        self.registry = registry

    def run(self, step: Step) -> dict[str, Any]:
        result = self.registry.run(step.tool, **step.args)
        result["tool"] = step.tool
        result["note"] = step.note
        return result


class Critic:
    """批判器。只打分，不改结果。

    判据：目标词在观测文本中的 IDF 覆盖度 + 工具是否全部成功。
    """

    name = "critic"

    def __init__(self, accept_threshold: float = 0.35) -> None:
        self.accept_threshold = accept_threshold

    def score(self, goal: str, observations: list[str], results: list[dict[str, Any]]) -> float:
        if not results:
            return 0.0
        goal_tokens = set(tokenize(goal))
        if not goal_tokens:
            return 1.0
        observation_tokens: set[str] = set()
        for text in observations:
            observation_tokens |= set(tokenize(text))
        coverage = len(goal_tokens & observation_tokens) / len(goal_tokens)
        success = sum(1 for r in results if r.get("ok")) / len(results)
        return round(min(1.0, 0.6 * coverage + 0.4 * success), 6)


class AgentLoop:
    """智能体循环：规划 -> 执行 -> 批判 -> 必要时重规划。"""

    def __init__(
        self,
        registry: ToolRegistry,
        planner: HeuristicPlanner | None = None,
        max_steps: int = 4,
        max_replan: int = 2,
        accept_threshold: float = 0.35,
    ) -> None:
        self.registry = registry
        self.planner = planner or HeuristicPlanner()
        self.executor = Executor(registry)
        self.critic = Critic(accept_threshold)
        self.max_steps = max_steps
        self.max_replan = max_replan
        self.accept_threshold = accept_threshold

    def run(self, goal: str) -> AgentResult:
        result = AgentResult(goal=goal, answer="")
        history: list[str] = []
        observations: list[str] = []
        results: list[dict[str, Any]] = []

        for attempt in range(self.max_replan + 1):
            steps = self.planner.plan(goal, history, self.max_steps)
            for step in steps:
                outcome = self.executor.run(step)
                results.append(outcome)
                observation = self._stringify(outcome)
                observations.append(observation)
                history.append(step.tool)
                result.steps.append(
                    {"tool": step.tool, "args": step.args, "note": step.note,
                     "ok": bool(outcome.get("ok"))}
                )
                self._collect_evidence(outcome, result)
            score = self.critic.score(goal, observations, results)
            result.critic_scores.append(score)
            if score >= self.accept_threshold:
                break
            result.replans = attempt
            history.append("critic_reject")

        result.observations = observations
        result.answer = self._compose(goal, results)
        return result

    @staticmethod
    def _stringify(outcome: dict[str, Any]) -> str:
        if not outcome.get("ok"):
            return f"{outcome.get('tool')} 失败: {outcome.get('error')}"
        tool = outcome.get("tool")
        if tool == "calculator":
            return f"计算结果 {outcome.get('expression')} = {outcome.get('result')}"
        if tool == "kb_search":
            hits = outcome.get("hits") or []
            return "检索到: " + " | ".join(h.get("text", "")[:160] for h in hits[:3])
        if tool == "text_stats":
            return f"文本统计: {outcome.get('chars')} 字符 / {outcome.get('lines')} 行"
        return str({k: v for k, v in outcome.items() if k != "ok"})

    @staticmethod
    def _compose(goal: str, results: list[dict[str, Any]]) -> str:
        if not results:
            return "未产生有效步骤。"
        parts: list[str] = []
        for outcome in results:
            if outcome.get("ok"):
                parts.append(AgentLoop._stringify(outcome))
        return "；".join(parts) if parts else "所有步骤均失败。"

    @staticmethod
    def _collect_evidence(outcome: dict[str, Any], result: AgentResult) -> None:
        """把 kb_search 的命中提升为证据，供下游归因校验使用。"""
        if outcome.get("tool") != "kb_search" or not outcome.get("ok"):
            return
        for item in outcome.get("hits") or []:
            result.evidence.append(
                Hit(
                    chunk_id=item.get("chunk_id", ""),
                    doc_id=item.get("doc_id", ""),
                    text=item.get("text", ""),
                    score=float(item.get("score", 0.0)),
                    source="agent",
                )
            )
