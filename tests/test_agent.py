"""智能体层单测。核心不变量：工具安全、闭环有产物、失败不炸、轮次有上限。作者: 晨星"""

from __future__ import annotations

import pytest

from stellarnx.agent.loop import AgentLoop, Critic, HeuristicPlanner
from stellarnx.agent.tools import ToolRegistry, build_default_tools, safe_eval


def test_safe_eval_basic():
    assert abs(safe_eval("12 * (3 + 4)") - 84.0) < 1e-9


@pytest.mark.parametrize("expression", [
    "__import__('os').system('echo hi')",
    "(1).__class__.__bases__[0].__subclasses__()",
    "open('x.txt')",
])
def test_safe_eval_blocks_escape(expression):
    # 断言具体类型：safe_eval 对任何非算术元素抛 ValueError。
    # 用 Exception 兜底会让「函数根本没实现拦截」也能通过测试。
    with pytest.raises(ValueError, match="不允许的表达式元素"):
        safe_eval(expression)


def test_registry_reports_unknown_tool():
    result = build_default_tools().run("not_exist")
    assert result["ok"] is False


def test_calculator_tool():
    result = build_default_tools().run("calculator", expression="7 * 8")
    assert result["ok"] and result["result"] == 56.0


def test_agent_loop_produces_answer(system):
    system.ingest_text("doc", "本手册说明向量数据库的部署方式与参数配置。" * 10)
    result = system.agent.run("计算 7 * 8")
    assert len(result.steps) >= 1
    assert "56" in result.answer


def test_agent_loop_bounded_replans(system):
    loop = AgentLoop(build_default_tools(None), max_steps=1, max_replan=2)
    result = loop.run("一个完全无法命中的目标描述")
    assert result.replans <= 2
    assert len(result.steps) <= 3


def test_critic_scores_zero_without_results():
    assert Critic().score("目标", [], []) == 0.0


def test_planner_emits_calculation_step():
    steps = HeuristicPlanner().plan("计算 3 + 5 等于多少", [], 4)
    assert any(s.tool == "calculator" for s in steps)


def test_tool_failure_does_not_crash_loop():
    registry = ToolRegistry()
    registry.register("boom", "总是失败", lambda: 1 / 0)
    loop = AgentLoop(registry, max_steps=1, max_replan=0)
    result = loop.run("触发失败")
    assert result.steps and result.steps[0]["ok"] is False
