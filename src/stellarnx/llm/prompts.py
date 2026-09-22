"""提示词构造。集中管理，保证 Mock 与真实 LLM 看到完全相同的输入结构。

作者: 晨星
"""

from __future__ import annotations

from stellarnx.core.types import Hit

SYSTEM_PROMPT = (
    "你是星枢（StellarNexus）智能助手。只依据给定资料回答问题，"
    "不得使用资料之外的知识。每个事实句末尾用方括号标注引用编号，例如 [1]。"
    "若资料不足以回答，直接说明资料不足，不要编造。"
)


def build_rag_messages(question: str, passages: list[Hit]) -> list[dict[str, str]]:
    """构造 RAG 消息。编号从 1 开始，与引用标记 [n] 一一对应。"""
    blocks = "\n".join(f"[{i + 1}] {p.text}" for i, p in enumerate(passages))
    user = (
        f"资料：\n{blocks}\n\n"
        f"问题：{question}\n\n"
        f"要求：只依据上述资料作答，句末标注引用编号，如 [1][2]。"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def build_agent_messages(goal: str, observations: list[str]) -> list[dict[str, str]]:
    """构造智能体单步消息：目标 + 已观察到的工具结果。"""
    history = "\n".join(f"- {o}" for o in observations) or "（暂无）"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"目标：{goal}\n已获得的信息：\n{history}\n请给出下一步。"},
    ]
