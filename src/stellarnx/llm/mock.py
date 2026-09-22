"""零依赖 Mock LLM：离线与 CI 的支点。

它不是"返回固定字符串"的占位，而是真正的**抽取式回答器**：
从资料中挑出与问题重叠度最高的句子拼成答案并带引用编号。
因此离线环境下，检索、重排、归因校验三段依然是真实验证，而非空跑。

作者: 晨星
"""

from __future__ import annotations

import re

from stellarnx.core.text import tokenize
from stellarnx.llm.prompts import SYSTEM_PROMPT

_PASSAGE = re.compile(r"^\[(\d+)\]\s*(.*)$")
_QUESTION = re.compile(r"^问题：(.*)$", re.M)
_SENTENCE = re.compile(r"(?<=[。！？；!?])")


class MockLLM:
    """满足 LLM 协议的确定性实现。同样输入永远同样输出。"""

    name = "mock"

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, messages: list[dict[str, str]], **kwargs: object) -> str:
        self.calls += 1
        content = "\n".join(m.get("content", "") for m in messages)
        passages = self._parse_passages(content)
        question = self._parse_question(content)
        if not passages:
            return "资料不足，无法回答。"
        ranked = self._rank(passages, question)
        top_idx, top_sentences = ranked[0]
        parts = [f"{s} [{top_idx}]" for s in top_sentences[:2]]
        if len(ranked) > 1:
            second_idx, second_sentences = ranked[1]
            if second_sentences:
                parts.append(f"{second_sentences[0]} [{second_idx}]")
        return "".join(parts)

    def is_available(self) -> bool:
        return True

    # --- 内部 ---
    @staticmethod
    def _parse_passages(content: str) -> dict[int, str]:
        out: dict[int, str] = {}
        for line in content.splitlines():
            m = _PASSAGE.match(line.strip())
            if m:
                out[int(m.group(1))] = m.group(2).strip()
        return out

    @staticmethod
    def _parse_question(content: str) -> str:
        m = _QUESTION.search(content)
        if m:
            return m.group(1).strip()
        # 退化：取最后一行非空内容
        lines = [line.strip() for line in content.splitlines() if line.strip()]
        return lines[-1] if lines else ""

    @staticmethod
    def _rank(passages: dict[int, str], question: str) -> list[tuple[int, list[str]]]:
        q_tokens = set(tokenize(question))
        scored: list[tuple[float, int, list[str]]] = []
        for idx, text in passages.items():
            sentences = [s.strip() for s in _SENTENCE.split(text) if s.strip()]
            best = 0.0
            for sentence in sentences:
                overlap = len(q_tokens & set(tokenize(sentence)))
                best = max(best, overlap / max(1, len(q_tokens)))
            scored.append((best, idx, sentences))
        scored.sort(key=lambda x: (-x[0], x[1]))
        return [(idx, sentences) for _, idx, sentences in scored]

    @property
    def system_prompt(self) -> str:
        return SYSTEM_PROMPT
