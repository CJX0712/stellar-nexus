"""LLM 列表式重排（RankGPT 风格）。

场景：本机没有交叉编码器权重（ModelScope 上 bge-reranker 只发 PyTorch 权重，
本机无 torch），但已有可用的生成模型。列表式重排让模型直接对候选编号排序，
是交叉编码器缺失时性价比最高的替代。

工程约束：解析失败一律保持原顺序，绝不因重排失败就把候选顺序打乱。

作者: 晨星
"""

from __future__ import annotations

import json
import logging
import re

from stellarnx.core.contracts import LLM
from stellarnx.core.types import Hit

logger = logging.getLogger(__name__)
_JSON_ARRAY = re.compile(r"\[[^\]]*\]")


class LLMReranker:
    """满足 Reranker 协议的 LLM 列表式实现。"""

    name = "llm-listwise"

    def __init__(self, llm: LLM, window: int = 8) -> None:
        self.llm = llm
        self.window = window

    def rerank(self, query: str, hits: list[Hit], k: int) -> list[Hit]:
        if len(hits) <= 1:
            return hits[:k]
        candidates = hits[: self.window]
        order = self._ask(query, candidates)
        if order is None:
            logger.warning("LLM 重排解析失败，保持原顺序")
            return hits[:k]
        ranked: list[Hit] = []
        for position, index in enumerate(order):
            if 0 <= index < len(candidates):
                ranked.append(
                    Hit(
                        chunk_id=candidates[index].chunk_id,
                        doc_id=candidates[index].doc_id,
                        text=candidates[index].text,
                        score=float(len(candidates) - position),
                        source=f"{candidates[index].source}+llm",
                        meta=dict(candidates[index].meta),
                    )
                )
        # 模型可能漏掉部分编号，补齐未提及的候选，保证不丢内容
        seen = {h.chunk_id for h in ranked}
        ranked.extend(h for h in candidates if h.chunk_id not in seen)
        return ranked[:k]

    def _ask(self, query: str, candidates: list[Hit]) -> list[int] | None:
        blocks = "\n".join(
            f"[{i}] {h.text[:220]}" for i, h in enumerate(candidates))
        prompt = (
            "下面是检索到的候选片段。请按它们与问题的相关程度从高到低排序，"
            "只输出编号的 JSON 数组，例如 [2, 0, 1]，不要输出任何解释。\n\n"
            f"问题：{query}\n\n候选：\n{blocks}\n\n排序："
        )
        try:
            raw = self.llm.complete([{"role": "user", "content": prompt}])
        except Exception as exc:
            logger.warning("LLM 重排调用失败: %s", exc)
            return None
        data = self._parse_json(raw)
        if data is None:
            # 小模型经常不给 JSON，退化成"按出现顺序抽取编号"，总比放弃重排强
            data = self._parse_loose(raw, len(candidates))
        if data is None:
            return None
        return data

    @staticmethod
    def _parse_json(raw: str) -> list[int] | None:
        match = _JSON_ARRAY.search(raw)
        if not match:
            return None
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
        if not isinstance(data, list) or not all(isinstance(x, int) for x in data):
            return None
        return list(data)

    @staticmethod
    def _parse_loose(raw: str, size: int) -> list[int] | None:
        numbers: list[int] = []
        for token in re.findall(r"\d+", raw):
            value = int(token)
            if 0 <= value < size and value not in numbers:
                numbers.append(value)
        return numbers or None
