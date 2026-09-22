"""可证性校验。

设计原则：**校验器必须能被反向测试打穿**。
把答案里的事实篡改掉，groundedness 必须显著下降；否则这个校验器是摆设。
因此除了词汇覆盖，还加了数字/实体一致性惩罚 —— 篡改往往体现在数字上。

作者: 晨星
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from stellarnx.core.contracts import LLM
from stellarnx.core.text import tokenize
from stellarnx.core.types import Claim, Hit, VerifyReport

_CITATION = re.compile(r"\[(\d+)\]")
_SENTENCE_SPLIT = re.compile(r"(?<=[。！？；!?\n])")
_NUMBER = re.compile(r"\d+(?:\.\d+)?")
_MIN_CLAIM_LEN = 2


def extract_citations(text: str) -> list[int]:
    """抽取文本中的引用编号，按出现顺序去重。"""
    seen: list[int] = []
    for m in _CITATION.finditer(text):
        value = int(m.group(1))
        if value not in seen:
            seen.append(value)
    return seen


def split_claims(answer: str) -> list[Claim]:
    """把答案拆成断言。只保留有实质内容的句子，并剥离引用标记。"""
    claims: list[Claim] = []
    for raw in _SENTENCE_SPLIT.split(answer or ""):
        sentence = _CITATION.sub("", raw).strip()
        if len(sentence) < _MIN_CLAIM_LEN:
            continue
        cites = extract_citations(raw)
        claims.append(
            Claim(text=sentence, supported=False, score=0.0,
                  evidence_id=str(cites[0]) if cites else None)
        )
    return claims


@dataclass
class _Idf:
    """轻量 IDF 表，让"的/了/是"这类高频词不主导覆盖度。"""

    df: dict[str, int]
    n: int

    def weight(self, token: str) -> float:
        count = self.df.get(token, 0)
        if count == 0:
            return 2.0
        return 1.0 + (self.n / (1.0 + count)) ** 0.5


def _build_idf(texts: list[str]) -> _Idf:
    df: dict[str, int] = {}
    for text in texts:
        for token in set(tokenize(text)):
            df[token] = df.get(token, 0) + 1
    return _Idf(df=df, n=max(1, len(texts)))


class EvidenceVerifier:
    """零依赖证据校验器：IDF 加权覆盖度 + 数字一致性惩罚。"""

    name = "evidence"

    def __init__(self, unsupported_penalty: float = 0.35) -> None:
        self.unsupported_penalty = unsupported_penalty

    def verify(self, answer: str, evidence: list[Hit], threshold: float = 0.6) -> VerifyReport:
        claims = split_claims(answer)
        if not claims:
            # 空答案不算"可证"，groundedness 必须为 0 —— 否则空输出会被误判为满分
            return VerifyReport(
                groundedness=0.0, claims=[], threshold=threshold,
                passed=threshold <= 0.0,
                detail={"reason": "no_claims", "evidence_count": len(evidence)},
            )

        idf = _build_idf([h.text for h in evidence] + [answer])
        evidence_tokens = [set(tokenize(h.text)) for h in evidence]
        evidence_numbers = [set(_NUMBER.findall(h.text)) for h in evidence]

        for claim in claims:
            claim_tokens = set(tokenize(claim.text))
            claim_numbers = set(_NUMBER.findall(claim.text))
            best = 0.0
            best_idx: str | None = None
            for pos, tokens in enumerate(evidence_tokens):
                overlap = claim_tokens & tokens
                if not overlap:
                    continue
                coverage = sum(idf.weight(t) for t in overlap) / max(
                    1e-9, sum(idf.weight(t) for t in claim_tokens))
                if claim_numbers:
                    # 数字必须能在证据里找到，否则视为编造
                    missing = claim_numbers - evidence_numbers[pos]
                    if missing:
                        coverage *= self.unsupported_penalty
                if coverage > best:
                    best = coverage
                    best_idx = evidence[pos].chunk_id
            claim.score = round(float(best), 6)
            claim.supported = best >= threshold
            if best_idx is not None:
                claim.evidence_id = best_idx

        groundedness = sum(1 for c in claims if c.supported) / len(claims)
        return VerifyReport(
            groundedness=round(groundedness, 6),
            claims=claims,
            threshold=threshold,
            passed=groundedness >= threshold,
            detail={
                "claims": len(claims),
                "supported": sum(1 for c in claims if c.supported),
                "evidence_count": len(evidence),
            },
        )


class LLMVerifier:
    """LLM 判定校验器：让模型逐条判定断言是否被支持。

    用于生产环境（质量更高）；离线环境用 EvidenceVerifier。
    判定失败时保守降级为"不支持"，绝不因为解析失败就放行。
    """

    name = "llm"

    def __init__(self, llm: LLM) -> None:
        self.llm = llm

    def verify(self, answer: str, evidence: list[Hit], threshold: float = 0.6) -> VerifyReport:
        claims = split_claims(answer)
        if not claims:
            return VerifyReport(0.0, [], threshold, threshold <= 0.0,
                                {"reason": "no_claims"})
        blocks = "\n".join(f"[{i + 1}] {h.text}" for i, h in enumerate(evidence))
        numbered = "\n".join(f"{i + 1}. {c.text}" for i, c in enumerate(claims))
        prompt = (
            "下面是资料与待判定的断言。逐条判定断言是否被资料支持，"
            "只输出 JSON 数组，元素形如 {\"i\": 1, \"supported\": true}。\n\n"
            f"资料：\n{blocks}\n\n断言：\n{numbered}\n\nJSON："
        )
        try:
            raw = self.llm.complete([{"role": "user", "content": prompt}])
            verdict = self._parse(raw, len(claims))
        except Exception:
            verdict = [False] * len(claims)
        # 模型可能少给或多给判定，长度不符时一律按「未支持」处理。
        # 宁可把不确定的当无据，也不要因为数量错位把判定张冠李戴。
        if len(verdict) != len(claims):
            verdict = [False] * len(claims)

        for claim, supported in zip(claims, verdict, strict=True):
            claim.supported = supported
            claim.score = 1.0 if supported else 0.0
        groundedness = sum(verdict) / len(claims)
        return VerifyReport(
            groundedness=round(groundedness, 6), claims=claims, threshold=threshold,
            passed=groundedness >= threshold,
            detail={"mode": "llm", "claims": len(claims)},
        )

    @staticmethod
    def _parse(raw: str, expected: int) -> list[bool]:
        import json
        import re

        match = re.search(r"\[.*\]", raw, re.S)
        if not match:
            return [False] * expected
        data = json.loads(match.group(0))
        verdict = [False] * expected
        for item in data:
            idx = int(item.get("i", 0)) - 1
            if 0 <= idx < expected:
                verdict[idx] = bool(item.get("supported", False))
        return verdict
