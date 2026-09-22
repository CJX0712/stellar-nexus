"""语义感知切分：段落优先，超长按句再按窗，带重叠。

中英混排适配：句子边界同时识别中文句号与西文句号，不依赖空格分词。

作者: 晨星
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from stellarnx.core.types import Chunk

_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n|\n(?=#{1,6}\s)")
_SENTENCE_SPLIT = re.compile(r"(?<=[。！？；!?;])\s*|(?<=[.。])\s+")


def split_sentences(text: str) -> list[str]:
    """按句中英混排切句。无标点时整体返回。"""
    parts = [p.strip() for p in _SENTENCE_SPLIT.split(text) if p and p.strip()]
    return parts or ([text.strip()] if text.strip() else [])


def _hard_window(text: str, size: int, overlap: int) -> list[str]:
    """纯字符窗切分，作为最后兜底。"""
    out: list[str] = []
    step = max(1, size - overlap)
    for start in range(0, len(text), step):
        piece = text[start : start + size].strip()
        if piece:
            out.append(piece)
        if start + size >= len(text):
            break
    return out


def split_text(text: str, size: int = 480, overlap: int = 80) -> list[str]:
    """把整篇文本切成若干不超过 size 的片段，尽量在段落/句子边界断开。"""
    if not text or not text.strip():
        return []
    if len(text) <= size:
        return [text.strip()]

    chunks: list[str] = []
    buffer = ""
    for para in _PARAGRAPH_SPLIT.split(text):
        para = para.strip()
        if not para:
            continue
        if len(para) > size:
            if buffer:
                chunks.append(buffer.strip())
                buffer = ""
            for sentence in split_sentences(para):
                if len(sentence) > size:
                    chunks.extend(_hard_window(sentence, size, overlap))
                    continue
                if len(buffer) + len(sentence) + 1 > size:
                    chunks.append(buffer.strip())
                    buffer = sentence
                else:
                    buffer = f"{buffer} {sentence}".strip()
            continue
        if len(buffer) + len(para) + 2 > size:
            chunks.append(buffer.strip())
            buffer = para
        else:
            buffer = f"{buffer}\n\n{para}".strip() if buffer else para

    if buffer.strip():
        chunks.append(buffer.strip())

    # 重叠：把上一片段尾部接到下一片段头部，避免跨边界语义被切断
    if overlap > 0 and len(chunks) > 1:
        merged: list[str] = [chunks[0]]
        for piece in chunks[1:]:
            tail = merged[-1][-overlap:]
            merged.append(f"{tail}{piece}" if tail else piece)
        chunks = merged
    return [c for c in chunks if c]


@dataclass
class Chunker:
    """切分器。对外唯一方法 chunk()。"""

    size: int = 480
    overlap: int = 80

    def chunk(
        self,
        doc_id: str,
        text: str,
        meta: dict[str, Any] | None = None,
    ) -> list[Chunk]:
        pieces = split_text(text, self.size, self.overlap)
        base_meta: dict[str, Any] = dict(meta or {})
        return [
            Chunk(
                id=f"{doc_id}:{idx}",
                doc_id=doc_id,
                text=piece,
                ordinal=idx,
                meta=dict(base_meta),
            )
            for idx, piece in enumerate(pieces)
        ]


@dataclass
class ChunkStats:
    """切分统计，供自检与评测使用。"""

    count: int = 0
    max_len: int = 0
    avg_len: float = 0.0
    coverage: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)
