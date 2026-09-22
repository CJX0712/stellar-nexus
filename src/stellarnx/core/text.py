"""共享文本处理。BM25 与哈希嵌入必须使用同一套切词，否则稀疏/稠密两路不可比。

作者: 晨星
"""

from __future__ import annotations

import re

_CJK = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]")
_ASCII_TOKEN = re.compile(r"[a-z0-9][a-z0-9_.\-]*")


def tokenize(text: str) -> list[str]:
    """中英混排切词。

    - CJK：产出单字（前缀 c）与相邻二字组（前缀 b），因此中文短文本也有重叠信号；
    - ASCII：产出词（前缀 w），并额外产出字符三元组以容忍拼写差异。
    返回带前缀的 token，避免中英文 token 意外撞车。
    """
    lowered = text.lower()
    tokens: list[str] = []
    for i, ch in enumerate(lowered):
        if _CJK.match(ch):
            tokens.append(f"c{ch}")
            if i + 1 < len(lowered) and _CJK.match(lowered[i + 1]):
                tokens.append(f"b{ch}{lowered[i + 1]}")
    for word in _ASCII_TOKEN.findall(lowered):
        tokens.append(f"w{word}")
        if len(word) > 4:
            tokens.extend(f"t{word[i:i + 3]}" for i in range(len(word) - 2))
    return tokens
