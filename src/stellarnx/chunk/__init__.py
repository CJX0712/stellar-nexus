"""切分层：文档 -> 片段。只做切分，不做嵌入、不碰存储。

作者: 晨星
"""

from stellarnx.chunk.chunker import Chunker, split_text

__all__ = ["Chunker", "split_text"]
