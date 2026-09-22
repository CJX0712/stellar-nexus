"""切分层单测。核心不变量：长文必多片段、不超阈值、内容不丢。作者: 晨星"""

from __future__ import annotations

from stellarnx.chunk import Chunker, split_text


def test_short_text_single_chunk():
    assert len(split_text("一句话。", size=100)) == 1


def test_long_text_produces_multiple_chunks():
    text = "这是第一段落的内容。" * 40 + "\n\n" + "这是第二段落的内容。" * 40
    pieces = split_text(text, size=120, overlap=20)
    assert len(pieces) > 1


def test_chunks_respect_size_limit():
    text = "内容内容内容。" * 300
    for piece in split_text(text, size=200, overlap=30):
        assert len(piece) <= 200 + 30 + 5


def test_no_content_lost():
    text = "段落一的关键信息。" * 20 + "\n\n" + "段落二的关键信息。" * 20
    joined = "".join(split_text(text, size=150, overlap=20))
    assert "段落一的关键信息" in joined
    assert "段落二的关键信息" in joined


def test_chunker_ids_are_stable():
    chunks = Chunker(120, 20).chunk("doc", "内容。" * 100)
    assert [c.id for c in chunks] == [f"doc:{i}" for i in range(len(chunks))]
    assert all(c.doc_id == "doc" for c in chunks)


def test_empty_text_returns_empty():
    assert Chunker().chunk("doc", "   ") == []
