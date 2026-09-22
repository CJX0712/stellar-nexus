"""重排层单测。核心不变量：相关片段必须被提到无关片段之前。作者: 晨星"""

from __future__ import annotations

from stellarnx.core.types import Hit
from stellarnx.rerank import IdentityReranker, LexicalReranker


def _hits() -> list[Hit]:
    return [
        Hit(chunk_id="irrelevant", doc_id="d", score=0.9,
            text="今天天气不错，适合出门散步。"),
        Hit(chunk_id="relevant", doc_id="d", score=0.1,
            text="向量数据库支持相似度检索，用于检索增强生成。"),
    ]


def test_identity_reranker_only_truncates():
    ranked = IdentityReranker().rerank("任意查询", _hits(), 1)
    assert len(ranked) == 1
    assert ranked[0].chunk_id == "irrelevant"


def test_lexical_reranker_promotes_relevant():
    ranked = LexicalReranker().rerank("向量数据库 检索增强", _hits(), 2)
    assert ranked[0].chunk_id == "relevant"


def test_lexical_reranker_respects_k():
    assert len(LexicalReranker().rerank("向量", _hits(), 1)) == 1


def test_lexical_reranker_empty_input():
    assert LexicalReranker().rerank("任意", [], 5) == []


def test_fitted_idf_changes_ranking():
    corpus = ["向量数据库用于检索", "向量数据库用于检索", "天气不错适合散步"]
    reranker = LexicalReranker().fit(corpus)
    ranked = reranker.rerank("向量 检索", _hits(), 2)
    assert ranked[0].chunk_id == "relevant"
