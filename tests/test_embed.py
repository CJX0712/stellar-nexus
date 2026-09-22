"""嵌入层单测。核心不变量：相似文本余弦高于无关文本，且跨进程确定。作者: 晨星"""

from __future__ import annotations

import numpy as np

from stellarnx.core.text import tokenize
from stellarnx.embed import HashEmbedder


def test_hash_embed_shape_and_normalized():
    emb = HashEmbedder(dim=64)
    matrix = emb.embed(["你好世界", "abc def"])
    assert matrix.shape == (2, 64)
    norms = np.linalg.norm(matrix, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_similar_texts_closer_than_unrelated():
    emb = HashEmbedder()
    matrix = emb.embed([
        "向量数据库支持相似度检索",
        "向量数据库用于相似度检索任务",
        "公司团建安排在海边的度假村",
    ])
    sim_same = float(matrix[0] @ matrix[1])
    sim_diff = float(matrix[0] @ matrix[2])
    assert sim_same > sim_diff, (sim_same, sim_diff)


def test_deterministic_across_instances():
    a = HashEmbedder().embed(["星枢系统的检索层"])
    b = HashEmbedder().embed(["星枢系统的检索层"])
    assert np.array_equal(a, b)


def test_empty_input_returns_zero_rows():
    assert HashEmbedder().embed([]).shape[0] == 0


def test_tokenize_handles_mixed_languages():
    tokens = tokenize("向量数据库 VectorDB 检索")
    assert any(t.startswith("c") for t in tokens)
    assert any(t.startswith("b") for t in tokens)
    assert "wvectordb" in tokens
