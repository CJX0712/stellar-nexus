"""测试公共夹具。作者: 晨星"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stellarnx.api.app import create_app  # noqa: E402
from stellarnx.core.config import load_config  # noqa: E402
from stellarnx.core.types import Chunk, Hit  # noqa: E402
from stellarnx.pipeline.orchestrator import NexusSystem  # noqa: E402


@pytest.fixture
def offline_config():
    """离线配置：哈希嵌入 + Mock 生成 + 词汇重排，无网络无密钥。"""
    return load_config(
        embed_backend="hash",
        llm_backend="mock",
        rerank_backend="lexical",
        offline=True,
    )


@pytest.fixture
def system(offline_config):
    return NexusSystem(offline_config, ":memory:")


@pytest.fixture
def client(offline_config):
    with TestClient(create_app(cfg=offline_config)) as test_client:
        yield test_client


@pytest.fixture
def sample_chunks() -> list[Chunk]:
    return [
        Chunk(id="d1:0", doc_id="d1", text="向量数据库用于存储嵌入向量并支持相似度检索。", ordinal=0),
        Chunk(id="d1:1", doc_id="d1", text="FAISS 提供 IndexFlatIP 精确索引与 HNSW 近似索引。", ordinal=1),
        Chunk(id="d2:0", doc_id="d2", text="年度团建安排在下周的海边度假村举行，请携带泳衣。", ordinal=0),
    ]


@pytest.fixture
def sample_hits(sample_chunks) -> list[Hit]:
    return [
        Hit(chunk_id=c.id, doc_id=c.doc_id, text=c.text, score=1.0 - i * 0.1, source="dense")
        for i, c in enumerate(sample_chunks)
    ]
