"""CLI 测试。

重点回归的缺陷：stats/ingest/search/chat 曾经各自 new 一个内存库，
于是 `ingest` 写完进程退出、`search` 再起一个新进程，知识库是空的。
表现是「命令全都返回 0，却什么都搜不到」——最不该出现在交付物里的坑。

作者: 晨星
"""

from __future__ import annotations

import json

import pytest

from stellarnx.cli import main


@pytest.fixture()
def cli_env(tmp_path, monkeypatch):
    """把 CLI 指向临时数据目录与离线后端，避免污染真实知识库。"""
    monkeypatch.setenv("SNX_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("SNX_EMBED_BACKEND", "hash")
    monkeypatch.setenv("SNX_LLM_BACKEND", "mock")
    monkeypatch.setenv("SNX_RERANK_BACKEND", "lexical")
    return tmp_path


def _write_corpus(tmp_path):
    path = tmp_path / "docs.jsonl"
    docs = [
        {"doc_id": "d1", "text": "星枢系统的向量索引基于 FAISS，并保留 numpy 回退实现。" * 4},
        {"doc_id": "d2", "text": "检索阶段并行执行 BM25 稀疏召回与稠密向量召回两路。" * 4},
    ]
    path.write_text("\n".join(json.dumps(d, ensure_ascii=False) for d in docs), encoding="utf-8")
    return path


def test_ingest_then_search_finds_data_across_processes(cli_env, capsys):
    path = _write_corpus(cli_env)
    assert main(["ingest", str(path)]) == 0
    out = capsys.readouterr().out
    assert "导入片段数" in out

    # 第二次调用是「新进程」，必须能读到上一次写进去的数据
    assert main(["search", "向量索引 FAISS", "-k", "3"]) == 0
    hits = capsys.readouterr().out
    assert "d1" in hits, f"跨进程检索失败，输出：{hits!r}"


def test_stats_reports_vectors_matching_chunks(cli_env, capsys):
    path = _write_corpus(cli_env)
    main(["ingest", str(path)])
    capsys.readouterr()
    assert main(["stats"]) == 0
    stats = json.loads(capsys.readouterr().out)
    assert stats["chunks"] > 0
    assert stats["vectors"] == stats["chunks"], "向量数必须等于片段数，否则说明索引没恢复"
    assert stats["documents"] == 2


def test_chat_json_output_is_machine_readable(cli_env, capsys):
    path = _write_corpus(cli_env)
    main(["ingest", str(path)])
    capsys.readouterr()
    assert main(["chat", "向量索引用什么实现？", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["text"]
    assert payload["citations"], "CLI 问答也必须带引用"
    assert payload["route"]["path"] in {"direct", "rag", "multihop", "agent"}


def test_ingest_missing_file_returns_2(cli_env, capsys):
    assert main(["ingest", str(cli_env / "nope.jsonl")]) == 2
    assert "文件不存在" in capsys.readouterr().err


def test_verify_selftest_passes(cli_env, capsys):
    assert main(["verify"]) == 0
    out = capsys.readouterr().out
    assert "通过" in out
    assert "[FAIL]" not in out
