"""文档与片段的持久化存储（sqlite，纯标准库，无额外依赖）。

作者: 晨星
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from stellarnx.core.types import Chunk

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id  TEXT PRIMARY KEY,
    title   TEXT NOT NULL DEFAULT '',
    source  TEXT NOT NULL DEFAULT '',
    meta    TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY,
    doc_id   TEXT NOT NULL,
    ordinal  INTEGER NOT NULL DEFAULT 0,
    text     TEXT NOT NULL,
    meta     TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
"""


class DocumentStore:
    """片段仓库。唯一职责：按 id 存取 Chunk，不理解检索打分。"""

    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    # --- 写入 ---
    def add_document(self, doc_id: str, title: str = "", source: str = "",
                     meta: dict[str, Any] | None = None) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO documents(doc_id, title, source, meta) VALUES (?,?,?,?)",
            (doc_id, title, source, json.dumps(meta or {}, ensure_ascii=False)),
        )
        self._conn.commit()

    def add_chunks(self, chunks: list[Chunk]) -> None:
        rows = [
            (c.id, c.doc_id, c.ordinal, c.text, json.dumps(c.meta, ensure_ascii=False))
            for c in chunks
        ]
        self._conn.executemany(
            "INSERT OR REPLACE INTO chunks(chunk_id, doc_id, ordinal, text, meta) "
            "VALUES (?,?,?,?,?)",
            rows,
        )
        self._conn.commit()

    # --- 读取 ---
    def get_chunk(self, chunk_id: str) -> Chunk | None:
        row = self._conn.execute(
            "SELECT chunk_id, doc_id, ordinal, text, meta FROM chunks WHERE chunk_id = ?",
            (chunk_id,),
        ).fetchone()
        return self._row_to_chunk(row) if row else None

    def get_chunks(self, chunk_ids: list[str]) -> list[Chunk]:
        """按给定顺序返回存在的片段；不存在的 id 直接跳过。"""
        result: list[Chunk] = []
        for cid in chunk_ids:
            chunk = self.get_chunk(cid)
            if chunk is not None:
                result.append(chunk)
        return result

    def chunks_of_doc(self, doc_id: str) -> list[Chunk]:
        rows = self._conn.execute(
            "SELECT chunk_id, doc_id, ordinal, text, meta FROM chunks "
            "WHERE doc_id = ? ORDER BY ordinal",
            (doc_id,),
        ).fetchall()
        return [self._row_to_chunk(r) for r in rows]

    def all_chunks(self) -> list[Chunk]:
        rows = self._conn.execute(
            "SELECT chunk_id, doc_id, ordinal, text, meta FROM chunks ORDER BY doc_id, ordinal"
        ).fetchall()
        return [self._row_to_chunk(r) for r in rows]

    def list_documents(self) -> list[dict[str, Any]]:
        """文档列表带片段计数。

        片段数用 LEFT JOIN 一次算出来，而不是让调用方每篇再查一次 ——
        控制台要按篇展示片段数，N+1 次查询在文档多时会明显变慢。
        """
        rows = self._conn.execute(
            "SELECT d.doc_id, d.title, d.source, d.meta, d.created_at, "
            "       COUNT(c.chunk_id) AS chunk_count "
            "FROM documents d LEFT JOIN chunks c ON c.doc_id = d.doc_id "
            "GROUP BY d.doc_id ORDER BY d.created_at, d.doc_id"
        ).fetchall()
        return [
            {
                "doc_id": r["doc_id"],
                "title": r["title"],
                "source": r["source"],
                "meta": json.loads(r["meta"]),
                "created_at": r["created_at"],
                "chunks": int(r["chunk_count"]),
            }
            for r in rows
        ]

    # --- 维护 ---
    def delete_document(self, doc_id: str) -> int:
        cur = self._conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
        removed = cur.rowcount or 0
        self._conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
        self._conn.commit()
        return removed

    def clear(self) -> None:
        self._conn.executescript("DELETE FROM chunks; DELETE FROM documents;")
        self._conn.commit()

    def count_chunks(self) -> int:
        return int(self._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])

    def count_documents(self) -> int:
        return int(self._conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0])

    def close(self) -> None:
        self._conn.close()

    @staticmethod
    def _row_to_chunk(row: sqlite3.Row) -> Chunk:
        return Chunk(
            id=row["chunk_id"],
            doc_id=row["doc_id"],
            text=row["text"],
            ordinal=row["ordinal"],
            meta=json.loads(row["meta"]) if row["meta"] else {},
        )
