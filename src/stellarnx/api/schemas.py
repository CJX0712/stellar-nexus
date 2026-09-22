"""API 数据模型。作者: 晨星"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from stellarnx import __author__, __version__


class IngestRequest(BaseModel):
    docs: list[dict[str, Any]] = Field(..., description="元素需含 doc_id 与 text，可选 meta")


class SearchRequest(BaseModel):
    query: str
    k: int = 5


class ChatRequest(BaseModel):
    query: str
    with_trace: bool = False


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = __version__
    author: str = __author__
    backends: dict[str, str] = Field(default_factory=dict)
