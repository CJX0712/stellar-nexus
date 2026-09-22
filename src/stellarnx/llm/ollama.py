"""Ollama LLM 后端（默认 qwen2.5:1.5b-instruct）。

只做 HTTP 适配与错误归一化，模型能力完全复用 Ollama。

作者: 晨星
"""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx

from stellarnx.core.config import Config
from stellarnx.core.errors import DependencyUnavailable, UpstreamTimeout

logger = logging.getLogger(__name__)


class OllamaLLM:
    """满足 LLM 协议的 Ollama 实现。"""

    def __init__(
        self,
        cfg: Config,
        model: str | None = None,
        host: str | None = None,
        timeout_s: float | None = None,
    ) -> None:
        self.cfg = cfg
        self.model = model or cfg.llm_model
        self.host = (host or cfg.ollama_host).rstrip("/")
        self.timeout_s = timeout_s or cfg.ollama_timeout_s
        # trust_env=False：本机 SOCKS5 代理会把本地请求也代理走，必须绕过
        self._client = httpx.Client(timeout=self.timeout_s, trust_env=False)

    @property
    def name(self) -> str:
        return f"ollama:{self.model}"

    def complete(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
        }
        options = kwargs.get("options")
        if options:
            payload["options"] = options
        try:
            resp = self._client.post(f"{self.host}/api/chat", json=payload)
        except httpx.TimeoutException as exc:
            raise UpstreamTimeout(f"Ollama 生成超时（{self.timeout_s}s）") from exc
        except httpx.HTTPError as exc:
            raise DependencyUnavailable(f"无法连接 Ollama: {exc}") from exc

        if resp.status_code != 200:
            raise DependencyUnavailable(
                f"Ollama 返回 {resp.status_code}: {resp.text[:200]}")
        try:
            body = resp.json()
        except json.JSONDecodeError as exc:
            raise DependencyUnavailable(f"Ollama 返回非 JSON: {exc}") from exc
        message = body.get("message") or {}
        return (message.get("content") or "").strip()

    def is_available(self) -> bool:
        """轻量探测：只问一个字，避免每次都拉长上下文。"""
        try:
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False,
                "options": {"num_predict": 1},
            }
            resp = self._client.post(
                f"{self.host}/api/chat", json=payload, timeout=min(20.0, self.timeout_s))
            return resp.status_code == 200
        except Exception:
            return False
