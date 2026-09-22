"""集中配置。所有可调参数在此声明，环境变量以 SNX_ 前缀覆盖。

作者: 晨星
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field


def _env(key: str, default: str) -> str:
    return os.getenv(f"SNX_{key}", default)


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(f"SNX_{key}", str(default)))
    except ValueError:
        return default


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.getenv(f"SNX_{key}", str(default)))
    except ValueError:
        return default


@dataclass
class Config:
    """全系统配置。默认值保证"离线零依赖"也能跑通整条链路。"""

    # --- 模型 ---
    embed_backend: str = field(default_factory=lambda: _env("EMBED_BACKEND", "hash"))
    embed_model: str = field(default_factory=lambda: _env("EMBED_MODEL", "bge-m3"))
    llm_backend: str = field(default_factory=lambda: _env("LLM_BACKEND", "mock"))
    llm_model: str = field(default_factory=lambda: _env("LLM_MODEL", "qwen2.5:1.5b-instruct"))
    rerank_backend: str = field(default_factory=lambda: _env("RERANK_BACKEND", "identity"))
    rerank_model_dir: str = field(default_factory=lambda: _env("RERANK_MODEL_DIR", "models/bge-reranker-base"))

    # --- 服务 ---
    ollama_host: str = field(default_factory=lambda: _env("OLLAMA_HOST", "http://127.0.0.1:11434"))
    ollama_timeout_s: float = field(default_factory=lambda: _env_float("OLLAMA_TIMEOUT_S", 120.0))
    api_host: str = field(default_factory=lambda: _env("API_HOST", "127.0.0.1"))
    api_port: int = field(default_factory=lambda: _env_int("API_PORT", 8000))

    # --- 切分 ---
    chunk_size: int = field(default_factory=lambda: _env_int("CHUNK_SIZE", 480))
    chunk_overlap: int = field(default_factory=lambda: _env_int("CHUNK_OVERLAP", 80))

    # --- 检索 ---
    top_k_retrieve: int = field(default_factory=lambda: _env_int("TOP_K_RETRIEVE", 20))
    top_k_rerank: int = field(default_factory=lambda: _env_int("TOP_K_RERANK", 5))
    rrf_k: int = field(default_factory=lambda: _env_int("RRF_K", 60))
    mmr_lambda: float = field(default_factory=lambda: _env_float("MMR_LAMBDA", 0.7))

    # --- 校验 ---
    groundedness_threshold: float = field(
        default_factory=lambda: _env_float("GROUNDEDNESS_THRESHOLD", 0.6))
    max_regenerate: int = field(default_factory=lambda: _env_int("MAX_REGENERATE", 1))

    # --- 智能体 ---
    agent_max_steps: int = field(default_factory=lambda: _env_int("AGENT_MAX_STEPS", 4))
    agent_max_replan: int = field(default_factory=lambda: _env_int("AGENT_MAX_REPLAN", 2))

    # --- 存储 ---
    data_dir: str = field(default_factory=lambda: _env("DATA_DIR", "data"))
    offline: bool = field(default_factory=lambda: _env("OFFLINE", "1") not in ("0", "false", "False"))

    def to_dict(self) -> dict:
        return asdict(self)


def load_config(**overrides: object) -> Config:
    """读取配置并应用显式覆盖。测试与 CLI 通过此入口注入。"""
    cfg = Config()
    for key, value in overrides.items():
        if not hasattr(cfg, key):
            raise AttributeError(f"未知配置项: {key}")
        setattr(cfg, key, value)
    return cfg
