"""大模型适配层：统一 complete(messages) -> str。

作者: 晨星
"""

from stellarnx.core.config import Config
from stellarnx.llm.mock import MockLLM
from stellarnx.llm.ollama import OllamaLLM
from stellarnx.llm.prompts import build_rag_messages

__all__ = ["MockLLM", "OllamaLLM", "build_llm", "build_rag_messages"]


def build_llm(cfg: Config):
    """按配置构造 LLM。Ollama 不可用时降级 Mock，保证链路永不中断。"""
    backend = (cfg.llm_backend or "mock").lower()
    if backend == "mock":
        return MockLLM()
    if backend == "ollama":
        llm = OllamaLLM(cfg)
        if llm.is_available():
            return llm
        import logging

        logging.getLogger(__name__).warning("Ollama 不可用，降级到 Mock LLM")
        return MockLLM()
    raise ValueError(f"未知 LLM 后端: {backend}")
