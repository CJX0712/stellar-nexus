"""重排层：对召回候选做精排。

四档实现，按成本从低到高：
identity（不做）/ lexical（零依赖 IDF 覆盖）/ llm（列表式，生产默认）/ onnx（交叉编码器，需自备权重）

作者: 晨星
"""

from stellarnx.core.config import Config
from stellarnx.rerank.base import IdentityReranker, LexicalReranker
from stellarnx.rerank.cross_encoder import OnnxCrossEncoderReranker
from stellarnx.rerank.llm_rerank import LLMReranker

__all__ = [
    "IdentityReranker",
    "LLMReranker",
    "LexicalReranker",
    "OnnxCrossEncoderReranker",
    "build_reranker",
]


def build_reranker(cfg: Config, llm=None):
    """按配置构造重排器。任何一档不可用时降级到下一档，不中断链路。"""
    backend = (cfg.rerank_backend or "identity").lower()
    if backend == "identity":
        return IdentityReranker()
    if backend == "lexical":
        return LexicalReranker()
    if backend == "llm":
        if llm is None:
            import logging

            logging.getLogger(__name__).warning("未注入 LLM，重排降级到 lexical")
            return LexicalReranker()
        return LLMReranker(llm)
    if backend == "onnx":
        try:
            return OnnxCrossEncoderReranker(cfg.rerank_model_dir)
        except Exception as exc:
            import logging

            logging.getLogger(__name__).warning("ONNX 重排不可用，降级到词汇重排: %s", exc)
            return LexicalReranker()
    raise ValueError(f"未知重排后端: {backend}")
