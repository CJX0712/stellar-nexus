"""ONNX 交叉编码器重排（bge-reranker-base 等）。

工程要点：
1. 不同导出带/不带 token_type_ids，输入名必须运行时探测后再喂，否则直接崩；
2. 输出可能是 logits 或已 sigmoid 的概率，统一映射成越大越相关；
3. 模型文件缺失抛 DependencyUnavailable，由上层降级，不静默。

作者: 晨星
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from stellarnx.core.errors import DependencyUnavailable
from stellarnx.core.types import Hit

_MAX_LEN = 512


class OnnxCrossEncoderReranker:
    """满足 Reranker 协议的 ONNX 实现。"""

    name = "onnx-cross-encoder"

    def __init__(self, model_dir: str, max_length: int = _MAX_LEN) -> None:
        try:
            import onnxruntime as ort
            from tokenizers import Tokenizer
        except Exception as exc:
            raise DependencyUnavailable(f"ONNX 运行时不可用: {exc}") from exc

        base = Path(model_dir)
        model_path = base / "model.onnx"
        tokenizer_path = base / "tokenizer.json"
        if not model_path.exists():
            raise DependencyUnavailable(f"缺少模型文件: {model_path}")
        if not tokenizer_path.exists():
            raise DependencyUnavailable(f"缺少分词器文件: {tokenizer_path}")

        self._ort = ort
        self._session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"])
        self._tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self._input_names = {i.name for i in self._session.get_inputs()}
        self._output_name = self._session.get_outputs()[0].name
        self.max_length = max_length

    def _encode(self, query: str, text: str) -> dict[str, np.ndarray]:
        encoded = self._tokenizer.encode(query, text)
        ids = encoded.ids[: self.max_length]
        mask = encoded.attention_mask[: self.max_length]
        type_ids = (encoded.type_ids or [0] * len(ids))[: self.max_length]

        def pad(seq: list[int], value: int = 0) -> list[int]:
            return seq + [value] * (self.max_length - len(seq))

        feeds = {
            "input_ids": np.array([pad(ids)], dtype="int64"),
            "attention_mask": np.array([pad(mask)], dtype="int64"),
            "token_type_ids": np.array([pad(type_ids)], dtype="int64"),
        }
        return {k: v for k, v in feeds.items() if k in self._input_names}

    def score(self, query: str, text: str) -> float:
        outputs = self._session.run([self._output_name], self._encode(query, text))
        value = float(np.asarray(outputs[0]).reshape(-1)[0])
        # 输出可能是 logits（任意实数）也可能已过 sigmoid；统一压到 [0,1] 后再用
        if value < 0.0 or value > 1.0:
            value = 1.0 / (1.0 + float(np.exp(-value)))
        return value

    def rerank(self, query: str, hits: list[Hit], k: int) -> list[Hit]:
        if not hits:
            return []
        scored = [
            Hit(
                chunk_id=h.chunk_id,
                doc_id=h.doc_id,
                text=h.text,
                score=float(self.score(query, h.text)),
                source=f"{h.source}+ce",
                meta=dict(h.meta),
            )
            for h in hits
        ]
        return sorted(scored, key=lambda h: -h.score)[:k]
