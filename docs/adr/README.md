# 架构决策记录（ADR）

作者 **晨星**

记录对本项目影响深远、且**当初有其它合理选项**的决策。每条包含：背景、备选方案、
决策、理由、代价与补救、以及「如果反悔怎么改」。

| 编号 | 决策 | 状态 |
|---|---|---|
| [0001](0001-protocol-plus-zero-dep-fallback.md) | 用协议 + 零依赖回退实现，而不是绑定具体库 | 采纳 |
| [0002](0002-faiss-and-handwritten-bm25.md) | 向量用 FAISS、BM25 自实现，不引第三方 BM25 包 | 采纳 |
| [0003](0003-rerank-default-llm-listwise.md) | 重排默认 LLM listwise，ONNX 交叉编码器为可插拔项 | 采纳（原计划为 ONNX 默认） |
| [0004](0004-idf-weighted-groundedness.md) | 归因校验用 IDF 加权词汇覆盖度，不训 NLI 模型 | 采纳 |
| [0005](0005-single-file-console.md) | 控制台单文件零构建，不上 React + Vite | 采纳 |
