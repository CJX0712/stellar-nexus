# ADR-0003 重排默认走 LLM listwise，ONNX 交叉编码器作为可插拔项

- 状态：采纳（**推翻了原始计划**）
- 日期：2026-09-23
- 作者：晨星

## 背景

原计划：召回 20 条 → ONNX 交叉编码器精排 → 取 5 条。
交叉编码器（如 `bge-reranker-base`）在效果上是重排的公认优选，
且 `onnxruntime` 有 `win_amd64` 轮子，零编译，看起来完全可行。

实施阶段做了一件事：**先去 ModelScope 上确认权重是否真的存在**，而不是默认它存在。

## 事实核查结果

```
搜索 ModelScope：bge-reranker-base / bge-reranker-v2-m3 / bge-reranker-large
结果：只有 PyTorch 权重（pytorch_model.bin / model.safetensors）
     没有 onnx/model.onnx
```

而本机**没有 `torch`**，也不打算为一个重排功能装 2GB 的 torch
（装了会拖慢冷启动、拉长复现时间，与「干净环境一键复现」冲突）。

可选项只剩：`bge-reranker` 的 PyTorch 权重 + 自行转换 ONNX，或换一条路。

自行转换需要在**有 torch 的机器**上导出，本机做不到；
即便做到了，也要多出一个「模型转换说明」步骤，复现链路变长。

## 备选方案

**A. 装 torch，跑 PyTorch 权重。** 体积、冷启动、CPU 推理速度三方面都不划算。

**B. 放弃重排，只做 RRF + MMR。** 检索质量会掉，但链路完整。
可以作为兜底，不该是默认。

**C. 默认用已就位的 Qwen 做 LLM listwise 重排；ONNX 后端保留为可插拔项。**（本决策）

**D. 换其它有 ONNX 权重的重排模型。** 需要在 ModelScope 上逐个核实，
且质量未必优于 BGE 系列；核实成本已花过一轮。

## 决策

采用 C，四个后端并存：

| 后端 | 名称 | 用途 | 额外依赖 |
|---|---|---|---|
| `identity` | `identity` | 原序截断，最省 | 无 |
| `lexical` | `lexical` | IDF 加权词项覆盖度 | 无 |
| **`llm`** | `llm-listwise` | **默认**，复用已在跑的 Qwen | Ollama |
| `onnx` | `onnx-cross-encoder` | 用户自备 `model.onnx` 即生效 | onnxruntime + tokenizers |

`LLMReranker` 的实现要点：

1. **窗口化**：每次只看 8 条候选，超出分批。
   一次性塞 20 条给 1.5B 模型，序号会错乱。
2. **三级解析降级**：严格 JSON 数组 → 宽松数字抽取 → **放弃并原序返回**。
3. **绝不抛异常**。重排是优化项，不该成为失败点。

## 理由

1. **零额外下载**。Qwen 已经是生成模型，重排复用它不增加任何依赖。
   复现路径不变，冷启动不变。
2. **对黄金集是净收益**。`recall@1` 从 0.50 提升到 0.88（真实模型下），
   代价是每个查询多一次约 1–3s 的生成。
3. **保留了升级通道**。用户若自备 `model.onnx` 放进 `SNX_RERANK_MODEL_DIR`，
   改一个环境变量就切到交叉编码器。协议是同一个 `Reranker`。
4. **保留了确定性测试**。离线测试用 `lexical`，指标逐位可复现 ——
   如果默认走 LLM，离线门禁就做不到确定性。

## 代价与补救

| 代价 | 补救 |
|---|---|
| 真实模型下每查询多 1–3s | 重排窗口限制为 8；`top_k_retrieve` 默认 20 不算激进 |
| 1.5B 模型的 listwise 排序能力有限 | 解析失败即原序返回，退化行为是安全的；换更大模型只需改 `SNX_LLM_MODEL` |
| ONNX 后端**没有在本机实测过**（无权重） | 代码路径完整、接口已实现，但 README 明确标注「用户自备权重」；`test_rerank.py` 只测模型缺失时抛 `DependencyUnavailable`，不假装测过推理 |
| 依赖 `onnxruntime` + `tokenizers` 仍然装在环境里 | 保留：装得上、体积可接受，且是 onnx 后端的前提。如果用户不要，从 `pyproject.toml` 的 `dependencies` 里移除即可 |

## 诚实性声明

本 ADR 记录了一次**计划被实测推翻**。原计划的 ADR 会写「用 ONNX 交叉编码器重排」，
而本机根本没有可用的 ONNX 权重 —— 如果当时不去核实，交付物里会留下一条
**从未真正跑通过**的默认路径。这类「文档说 A、代码走 B」的偏差比功能缺失更难发现。

因此本项目立了一条规则：**任何写在文档里的默认值，必须是本机实测跑过的那条路径。**

## 反悔路径

拿到 `model.onnx` 后：

```bash
export SNX_RERANK_BACKEND=onnx
export SNX_RERANK_MODEL_DIR=models/bge-reranker-base
```

## 相关

- 代码：`src/stellarnx/rerank/`（`base.py` / `llm_rerank.py` / `cross_encoder.py` / `__init__.py`）
- 测试：`tests/test_rerank.py`
- 验证脚本：`scripts/download_models.py`（从 ModelScope 拉模型）
