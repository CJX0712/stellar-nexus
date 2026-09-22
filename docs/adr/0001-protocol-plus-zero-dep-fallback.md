# ADR-0001 用协议 + 零依赖回退实现，而不是绑定具体库

- 状态：采纳
- 日期：2026-09-23
- 作者：晨星

## 背景

系统里六种能力（嵌入、向量索引、召回、重排、生成、校验）都需要「真实实现」和
「能在没有网络、没有模型、没有 GPU 的机器上跑」的实现。硬约束是
**干净环境必须一键复现**，而 CI runner 上没有 Ollama，也不可能在 CI 里下 400MB 模型。

## 备选方案

**A. 只做真实实现，CI 跳过需要模型的测试。**
最简单，但门禁形同虚设：评测指标、端到端链路、控制台交互全都没法在 CI 里验证。
「CI 绿」和「系统能跑」变成两件事，这正是我们要避免的。

**B. 只做离线实现，真实模型作为「以后再说」。**
能测，但它不是真的要交付的东西。交付物会退化成另一个玩具。

**C. 每个能力定义 `typing.Protocol`，提供两个实现：一个生产级、一个零依赖。**
（本决策）

**D. 用 `unittest.mock` 打桩替代真实实现。**
Mock 无法验证「相似句余弦确实更大」这类**算法性质**，只能验证调用顺序。
它会让我们误以为检索质量被测过了。

## 决策

采用 C。契约层只放 `Protocol`，零实现：

```python
class Embedder(Protocol):
    name: str
    dim: int
    def embed(self, texts: list[str]) -> np.ndarray: ...
```

每个能力两个实现：

| 能力 | 零依赖实现 | 生产实现 |
|---|---|---|
| 嵌入 | `HashEmbedder`（哈希特征，512 维） | `OllamaEmbedder`（BGE-M3，1024 维） |
| 索引 | `InMemoryIndex`（numpy 暴力内积） | `FaissIndex`（`IndexFlatIP`） |
| 生成 | `MockLLM`（抽取式，只挑证据原句） | `OllamaLLM`（Qwen2.5） |
| 重排 | `LexicalReranker`（IDF 覆盖度） | `LLMReranker` / `OnnxCrossEncoderReranker` |

构造统一走工厂：`build_embedder(cfg)` / `build_llm(cfg)` / `build_reranker(cfg, llm)`，
按 `SNX_*_BACKEND` 分派。

## 理由

1. **可复现性可被证明，而不是被声称**。离线路径下指标确定：`EvalRunner` 每次新建内存管道，
   同配置两次运行指标逐位相同（`test_eval.py` 有断言）。
2. **零依赖实现不是玩具**。`HashEmbedder` 在黄金集上 `recall@5 = 1.00`，
   而 `MockLLM` 是抽取式的 —— 它只会复述证据原句，永远不会编造。
   这两者组合出了一个**质量可接受且完全确定**的基线。
3. **门禁阈值可以定得诚实**。离线基线 `latency_p95 ≈ 33ms`，真实模型 `≈ 31s`，
   差三个数量级。门禁只跑离线路径，阈值 `8000ms` 留了足够余量，
   避免「因为 CI 机器慢就误判失败」。
4. **协议让替换成本可预期**。把 `MockLLM` 换成自研模型，只要 `complete(messages) -> str`
   的形状不变，下游一行都不用改。

## 代价与补救

| 代价 | 补救 |
|---|---|
| 两套实现要同步维护 | 契约测试（`test_contracts` 系列）同时跑两套实现，接口漂移立刻暴露 |
| 离线指标不等于真实指标 | README 与评测面板**同时**展示两组数字，绝不只放好看的那组 |
| 零依赖实现可能被当成「够用」而不再换真实模型 | `stellarnx stats` 与 `/health` 显式报告当前生效的后端名；控制台头部常驻显示 |

## 反悔路径

删掉 `build_*` 工厂里的 `else` 分支即可。协议层不用动，`pipeline` 也不用动。

## 相关

- 代码：`src/stellarnx/core/contracts.py`、各模块的 `__init__.py` 工厂
- 测试：`tests/test_embed.py`、`tests/test_llm` 相关断言
- 文档：[`SPEC.md` 第 1 章](../../SPEC.md)
