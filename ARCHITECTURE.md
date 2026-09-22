# 星枢 StellarNexus 架构说明

作者 **晨星** · 版本 1.0.0

本文回答三个问题：**系统由哪些部件构成**、**部件之间怎么调**、**为什么这样切**。
接口的逐字段契约在 [`SPEC.md`](SPEC.md)，代码级注释在源码里，不在这里重复。

---

## 1. 设计目标与约束

### 目标

| # | 目标 | 可验证的判据 |
|---|---|---|
| G1 | 每个模块可独立验证 | 每个模块至少有 1 条不依赖其它模块实现的测试 |
| G2 | 整条链路可离线跑通 | 断网 + 无 Key 时 `stellarnx verify` 15/15 —— 实测通过 |
| G3 | 每个答案可溯源 | 每条输出都带引用片段 id 与断言级支持度，且可被「篡改证据」反向打穿 |
| G4 | 构建可复现 | 干净环境 `pip install -e .` 一次成功，零编译；依赖全部锁版本 |
| G5 | 质量可门禁 | `stellarnx eval` 退出码即门禁结论；CI 在干净 runner 上复现 |

### 硬约束

- **不重造模型**。生成、嵌入、重排的算法能力全部复用现有成果（Ollama / Qwen / BGE-M3 / ONNX）。
- **不引入编译器**。所有依赖必须存在 `win_amd64` 与 `manylinux` 的预编译轮子。
- **不引入重型框架**。控制台不构建、不打包、不联网取资源。
- **单一职责**。一个模块只做一件事；发现某模块开始理解两个概念，就拆。

---

## 2. 分层

```
┌──────────────────────────────────────────────────────────────────┐
│  L4 边界层      api/ (FastAPI)          cli.py (argparse)         │
│                 只做协议转换与错误映射，零业务逻辑                 │
├──────────────────────────────────────────────────────────────────┤
│  L3 编排层      pipeline/orchestrator.py                          │
│                 唯一装配点：决定「路由之后走哪条路」               │
│                 不实现任何具体能力                                │
├──────────────────────────────────────────────────────────────────┤
│  L2 能力层      route  agent  retrieve  rerank  llm  verify       │
│                 每个都是可通过协议替换的实现                       │
│                 trace 横切全部模块                                │
├──────────────────────────────────────────────────────────────────┤
│  L1 基础设施    embed  chunk  store                               │
│                 纯数据变换与持久化，不含决策                       │
├──────────────────────────────────────────────────────────────────┤
│  L0 契约层      core/  协议 · 类型 · 配置 · 路径 · 线性代数 · 切词 │
│                 零实现依赖，被所有层依赖，自己不依赖任何业务模块   │
└──────────────────────────────────────────────────────────────────┘
```

**依赖方向严格单向**：L4 → L3 → L2 → L1 → L0。
CI 里由 `scripts/check_imports.py` 静态校验，出现反向依赖直接失败。

> 为什么要专门写一个导入方向检查器？因为「循环导入」在本项目真实发生过：
> `embed/__init__.py` 导出 `normalize`，`hash_embed.py` 又回头 `from stellarnx.embed import normalize`，
> 结果是「单独 import 能过、从入口 import 就炸」。当时的修法不是调整 import 顺序，
> 而是把 `normalize` 下沉到 `core/linalg.py` —— **把共用能力放到依赖图的下层**，
> 这才是根治。这个教训后来固化成了下面的 L0 规则。

---

## 3. 模块清单

| 模块 | 职责（一句话） | 依赖 | 关键不变量 |
|---|---|---|---|
| `core.contracts` | 定义 6 个 `Protocol` | 无 | 任何实现必须可 `isinstance` 判定 |
| `core.types` | `Chunk/Hit/Citation/Claim/VerifyReport/RouteDecision/Answer` | 无 | `Answer.to_dict()` 必须 JSON 可序列化 |
| `core.config` | 集中配置，`SNX_*` 覆盖 | 无 | 未知配置项设值必须抛 `AttributeError` |
| `core.paths` | 仓库根/数据/Web/模型目录解析 | 无 | 相对路径一律相对仓库根，绝不相对 CWD |
| `core.linalg` | `normalize / cosine / softmax / logsumexp` | numpy | 全零行归一化不产生 NaN |
| `core.text` | 中英混排切词（CJK 单字 + 相邻二字组 + ASCII 词） | 无 | 同一文本两次切词结果逐位相同 |
| `embed.hash` | 哈希特征嵌入，零依赖 | core | 相似句余弦 > 不相似句 |
| `embed.ollama` | BGE-M3 嵌入 | core, httpx | 归一化后 `‖v‖≈1`；`trust_env=False` |
| `chunk.chunker` | 滑窗切分，带重叠 | core | 片段拼回原文不丢字符；超阈值必多片段 |
| `store.document_store` | sqlite 片段与文档持久化 | core | `count_chunks()` 与实际行数一致 |
| `store.vector_index` | FAISS 索引（numpy 回退） | core | `count()` == 已 add 的唯一 id 数 |
| `retrieve.bm25` | BM25 稀疏召回，自实现 | core | 查询词全不在语料时返回空列表 |
| `retrieve.dense` | 稠密召回 | core | 自查询首位必是自己 |
| `retrieve.hybrid` | RRF 融合 + MMR 去冗余 | core | 双路都命中的项排名不低于单路命中项 |
| `rerank.lexical` | 词法重排（零依赖，可测） | core | 正例分数 > 负例 |
| `rerank.llm` | LLM listwise 重排 | core, llm | 模型输出不可解析时**原序返回**，不抛异常 |
| `rerank.cross_encoder` | ONNX 交叉编码器 | core, onnxruntime | 模型缺失即抛 `DependencyUnavailable`，不静默降级 |
| `llm.mock` | 离线生成，抽取式 | core | 输出必为证据中的原句拼接 |
| `llm.ollama` | 真实生成 | core, httpx | 超时映射为 `UpstreamTimeout` |
| `route.router` | 规则路由 | core | 返回必属 4 个枚举之一，且 `reason` 非空 |
| `agent.loop` | Planner–Executor–Critic | core, llm | 轮次不超上限；`observations` 记录每步 |
| `agent.tools` | 计算器 / 检索 / 语料统计 | core | 计算器拒绝任何非算术逃逸 |
| `verify.groundedness` | 断言拆分 + IDF 加权归因 | core | 证据被篡改后忠实度必降 |
| `pipeline.orchestrator` | 装配与路由分发 | 全部 | `index_is_consistent()` 为真 |
| `trace.tracer` | Span 树 | 无 | 子 span 耗时 ≤ 父 span 耗时 |
| `eval.metrics` | Recall@K / MRR / nDCG / 分位数 | 无 | nDCG ≤ 1，Recall ∈ [0,1] |
| `eval.runner` | 黄金集执行与门禁 | pipeline | 同配置两次运行指标逐位相同 |

---

## 4. 一次问答的完整调用链

以「星枢系统默认采用什么向量索引？」（走 `rag`）为例：

```
POST /chat
 └─ pipeline.answer(query)
     ├─ Tracer 建立 root span
     ├─ span[route]         → route.decide(q, corpus_ready=True)
     │                         命中事实型特征 → RoutePath.RAG
     ├─ span[retrieve]      → retrieve.hybrid.search(q, top_k=20)
     │    ├─ bm25.search         稀疏候选（字面：向量/索引/FAISS）
     │    ├─ dense.search        稠密候选（语义：嵌入 → FAISS 内积）
     │    ├─ rrf_fuse            两路排名倒数融合，无需分数归一化
     │    └─ mmr_select          按 λ=0.7 在相关性与多样性间取平衡
     ├─ span[rerank]        → rerank.llm.rerank(q, candidates, k=5)
     │    └─ 模型返回序号数组；解析失败则原序返回（绝不因重排挂掉整条链）
     ├─ span[generate]      → llm.complete(build_rag_messages(q, evidence))
     │    └─ 提示词强制「每条事实后标注 [n]」
     ├─ span[verify]        → verify.verify(text, evidence, 0.6)
     │    ├─ 拆断言（按句 + 转折词）
     │    ├─ 逐条算 IDF 加权覆盖度
     │    └─ 未过 → 收紧提示重生成（最多 SNX_MAX_REGENERATE 次）
     └─ _finish()           → 按答案里的 [n] 标记挑引用 → Answer
```

**四条链路的差异**

| 链路 | 触发条件 | 差异点 |
|---|---|---|
| `direct` | 寒暄、纯常识、语料为空 | 跳过检索与校验，一次生成即返回。**「你好」0.6 s 而 RAG 要 13 s**，这就是路由存在的意义 |
| `rag` | 事实型问题且有语料 | 单跳检索 |
| `multihop` | 含「和 / 与 / 对比 / 分别」等并列结构 | 拆子查询各自召回后 RRF 融合，再取前 K |
| `agent` | 含算术、换算、多步操作意图 | 走 Planner–Executor–Critic；工具观测**并入证据集**后再校验 |

> **工具观测必须并入证据**，这一条是实测出来的：系统提示词要求「有据可依」，
> 而计算器给出的 `512` 在任何文档里都不存在，若不把它作为证据，
> 归因校验会把**完全正确的工具答案**判为无据。这类「正确却被判错」的故障
> 比「错误却被放过」更难发现，因为它只在 `agent` 链路上出现。

---

## 5. 关键设计决策

完整 ADR 见 [`docs/adr/`](docs/adr/)，此处摘要。

### 5.1 协议 + 零依赖回退，而不是绑定具体库

每个能力都定义为 `typing.Protocol`，提供两个实现：一个生产级、一个零依赖。

| 能力 | 零依赖实现 | 生产实现 |
|---|---|---|
| 嵌入 | `HashEmbedder`（哈希特征，512 维） | `OllamaEmbedder`（BGE-M3，1024 维） |
| 索引 | `NumpyIndex`（暴力内积） | `FaissIndex`（`IndexFlatIP`） |
| 生成 | `MockLLM`（抽取式） | `OllamaLLM`（Qwen2.5） |
| 重排 | `LexicalReranker` | `LLMReranker` / `OnnxCrossEncoder` |

收益：CI 无需模型与网络、指标确定可复现；同时生产路径不被削弱。
代价：两套实现要同步维护。**接受**，因为「可复现」优先于「少写代码」。

### 5.2 检索引擎：FAISS + 自实现 BM25

FAISS 有 `win_amd64` 轮子，零编译可用。BM25 只有 80 行，自己写胜过引入依赖 ——
第三方 BM25 包常自带一套切词器，而我们需要与稠密路**共享同一个切词器**，
否则两路召回的词空间不一致，融合就成了玄学。

### 5.3 重排默认走 LLM listwise

原计划用 ONNX 交叉编码器。实测 ModelScope 上 `bge-reranker` 只提供 PyTorch 权重，
本机无 `torch`（且装了也会拖慢冷启动），因此：

- **默认 `llm`**：复用已就位的 Qwen 做 listwise 排序，零额外下载；
- **保留 `onnx`**：用户自备 `model.onnx` 放入 `SNX_RERANK_MODEL_DIR` 即生效；
- **保留 `lexical`**：离线测试与 88 项单测的确定性来源。

### 5.4 归因校验：IDF 加权词汇覆盖度

不训 NLI 模型，理由：目标不是「判断蕴含」，而是「判断答案里的事实是否能在这几条证据里找到落脚点」。
用 IDF 加权的词项覆盖度即可，且**必须加权** —— 否则「的 / 了 / 是」会让所有断言看起来都被支持，
校验彻底失去区分度（这一点由 `test_verify.py` 的反向断言守着：换掉证据，忠实度必须掉到近 0）。

### 5.5 持久化启动必须重建内存索引

文档在 sqlite，向量索引与 BM25 在内存。进程重启后「库里有数据、检索全落空」。
修法是启动时 `restore()`，且**完成判据是「向量数 == 片段数」而不是「跑过一遍」**。
`test_persistence.py` 断言的是「重启前后检索结果逐位一致」，而不是「restore 被调用了」。

### 5.6 控制台单文件零构建

一份 `web/index.html`，内联全部 CSS/JS/SVG，服务端 `FileResponse` 直接吐出。
测试用 jsdom 真跑内联脚本（30 项断言），而不是正则匹配 HTML 字符串 ——
后者只能证明字符串存在，证明不了交互可用。

---

## 6. 可观测性

`Tracer` 用 `with tracer.span("name", **attrs)` 包裹任意代码块，产出 Span 树：

```json
{
  "name": "root", "duration_ms": 13425.1,
  "children": [
    {"name": "route",    "duration_ms": 1.2,    "attrs": {"path": "rag", "reason": "..."}},
    {"name": "retrieve", "duration_ms": 18.4,   "attrs": {"mode": "rag"}},
    {"name": "rerank",   "duration_ms": 96.1,   "attrs": {"candidates": 20}},
    {"name": "generate", "duration_ms": 13290.7,"attrs": {"evidence": 5}},
    {"name": "verify",   "duration_ms": 18.7,   "attrs": {}}
  ]
}
```

控制台把它渲染成瀑布图。**这个设计直接暴露了瓶颈**：真实模型下 `generate` 占 99% 耗时，
所以优化方向不是调检索参数，而是换更小的模型或上 GPU —— 没有 Span 树就只能靠猜。

---

## 7. 错误处理策略

| 场景 | 策略 | 理由 |
|---|---|---|
| 嵌入后端不可用 | 降级到 Hash，记日志 | 宁可质量降级也不能让整条链断掉 |
| 重排模型输出不可解析 | 原序返回 | 重排是「优化项」，不该成为失败点 |
| ONNX 模型文件缺失 | 抛 `DependencyUnavailable` | 用户显式选了 onnx 后端，静默降级会让人误以为在用 ONNX |
| LLM 超时 | 抛 `UpstreamTimeout`，API 映射 504 | 超时是上游问题，不该伪装成业务错误 |
| 归因校验不通过 | 重生成（有上限）→ 仍不过则标注 `passed=false` | **绝不把不确定的内容当成事实输出** |
| 索引恢复不完整 | 抛 `RuntimeError`，拒绝启动 | 带病启动比启动失败更危险 |

错误码统一在 `core/errors.py`，API 层映射为 HTTP 状态码，不泄漏堆栈。

---

## 8. 已知边界

1. **CPU 推理慢**。Qwen2.5-1.5B 在本机 `generate` 阶段约 13 s；`latency_p95` 门禁在真实模型下不适用，
   因此门禁只跑离线路径。
2. **中文切词是轻量的**。单字 + 二字组已够用（黄金集 Recall@5 = 1.0），
   但替换为 `jieba` 或 BGE-M3 自带 tokenizer 会更好，接口已留好。
3. **路由是规则实现**。复杂意图会误判；协议已定义，可整体替换为模型分类器。
4. **无鉴权**。服务默认绑 `127.0.0.1`；对外暴露前必须加网关鉴权。
5. **重排用 LLM 会增加一次生成**。离线后端下成本为零，真实模型下每个查询多约 1–3 s。
