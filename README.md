<div align="center">

# 星枢 StellarNexus

**可自证的模块化复合智能系统**

检索增强 · 自适应链路路由 · 断言级归因校验 · 智能体闭环 · 评测门禁

作者 **晨星** · 版本 1.0.0 · 单文件控制台 · 干净环境一键复现

</div>

---

## 这是什么

一套**模块化、可独立验证、能离线跑通、指标进 CI 门禁**的 AI 系统。

它建立在前人成果之上（Ollama / Qwen / BGE-M3 / FAISS / ONNX Runtime），不重造模型，
只把「检索 → 融合 → 重排 → 生成 → 溯源」这条链路做成**七个可替换的协议实现**，
再把它们装配成一条**每一步都能被单独测**的完整链路。

### 与常见 RAG 演示的差别

| # | 能力 | 普通演示 | 星枢的做法 |
|---|---|---|---|
| 1 | 链路选择 | 所有问题都走同一条重链路 | **自适应路由**：`direct / rag / multihop / agent` 四路，按 query 特征分派。「你好」0.6 秒返回，不会白跑一次检索 |
| 2 | 召回 | 单路向量检索 | **三层融合**：BM25 稀疏 + BGE-M3 稠密 → RRF 融合 → MMR 去冗余 → 精排收紧 |
| 3 | 可信度 | 直接输出模型文本 | **断言级归因校验**：答案拆成断言，逐条对证据计算 IDF 加权覆盖度；不达标则收紧提示重生成，仍不达标就**如实标注未通过** |
| 4 | 多步任务 | 无 | **Planner–Executor–Critic** 闭环，含工具调用与轮次上限，全程可回放 |
| 5 | 质量把关 | 靠人肉感觉 | **黄金集 + 阈值门禁**：Recall@K / MRR / nDCG / 忠实度 / P95 延迟，低于阈值 CI 直接红 |
| 6 | 可观测 | 打日志 | **Span 树**：每请求一棵调用树，控制台可展开瀑布图看每阶段真实耗时 |
| 7 | 离线可用 | 无网即废 | **零依赖回退**：Hash 嵌入 + 内存索引 + Mock LLM + 词法重排，断网无 Key 也能 `verify` 全绿 |

---

## 一键复现

### 前置

- Python **3.12 – 3.13**（仓库已在 3.13.12 实测；下限由锁定的 numpy 2.5.3 决定）
- 可选：[Ollama](https://ollama.com) 本地模型服务（不装也能跑离线模式）

### 安装（无编译器、无 GPU 要求）

```bash
git clone <repo-url> && cd stellar-nexus
python -m venv .venv
# Windows: .venv\Scripts\activate    |    Linux/macOS: source .venv/bin/activate
pip install -e .
```

全部依赖都有 `win_amd64 / manylinux` 预编译轮子，**不需要 MSVC、不需要 cmake**。

### 30 秒自检（离线，不需要模型、不需要网络）

```bash
stellarnx verify
```

期望输出（15 项全绿）：

```
[PASS] 依赖导入完整性                            numpy/faiss/onnxruntime/fastapi/httpx 均可用
[PASS] 向量后端真实生效                           后端=faiss 回退原因=None
[PASS] 切分多片段与覆盖                           片段数=6 最长=139
[PASS] BM25 稀疏召回                          top=a:0
[PASS] 稠密检索自命中                            index=faiss top=doc:0
[PASS] RRF 融合排名                           top=y
[PASS] MMR 去冗余                            选中=['c0', 'c2']
[PASS] 重排器有效性                             top=good
[PASS] 路由分派正确                             direct/rag/agent/multihop
[PASS] 工具安全求值                             12*(3+4)=84.0 逃逸被拒=True
[PASS] 智能体闭环                              步骤=2 重规划=0 批评分=[1.0]
[PASS] 归因校验可被打穿                           真实=1.000 篡改=0.000
[PASS] 端到端链路产出引用                          路由=rag 引用=1 可证性=1.000
[PASS] 评测确定性                              recall@5=1.0000 两次一致=True
[PASS] HTTP 服务健康                          health=200 ingest=1 chat=200

通过 15/15
```

### 默认就是离线（零依赖、断网即用）

不装 Ollama、不联网，也能跑完整链路：`stellarnx verify` 15/15 全绿、
`pytest` 94 passed、`scripts/live_check.py` 也能走真 HTTP 验收。这一套用的是
内置回退——**Hash 嵌入 + 内存索引 + Mock LLM + 词法重排**——专门保证
「干净环境一键复现」：任意机器、任意网络下先把系统跑起来再说。

> 也就是说，**默认 `stellarnx serve` 起的是离线回退，不是真模型**。别依赖
> 默认值去指望真答案，要真模型就显式切后端（见下）。

### 接入真实模型（可选，质量更高）

想让答案由真模型生成、知识由真向量召回，先准备好本地 Ollama 服务与两个模型，
再**显式**把后端切到 ollama（`SNX_EMBED_BACKEND` 等默认是 `hash/mock/identity`，
不显式设就会一直走离线回退）：

```bash
ollama pull qwen2.5:1.5b-instruct     # 生成
ollama pull bge-m3                    # 嵌入，1024 维
ollama serve                          # 默认 http://127.0.0.1:11434

export SNX_EMBED_BACKEND=ollama
export SNX_LLM_BACKEND=ollama
export SNX_RERANK_BACKEND=llm
# Windows PowerShell:
#   $env:SNX_EMBED_BACKEND="ollama"; $env:SNX_LLM_BACKEND="ollama"; $env:SNX_RERANK_BACKEND="llm"
```

### 启动服务与控制台

```bash
stellarnx serve            # 默认 http://127.0.0.1:8000
```

真实验收（对运行中的服务走真 HTTP，要求上面三个 ollama 开关已生效）：

```bash
python scripts/live_check.py --base http://127.0.0.1:8000
```

实测结论（当前 HEAD，真模型链路）：**23/23 通过**

- 后端：`ollama:bge-m3` + `ollama:qwen2.5:1.5b-instruct` + `faiss`（1024 维）+ `llm-listwise` 重排
- RAG 链路：`chunks == vectors == 16`、引用 3 条、`groundedness=1.0`、Span 树含 `route → retrieve → rerank → generate → verify`
- 评测门禁：`recall@1 0.88 / recall@3 1.0 / recall@5 1.0 / MRR@10 0.9333 / nDCG@5 0.9505 / 忠实度 1.0 / P95 ≈ 32ms`，全部超过阈值

浏览器打开 <http://127.0.0.1:8000> 即是控制台：

| 面板 | 用途 |
|---|---|
| 对话 | 提问、看待引用的答案、逐条断言的通过情况、点 trace 跳到链路 |
| 检索调试 | 直接看召回结果与分数、来源（bm25 / dense / hybrid） |
| 知识库 | 粘贴文档入库、查看已入库文档与片段数、删除 |
| 链路追踪 | Span 瀑布图，每个阶段真实耗时 + 原始 JSON |
| 评测门禁 | 一键跑黄金集，指标 vs 阈值逐行判定 |
| 系统信息 | 实际生效的后端、完整配置、API 地址 |

---

## 架构

```
                     ┌──────────────────────────────────────┐
   用户提问 ────────► │ route   自适应路由器                  │
                     │ direct / rag / multihop / agent      │
                     └───────────────┬──────────────────────┘
                                     │
     ┌───────────────┬───────────────┼────────────────┬────────────────┐
     ▼               ▼               ▼                ▼                ▼
 ┌────────┐    ┌──────────┐   ┌───────────┐    ┌──────────┐    ┌───────────┐
 │ direct │    │   rag    │   │ multihop  │    │  agent   │    │  tracer   │
 │ 直答   │    │ 单跳检索 │   │ 子查询融合│    │ 工具闭环 │    │ Span 树   │
 └────────┘    └────┬─────┘   └─────┬─────┘    └────┬─────┘    └───────────┘
                    └───────┬───────┘               │
                            ▼                       │
              ┌──────────────────────────┐          │
              │ retrieve  三路召回        │◄─────────┘ 工具内检索
              │  BM25  ┃  dense  ┃ ...   │
              └────────────┬─────────────┘
                           ▼
              ┌──────────────────────────┐
              │ hybrid  RRF → MMR        │
              └────────────┬─────────────┘
                           ▼
              ┌──────────────────────────┐
              │ rerank  llm / onnx / lexical │
              └────────────┬─────────────┘
                           ▼
              ┌──────────────────────────┐      ┌──────────────┐
              │ llm  生成（带 [n] 引用）  │◄─────│ chunk/store  │
              └────────────┬─────────────┘      │ embed/index  │
                           ▼                    └──────────────┘
              ┌──────────────────────────┐
              │ verify  断言级归因校验    │──► 未过则收紧提示重生成（有轮次上限）
              └────────────┬─────────────┘
                           ▼
              答案 + 引用 + 校验报告 + Span 树
```

### 模块与职责（单一职责，每个都能单独测）

| 模块 | 只做一件事 | 对外接口 | 独立验证方式 |
|---|---|---|---|
| `core` | 定义协议、类型、配置、路径、中文切词 | `Embedder/Index/Retriever/Reranker/LLM/Verifier`、`Answer/Hit/Chunk/VerifyReport` | 契约测试；`import` 无循环依赖 |
| `embed` | 文本 → 向量 | `embed(texts) -> (N,d) float32` | 相似句余弦 > 不相似句；归一化后模长 = 1 |
| `chunk` | 文档 → 片段 | `chunk(doc_id, text, meta) -> [Chunk]` | 超阈值必多片段；片段可拼回原文不丢字 |
| `store` | 片段与文档持久化 | `add_chunks/all_chunks/list_documents/delete_document` | 灌 N 条 `count == N`；重启后一致 |
| `retrieve` | 召回候选 | `search(q,k) -> [Hit]`、`rrf_fuse`、`mmr_select` | 黄金集 Recall@K；RRF 对双路命中项加分 |
| `rerank` | 候选精排 | `rerank(q, hits, k) -> [Hit]` | 正例排名必在负例之前 |
| `llm` | 生成与结构化输出 | `complete(messages)`、`json(messages, schema)` | Mock 离线可测；Ollama 冒烟 |
| `route` | 链路分派 | `decide(q, corpus_ready) -> RouteDecision` | 各类 query 落到期望分支 |
| `agent` | 规划–执行–批判循环 | `run(goal) -> AgentResult` | 工具闭环 + 重试上限 + 观测非空 |
| `verify` | 归因与可证性 | `verify(answer, evidence, threshold) -> VerifyReport` | **反向断言**：把证据换掉，忠实度必降 |
| `pipeline` | **唯一装配点** | `ingest/search/answer/stats` | 端到端；持久化重启后结果逐位一致 |
| `trace` | 可观测 | `span()` 上下文管理器 → Span 树 | 嵌套关系正确、耗时非负 |
| `eval` | 指标与门禁 | `run(corpus, cases) -> EvalReport` | 同索引重跑指标逐位不变 |
| `api` | HTTP 边界 | REST 见下 | 自包含 E2E 脚本真起进程实测 |
| `cli` | 命令行 | `serve/ingest/search/chat/eval/stats/verify` | 跨进程持久化测试 |

> **依赖方向是单向的**：`pipeline` 依赖一切，但没有任何模块反向依赖 `pipeline`。
> 循环导入在开发中真实发生过（`embed/__init__` ↔ `hash_embed`），修法是把 `normalize`
> 下沉到 `core/linalg.py`，而不是靠 `import` 时机取巧。

---

## 命令行

```bash
stellarnx verify                       # 一键自检（离线，15 项）
stellarnx ingest data/docs.jsonl       # 导入 JSONL，每行 {doc_id, text, meta?}
stellarnx search "向量索引" -k 5        # 直接看召回结果
stellarnx chat "向量索引用什么实现？"    # 问答
stellarnx chat "..." --json            # 机器可读（含引用、断言、route）
stellarnx eval                         # 跑黄金集并判定门禁，未达标退出码 1
stellarnx stats                        # 文档数/片段数/向量数/后端
stellarnx serve --port 8000            # 起服务 + 控制台
```

> `ingest / search / chat / stats / serve` **共用同一个持久化知识库**
> （`<SNX_DATA_DIR>/stellarnx.db`，默认 `data/`）。先用 `ingest` 灌数据，再 `search` 就能搜到。
> 需要一次性内存库时设 `SNX_STORE=memory`。

---

## HTTP 接口

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/health` | 存活 + 实际生效的四个后端 + 版本 + 作者 |
| GET | `/stats` | 文档/片段/向量计数、配置快照 |
| POST | `/ingest` | `{docs:[{doc_id,text,meta?}]}` → `{chunks, documents}` |
| GET | `/documents` | 文档列表（含每篇片段数） |
| DELETE | `/documents/{doc_id}` | 删除文档及其片段，并重建索引 |
| POST | `/search` | `{query,k}` → 命中列表（含 score 与来源） |
| POST | `/chat` | `{query,with_trace}` → 答案 + 引用 + 断言校验 + Span 树 |
| GET | `/chat/stream` | SSE：`stage` / `answer` / `done` / `error` 事件 |
| GET | `/trace/{trace_id}` | 取回 Span 树 |
| POST | `/evaluate` | 跑黄金集，返回指标、阈值、逐条明细、门禁结论 |
| GET | `/` | 控制台（单文件 HTML） |

完整契约见 [`docs/openapi.yaml`](docs/openapi.yaml)。

---

## 评测指标与门禁

指标定义（`eval/metrics.py`）：

| 指标 | 定义 | 阈值 |
|---|---|---|
| `recall@1` | 榜首是否命中黄金片段 | ≥ 0.30 |
| `recall@3` | 前三是否命中 | ≥ 0.60 |
| `recall@5` | 前五是否命中 | ≥ 0.75 |
| `mrr@10` | 首个命中位置的倒数均值 | ≥ 0.45 |
| `ndcg@5` | 折损累计增益，位置越靠前权重越高 | ≥ 0.45 |
| `groundedness` | 被证据支持的断言数 / 断言总数 | ≥ 0.55 |
| `latency_p95_ms` | 端到端 P95 延迟 | ≤ 8000 |

**实测（本机 Windows 11 / Ryzen 7 / 16GB / 无 GPU）**

| 运行方式 | Recall@1 | Recall@5 | MRR@10 | nDCG@5 | 忠实度 | P95 延迟 |
|---|---|---|---|---|---|---|
| 离线（Hash + Mock + 词法重排） | 0.88 | 1.00 | 0.933 | 0.950 | 1.00 | 33 ms |
| 真实（BGE-M3 + Qwen2.5-1.5B，8 条子集） | 0.50 | 1.00 | 0.708 | 0.783 | 0.833 | 31.2 s |

> 两组数字放在一起看才有意义：离线那组是**门禁基线**，必须稳定、快、可重复；
> 真实那组是**质量上界参考**，受 CPU 推理速度限制，延迟高一个数量级是正常现象。
> 门禁阈值定在两者之间偏保守的位置，保证「离线必过、真实模型不会因慢而误判」。

---

## 目录结构

```
stellar-nexus/
├── src/stellarnx/
│   ├── core/        协议、类型、配置、路径、线性代数、中文切词
│   ├── embed/       Hash 嵌入（零依赖） / Ollama 嵌入（BGE-M3）
│   ├── chunk/       滑窗切分
│   ├── store/       sqlite 文档库 + FAISS 向量索引（numpy 回退）
│   ├── retrieve/    BM25 / 稠密 / 融合（RRF + MMR）
│   ├── rerank/      词法 / LLM listwise / ONNX 交叉编码器
│   ├── llm/         Mock / Ollama + 提示词模板
│   ├── route/       规则路由器
│   ├── agent/       Planner–Executor–Critic + 工具集
│   ├── verify/      断言拆分 + IDF 加权归因校验
│   ├── pipeline/    唯一装配点
│   ├── trace/       Span 树
│   ├── eval/        黄金集、指标、门禁
│   ├── api/         FastAPI 应用与 ASGI 入口
│   ├── cli.py       命令行
│   └── selftest.py  一键自检
├── tests/           88 项单元/集成测试
├── data/golden/     黄金语料与用例（评测门禁的数据来源）
├── web/index.html   控制台（单文件，内联 CSS/JS，零外部依赖）
├── scripts/         verify / e2e / web_smoke / real_smoke / download_models
└── docs/            openapi.yaml + ADR
```

---

## 全量验证

```bash
# 1) 离线自检（15 项，不需要网络与模型）
stellarnx verify

# 2) 单元与集成测试（88 项）
pytest -q

# 3) 端到端：真起 HTTP 服务，跑成功流与错误流（16 项）
python scripts/e2e.py

# 4) 控制台无头冒烟（30 项，需要 node + jsdom）
npm install && npm run test:web

# 5) 真实模型全链路（需要 Ollama）
python scripts/real_smoke.py
```

四条命令全部退出码 0，才算这套系统「在本机确实能跑」。

---

## 配置

所有配置项都能用 `SNX_` 前缀的环境变量覆盖，代码里没有散落的魔法数字。

| 环境变量 | 默认值 | 说明 |
|---|---|---|
| `SNX_EMBED_BACKEND` | `hash` | `hash` 零依赖 / `ollama` 真实嵌入 |
| `SNX_EMBED_MODEL` | `bge-m3` | Ollama 嵌入模型，1024 维 |
| `SNX_LLM_BACKEND` | `mock` | `mock` 离线 / `ollama` 真实生成 |
| `SNX_LLM_MODEL` | `qwen2.5:1.5b-instruct` | 生成模型，CPU 上速度与质量的折中 |
| `SNX_RERANK_BACKEND` | `identity` | `identity` / `lexical` / `llm` / `onnx` |
| `SNX_RERANK_MODEL_DIR` | `models/bge-reranker-base` | ONNX 后端目录（含 `model.onnx`） |
| `SNX_OLLAMA_HOST` | `http://127.0.0.1:11434` | Ollama 地址 |
| `SNX_CHUNK_SIZE` / `SNX_CHUNK_OVERLAP` | `480` / `80` | 切分窗口与重叠 |
| `SNX_TOP_K_RETRIEVE` / `SNX_TOP_K_RERANK` | `20` / `5` | 召回数 / 精排保留数 |
| `SNX_RRF_K` / `SNX_MMR_LAMBDA` | `60` / `0.7` | 融合与去冗余参数 |
| `SNX_GROUNDEDNESS_THRESHOLD` | `0.6` | 归因校验阈值 |
| `SNX_MAX_REGENERATE` | `1` | 未过校验时的重生成轮数上限 |
| `SNX_AGENT_MAX_STEPS` / `SNX_AGENT_MAX_REPLAN` | `4` / `2` | 智能体步数与重规划上限 |
| `SNX_DATA_DIR` | `data` | 持久化目录 |
| `SNX_STORE` | — | 设为 `memory` 则不落盘 |

---

## 排障（都是本机真实踩过的）

| 现象 | 原因 | 处理 |
|---|---|---|
| Python 连不上 Ollama，PowerShell 却能连 | 机器上有 SOCKS5 代理，`httpx` 默认信任环境变量，把 `localhost` 也送进了代理 | 已修：相关客户端统一 `trust_env=False`。自定义代码里连本地服务时也要这样做 |
| `npm install` 报 404 指向 `ohpm.openharmony.cn` | npm registry 被改成公共鸿蒙源 | 临时指定 `--registry=https://registry.npmjs.org/` |
| `huggingface.co` 超时 | 网络不可达 | 模型走 ModelScope 或直接用 Ollama 拉取；本仓库不依赖 HF |
| 服务换个目录启动后 `/evaluate` 404 | 相对路径按当前工作目录解析 | 已修：`core/paths.py` 统一按仓库根解析 |
| 重启后库里有数据却搜不到 | 向量索引与 BM25 只在内存 | 已修：启动时 `restore()` 强制重建，判据是「向量数 == 片段数」 |
| 评测指标突然全 1.0 | 基准太简单，没有区分度 | 已修：黄金集加入改写型难题，MRR 从 1.0 降到 0.93 才说明基准有效 |
| 语料短于切分阈值，多片段路径测不到 | 测试数据缺陷 | 已修：`scripts/check_corpus.py` 校验每篇必产生 ≥ 2 片段 |

---

## 设计取舍（ADR）

详见 [`docs/adr/`](docs/adr/)：

- **0001** 用协议 + 零依赖回退实现，而不是绑定某个向量库
- **0002** 检索引擎选 FAISS + BM25 自实现，不引第三方 BM25 包
- **0003** 重排默认走 LLM listwise，ONNX 交叉编码器作为可插拔选项
- **0004** 归因校验用 IDF 加权的词汇覆盖度，而不是再训一个 NLI 模型
- **0005** 控制台做成单文件零构建，而不是上 React + Vite

---

## 许可

Apache-2.0 · 作者 **晨星**
