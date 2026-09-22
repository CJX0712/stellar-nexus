# 星枢 StellarNexus 规格说明（SPEC）

作者 **晨星** · 版本 1.0.0

本文是**接口契约**：每个模块对外暴露什么、参数与返回值的确切形状、以及**违反契约时应该发生什么**。
架构与设计理由见 [`ARCHITECTURE.md`](ARCHITECTURE.md)。

约定：
- 所有 `Hit` 的 `score` 为浮点，**越大越相关**（不同实现的量纲不必可比）。
- 所有返回列表的方法**不得返回 `None`**，无结果时返回空列表。
- 所有「形状」断言都是可测的，`tests/` 里逐条有对应测试。

---

## 1. 契约层 `core`

### 1.1 协议 `core/contracts.py`

```python
class Embedder(Protocol):
    name: str
    dim: int
    def embed(self, texts: list[str]) -> np.ndarray: ...   # -> (N, dim) float32

class VectorIndex(Protocol):
    name: str
    def add(self, ids: list[str], vectors: np.ndarray) -> None: ...
    def search(self, vectors: np.ndarray, k: int) -> list[list[tuple[str, float]]]: ...
    def delete(self, ids: list[str]) -> None: ...
    def clear(self) -> None: ...
    def count(self) -> int: ...

class Retriever(Protocol):
    name: str
    def search(self, query: str, k: int) -> list[Hit]: ...

class Reranker(Protocol):
    name: str
    def rerank(self, query: str, hits: list[Hit], k: int) -> list[Hit]: ...

class LLM(Protocol):
    name: str
    def complete(self, messages: list[dict[str, str]], **kwargs: Any) -> str: ...
    def is_available(self) -> bool: ...

class Verifier(Protocol):
    name: str
    def verify(self, answer: str, evidence: list[Hit], threshold: float) -> VerifyReport: ...

class Tool(Protocol):
    name: str
    description: str
    def run(self, **kwargs: Any) -> dict[str, Any]: ...
```

**硬性约束**

| 编号 | 约束 | 违反后果 |
|---|---|---|
| C-1 | `embed([])` 返回形状 `(0, dim)`，不抛异常 | 空语料入库会崩 |
| C-2 | `embed` 返回 `float32` | 与 FAISS 交互时隐式转换，慢且可能精度异常 |
| C-3 | `index.search(v, k)` 返回长度 == `v` 行数，每行 ≤ k 个 `(id, score)`，按 score 降序 | 调用方按行取用会错位 |
| C-4 | `rerank` 返回长度 ≤ k，且元素必来自入参 `hits` | 引入不存在片段 → 引用悬空 |
| C-5 | `count()` == 已 `add` 的唯一 id 数（`delete` 后同步减少） | 与 `DocumentStore.count_chunks()` 不一致 → 检索漏召回 |
| C-6 | 任何实现抛出的异常必须是 `StellarNexusError` 子类 | API 层错误映射失效，500 泄漏堆栈 |

### 1.2 数据类型 `core/types.py`

```python
class RoutePath(str, Enum):
    DIRECT = "direct"; RAG = "rag"; MULTIHOP = "multihop"; AGENT = "agent"

@dataclass(slots=True)
class Hit:
    chunk_id: str; doc_id: str; text: str; score: float
    source: str = "dense"; meta: dict = {}

@dataclass(slots=True)
class Chunk:
    id: str; doc_id: str; text: str; ordinal: int = 0; meta: dict = {}

@dataclass(slots=True)
class Citation:
    chunk_id: str; doc_id: str; text: str; score: float

@dataclass(slots=True)
class Claim:
    text: str; supported: bool; score: float; evidence_id: str | None = None

@dataclass(slots=True)
class VerifyReport:
    groundedness: float; claims: list[Claim]
    threshold: float; passed: bool; detail: dict = {}

@dataclass(slots=True)
class RouteDecision:
    path: RoutePath; reason: str; confidence: float; features: dict = {}

@dataclass(slots=True)
class Answer:
    text: str
    citations: list[Citation] = []
    route: RouteDecision | None = None
    verification: VerifyReport | None = None
    trace_id: str = ""
    latency_ms: float = 0.0
    metadata: dict = {}
```

**关键语义**

- `groundedness ∈ [0, 1]`，定义是 `被证据支持的断言数 / 断言总数`。
- **断言总数为 0 时 `groundedness == 0.0`，不是 1.0**。空答案不算「可证」——
  这一条决定了「模型输出空串」不会骗过门禁。
- `Answer.to_dict()` 必须整体 `json.dumps` 可序列化（`trace` 也会被塞进 `metadata`）。

### 1.3 异常 `core/errors.py`

```
StellarNexusError            基类
├── ConfigError              配置非法（未知项、越界值）
├── DependencyUnavailable    依赖缺失（ONNX 模型文件、Ollama 不可达）
├── UpstreamTimeout          上游超时
├── VerificationFailed       校验未通过且超过重生成上限
└── NotFound                 请求的资源不存在
```

HTTP 映射：`NotFound → 404`、`UpstreamTimeout → 504`、`DependencyUnavailable → 503`、
`ConfigError → 500`、`StellarNexusError → 500`（均返回 `{detail, code}`，不泄漏堆栈）。

### 1.4 路径 `core/paths.py`

```python
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"
GOLDEN_DIR = DATA_DIR / "golden"
WEB_DIR = PROJECT_ROOT / "web"
MODELS_DIR = PROJECT_ROOT / "models"
CORPUS_PATH = GOLDEN_DIR / "corpus.jsonl"
QUESTIONS_PATH = GOLDEN_DIR / "questions.jsonl"

def resolve(path_like: str | Path) -> Path: ...   # 相对路径按仓库根解析
```

**约束**：任何内置数据文件的默认位置都必须经 `resolve()` 或上述常量得到，
**不得**出现相对当前工作目录的裸字符串。`/evaluate` 曾因此 404。

### 1.5 切词 `core/text.py`

```python
def tokenize(text: str) -> list[str]: ...
```

产出四类 token：CJK 单字 `cX`、CJK 相邻二字组 `bXY`、ASCII 词 `wfoo`、数字同上。
**约束**：确定性 —— 同一输入两次调用结果逐位相同；不引入词典或模型。

---

## 2. 基础设施层

### 2.1 `embed`

```python
def build_embedder(cfg: Config) -> Embedder: ...

class HashEmbedder:   # name="hash", dim=512（默认）
    def __init__(self, dim: int = 512) -> None: ...
class OllamaEmbedder: # name="ollama:<model>", dim 由首次探测得到（bge-m3 -> 1024）
    def __init__(self, cfg: Config, model: str | None = None, host: str | None = None) -> None: ...
```

**验收断言**：相似句余弦相似度 > 不相似句；归一化后每行 `‖v‖ ≈ 1`（全零行除外）。
`OllamaEmbedder` 必须 `httpx.Client(trust_env=False)` —— 否则本地代理会把 `localhost` 请求也劫持。

### 2.2 `chunk`

```python
class Chunker:
    def __init__(self, size: int = 480, overlap: int = 80) -> None: ...
    def chunk(self, doc_id: str, text: str, meta: dict | None = None) -> list[Chunk]: ...

def split_text(text: str, size: int = 480, overlap: int = 80) -> list[str]: ...
```

**验收断言**

1. `text` 长度 > `size` 时，片段数 ≥ 2。
2. 相邻片段有 `overlap` 字符重叠。
3. `chunk_id` 形如 `<doc_id>:<ordinal>`，`ordinal` 从 0 连续递增。
4. 空白输入返回 `[]`，不返回 `[Chunk(text="")]`。

### 2.3 `store`

```python
class DocumentStore:
    def add_document(self, doc_id, title="", source="", meta=None) -> None
    def add_chunks(self, chunks: list[Chunk]) -> None
    def get_chunk(self, chunk_id: str) -> Chunk | None
    def get_chunks(self, chunk_ids: list[str]) -> list[Chunk]      # 缺失 id 静默跳过
    def chunks_of_doc(self, doc_id: str) -> list[Chunk]            # 按 ordinal 升序
    def all_chunks(self) -> list[Chunk]                            # 按 (doc_id, ordinal)
    def list_documents(self) -> list[dict]                         # 含 "chunks" 片段数
    def delete_document(self, doc_id: str) -> int                  # 返回删除的片段数
    def clear(self) -> None
    def count_chunks(self) -> int
    def count_documents(self) -> int
    def close(self) -> None

def build_index(dim: int, prefer: str = "faiss") -> InMemoryIndex | FaissIndex: ...
```

**验收断言**：灌 N 条 → `count_chunks() == N`；`delete_document` 后 `count_documents()` 递减；
`list_documents()` 每条必含 `chunks` 字段且等于该文档实际片段数（一次 JOIN 算出，不做 N+1 查询）。

---

## 3. 能力层

### 3.1 `retrieve`

```python
class BM25Retriever:                     # k1=1.5, b=0.75
    def index(self, chunks: list[Chunk]) -> None
    def all_ids(self) -> list[str]
    def scores(self, query: str) -> list[float]   # 与 _chunks 等长
    def search(self, query: str, k: int) -> list[Hit]   # 仅返回 score > 0

class DenseRetriever:
    def __init__(self, embedder, index, store) -> None
    def search(self, query: str, k: int) -> list[Hit]

def rrf_fuse(ranked_lists: list[list[Hit]], k: int = 60) -> list[Hit]
def mmr_select(query_vec, hits, embedder, k, lam=0.7) -> list[Hit]

class HybridRetriever:
    def __init__(self, retrievers, embedder=None, rrf_k=60, mmr_lambda=0.7) -> None
    def search(self, query: str, k: int) -> list[Hit]
```

**验收断言**

| 断言 | 说明 |
|---|---|
| 查询词全不在语料 → `search` 返回 `[]` | 不允许返回一堆 0 分噪音 |
| 自查询首位必为自身片段 | 稠密路自洽性 |
| RRF：同时出现在两路的项排名 ≥ 仅出现在一路的项 | 融合真的在起作用，不只是拼接 |
| MMR：高度重复的片段不会被同时选中 | 去冗余有效 |

### 3.2 `rerank`

```python
def build_reranker(cfg: Config, llm=None) -> Reranker: ...

class IdentityReranker:            # name="identity"  原序截断
class LexicalReranker:             # name="lexical"   IDF 加权词项覆盖度
    def fit(self, corpus: list[str]) -> "LexicalReranker"
class LLMReranker:                 # name="llm-listwise"  默认
    def __init__(self, llm: LLM, window: int = 8) -> None
class OnnxCrossEncoderReranker:    # name="onnx-cross-encoder"
    def __init__(self, model_dir: str, max_length: int = 512) -> None
    def score(self, query: str, text: str) -> float
```

**验收断言**

- 正例（真正相关的片段）在重排后排名必须高于负例。
- **LLM 输出不可解析时返回原序**，绝不抛异常。重排是优化项，不该成为失败点。
  解析策略依次为：严格 JSON 数组 → 宽松数字抽取 → 放弃并原序返回。
- `OnnxCrossEncoderReranker` 在模型文件缺失时抛 `DependencyUnavailable`（**不静默降级**）。

### 3.3 `llm`

```python
def build_llm(cfg: Config) -> LLM: ...
def build_rag_messages(question: str, passages: list[Hit]) -> list[dict[str, str]]: ...

class MockLLM:      # name="mock"     抽取式：只从证据里挑原句，离线确定性
class OllamaLLM:    # name="ollama:<model>"
```

提示词契约（`build_rag_messages`）：每条资料标注 `[n]`，要求模型答句中用 `[n]` 回指。
`verify.extract_citations` 依赖这个格式把答案与证据对齐。

### 3.4 `route`

```python
class ComplexityRouter:
    def decide(self, query: str, corpus_ready: bool) -> RouteDecision
```

**验收断言**：`path` 必属 4 个枚举；`reason` 非空；`features` 至少含
`has_question_mark / corpus_ready / calc_hint / compare_hint`。

分派规则（实测校准）：

| 条件 | 结果 |
|---|---|
| 语料为空 | `DIRECT`（无据可查，检索无意义） |
| 寒暄/致谢/自我介绍 | `DIRECT` |
| 含算术表达式且期望数值结果 | `AGENT` |
| 含「和 / 与 / 对比 / 比较 / 分别 / 、」等并列结构 | `MULTIHOP` |
| 其余事实型问题 | `RAG` |

> 校准过程中的坑：最初把「**多少**」当作计算意图的关键词，
> 结果「星枢系统有多少个模块？」这类**概念题**被误判为计算题送去工具链路。
> 现在只认「表达式 + 数值意图」的组合，见 `tests/test_router.py`。

### 3.5 `agent`

```python
@dataclass
class Step:        tool: str; args: dict; note: str = ""

@dataclass
class AgentResult:
    goal: str; answer: str
    steps: list[dict]; observations: list[str]
    critic_scores: list[float]; replans: int; evidence: list[Hit]
    def to_dict(self) -> dict     # metadata["agent"] 的形状

class HeuristicPlanner:
    def plan(self, goal, history: list[str], budget: int) -> list[Step]
class Executor:
    def run(self, step: Step) -> dict
class Critic:                       # accept_threshold=0.35
    def score(self, goal, observations, results) -> float
class AgentLoop:
    def run(self, goal: str) -> AgentResult

class ToolRegistry:
    def register(self, name, description, func) -> None
    def names(self) -> list[str]
    def describe(self) -> list[dict]
    def run(self, name: str, **kwargs) -> dict
def build_default_tools(retriever: Retriever | None = None) -> ToolRegistry
def safe_eval(expression: str) -> float
```

**验收断言**

1. 步数 ≤ `SNX_AGENT_MAX_STEPS`，重规划数 ≤ `SNX_AGENT_MAX_REPLAN`。
2. `safe_eval` 只接受算术 AST（`BinOp/UnaryOp/Num/Constant`），
   `__import__("os")` 之类一律抛错。
3. `observations` 非空 —— 否则「规划了但没执行」会静默通过。

### 3.6 `verify`

```python
def extract_citations(text: str) -> list[int]      # 抽取 [1] [2] 形式的编号
def split_claims(answer: str) -> list[Claim]       # 按句切分 + 转折词
class EvidenceVerifier:                            # unsupported_penalty=0.35
    def verify(self, answer, evidence, threshold=0.6) -> VerifyReport
class LLMVerifier:                                 # 可选，用模型做蕴含判定
    def verify(self, answer, evidence, threshold=0.6) -> VerifyReport
```

**算法**：对每条断言，计算其 token 在证据中的 IDF 加权覆盖度；未覆盖部分按
`unsupported_penalty` 折扣。`groundedness = 支持断言数 / 总断言数`。

**验收断言（反向测试）**

- 答案与证据一致 → `groundedness` 接近 1。
- **把证据替换成无关文本 → `groundedness` 必须显著下降**（实测 1.000 → 0.000）。
  只测「正确时分数高」是不够的，那只证明它会给高分，证明不了它能识别错误。
- 空答案 → `groundedness == 0.0`、`passed == False`。

---

## 4. 编排层 `pipeline`

```python
class NexusSystem:
    def __init__(self, cfg: Config | None = None, store_path: str = ":memory:") -> None
    def ingest_text(self, doc_id, text, meta=None) -> int
    def ingest_many(self, docs: list[dict]) -> int          # 元素需含 doc_id, text
    def search(self, query: str, k: int | None = None) -> list[Hit]
    def answer(self, query: str) -> Answer
    def restore(self) -> int                                 # 从存储重建索引，返回片段数
    def index_is_consistent(self) -> bool
    def clear(self) -> None
    def stats(self) -> dict

def build_system(cfg=None, store_path: str | None = None) -> NexusSystem
```

**验收断言**

1. `ingest_many` 后 `store.count_chunks() == index.count() == len(bm25.all_ids())`。
2. **`restore()` 的完成判据是「向量数 == 片段数」**，不等则抛 `RuntimeError`。
   不一致时带病启动，比启动失败更危险。
3. 重启前后同一 query 的检索结果**逐位一致**。
4. `answer()` 在四种路由下的返回：
   - `direct`：`verification is None`，`citations == []`
   - `rag` / `multihop`：`citations` 非空，`verification` 非空
   - `agent`：`metadata["agent"]` 存在，且**工具观测被并入证据**后再校验
5. `Answer.metadata["trace"]` 是完整 Span 树。

---

## 5. 可观测层 `trace`

```python
@dataclass
class Span:
    name: str; trace_id: str; start_ms: float
    end_ms: float | None; attrs: dict; children: list[Span]
    @property duration_ms -> float
    def to_dict(self) -> dict

class Tracer:
    def __init__(self, trace_id: str | None = None) -> None   # 缺省 12 位 hex
    def span(self, name: str, **attrs) -> ContextManager[Span]
    def finish(self) -> Span
    def to_dict(self) -> dict
    def flat(self) -> list[Span]
```

**验收断言**：子 span 的 `duration_ms` ≤ 父 span；`attrs` 可 JSON 序列化；
`span()` 上下文里抛异常时 `end_ms` 仍被填上（`finally` 保证），且异常继续向上抛。

---

## 6. 评测层 `eval`

```python
def recall_at_k(retrieved: Sequence[str], gold: Sequence[str], k: int) -> float
def mrr_at_k(retrieved, gold, k) -> float
def ndcg_at_k(retrieved, gold, k) -> float
def percentile(values: Sequence[float], p: float) -> float
def mean(values: Sequence[float]) -> float

class EvalRunner:
    def __init__(self, cfg: Config | None = None, thresholds: dict[str, float] | None = None)
    def run(self, corpus: list[dict], cases: list[EvalCase], k: int = 10) -> EvalReport
```

**门禁阈值**（`DEFAULT_THRESHOLDS`）

```python
recall@1 >= 0.30   recall@3 >= 0.60   recall@5 >= 0.75
mrr@10   >= 0.45   ndcg@5   >= 0.45   groundedness >= 0.55
latency_p95_ms <= 8000
```

**验收断言**

1. `run()` 每次新建内存管道 → 同配置两次运行**指标逐位相同**。
2. 自定义阈值与默认**合并且以自定义为准**（只传一项不得让其它项 `KeyError`）。
3. `latency*` 判定为「越小越好」，其余为「越大越好」。
4. 语料为空时抛 `ValueError`，不返回一份「全 0 但通过」的假报告。
5. `EvalReport.to_dict()` 含 `metrics / thresholds / passed / failures / cases`。
6. `/evaluate?offline=true` 强制零依赖后端，响应里 `mode == "offline"`，
   且 `backends` 反映实际生效的后端名。
   **为什么需要这个开关**：真实模型下 8 条用例要几分钟、指标还随采样波动，
   而门禁的价值恰恰在于「快、确定、能反复跑」。把两者混成一个默认行为，
   只会让人以为系统坏了。

**黄金集格式**

`data/golden/corpus.jsonl`：每行 `{doc_id, text, meta?}`，`#` 开头为注释。
每篇长度必须超过 `chunk_size`，否则多片段路径永远测不到（由 `scripts/check_corpus.py` 校验）。

`data/golden/questions.jsonl`：每行
`{query, gold_doc, gold_text, must_include[], route_expect}`。
`gold_text` 是目标片段里的子串，**不写死 chunk_id** —— 切分参数一改，写死的 id 全会失效。

---

## 7. HTTP 接口

完整 OpenAPI 见 [`docs/openapi.yaml`](docs/openapi.yaml)。摘要：

| 方法 | 路径 | 请求 | 响应 |
|---|---|---|---|
| GET | `/health` | — | `{status, version, author, backends:{embedder,llm,reranker,vector}}` |
| GET | `/stats` | — | `{documents, chunks, vectors, embedder, dim, llm, reranker, vector_backend, config}` |
| POST | `/ingest` | `{docs:[{doc_id,text,meta?}]}` | `{chunks, documents}`；缺字段 → 400 |
| GET | `/documents` | — | `[{doc_id,title,source,meta,created_at,chunks}]` |
| DELETE | `/documents/{doc_id}` | — | `{removed_chunks, doc_id}` |
| POST | `/search` | `{query, k}` | `[{chunk_id,doc_id,score,source,text}]` |
| POST | `/chat` | `{query, with_trace}` | `Answer.to_dict()`；`with_trace=false` 时去掉 `metadata.trace` |
| GET | `/chat/stream` | `?query=` | SSE：`stage` / `answer` / `done` / `error` |
| GET | `/trace/{trace_id}` | — | Span 树；不存在 → 404 |
| POST | `/evaluate` | `?corpus=&questions=&offline=` | `EvalReport.to_dict()` + `mode` + `backends`；文件缺失 → 404 |
| GET | `/` | — | 控制台 HTML |

**验收断言**（`scripts/e2e.py`，真起 HTTP 服务）

- `/health` 的 `author == "晨星"`。
- 非法 `/ingest` → 400；`/trace/not-exist` → 404。
- `/chat` 返回非空 `text` 且 `citations` 非空，`groundedness ≥ 0.6`。
- `/` 返回 HTML 且**零外部资源引用**（无 CDN、无外链 script/link、无 `@import`）。
- `/evaluate` 的 `recall@5 ≥ 0.75`。

---

## 8. 控制台契约 `web/index.html`

单文件，内联全部 CSS/JS/SVG，**禁止**任何外部资源。要求：

| 编号 | 要求 | 验证方式 |
|---|---|---|
| W-1 | 六个面板可切换 | jsdom 点击 `nav.tabs button[data-tab]` 后断言面板 `active` |
| W-2 | 头部回填四个后端名与知识库统计 | 断言 `#chip-*` 文本 |
| W-3 | 对话渲染答案 + 路由 + 忠实度环形图 + 引用 + 逐条断言 | 断言 `#chatlog .cite` 与 `.claim.sup` 数量 |
| W-4 | 追踪瀑布图按真实耗时等比绘制 | 断言阶段行数与耗时文本 |
| W-5 | 评测面板逐指标显示「实测 / 阈值 / 判定」 | 断言表格行数 |
| W-6 | 深/浅色主题可切换，默认跟随系统 | 断言 `data-theme` |
| W-7 | 脚本零未捕获异常 | 捕获 `window.onerror` 与 jsdom 错误 |

实现方式：`scripts/web_smoke.mjs` 用 jsdom **真执行**内联脚本，把 `fetch` 换成桩。
比「正则匹配 HTML 里有没有某个字符串」强得多 —— 后者只能证明字符串存在。

---

## 9. 运行期契约

### 退出码

| 命令 | 0 | 非 0 |
|---|---|---|
| `stellarnx verify` | 全部自检通过 | 1 = 存在失败项 |
| `stellarnx eval` | 门禁通过 | 1 = 指标未达标 |
| `stellarnx ingest <path>` | 导入成功 | 2 = 文件不存在 |
| `scripts/e2e.py` | 全部断言通过 | 1 = 存在失败项 |
| `scripts/web_smoke.mjs` | 全部断言通过 | 1 = 存在失败项；0 + SKIP 提示 = 未装 jsdom |

### 持久化

- 默认落 `<SNX_DATA_DIR>/stellarnx.db`（默认 `data/`），`SNX_STORE=memory` 则不落盘。
- `ingest / search / chat / stats / serve` **共用同一个库**。
- 启动时若 `count_chunks() > 0` 则自动 `restore()` 重建 FAISS 与 BM25。
- 相对路径的 `data_dir` 按仓库根解析，不按当前工作目录。
