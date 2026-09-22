# 使用指南

作者 **晨星** · 版本 1.0.0

面向「已经部署好，开始用它做事」。部署见 [`DEPLOYMENT.md`](DEPLOYMENT.md)。

---

## 1. 三种用法

| 入口 | 适合 | 特点 |
|---|---|---|
| 命令行 `stellarnx` | 脚本、批量、排查 | 可组合，可进 CI |
| HTTP API | 集成到别的系统 | 语言无关 |
| 控制台（浏览器） | 人工查看与调试 | 可视化引用、断言、链路瀑布 |

三者共享同一个知识库，**用哪个入口灌的数据，另外两个入口都能查到**。

---

## 2. 命令行

### 2.1 灌数据

```bash
stellarnx ingest docs.jsonl
```

`docs.jsonl` 每行一个 JSON 对象：

```jsonl
{"doc_id": "handbook-001", "text": "正文内容……", "meta": {"title": "运维手册", "source": "wiki"}}
{"doc_id": "handbook-002", "text": "另一篇正文……"}
```

- `doc_id` 与 `text` 必填；`meta` 可选。
- 同一 `doc_id` 重复导入是**覆盖**，不是追加。
- 切分、嵌入、建索引、重建 BM25 一次完成。

也可以不写文件，直接打 API（见第 3 节）。

### 2.2 检索

```bash
stellarnx search "向量索引" -k 5
```

输出形如：

```
1. [0.8213] vecdb:0 (hybrid) 星枢系统使用 FAISS 作为向量索引，同时保留 numpy 回退实现。
2. [0.6104] vecdb:1 (bm25)   检索阶段采用 BM25 与稠密向量双路召回。
```

`source` 的含义：`bm25` 稀疏命中、`dense` 稠密命中、`hybrid` 融合后、
`rerank` 精排后。看这一列能快速判断是哪一路在起作用。

### 2.3 问答

```bash
stellarnx chat "星枢系统默认采用什么向量索引？"
```

```
星枢系统默认采用 FAISS 的 IndexFlatIP 索引，配合 L2 归一化后的向量。[1]

引用:
  - vecdb:0: 星枢系统使用 FAISS 作为向量索引……
  - vecdb:2: 归一化后内积等价于余弦相似度……

可证性: 1.000 (阈值 0.6)
```

机器可读：

```bash
stellarnx chat "..." --json
```

返回完整 `Answer` 结构（`text` / `citations` / `route` / `verification` / `trace_id` /
`latency_ms` / `metadata.trace`），可直接喂给下游。

### 2.4 评测门禁

```bash
stellarnx eval
echo $?      # 0 = 门禁通过；1 = 有指标未达标
```

这一条命令就是 CI 的关卡。想只跑离线路径（速度快、指标确定）：

```bash
SNX_EMBED_BACKEND=hash SNX_LLM_BACKEND=mock SNX_RERANK_BACKEND=lexical stellarnx eval
```

### 2.5 状态与自检

```bash
stellarnx stats    # 文档数 / 片段数 / 向量数 / 生效后端 / 完整配置
stellarnx verify   # 15 项离线自检
```

`stats` 里 `chunks` 与 `vectors` 必须相等。不等就重启（启动会重建索引）。

### 2.6 起服务

```bash
stellarnx serve --host 127.0.0.1 --port 8000
```

---

## 3. HTTP API

### 3.1 灌数据

```bash
curl -X POST http://127.0.0.1:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "docs": [
      {"doc_id": "handbook-001",
       "text": "星枢系统的检索层由三路组成……",
       "meta": {"title": "运维手册"}}
    ]
  }'
# {"chunks": 3, "documents": 1}
```

### 3.2 问答

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "星枢系统默认采用什么向量索引？", "with_trace": true}'
```

响应关键部分：

```json
{
  "text": "星枢系统默认采用 FAISS 的 IndexFlatIP 索引，配合 L2 归一化。[1]",
  "citations": [
    {"chunk_id": "vecdb:0", "doc_id": "vecdb", "score": 0.9123,
     "text": "星枢系统使用 FAISS 作为向量索引……"}
  ],
  "route": {"path": "rag", "reason": "知识库已就绪且问题指向事实", "confidence": 0.82},
  "verification": {
    "groundedness": 1.0, "threshold": 0.6, "passed": true,
    "claims": [{"text": "默认采用 FAISS", "supported": true, "score": 0.91,
                "evidence_id": "vecdb:0"}]
  },
  "trace_id": "a1b2c3d4e5f6",
  "latency_ms": 13425.1
}
```

**读这个响应的正确顺序**

1. 先看 `verification.passed`。`false` 意味着答案里有断言没找到证据落脚点，
   **不要直接采信 `text`**。
2. 再看 `verification.claims` 里哪些 `supported == false`，那些就是可疑句。
3. 想核对细节，用 `citations[].chunk_id` 回原始片段，或直接把 `[n]` 与
   `citations[n-1]` 对照 —— 两者下标是对齐的。
4. `route.path == "direct"` 时 `verification` 为 `null`，这是**设计如此**：
   直答链路没有检索，本就没有证据可校验，不是漏做。

### 3.3 事件流（SSE）

```bash
curl -N "http://127.0.0.1:8000/chat/stream?query=向量索引是什么"
```

```
event: stage
data: {"stage": "start", "query": "向量索引是什么"}

event: stage
data: {"stage": "route", "duration_ms": 1.2, "attrs": {"path": "rag"}}

event: answer
data: {"text": "...", "trace_id": "...", "citations": ["vecdb:0"]}

event: done
data: {"groundedness": 1.0}
```

适合放进 UI 做「正在处理」的进度提示。注意 `stage: start` 之后会有较长的静默期
（生成阶段耗时），这不是卡死。

### 3.4 Python 客户端

```python
import httpx

BASE = "http://127.0.0.1:8000"
# 本机若挂了代理，必须显式关掉，否则 localhost 请求会被送进代理
client = httpx.Client(base_url=BASE, timeout=300.0, trust_env=False)

client.post("/ingest", json={"docs": [
    {"doc_id": "d1", "text": "星枢系统使用 FAISS 作为向量索引。"}
]})

ans = client.post("/chat", json={"query": "用什么向量索引？", "with_trace": True}).json()

print(ans["text"])
print("路由:", ans["route"]["path"])
print("忠实度:", ans["verification"]["groundedness"])
for i, c in enumerate(ans["citations"], 1):
    print(f"[{i}] {c['chunk_id']}: {c['text'][:60]}")
```

### 3.5 检索与文档管理

```bash
# 只看检索结果，不生成
curl -X POST http://127.0.0.1:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "向量索引", "k": 5}'

# 文档列表（含每篇片段数）
curl http://127.0.0.1:8000/documents

# 删除一篇
curl -X DELETE http://127.0.0.1:8000/documents/handbook-001
```

---

## 4. 控制台

打开 `http://127.0.0.1:8000`。六个面板：

### 对话

输入问题回车。答案区自下而上读：

1. **路由标签** —— `direct` / `rag` / `multihop` / `agent`，以及路由依据
2. **校验标签** —— `校验通过` / `校验未过` / `未经校验`
3. **忠实度环形图** —— 绿 ≥ 0.6、黄 0.3–0.6、红 < 0.3
4. **耗时与 trace 链接** —— 点 trace 跳到链路瀑布
5. **正文** —— 带 `[n]` 回指
6. **引用证据** —— 每条含 doc_id / chunk_id / score / 原文
7. **断言级校验**（默认折叠）—— 展开看每条断言是否被支持及对应证据编号

下方示例按钮可直接试四种链路：

| 示例问题 | 走哪条路 | 观察点 |
|---|---|---|
| 星枢系统用了哪几种检索方式？ | `rag` | 引用与断言 |
| 为什么归因校验要用 IDF 加权？ | `rag` | 改写型问法仍能召回 |
| 对比 BM25 和稠密向量两路召回的区别 | `multihop` | 子查询融合 |
| 12 乘以 43 等于多少 | `agent` | 工具调用，证据含工具观测 |
| 你好 | `direct` | 0.6s 返回，不做检索 |

### 检索调试

看召回原文与分数，判断「是检索没召回到」还是「召回到了但生成没用好」——
这两类问题的修法完全不同，这一步是分诊。

### 知识库

粘贴正文即可入库。演示数据点「载入示例语料」。右侧列出已入库文档与片段数。

### 链路追踪

**瀑布图是关键诊断工具。** 一次真实模型问答的典型分布：

```
route        1 ms   ▏
retrieve    18 ms   ▎
rerank      96 ms   ▍
generate 13290 ms   ████████████████████████████
verify      19 ms   ▎
```

`generate` 吃掉 99% 时间 —— 所以优化方向是换小模型或上 GPU，
而不是调检索参数。没有这张图只能靠猜。

### 评测门禁

点「运行评测」，逐指标显示实测值 / 阈值 / 判定。展开可见逐条用例明细。

### 系统信息

核对生效后端（同 `/health`）、完整配置，也可在此改 API 地址
（控制台被单独部署在别处时用）。

---

## 5. 参数调优

改环境变量后重启服务。

| 想达到的效果 | 调什么 | 说明 |
|---|---|---|
| 答案更准 | `SNX_TOP_K_RETRIEVE` 20 → 30 | 多召回一些，靠精排收紧，通常比直接减少召回数更有效 |
| 答案更全 | `SNX_TOP_K_RERANK` 5 → 8 | 进提示词的证据更多；但注意窗口与延迟 |
| 减少幻觉 | `SNX_GROUNDEDNESS_THRESHOLD` 0.6 → 0.75 | 阈值越高越严格，代价是更多重生成 |
| 不想重生成 | `SNX_MAX_REGENERATE` 1 → 0 | 未过校验直接如实标注 `passed=false` |
| 更快 | `SNX_LLM_MODEL` 换更小模型，或 `SNX_RERANK_BACKEND=identity` | 重排是延迟大户之一 |
| 更省内存 | `SNX_CHUNK_SIZE` 调小，减少单次提示词长度 | 影响召回粒度，需重跑评测确认 |
| 多跳更激进 | `SNX_AGENT_MAX_STEPS` 4 → 6 | 步数多不一定更好，先看 span 树 |

**调整后必须重跑 `stellarnx eval`**，指标掉下阈值就是负优化。

---

## 6. 常见任务

### 6.1 把一篇长文档入库

```bash
python - <<'PY'
import json, pathlib
text = pathlib.Path("manual.md").read_text(encoding="utf-8")
docs = [{"doc_id": "manual", "text": text,
         "meta": {"title": "产品手册", "source": "manual.md"}}]
pathlib.Path("manual.jsonl").write_text(
    "\n".join(json.dumps(d, ensure_ascii=False) for d in docs), encoding="utf-8")
PY

stellarnx ingest manual.jsonl
```

切分是自动的，不需要预先分段。文档超过 `SNX_CHUNK_SIZE`（默认 480 字）会被切成多个片段。

### 6.2 判断「是检索不行还是生成不行」

```bash
# 第一步：看检索是否召回了正确片段
stellarnx search "你的问题" -k 5

# 第二步：看生成与校验
stellarnx chat "你的问题" --json | python -m json.tool
```

- 检索没召回 → 调 `SNX_TOP_K_RETRIEVE` 或换嵌入模型。
- 检索召回了但答案不对 → 看 `verification.claims`，是提示词问题还是模型能力问题。
- 答案正确但 `passed=false` → 阈值偏严，或证据片段被截断导致校验看不到关键词。

### 6.3 造假答案自检归因校验器

校验器必须能识别错误，而不是只给高分：

```bash
SNX_EMBED_BACKEND=hash SNX_LLM_BACKEND=mock SNX_RERANK_BACKEND=lexical \
  stellarnx verify
# 关注这一行：
# [PASS] 归因校验可被打穿    真实=1.000 篡改=0.000
```

`篡改=0.000` 说明把证据换成无关文本后，忠实度确实掉到 0 —— 校验器是有效的。

### 6.4 增量更新知识库

```bash
stellarnx ingest new-docs.jsonl     # 新增
stellarnx ingest revised.jsonl      # 同 doc_id 覆盖
curl -X DELETE http://127.0.0.1:8000/documents/obsolete   # 删除
```

删除后索引自动重建，无需重启。

### 6.5 接入 CI

```yaml
- run: pip install -e ".[dev]"
- run: stellarnx verify                                    # 离线自检
- run: pytest -q                                           # 单测
- run: python scripts/check_imports.py                    # 分层依赖
- run: python scripts/e2e.py                               # 端到端
- run: stellarnx eval                                     # 评测门禁
- run: npm i && node scripts/web_smoke.mjs                # 控制台
```

任一退出码非 0 即失败。

---

## 7. 输出怎么读

### `verification.groundedness` 的取值含义

| 区间 | 含义 | 建议动作 |
|---|---|---|
| 1.0 | 全部断言都能在证据里找到落脚点 | 可直接采信，仍建议抽查引用 |
| 0.6 – 1.0 | 部分断言缺据 | 展开 `claims` 看哪条 `supported == false` |
| 0.3 – 0.6 | 大概率有推断成分 | 不要直接对外发布 |
| < 0.3 | 基本无据 | 当作不可用；通常是检索没召回到正确片段 |
| `null` | 走的是 `direct` 链路，没有检索 | 正常，非异常 |

### `route.path` 的含义

| 值 | 含义 | 典型延迟（本机 CPU） |
|---|---|---|
| `direct` | 寒暄/常识/语料为空，不检索 | < 1 s |
| `rag` | 事实型问题，单跳检索 | 13–30 s（真实模型） |
| `multihop` | 并列/对比结构，子查询融合 | 略高于 `rag` |
| `agent` | 需工具（算术、换算、多步） | 7–15 s |

---

## 8. 限制（请据此设定预期）

1. **CPU 推理慢**。1.5B 模型单次问答 13–30 s，这是硬件限制而非实现问题。
2. **校验是启发式**。IDF 加权词汇覆盖度能抓住「无中生有」，但抓不住
   「同义改写但语义正确」的边界情况。要更准可换 `LLMVerifier`。
3. **路由是规则实现**。复杂或混合意图可能误判，`route.reason` 会说明依据，可据此判断。
4. **无多用户与鉴权**。设计为单机单实例；对外必须加代理鉴权。
5. **中文切词是轻量的**。黄金集表现良好，专业领域术语密集时可能需换分词器。
6. **`direct` 链路不校验**。没有证据就没有归因，这是设计取舍，输出会明确标注「未经校验」。
