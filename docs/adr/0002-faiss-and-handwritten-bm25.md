# ADR-0002 向量用 FAISS、BM25 自实现，不引第三方 BM25 包

- 状态：采纳
- 日期：2026-09-23
- 作者：晨星

## 背景

混合检索需要一路稀疏召回。技术上最省事的做法是 `pip install rank-bm25`。
同时向量检索需要 ANN 能力，候选是 FAISS、hnswlib、annoy、scann。

前置约束：**机器上没有 MSVC / cmake，不能编译任何 C 扩展**。
所以选型的第一个筛子是「有没有现成的预编译轮子」。

## 备选方案

### 向量索引

| 方案 | 轮子情况 | 结论 |
|---|---|---|
| **faiss-cpu** | `win_amd64` + `manylinux` 均有，实测 `IndexFlatIP` 可用 | 采纳 |
| hnswlib | 常见版本无 windows 轮子，需编译 | 否决 |
| annoy | 维护停滞，windows 轮子来源不可靠 | 否决 |
| scann | 依赖 TensorFlow，体积与复杂度都不划算 | 否决 |
| 纯 numpy 暴力检索 | 无依赖但 O(N·d)，语料上万后明显变慢 | 保留为回退 |

### 稀疏召回

| 方案 | 问题 |
|---|---|
| `rank-bm25` | 自带一套英文向切词器，与我们稠密路用的切词器**不一致** |
| Elasticsearch / OpenSearch | 为一个功能引入外部服务，与「单机可复现」冲突 |
| **自实现 BM25（约 80 行）** | 采纳 |

## 决策

- 向量索引：`FaissIndex`（`IndexFlatIP` + L2 归一化，内积等价余弦），
  `InMemoryIndex`（numpy 暴力）作为回退。`build_index(dim, prefer="faiss")` 分派。
- 稀疏召回：`BM25Retriever` 自实现，**复用 `core.text.tokenize`**。

## 理由

### 为什么 FAISS 而不是纯 numpy

`FAISS` 有可靠的双平台预编译轮子，不需要编译器，冷启动无额外进程。
主场景是「文档规模几千到几十万」，`IndexFlatIP` 用 SIMD 已足够快；
真到百万级再把 `IndexFlatIP` 换成 `IVF` 只需改一行，协议不变。

### 为什么 BM25 要自己写

**切词器必须共用。** RRF 融合的是两路**排名**，但如果两路对「同一个词」的理解不同，
融合出来的结果就无法解释：分数不可比、排序不可预测、调参变成玄学。

`rank-bm25` 默认按空白切词，中文查询会被切成整串，直接失效；
改成自定义切词器要传 `tokenizer` 参数，那已经用上了它全部的核心逻辑 ——
不如自己写 80 行，顺便把 `all_ids()` 这类自检需要的方法一起加上。

自实现还带来两个好处：

1. `search()` 明确**只返回 `score > 0` 的项**。查询词全不在语料时返回空列表，
   而不是一堆 0 分噪音污染 RRF（噪音会稀释真正命中的项）。
2. 可以暴露 `all_ids()`，让 `index_is_consistent()` 能一次性校验
   「片段库 / 向量索引 / BM25 索引」三者条数是否一致。

## 代价与补救

| 代价 | 补救 |
|---|---|
| BM25 参数（k1=1.5, b=0.75）是我们自己实现的，需自行验证 | `test_retrieve.py` 断言经典性质：词频越高分越高、文档越长归一化越低、未出现词得 0 分 |
| FAISS `delete` 不支持原地删除，需重建 | `FaissIndex.delete` 用 `IndexIDMap2` + 重建；`count()` 与 `_ids` 长度严格同步（有契约测试） |
| 中文切词只有单字 + 二字组，不如分词器精细 | 实测黄金集 `recall@5 = 1.00`，够用。接口留好，换 `jieba` 或 BGE-M3 tokenizer 只改 `core/text.py` |

## 反悔路径

- 换 ANN：改 `build_index` 的工厂分支。
- 换切词：改 `core/text.tokenize`，`BM25Retriever` 与 `HashEmbedder` 自动跟随
  （这正是共用切词器的价值）。

## 相关

- 代码：`src/stellarnx/retrieve/bm25.py`、`src/stellarnx/store/vector_index.py`
- 测试：`tests/test_retrieve.py`、`tests/test_store.py`
