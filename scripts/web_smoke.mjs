/**
 * 控制台无头冒烟测试。
 *
 * 为什么不用「正则匹配 HTML」充当测试：那种测法只能证明字符串存在，
 * 证明不了标签切换、请求发得出去、答案渲染得出来。这里用 jsdom 真跑一遍
 * 内联脚本，把 fetch 换成桩，逐项断言 DOM 结果。
 *
 * 依赖：jsdom（仅测试期，不进入运行时依赖）。缺失时明确跳过而不是静默通过。
 *
 * 作者: 晨星
 */

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";

const HERE = dirname(fileURLToPath(import.meta.url));
const INDEX = resolve(HERE, "..", "web", "index.html");

let JSDOM;
try {
  ({ JSDOM } = await import("jsdom"));
} catch {
  console.log("[SKIP] 未安装 jsdom，跳过控制台无头冒烟（npm i jsdom 后可运行）");
  process.exit(0);
}

let passed = 0;
let failed = 0;
function check(name, ok, detail = "") {
  if (ok) {
    passed += 1;
    console.log(`[PASS] ${name} ${detail}`);
  } else {
    failed += 1;
    console.log(`[FAIL] ${name} ${detail}`);
  }
}

/* ---------------- 假后端 ---------------- */
const TRACE = {
  name: "root",
  trace_id: "smoke0001",
  duration_ms: 412.5,
  attrs: {},
  children: [
    { name: "route", trace_id: "smoke0001", duration_ms: 1.2, attrs: { path: "rag" }, children: [] },
    { name: "retrieve", trace_id: "smoke0001", duration_ms: 18.4, attrs: { mode: "rag" }, children: [] },
    { name: "rerank", trace_id: "smoke0001", duration_ms: 96.1, attrs: { candidates: 20 }, children: [] },
    { name: "generate", trace_id: "smoke0001", duration_ms: 280.3, attrs: { evidence: 5 }, children: [] },
    { name: "verify", trace_id: "smoke0001", duration_ms: 16.5, attrs: {}, children: [] },
  ],
};

const ANSWER = {
  text: "星枢系统默认使用 FAISS 作为向量索引[1]，并保留 numpy 回退实现。",
  citations: [
    { chunk_id: "vecdb:0", doc_id: "vecdb", text: "星枢系统使用 FAISS 作为向量索引。", score: 0.9123 },
    { chunk_id: "vecdb:1", doc_id: "vecdb", text: "同时保留 numpy 回退实现。", score: 0.7712 },
  ],
  route: { path: "rag", reason: "知识库已就绪且问题指向事实", confidence: 0.82, features: {} },
  verification: {
    groundedness: 1.0,
    threshold: 0.6,
    passed: true,
    claims: [
      { text: "默认使用 FAISS", supported: true, score: 0.91, evidence_id: "vecdb:0" },
      { text: "保留 numpy 回退", supported: true, score: 0.77, evidence_id: "vecdb:1" },
    ],
  },
  trace_id: "smoke0001",
  latency_ms: 412.5,
  metadata: { note: "", embedder: "ollama:bge-m3", llm: "ollama:qwen2.5:1.5b-instruct", trace: TRACE },
};

const EVAL = {
  metrics: {
    "recall@1": 0.88, "recall@3": 1.0, "recall@5": 1.0, "mrr@10": 0.933,
    "ndcg@5": 0.95, groundedness: 1.0, latency_p50_ms: 26.65, latency_p95_ms: 32.66, cases: 8,
  },
  thresholds: {
    "recall@1": 0.3, "recall@3": 0.6, "recall@5": 0.75, "mrr@10": 0.45,
    "ndcg@5": 0.45, groundedness: 0.55, latency_p95_ms: 8000,
  },
  passed: true,
  failures: [],
  cases: [
    {
      query: "星枢系统默认采用什么向量索引？", route: "rag", "recall@1": 1, "recall@3": 1,
      "recall@5": 1, "mrr@10": 1, "ndcg@5": 1, groundedness: 1, latency_ms: 26.65,
      top_chunk: "vecdb:0", passed: true,
    },
  ],
};

const ROUTES = {
  "GET /health": () => ({
    status: "ok", version: "1.0.0", author: "晨星",
    backends: {
      embedder: "ollama:bge-m3", llm: "ollama:qwen2.5:1.5b-instruct",
      reranker: "llm-listwise", vector: "faiss",
    },
  }),
  "GET /stats": () => ({
    documents: 8, chunks: 16, vectors: 16, embedder: "ollama:bge-m3", dim: 1024,
    llm: "ollama:qwen2.5:1.5b-instruct", reranker: "llm-listwise", vector_backend: "faiss",
    config: { chunk_size: 480, top_k_rerank: 5 },
  }),
  "GET /documents": () => ([
    { doc_id: "handbook-001", title: "星枢运维手册", source: "console", chunks: 2 },
    { doc_id: "vecdb", title: "向量索引说明", source: "golden", chunks: 2 },
  ]),
  "POST /chat": () => ANSWER,
  "POST /search": () => ([
    { chunk_id: "vecdb:0", doc_id: "vecdb", score: 0.9123, source: "hybrid", text: "星枢系统使用 FAISS 作为向量索引。" },
    { chunk_id: "vecdb:1", doc_id: "vecdb", score: 0.7712, source: "bm25", text: "同时保留 numpy 回退实现。" },
  ]),
  "POST /evaluate": () => EVAL,
  "POST /ingest": () => ({ chunks: 3, documents: 9 }),
  "DELETE /documents/tmp": () => ({ removed_chunks: 2, doc_id: "tmp" }),
};

const calls = [];

function makeFetch() {
  return async function fetch(input, init) {
    const url = String(input);
    // 去掉主机与查询串：同一个端点带不带 ?offline=true 都走同一份桩
    const path = url.replace(/^https?:\/\/[^/]+/, "").split("?")[0];
    const method = (init && init.method) || "GET";
    const key = `${method} ${path}`;
    calls.push(key);
    const handler = ROUTES[key];
    if (!handler) {
      return { ok: false, status: 404, statusText: "Not Found", text: async () => JSON.stringify({ detail: "no stub for " + key }) };
    }
    const body = handler();
    return {
      ok: true, status: 200, statusText: "OK",
      text: async () => JSON.stringify(body),
    };
  };
}

/* ---------------- 装载 ---------------- */
const html = readFileSync(INDEX, "utf-8");
const errors = [];

const dom = new JSDOM(html, {
  runScripts: "dangerously",
  pretendToBeVisual: true,
  url: "http://127.0.0.1:8010/",
  beforeParse(window) {
    window.fetch = makeFetch();
    window.matchMedia = () => ({ matches: false, addEventListener() {}, removeEventListener() {} });
    window.addEventListener("error", (e) => errors.push(String(e.message || e.error)));
  },
});
const { window } = dom;
const { document } = window;

const tick = (n = 6) => new Promise((r) => setTimeout(r, n));

await tick(30);

/* ---------------- 断言 ---------------- */
check("页面无未捕获脚本错误", errors.length === 0, errors.join(" | "));

check("标题与署名正确",
  document.title.includes("星枢") && document.body.textContent.includes("晨星"),
  document.title);

check("健康检查写回后端标签",
  document.querySelector("#chip-embed").textContent === "ollama:bge-m3" &&
  document.querySelector("#chip-llm").textContent === "ollama:qwen2.5:1.5b-instruct" &&
  document.querySelector("#chip-rank").textContent === "llm-listwise" &&
  document.querySelector("#chip-vec").textContent === "faiss",
  `${document.querySelector("#chip-embed").textContent} / ${document.querySelector("#chip-vec").textContent}`);

check("服务状态指示为在线",
  document.querySelector("#dot-health").className.includes("on"),
  document.querySelector("#chip-health").textContent.trim());

check("知识库统计写入头部",
  document.querySelector("#chip-corpus").textContent.includes("8") &&
  document.querySelector("#chip-corpus").textContent.includes("16"),
  document.querySelector("#chip-corpus").textContent);

check("系统信息面板填充运行时后端",
  document.querySelector("#ab-stats").textContent.includes("ollama:bge-m3") &&
  document.querySelector("#ab-config").textContent.includes("chunk_size"),
  "");

check("文档列表渲染 2 条", document.querySelectorAll("#doc-list [data-del]").length === 2,
  `count=${document.querySelectorAll("#doc-list [data-del]").length}`);

/* --- 标签切换 --- */
document.querySelector('nav.tabs button[data-tab="search"]').click();
check("切换到检索面板",
  document.querySelector("#p-search").classList.contains("active") &&
  !document.querySelector("#p-chat").classList.contains("active"), "");
document.querySelector('nav.tabs button[data-tab="chat"]').click();
check("切回对话面板", document.querySelector("#p-chat").classList.contains("active"), "");

/* --- 对话 --- */
document.querySelector("#chat-in").value = "星枢系统默认采用什么向量索引？";
document.querySelector("#chat-send").click();
await tick(60);

const log = document.querySelector("#chatlog").textContent;
check("对话发起 POST /chat", calls.includes("POST /chat"), calls.filter((c) => c.includes("chat")).join(","));
check("答案正文渲染", log.includes("FAISS"), "");
check("路由标签渲染", log.includes("rag"), "");
check("引用证据渲染 2 条", document.querySelectorAll("#chatlog .cite").length >= 2,
  `cites=${document.querySelectorAll("#chatlog .cite").length}`);
check("断言校验渲染", log.includes("2/2") && document.querySelectorAll("#chatlog .claim.sup").length === 2,
  `sup=${document.querySelectorAll("#chatlog .claim.sup").length}`);
check("忠实度环形图与耗时展示", log.includes("1.00") && log.includes("413"), "");
check("侧栏历史更新", document.querySelector("#chat-count").textContent === "1", "");

/* --- 追踪 --- */
check("追踪列表已记录",
  document.querySelector("#tr-count").textContent === "1" &&
  document.querySelectorAll("#tr-list [data-t]").length === 1, "");
document.querySelector("#tr-list [data-t]").click();
await tick(10);
const wf = document.querySelector("#tr-wf");
check("瀑布图渲染 5 个阶段", wf.querySelectorAll(".row2").length === 5,
  `rows=${wf.querySelectorAll(".row2").length}`);
check("瀑布图显示真实耗时", wf.textContent.includes("280 ms") || wf.textContent.includes("280ms"),
  wf.textContent.replace(/\s+/g, " ").slice(0, 80));
check("原始 Span 树可展开为 JSON",
  document.querySelector("#tr-json").textContent.includes('"name": "root"'), "");

/* --- 检索 --- */
document.querySelector('nav.tabs button[data-tab="search"]').click();
document.querySelector("#se-q").value = "向量索引";
document.querySelector("#se-go").click();
await tick(40);
check("检索结果表格 2 行",
  document.querySelectorAll("#se-out tbody tr").length === 2,
  `rows=${document.querySelectorAll("#se-out tbody tr").length}`);

/* --- 知识库写入 --- */
document.querySelector('nav.tabs button[data-tab="corpus"]').click();
document.querySelector("#ing-id").value = "smoke-doc";
document.querySelector("#ing-text").value = "这是一段用于冒烟测试的正文。";
document.querySelector("#ing-go").click();
await tick(40);
check("导入触发 POST /ingest", calls.includes("POST /ingest"), "");
check("导入后刷新文档列表（桩仍返回 2 条）",
  document.querySelector("#doc-count").textContent === "2",
  document.querySelector("#doc-count").textContent);

/* --- 评测 --- */
document.querySelector('nav.tabs button[data-tab="eval"]').click();
check("评测默认走离线基线（门禁要快且确定）",
  document.querySelector("#ev-offline").checked &&
  document.querySelector("#ev-hint").textContent.includes("离线"),
  document.querySelector("#ev-hint").textContent.slice(0, 24));

document.querySelector("#ev-go").click();
await tick(60);
const verdict = document.querySelector("#ev-verdict").textContent;
const metrics = document.querySelector("#ev-metrics").textContent;
check("评测触发 POST /evaluate", calls.includes("POST /evaluate"), "");
check("门禁结论渲染", verdict.includes("门禁通过") && verdict.includes("离线基线"), verdict.trim());
check("指标表行数完整", document.querySelectorAll("#ev-metrics tbody tr").length === 9,
  `rows=${document.querySelectorAll("#ev-metrics tbody tr").length}`);
check("指标含延迟与忠实度", metrics.includes("P95") && metrics.includes("8.00 s"), "");
check("逐条用例表渲染", document.querySelectorAll("#ev-cases tbody tr").length === 1, "");

/* --- 主题切换 --- */
document.querySelector("#btn-theme").click();
check("主题可切换到深色",
  document.documentElement.getAttribute("data-theme") === "dark", "");
document.querySelector("#btn-theme").click();

check("运行期仍无脚本错误", errors.length === 0, errors.join(" | "));

console.log(`\n通过 ${passed} / 失败 ${failed}`);
dom.window.close();
process.exit(failed === 0 ? 0 : 1);
