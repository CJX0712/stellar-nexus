"""对着**正在运行的服务**做真实模型验收。

与 scripts/real_smoke.py 的区别：那个是进程内直接调管道，这个走真 HTTP，
验证的是「控制台消费的那份 JSON 契约在真实模型下是否成立」——
字段齐全、类型正确、可 JSON 往返、引用下标与 [n] 对齐。

用法：
    stellarnx serve          # 另开一个窗口，先让它跑起来
    python scripts/live_check.py --base http://127.0.0.1:8010

作者: 晨星
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROUTES = ("direct", "rag", "multihop", "agent")


def request(method: str, url: str, payload: dict | None = None, timeout: float = 300.0):
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(
        url, data=data, method=method, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
            return resp.status, (json.loads(body) if body else None)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")
        try:
            return exc.code, json.loads(body)
        except json.JSONDecodeError:
            return exc.code, {"raw": body}


def fetch_html(url: str, timeout: float = 30.0) -> tuple[int, str]:
    req = urllib.request.Request(url, headers={"User-Agent": "stellarnx-live-check"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        return exc.code, ""


passed = 0
failed = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global passed, failed
    if ok:
        passed += 1
        print(f"[PASS] {name} {detail}")
    else:
        failed += 1
        print(f"[FAIL] {name} {detail}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="真实模型在线验收（HTTP）")
    parser.add_argument("--base", default="http://127.0.0.1:8010")
    parser.add_argument("--query", default="星枢系统默认采用什么向量索引？")
    parser.add_argument("--no-ingest", action="store_true", help="跳过语料导入")
    args = parser.parse_args(argv)
    base = args.base.rstrip("/")

    status, health = request("GET", f"{base}/health")
    check("服务可达", status == 200, f"status={status}")
    if status != 200:
        return 1
    backends = health.get("backends", {})
    check("署名正确", health.get("author") == "晨星", health.get("author", ""))
    print(f"       后端: {json.dumps(backends, ensure_ascii=False)}")
    check("报告了四个后端", len(backends) == 4, str(list(backends)))

    if not args.no_ingest:
        corpus = [json.loads(line) for line in
                  (ROOT / "data/golden/corpus.jsonl").read_text(encoding="utf-8").splitlines()
                  if line.strip() and not line.strip().startswith("#")]
        started = time.perf_counter()
        status, ingested = request("POST", f"{base}/ingest", {"docs": corpus})
        elapsed = (time.perf_counter() - started) * 1000
        check("导入黄金语料", status == 200 and ingested.get("chunks", 0) > 0,
              f"片段={ingested.get('chunks')} 耗时={elapsed:.0f}ms")

    status, stats = request("GET", f"{base}/stats")
    check("索引与片段库一致（chunks == vectors）",
          status == 200 and stats.get("chunks") == stats.get("vectors"),
          f"chunks={stats.get('chunks')} vectors={stats.get('vectors')} dim={stats.get('dim')}")

    status, docs = request("GET", f"{base}/documents")
    check("文档列表含片段计数字段",
          status == 200 and all("chunks" in d for d in docs) and len(docs) > 0,
          f"文档数={len(docs)}")

    # ---- 核心：真实模型问答 ----
    started = time.perf_counter()
    status, answer = request("POST", f"{base}/chat",
                             {"query": args.query, "with_trace": True})
    latency = (time.perf_counter() - started) * 1000
    check("问答返回 200", status == 200, f"status={status}")
    if status != 200:
        print(json.dumps(answer, ensure_ascii=False)[:400])
        return 1

    check("答案非空", bool(answer.get("text", "").strip()),
          f"长度={len(answer.get('text', ''))}")
    check("路由合法", answer.get("route", {}).get("path") in ROUTES,
          str(answer.get("route", {}).get("path")))
    check("含引用", len(answer.get("citations", [])) > 0,
          f"引用={len(answer.get('citations', []))}")

    verification = answer.get("verification") or {}
    check("含归因校验报告", bool(verification), f"groundedness={verification.get('groundedness')}")
    check("断言列表非空且逐条带判定",
          bool(verification.get("claims"))
          and all(isinstance(c.get("supported"), bool) for c in verification["claims"]),
          f"断言={len(verification.get('claims', []))}")
    grounded = verification.get("groundedness")
    check("忠实度达阈值", grounded is not None and grounded >= 0.6, f"groundedness={grounded}")

    trace = (answer.get("metadata") or {}).get("trace") or {}
    names = [c.get("name") for c in trace.get("children", [])]
    check("Span 树含各阶段", {"route", "generate"} <= set(names), " <- ".join(names))
    check("Span 耗时非负",
          all(float(c.get("duration_ms", -1)) >= 0 for c in trace.get("children", [])), "")

    trace_id = answer.get("trace_id")
    status, fetched = request("GET", f"{base}/trace/{trace_id}")
    check("追踪可通过 HTTP 取回", status == 200 and "children" in fetched, str(trace_id))

    # 引用下标与正文 [n] 必须对齐，否则控制台会把证据张冠李戴
    import re as _re
    numbers = sorted({int(x) for x in _re.findall(r"\[(\d+)\]", answer.get("text", ""))})
    check("正文 [n] 下标都在引用范围内",
          all(1 <= n <= len(answer.get("citations", [])) for n in numbers),
          f"正文标记={numbers} 引用数={len(answer.get('citations', []))}")

    # 可 JSON 往返：控制台拿到的是序列化后的东西
    try:
        round_trip = json.loads(json.dumps(answer, ensure_ascii=False))
        check("响应可 JSON 往返", round_trip["trace_id"] == trace_id, "")
    except (TypeError, ValueError) as exc:
        check("响应可 JSON 往返", False, f"{type(exc).__name__}: {exc}")

    # ---- 控制台 ----
    status, page = fetch_html(f"{base}/")
    check("控制台 HTML 可访问", status == 200 and "星枢 StellarNexus" in page, f"bytes={len(page)}")
    check("控制台零外部资源",
          all(tag not in page for tag in ("cdn.", "unpkg", "jsdelivr", 'src="http', "@import")), "")

    # ---- 评测门禁 ----
    # 走 offline=true：门禁要的是快且确定，真实模型下 8 条用例要好几分钟。
    status, report = request("POST", f"{base}/evaluate?offline=true")
    metrics = (report or {}).get("metrics", {})
    check("评测端点返回报告", status == 200 and bool(metrics),
          json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                      for k, v in metrics.items() if k != "cases"}, ensure_ascii=False))
    check("评测报告标注了基线模式", (report or {}).get("mode") == "offline",
          str((report or {}).get("mode")))
    check("门禁通过", bool((report or {}).get("passed")),
          "；".join((report or {}).get("failures", [])) or "无失败项")

    print(f"\n本次问答真实耗时: {latency:.0f} ms")
    print(f"\n通过 {passed} / 失败 {failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
