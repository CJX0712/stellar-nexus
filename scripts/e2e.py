"""端到端冒烟：真实拉起 HTTP 服务，跑核心成功流与关键错误流。

不用 docker/curl（本机不可靠），改为 Python 子进程 + 内置 urllib，
失败时打印进程 stderr，绝不静默跳过。

作者: 晨星
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
TIMEOUT = 90
_children: list[subprocess.Popen] = []


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def request(method: str, url: str, payload: dict | None = None, expect: int = 200):
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(
        url, data=data, method=method,
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")
        try:
            return exc.code, json.loads(body)
        except json.JSONDecodeError:
            return exc.code, {"raw": body}


def fetch_text(url: str) -> tuple[int, str]:
    """取回非 JSON 响应（控制台 HTML），失败时返回状态码与错误文本。"""
    req = urllib.request.Request(url, headers={"User-Agent": "stellarnx-e2e"})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            return resp.status, resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", "replace")


class _ServerHandle:
    """统一子进程与进程内服务的行为，便于 wait_health / kill 复用。"""

    def __init__(self, server, thread) -> None:
        self.server = server
        self.thread = thread

    def poll(self) -> int | None:
        return None if self.thread.is_alive() else 0

    @property
    def stdout(self):
        return None


def wait_health(base: str, proc, deadline: float = 60.0) -> bool:
    end = time.time() + deadline
    last_error = ""
    while time.time() < end:
        if proc.poll() is not None:
            return False
        try:
            status, _ = request("GET", f"{base}/health")
            if status == 200:
                return True
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        time.sleep(0.6)
    # 用 print 而不是直接调 log：log 是 main 里的局部函数，直接引用会 NameError。
    # main 期间 builtins.print 已被替换为 log，因此 print 同样落到报告文件，
    # 且超时分支不会自己把脚本打崩。
    print(f"健康检查超时，最后一次错误: {last_error}")
    return False


def drain(proc: subprocess.Popen) -> str:
    """安全读取子进程输出。

    直接 read() 在进程仍存活时会永久阻塞（管道不会自己到 EOF），
    因此只在进程已退出时读，否则返回提示。
    """
    if proc.poll() is None:
        return "<进程仍在运行，输出不可读>"
    try:
        out, _ = proc.communicate(timeout=10)
    except Exception:
        return "<读取输出超时>"
    return (out or "")[-3000:]


def kill(proc) -> None:
    if isinstance(proc, subprocess.Popen):
        if proc.poll() is None:
            subprocess.run(["taskkill", "/pid", str(proc.pid), "/t", "/f"],
                           capture_output=True, check=False)
            try:
                proc.wait(timeout=10)
            except Exception:
                proc.kill()
        return
    proc.server.should_exit = True
    proc.thread.join(timeout=15)


def main() -> int:
    # 沙箱里 stdout 常被吞，报告同时落盘，保证结果可读
    report_path = ROOT / "data" / "e2e-report.txt"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_fh = open(report_path, "w", encoding="utf-8")

    import builtins

    real_print = builtins.print

    def log(*args: object, **kwargs: object) -> None:
        text = " ".join(str(a) for a in args)
        real_print(text, **kwargs)
        report_fh.write(text + "\n")
        report_fh.flush()

    # 必须替换 builtins.print，不能只改 globals —— 否则 log 里再调 print 会自我递归
    builtins.print = log
    port = free_port()
    base = f"http://127.0.0.1:{port}"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(SRC) + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"

    # 沙箱里 spawn 子进程再走本地 socket 会被随机 reset，改为进程内起 uvicorn 线程。
    # 链路仍然是真实 HTTP 协议栈（真 socket + 真 ASGI 应用），不是直接调函数。
    sys.path.insert(0, str(SRC))
    import threading

    import uvicorn

    from stellarnx.api.app import create_app

    config = uvicorn.Config(create_app(), host="127.0.0.1", port=port,
                            log_level="warning", timeout_graceful_shutdown=2)
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    proc = _ServerHandle(server, thread)

    passed, failed = 0, 0

    def check(name: str, ok: bool, detail: str = "") -> None:
        nonlocal passed, failed
        if ok:
            passed += 1
            print(f"[PASS] {name} {detail}")
        else:
            failed += 1
            print(f"[FAIL] {name} {detail}")

    try:
        healthy = wait_health(base, proc)
        check("服务启动健康检查", healthy, f"port={port}")
        if not healthy:
            print("服务未能就绪，进程输出：\n", drain(proc))

        status, health = request("GET", f"{base}/health")
        check("健康检查返回作者与后端", status == 200 and health.get("author") == "晨星",
              json.dumps(health.get("backends", {}), ensure_ascii=False))

        corpus = [json.loads(line) for line in
                  (ROOT / "data/golden/corpus.jsonl").read_text(encoding="utf-8").splitlines()
                  if line.strip()]
        status, ingested = request("POST", f"{base}/ingest", {"docs": corpus})
        check("导入黄金语料", status == 200 and ingested.get("chunks", 0) > 8,
              f"片段={ingested.get('chunks')}")

        status, docs = request("GET", f"{base}/documents")
        check("文档列表可见", status == 200 and len(docs) == len(corpus), f"文档数={len(docs)}")

        status, hits = request("POST", f"{base}/search", {"query": "向量索引有哪些结构", "k": 5})
        check("检索端点返回命中", status == 200 and len(hits) > 0,
              f"top={hits[0]['chunk_id'] if hits else 'none'}")

        status, chat = request("POST", f"{base}/chat",
                               {"query": "星枢系统默认采用什么向量索引？", "with_trace": True})
        check("问答返回文本与引用", status == 200 and bool(chat.get("text")) and len(chat.get("citations", [])) > 0,
              f"路径={chat.get('route', {}).get('path')} 引用={len(chat.get('citations', []))}")
        grounded = (chat.get("verification") or {}).get("groundedness")
        check("问答通过归因校验", grounded is not None and grounded >= 0.6, f"可证性={grounded}")

        status, trace = request("GET", f"{base}/trace/{chat.get('trace_id')}")
        check("追踪树可取回", status == 200 and "children" in trace, f"trace={chat.get('trace_id')}")

        status, _ = request("POST", f"{base}/ingest", {"docs": [{"doc_id": "bad"}]})
        check("非法导入返回 400", status == 400, f"status={status}")

        status, _ = request("GET", f"{base}/trace/not-exist")
        check("缺失追踪返回 404", status == 404, f"status={status}")

        status, report = request("POST", f"{base}/evaluate")
        check("评测端点达标", status == 200 and report.get("metrics", {}).get("recall@5", 0) >= 0.75,
              json.dumps({k: round(v, 3) for k, v in report.get("metrics", {}).items() if k != "cases"},
                         ensure_ascii=False))

        status, _ = request("POST", f"{base}/ingest", {"docs": [
            {"doc_id": "tmp", "text": "临时文档用于验证删除流程。" * 40}]})
        status, removed = request("DELETE", f"{base}/documents/tmp")
        check("文档删除生效", status == 200 and removed.get("removed_chunks", 0) > 0,
              f"删除片段={removed.get('removed_chunks')}")

        # --- 控制台：单文件 HTML 必须能直接由服务吐出，且内联零外部依赖 ---
        status, page = fetch_text(f"{base}/")
        check("根路径返回控制台 HTML", status == 200 and "星枢 StellarNexus" in page,
              f"status={status} bytes={len(page)}")
        check("控制台零外部依赖（无外链 script/link，无远程字体）",
              "cdn." not in page and "unpkg" not in page and "jsdelivr" not in page
              and 'src="http' not in page and "@import" not in page,
              f"inline_js={'<script>' in page}")
        check("控制台署名晨星", "晨星" in page, "")
        status, asset = fetch_text(f"{base}/web/index.html")
        check("/web 静态目录可访问", status == 200 and "StellarNexus" in asset, f"status={status}")
    finally:
        for child in _children:
            kill(child)

    log(f"\n通过 {passed} / 失败 {failed}")
    report_fh.close()
    builtins.print = real_print
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        import traceback

        target = ROOT / "data" / "e2e-report.txt"
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "a", encoding="utf-8") as handle:
            handle.write(traceback.format_exc())
        raise
