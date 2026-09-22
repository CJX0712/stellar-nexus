"""命令行入口。只做参数解析与结果打印，不含业务逻辑。

作者: 晨星
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from stellarnx import __author__, __version__
from stellarnx.core.paths import CORPUS_PATH, QUESTIONS_PATH


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="stellarnx", description="星枢 StellarNexus 命令行工具")
    parser.add_argument("--version", action="version", version=f"{__version__} by {__author__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_serve = sub.add_parser("serve", help="启动 HTTP 服务")
    p_serve.add_argument("--host", default=None)
    p_serve.add_argument("--port", type=int, default=None)
    p_serve.add_argument("--reload", action="store_true")

    p_ingest = sub.add_parser("ingest", help="导入文档")
    p_ingest.add_argument("path", help="JSONL 文件，每行 {doc_id, text, meta?}")

    p_search = sub.add_parser("search", help="检索")
    p_search.add_argument("query")
    p_search.add_argument("-k", type=int, default=5)

    p_chat = sub.add_parser("chat", help="问答")
    p_chat.add_argument("query")
    p_chat.add_argument("--json", action="store_true")

    sub.add_parser("eval", help="运行评测")
    sub.add_parser("stats", help="查看系统状态")
    sub.add_parser("verify", help="一键自检")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    command = args.command

    if command == "serve":
        import uvicorn

        from stellarnx.api.app import create_app
        from stellarnx.core.config import load_config
        from stellarnx.pipeline.orchestrator import build_system

        cfg = load_config()
        app = create_app(system=build_system(cfg), cfg=cfg)
        uvicorn.run(app, host=args.host or cfg.api_host,
                    port=args.port or cfg.api_port, reload=args.reload)
        return 0

    if command == "verify":
        from stellarnx.selftest import run_selftest

        results = run_selftest()
        failed = [r for r in results if not r["ok"]]
        for item in results:
            mark = "PASS" if item["ok"] else "FAIL"
            print(f"[{mark}] {item['name']:<34} {item['detail']}")
        print(f"\n通过 {len(results) - len(failed)}/{len(results)}")
        return 0 if not failed else 1

    from stellarnx.core.config import load_config
    from stellarnx.pipeline.orchestrator import build_system

    if command == "eval":
        from stellarnx.eval.dataset import load_cases, load_corpus
        from stellarnx.eval.runner import EvalRunner

        cfg = load_config()
        report = EvalRunner(cfg).run(
            load_corpus(CORPUS_PATH),
            load_cases(QUESTIONS_PATH),
        )
        print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
        return 0 if report.passed else 1

    # stats/ingest/search/chat 共用同一个持久化知识库。
    # 如果每个子命令各起一个内存库，"先 ingest 再 search" 就会搜不到 —— 这是
    # 用户第一次上手最容易撞上的坑，所以这里必须落库而不是用 :memory:。
    if command == "stats":
        system = build_system(load_config())
        print(json.dumps(system.stats(), ensure_ascii=False, indent=2))
        return 0

    system = build_system(load_config())

    if command == "ingest":
        path = Path(args.path)
        if not path.exists():
            print(f"文件不存在: {path}", file=sys.stderr)
            return 2
        docs = [json.loads(line) for line in
                path.read_text(encoding="utf-8").splitlines() if line.strip()]
        print(f"导入片段数: {system.ingest_many(docs)}")
        return 0

    if command == "search":
        hits = system.search(args.query, args.k)
        for rank, hit in enumerate(hits, 1):
            print(f"{rank}. [{hit.score:.4f}] {hit.chunk_id} ({hit.source}) {hit.text[:100]}")
        return 0

    if command == "chat":
        answer = system.answer(args.query)
        if args.json:
            print(json.dumps(answer.to_dict(), ensure_ascii=False, indent=2))
        else:
            print(answer.text)
            if answer.citations:
                print("\n引用:")
                for citation in answer.citations:
                    print(f"  - {citation.chunk_id}: {citation.text[:100]}")
            if answer.verification:
                print(f"\n可证性: {answer.verification.groundedness:.3f} "
                      f"(阈值 {answer.verification.threshold})")
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
