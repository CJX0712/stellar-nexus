"""核查模型源里到底有哪些权重文件。

存在理由：ADR-0003 推翻过一次设计 —— 原计划默认用 ONNX 交叉编码器重排，
直到真的去 ModelScope 看了一眼，才发现 bge-reranker 只发 PyTorch 权重，
没有 ONNX。如果当时不去核实，交付物里会留下一条**从未真正跑通过**的默认路径。

所以这个脚本把「核实」这一步固化成可重复执行的动作，
而不是留在某次会话的记忆里。它只读，不下载。

作者: 晨星
"""

from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request

MODELSCOPE_API = "https://www.modelscope.cn/api/v1/models/{repo}/repo/files?Revision=master"

# ADR-0003 核查过的候选仓库
CANDIDATES = [
    "BAAI/bge-reranker-base",
    "BAAI/bge-reranker-large",
    "BAAI/bge-reranker-v2-m3",
    "Xorbits/bge-reranker-base",
]

ONNX_HINTS = (".onnx", "onnx")
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "vocab.txt", "sentencepiece.bpe.model")


def list_files(repo: str, timeout: float = 30.0) -> list[dict]:
    url = MODELSCOPE_API.format(repo=repo)
    request = urllib.request.Request(url, headers={"User-Agent": "stellarnx-model-check"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.load(response)
    return payload.get("Data", {}).get("Files", [])


def classify(files: list[dict]) -> dict[str, list[str]]:
    buckets: dict[str, list[str]] = {"onnx": [], "torch": [], "tokenizer": [], "other": []}
    for item in files:
        path = item.get("Path") or item.get("Name") or ""
        lower = path.lower()
        if lower.endswith(".onnx"):
            buckets["onnx"].append(path)
        elif lower.endswith((".bin", ".safetensors", ".pt", ".pth")):
            buckets["torch"].append(path)
        elif lower.endswith(TOKENIZER_FILES):
            buckets["tokenizer"].append(path)
        else:
            buckets["other"].append(path)
    return buckets


def probe(repo: str) -> tuple[bool, dict[str, list[str]], str]:
    try:
        files = list_files(repo)
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as exc:
        return False, {}, f"{type(exc).__name__}: {exc}"
    except json.JSONDecodeError as exc:
        return False, {}, f"响应不是合法 JSON: {exc}"
    return True, classify(files), ""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="核查模型源的权重类型（只读）")
    parser.add_argument("repos", nargs="*", default=None,
                        help=f"要核查的仓库，默认：{' '.join(CANDIDATES)}")
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args(argv)
    repos = args.repos or CANDIDATES

    usable_onnx: list[str] = []
    unreachable: list[str] = []

    for repo in repos:
        ok, buckets, error = probe(repo)
        if not ok:
            unreachable.append(repo)
            print(f"[网络不可达] {repo}  {error}")
            continue
        has_onnx = "已具备 ONNX 权重" if buckets["onnx"] else "无 ONNX 权重"
        print(f"[{has_onnx}] {repo}")
        for kind in ("onnx", "torch", "tokenizer"):
            names = buckets[kind]
            if names:
                preview = ", ".join(names[:4]) + (" …" if len(names) > 4 else "")
                print(f"    {kind:<9} {len(names):>3} 个  {preview}")
        if buckets["onnx"]:
            usable_onnx.append(repo)

    print()
    if unreachable and len(unreachable) == len(repos):
        print("所有候选仓库都不可达。本机若走代理，请确认允许访问 modelscope.cn。")
        return 2

    if usable_onnx:
        print(f"可直接启用 ONNX 交叉编码器：{', '.join(usable_onnx)}")
        print("  python scripts/download_models.py --rerank-repo <repo> --with-tokenizer")
        print("  export SNX_RERANK_BACKEND=onnx")
    else:
        print("没有找到带 ONNX 权重的仓库 —— 与 ADR-0003 的核查结论一致。")
        print("当前默认走 LLM 列表式重排（SNX_RERANK_BACKEND=llm），"
              "ONNX 后端保留为可插拔项：自备 model.onnx + tokenizer.json 即生效。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
