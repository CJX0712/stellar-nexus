"""模型下载脚本。

ModelScope 为本机唯一可达的模型源（HuggingFace 超时）。
注意：BAAI/bge-reranker-base 在 ModelScope 上只发布 PyTorch 权重，没有 ONNX，
因此交叉编码器后端需要自备 model.onnx + tokenizer.json 放到 models 目录。
缺失时系统会走 LLM 列表式重排或词汇重排，不会中断。

作者: 晨星
"""

from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TIMEOUT = 120


def list_files(repo: str) -> list[dict]:
    url = f"https://www.modelscope.cn/api/v1/models/{repo}/repo/files?Revision=master"
    with urllib.request.urlopen(url, timeout=30) as resp:
        data = json.load(resp)
    return data.get("Data", {}).get("Files", [])


def download(repo: str, filename: str, out_dir: Path) -> Path | None:
    url = (f"https://www.modelscope.cn/api/v1/models/{repo}/repo"
           f"?Revision=master&FilePath={filename}")
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / Path(filename).name
    if target.exists() and target.stat().st_size > 0:
        print(f"已存在，跳过: {target}")
        return target
    print(f"下载 {repo}/{filename} -> {target}")
    with urllib.request.urlopen(url, timeout=TIMEOUT) as resp:
        target.write_bytes(resp.read())
    print(f"完成，大小 {target.stat().st_size / 1e6:.1f} MB")
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="下载星枢所需模型")
    parser.add_argument("--rerank-repo", default="BAAI/bge-reranker-base",
                        help="重排模型仓库（ModelScope）")
    parser.add_argument("--out", default="models/bge-reranker-base", help="输出目录")
    parser.add_argument("--with-tokenizer", action="store_true",
                        help="同时下载 tokenizer.json")
    args = parser.parse_args(argv)

    out_dir = ROOT / args.out
    print(f"== 扫描 {args.rerank_repo} ==")
    try:
        files = list_files(args.rerank_repo)
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as exc:
        print(f"无法访问仓库: {type(exc).__name__}: {exc}")
        return 1
    except json.JSONDecodeError as exc:
        print(f"仓库接口返回了非法 JSON: {exc}")
        return 1

    onnx = [f for f in files if f.get("Path", "").endswith(".onnx")]
    if not onnx:
        print("该仓库没有 ONNX 权重（ModelScope 上的 bge-reranker 只发 PyTorch 权重）。")
        print("可选做法：")
        print("  1) 在有网络的环境用 optimum 导出 model.onnx，连同 tokenizer.json 放入", out_dir)
        print("  2) 不启用交叉编码器，使用 LLM 列表式重排（SNX_RERANK_BACKEND=llm）")
        print("  3) 使用零依赖词汇重排（SNX_RERANK_BACKEND=lexical）")
        if not args.with_tokenizer:
            return 0

    for item in onnx:
        download(args.rerank_repo, item["Path"], out_dir)
    if args.with_tokenizer:
        download(args.rerank_repo, "tokenizer.json", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
