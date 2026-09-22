"""依赖轮子可用性预检。

存在理由：本项目的硬约束是「干净环境一键复现，不装编译器」。
一旦某个依赖在目标平台没有预编译轮子，pip 会转向源码编译，
在没装 MSVC/gcc 的机器上直接失败 —— 而失败发生在**别人 clone 之后**，
不是在提交之前。这个脚本把检查提前到提交前。

做法：读 pyproject.toml 的 dependencies，逐个查 PyPI，
判断目标平台 + 目标 Python 版本是否存在可用轮子。

作者: 晨星
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYPI = "https://pypi.org/pypi/{name}/{version}/json"

PLATFORM_KEYS = {
    "linux": ("manylinux", "musllinux"),
    "windows": ("win_amd64", "win32"),
    "macos": ("macosx",),
}

# 纯 Python 包通常以 py3-none-any 发布，任何平台都算可用
PURE_TAGS = ("py3-none-any",)


def parse_dependencies(pyproject: Path) -> list[tuple[str, str]]:
    """从 pyproject 里取出 (名字, 版本) 列表。只处理 `name==version` 形式。"""
    text = pyproject.read_text(encoding="utf-8")
    start = text.find("dependencies = [")
    if start < 0:
        raise SystemExit("pyproject.toml 里找不到 dependencies")
    end = text.find("]", start)
    block = text[start:end]
    out: list[tuple[str, str]] = []
    for line in block.splitlines():
        line = line.strip().strip('",')
        if not line or line.startswith("#") or line.startswith("dependencies"):
            continue
        if "==" not in line:
            continue
        name, version = line.split("==", 1)
        out.append((name.strip(), version.strip()))
    return out


def fetch(name: str, version: str, timeout: float) -> dict | None:
    url = PYPI.format(name=name, version=version)
    request = urllib.request.Request(url, headers={"User-Agent": "stellarnx-wheel-check"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise


def has_wheel(data: dict, platform: str, python_tag: str) -> bool:
    markers = PLATFORM_KEYS[platform]
    for item in data.get("urls", []):
        filename = item.get("filename", "")
        if not filename.endswith(".whl"):
            continue
        if any(tag in filename for tag in PURE_TAGS):
            return True
        if not any(marker in filename for marker in markers):
            continue
        # cp313 / abi3 / cp311-abi3 都能在目标解释器上安装
        if python_tag in filename or "abi3" in filename:
            return True
    return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="检查依赖是否有预编译轮子")
    parser.add_argument("--platform", default="linux", choices=sorted(PLATFORM_KEYS))
    parser.add_argument("--python", default="cp313", help="目标解释器标签，如 cp313")
    parser.add_argument("--pyproject", default=str(ROOT / "pyproject.toml"))
    parser.add_argument("--timeout", type=float, default=25.0)
    args = parser.parse_args(argv)

    deps = parse_dependencies(Path(args.pyproject))
    print(f"检查 {len(deps)} 个依赖 · 目标 {args.platform} / {args.python}\n")

    missing: list[str] = []
    unknown: list[str] = []
    for name, version in deps:
        try:
            data = fetch(name, version, args.timeout)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            unknown.append(name)
            print(f"[网络不可达] {name}=={version}  {type(exc).__name__}")
            continue
        if data is None:
            missing.append(name)
            print(f"[版本不存在] {name}=={version}")
            continue
        if has_wheel(data, args.platform, args.python):
            print(f"[有轮子] {name}=={version}")
        else:
            missing.append(name)
            print(f"[缺轮子] {name}=={version}  —— 该平台会转为源码编译")

    print()
    if unknown and len(unknown) == len(deps):
        print("所有依赖都查不到，网络不可用。")
        return 2
    if missing:
        print(f"以下依赖在 {args.platform}/{args.python} 上没有预编译轮子：")
        for name in missing:
            print(f"  - {name}")
        print("这意味着干净环境需要编译器。请换版本，或把它移到可选依赖。")
        return 1

    print(f"[PASS] {len(deps)} 个依赖在 {args.platform}/{args.python} 上都有预编译轮子，无需编译器")
    return 0


if __name__ == "__main__":
    sys.exit(main())
