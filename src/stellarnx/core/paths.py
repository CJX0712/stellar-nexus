"""路径解析。所有内置数据文件的默认位置都相对"仓库根"，绝不依赖当前工作目录。

踩过的坑：服务端以 src 为工作目录启动时，"data/golden/corpus.jsonl" 会解析到
src/data/... 导致评测端点返回 404，而单元测试（仓库根运行）却全绿。

作者: 晨星
"""

from __future__ import annotations

from pathlib import Path

# src/stellarnx/core/paths.py -> 上四级即仓库根
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"
GOLDEN_DIR = DATA_DIR / "golden"
WEB_DIR = PROJECT_ROOT / "web"
MODELS_DIR = PROJECT_ROOT / "models"

CORPUS_PATH = GOLDEN_DIR / "corpus.jsonl"
QUESTIONS_PATH = GOLDEN_DIR / "questions.jsonl"


def resolve(path_like: str | Path) -> Path:
    """把相对路径解析到仓库根之下；绝对路径原样返回。"""
    path = Path(path_like)
    return path if path.is_absolute() else PROJECT_ROOT / path
