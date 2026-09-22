"""语料体检：报告每篇文档长度与切分后片段数。作者: 晨星"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stellarnx.chunk import Chunker
from stellarnx.eval.dataset import load_corpus

corpus = load_corpus(Path(__file__).resolve().parents[1] / "data/golden/corpus.jsonl")
chunker = Chunker()
short = []
for doc in corpus:
    count = len(chunker.chunk(doc["doc_id"], doc["text"]))
    flag = "" if count > 1 else "  <== 太短"
    if count <= 1:
        short.append(doc["doc_id"])
    print(f"{doc['doc_id']:10s} 字符={len(doc['text']):5d} 片段={count}{flag}")
print("过短文档:", short or "无")
