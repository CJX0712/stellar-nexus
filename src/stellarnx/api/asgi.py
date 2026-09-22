"""ASGI 入口。

默认持久化到 ``<仓库根>/data/stellarnx.db``，重启后自动重建向量索引与 BM25；
``SNX_STORE=memory`` 可切成内存库（评测、演示用，退出即清空）。

作者: 晨星
"""

from __future__ import annotations

import logging
import os

from stellarnx import __author__, __version__
from stellarnx.api.app import create_app
from stellarnx.core.config import load_config
from stellarnx.pipeline.orchestrator import build_system

logger = logging.getLogger(__name__)

cfg = load_config()
if os.getenv("SNX_STORE", "").strip().lower() == "memory":
    from stellarnx.pipeline.orchestrator import NexusSystem

    _system = NexusSystem(cfg, ":memory:")
    logger.info("星枢 %s（作者 %s）启动，存储=内存", __version__, __author__)
else:
    _system = build_system(cfg)
    logger.info(
        "星枢 %s（作者 %s）启动，存储=%s，片段=%d",
        __version__, __author__, _system.store.path, _system.store.count_chunks(),
    )

app = create_app(system=_system, cfg=cfg)
