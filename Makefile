# 星枢 StellarNexus —— 统一入口
# 作者: 晨星
#
# 所有目标都可重复执行；`make ci` 就是 GitHub Actions 里跑的同一套。

PY      ?= python
VENV    ?= .venv
ifeq ($(OS),Windows_NT)
  BIN := $(VENV)/Scripts
else
  BIN := $(VENV)/bin
endif
PYBIN   := $(BIN)/python
NODE    ?= node

.DEFAULT_GOAL := help
.PHONY: help venv install dev verify test imports e2e web eval check ci serve clean

help: ## 显示所有可用目标
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

venv: ## 创建虚拟环境
	$(PY) -m venv $(VENV)

install: venv ## 安装运行时依赖（零编译）
	$(PYBIN) -m pip install -U pip
	$(PYBIN) -m pip install -e .

dev: install ## 安装开发依赖
	$(PYBIN) -m pip install -e ".[dev]"

verify: ## 一键自检（离线，15 项）
	$(PYBIN) -m stellarnx.cli verify

test: ## 单元与集成测试
	$(PYBIN) -m pytest -q

imports: ## 分层依赖与循环依赖检查
	$(PYBIN) scripts/check_imports.py

e2e: ## 端到端：真起 HTTP 服务跑成功流与错误流
	$(PYBIN) scripts/e2e.py

web: ## 控制台无头冒烟（需 node + jsdom）
	$(NODE) scripts/web_smoke.mjs

eval: ## 离线评测与门禁（退出码即结论）
	SNX_EMBED_BACKEND=hash SNX_LLM_BACKEND=mock SNX_RERANK_BACKEND=lexical \
		$(PYBIN) -m stellarnx.cli eval

check: imports verify test e2e ## 离线全量检查（不含需要 node/模型的部分）

ci: imports verify test e2e eval ## CI 门禁：与流水线完全一致

serve: ## 启动服务与控制台
	$(PYBIN) -m uvicorn stellarnx.api.asgi:app --host 127.0.0.1 --port 8000

clean: ## 清理临时产物（不动 data/*.db）
	find . -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache data/.pytest-tmp data/e2e-report.txt 2>/dev/null || true
