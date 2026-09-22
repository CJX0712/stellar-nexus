# 部署指南

作者 **晨星** · 版本 1.0.0

面向「把星枢部署到一台机器上并让它稳定提供服务」。

---

## 1. 环境要求

| 项 | 最低 | 推荐 | 说明 |
|---|---|---|---|
| 操作系统 | Windows 10 / Ubuntu 20.04 | Windows 11 / Ubuntu 22.04 | 双平台均有两套轮子 |
| Python | 3.12 | 3.13 | **不需要** MSVC / cmake / gcc |
| CPU | 4 核 | 8 核以上 | 生成阶段是纯 CPU 瓶颈 |
| 内存 | 4 GB | 16 GB | 7B 模型量化后约需 6 GB |
| 磁盘 | 1 GB | 5 GB | 含模型与向量库 |
| GPU | 不需要 | 有则更快 | 当前实现走 Ollama，GPU 加速由 Ollama 负责 |

---

## 2. 安装

### 2.1 取代码

```bash
git clone <repo-url> stellar-nexus
cd stellar-nexus
```

### 2.2 建虚拟环境并安装

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e .
```

全部依赖均有预编译轮子，**零编译**。实测本机 40 个包装完无需任何编译器。

### 2.3 自检（离线，不需要模型）

```bash
stellarnx verify
```

必须 `通过 15/15`。这一步不通过就不要继续，先按输出定位失败项。

---

## 3. 模型准备

### 3.1 方式一：Ollama（推荐）

```bash
# 安装 Ollama 后
ollama pull qwen2.5:1.5b-instruct    # 生成，约 1 GB
ollama pull bge-m3                   # 嵌入，1024 维，约 1.2 GB
ollama serve                          # 默认监听 127.0.0.1:11434
```

验证：

```bash
curl http://127.0.0.1:11434/api/tags
```

### 3.2 方式二：不装模型（离线模式）

不装任何模型也能跑完整链路，质量略降但确定可复现。
此时保持默认后端即可：`hash` 嵌入 + `mock` 生成 + `lexical` 重排。

### 3.3 生产后端组合对照

| 场景 | 嵌入 | 生成 | 重排 |
|---|---|---|---|
| 离线开发 / CI | `hash` | `mock` | `lexical` |
| 本地体验 | `ollama` | `ollama` | `llm` |
| 有 ONNX 重排权重 | `ollama` | `ollama` | `onnx` |
| 只想要最快响应 | `ollama` | `ollama` | `identity` |

---

## 4. 配置

所有配置通过 `SNX_` 前缀环境变量注入，不需要改代码。

```bash
# Linux / macOS
export SNX_EMBED_BACKEND=ollama
export SNX_LLM_BACKEND=ollama
export SNX_RERANK_BACKEND=llm
export SNX_OLLAMA_HOST=http://127.0.0.1:11434
export SNX_DATA_DIR=/var/lib/stellarnx
export SNX_API_HOST=127.0.0.1
export SNX_API_PORT=8000
```

```powershell
# Windows PowerShell
$env:SNX_EMBED_BACKEND="ollama"
$env:SNX_LLM_BACKEND="ollama"
$env:SNX_RERANK_BACKEND="llm"
$env:SNX_OLLAMA_HOST="http://127.0.0.1:11434"
```

完整清单见 [`README.md` 配置章节](../README.md#配置)。

### 建议直接落成 `.env` 风格的启动脚本

```bash
#!/usr/bin/env bash
# run.sh
set -euo pipefail
cd "$(dirname "$0")"
source .venv/bin/activate
export SNX_EMBED_BACKEND=ollama
export SNX_LLM_BACKEND=ollama
export SNX_RERANK_BACKEND=llm
export SNX_DATA_DIR="$PWD/data"
exec stellarnx serve --host 127.0.0.1 --port 8000
```

```powershell
# run.ps1
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
& .\.venv\Scripts\Activate.ps1
$env:SNX_EMBED_BACKEND = "ollama"
$env:SNX_LLM_BACKEND = "ollama"
$env:SNX_RERANK_BACKEND = "llm"
& .\.venv\Scripts\python.exe -m uvicorn stellarnx.api.asgi:app --host 127.0.0.1 --port 8000
```

---

## 5. 启动与验证

```bash
stellarnx serve --port 8000
```

### 5.1 健康检查

```bash
curl -s http://127.0.0.1:8000/health
# {"status":"ok","version":"1.0.0","author":"晨星",
#  "backends":{"embedder":"ollama:bge-m3","llm":"ollama:qwen2.5:1.5b-instruct",
#              "reranker":"llm-listwise","vector":"faiss"}}
```

**逐个核对 `backends`** —— 这是确认「配置真的生效」而不是「配了但没起来」的唯一可靠方式。
如果配了 `ollama` 却显示 `hash`，说明环境变量没传到进程里。

### 5.2 状态检查

```bash
curl -s http://127.0.0.1:8000/stats
```

关键字段：`chunks` 与 `vectors` **必须相等**。不等说明索引与片段库不同步，
此时应重启（启动时会自动 `restore()`）。

### 5.3 冒烟

```bash
curl -s -X POST http://127.0.0.1:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"docs":[{"doc_id":"t1","text":"这是一段用于验证部署的测试文本。"}]}'

curl -s -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query":"这段文本是做什么的？"}'
```

返回中应含非空 `text`、`citations`，以及 `verification.groundedness`。

浏览器打开 <http://127.0.0.1:8000> 看控制台。

---

## 6. 数据与持久化

| 项 | 位置 | 说明 |
|---|---|---|
| 知识库 | `<SNX_DATA_DIR>/stellarnx.db` | sqlite，单文件，含文档与片段 |
| 向量索引 | 内存（启动时重建） | `restore()` 从 sqlite 重新嵌入 |
| BM25 索引 | 内存（启动时重建） | 同上 |
| 模型权重 | Ollama 管理 | 不在本仓库内 |
| ONNX 模型（可选） | `SNX_RERANK_MODEL_DIR` | 默认 `models/`，已 gitignore |
| 评测报告 | `data/e2e-report.txt` | 端到端脚本产出 |

### 备份

只需备份 `<SNX_DATA_DIR>/stellarnx.db` 一个文件（先停服务或用 sqlite 在线备份）：

```bash
sqlite3 /var/lib/stellarnx/stellarnx.db ".backup '/backup/stellarnx-$(date +%F).db'"
```

### 恢复

把 `.db` 文件放回 `<SNX_DATA_DIR>/`，重启服务。启动时自动重建向量与 BM25 索引
（大语料下这会花一些时间，属于正常）。

### 一致性校验

```bash
stellarnx stats | grep -E '"chunks"|"vectors"'
```

两个数字必须相等。

---

## 7. 反向代理

服务本身无鉴权，**默认只绑 `127.0.0.1`**。对外暴露必须加代理与鉴权。

### Nginx

```nginx
server {
    listen 443 ssl http2;
    server_name nexus.example.com;

    ssl_certificate     /etc/ssl/certs/nexus.crt;
    ssl_certificate_key /etc/ssl/private/nexus.key;

    # 生成阶段可能耗时数十秒（CPU 推理），超时必须放宽
    proxy_read_timeout 300s;
    proxy_send_timeout 300s;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # SSE 必须关闭缓冲，否则 stage 事件会被攒到最后一次性吐出
    location /chat/stream {
        proxy_pass http://127.0.0.1:8000;
        proxy_buffering off;
        proxy_cache off;
        proxy_set_header Connection "";
        chunked_transfer_encoding on;
    }
}
```

> `proxy_read_timeout` 是必调项。默认 60s，而 CPU 上单次问答可能 30s+，
> 叠加重排与重生成会超时 —— 表现为「浏览器一直转圈，日志里什么都没有」。

### 鉴权（在代理层做）

```nginx
location / {
    auth_basic "StellarNexus";
    auth_basic_user_file /etc/nginx/.htpasswd;
    proxy_pass http://127.0.0.1:8000;
}
```

---

## 8. 容器化

```dockerfile
FROM python:3.13-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir -e .

COPY web ./web
COPY data ./data
ENV SNX_OFFLINE=1 \
    SNX_DATA_DIR=/data \
    SNX_EMBED_BACKEND=hash \
    SNX_LLM_BACKEND=mock \
    SNX_RERANK_BACKEND=lexical
VOLUME ["/data"]
EXPOSE 8000
CMD ["uvicorn", "stellarnx.api.asgi:app", "--host", "0.0.0.0", "--port", "8000"]
```

要点：

- 容器内默认用离线后端，**镜像不含模型权重**。要用真实模型，让容器访问宿主的 Ollama
  （`SNX_OLLAMA_HOST=http://host.docker.internal:11434`）。
- `/data` 挂卷，否则重启丢库。
- 如果需要 ONNX 重排，把 `models/` 一起挂进来。

---

## 9. 常驻运行

### systemd（Linux）

```ini
# /etc/systemd/system/stellarnx.service
[Unit]
Description=StellarNexus AI Service
After=network.target ollama.service

[Service]
Type=simple
User=stellarnx
WorkingDirectory=/opt/stellar-nexus
Environment=SNX_LLM_BACKEND=ollama
Environment=SNX_EMBED_BACKEND=ollama
Environment=SNX_RERANK_BACKEND=llm
Environment=SNX_DATA_DIR=/var/lib/stellarnx
ExecStart=/opt/stellar-nexus/.venv/bin/stellarnx serve --host 127.0.0.1 --port 8000
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now stellarnx
sudo systemctl status stellarnx
```

### Windows 计划任务

```powershell
$action  = New-ScheduledTaskAction -Execute "powershell.exe" `
           -Argument "-NoProfile -WindowStyle Hidden -File C:\stellar-nexus\run.ps1"
$trigger = New-ScheduledTaskTrigger -AtStartup
Register-ScheduledTask -TaskName "StellarNexus" -Action $action -Trigger $trigger -RunLevel Highest
```

或用 `nssm` 注册为服务。

---

## 10. 升级与回滚

```bash
git fetch --tags
git checkout v1.1.0
source .venv/bin/activate
pip install -e .          # 依赖有变化时
stellarnx verify          # 必须先过
systemctl restart stellarnx
curl -s http://127.0.0.1:8000/health
```

向量维度变化时（换嵌入模型）**必须重建索引**：

```bash
# 备份
cp data/stellarnx.db data/stellarnx.db.bak
# 换嵌入模型后，旧向量维度与新嵌入不匹配，需要重新入库
stellarnx ingest data/docs.jsonl
```

回滚：切回旧 tag、装回旧依赖、恢复 `.db` 备份、重启。

---

## 11. 故障排查

| 现象 | 根因 | 处理 |
|---|---|---|
| 配了 ollama 却显示 hash | 环境变量没传进进程（sudo 丢环境、systemd 未声明） | `curl /health` 核对 `backends`；systemd 用 `Environment=` 显式声明 |
| Python 连不上 Ollama，curl 却能连 | 本地有代理，httpx 默认信任 `HTTP_PROXY` 把 localhost 也走了代理 | 已内置 `trust_env=False`；自己写的客户端也要这样 |
| 重启后库里有数据但搜不到 | 向量索引与 BM25 只在内存 | 已修：启动自动 `restore()`。若仍异常，`chunks` 与 `vectors` 不等即为症状，重启观察 |
| 前端一直转圈，后端无日志 | 反向代理 `proxy_read_timeout` 太小 | 调到 300s |
| SSE 阶段事件一次性全出 | 代理开了缓冲 | `proxy_buffering off` |
| 首次请求特别慢（几十秒） | Ollama 首次加载模型进内存 | 预热：启动后先发一个简单问题 |
| `latency_p95` 门禁失败 | 在真实模型下跑了门禁 | 门禁只跑离线路径；真实模型延迟高一个数量级属正常 |
| `pip install` 慢或失败 | 网络受限 | 换国内镜像；本仓库不依赖 HuggingFace |
| 中文乱码 | Windows 控制台编码 | 设 `PYTHONIOENCODING=utf-8`、`PYTHONUTF8=1` |

### 收集诊断信息

```bash
stellarnx stats > diag-stats.json
stellarnx verify > diag-verify.txt 2>&1
curl -s http://127.0.0.1:8000/health > diag-health.json
python scripts/check_imports.py > diag-imports.txt
```

---

## 12. 部署验收清单

部署完成后，逐项确认：

- [ ] `stellarnx verify` → `通过 15/15`
- [ ] `pytest -q` → 全绿
- [ ] `python scripts/e2e.py` → `通过 16 / 失败 0`
- [ ] `python scripts/check_imports.py` → `通过 2/2`
- [ ] `curl /health` 的 `author == "晨星"`，且 `backends` 与预期一致
- [ ] `curl /stats` 的 `chunks == vectors`
- [ ] 浏览器打开根路径能看到控制台，头部四个后端名正确
- [ ] 控制台「知识库」能写入文档，「对话」能返回带引用的答案
- [ ] 「链路追踪」能展开 Span 瀑布图
- [ ] 「评测门禁」能跑出结论并显示逐项判定
- [ ] 重启服务后，之前入库的文档仍可检索（持久化生效）
- [ ] 对外暴露时已加鉴权与 TLS
