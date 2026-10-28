# 换机部署指南

在另一台电脑上从零跑起 MedCheck 的完整步骤。全程约 20–40 分钟（大头是下载模型）。

## 前提认知：什么会跟过来，什么不会

| 内容 | 在 git 里？ | 新电脑要做什么 |
| --- | --- | --- |
| 全部代码、测试、README | ✅ | 无 |
| 语料 `data/corpus/*.jsonl`（657 药 / 1659 条） | ✅ | 无 |
| 药名索引 `drug_name_index.json`（663 条） | ✅ | 无 |
| 评测集 `eval/*.jsonl` | ✅ | 无 |
| `.env`（模型端点、key） | ❌ | 手动创建 |
| 向量库 `data/chroma/` | ❌ | **重新建库**（约 17 分钟） |
| 重排模型缓存 `data/models/`（约 570MB） | ❌ | 首次用到时自动下载 |
| Ollama 模型（bge-m3 1.2GB / qwen2.5:7b 4.7GB） | ❌ | 重新 pull |

> **为什么向量库不拷贝**：它必须由目标机器的 embedding 模型生成。就算拷过去，模型版本不一致时向量也对不上。重建是唯一可靠的方式。

---

## 方式一：直接跑（推荐先用这个验证）

### ① 装三个工具

```
git      https://git-scm.com/download/win
uv       powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
Ollama   https://ollama.com/download
```

**每装完一个都要新开终端**，否则命令找不到（PATH 只对新进程生效）。

### ② 拉代码

```powershell
cd C:\Users\T14\Desktop        # 位置随意
git clone https://github.com/knox1019/MedCheck.git
cd MedCheck
```

### ③ 重建 Python 环境

```powershell
uv sync
```

按 `uv.lock` 装同一批版本，约 30 秒。

### ④ 配置 `.env`

```powershell
copy .env.example .env
```

然后编辑 `.env`，核心是 **LLM 部分**。三种选择：

```ini
# 选择 A：本地 Ollama（完全离线，但慢，约 19s/次）
LLM_BASE_URL=http://localhost:11434/v1
LLM_MODEL=qwen2.5:7b
LLM_API_KEY=ollama

# 选择 B：DeepSeek 官方（快，约 2-3s/次，需 key）
# LLM_BASE_URL=https://api.deepseek.com        ← 不带 /v1
# LLM_MODEL=deepseek-flash
# LLM_API_KEY=sk-你的key

# 选择 C：本地模型网关（原机器用的方案）
# LLM_BASE_URL=http://localhost:7863/v1
# LLM_MODEL=cn:hy3
```

Embedding 部分保持默认（本地 Ollama + bge-m3）即可。

### ⑤ 拉 Ollama 模型并启动

```powershell
ollama pull bge-m3
ollama pull qwen2.5:7b        # 仅选择 A 需要
```

**保持 Ollama 在托盘运行**（小羊驼图标）。它没启动的话，后面建库会报 `embedding 连续失败 3 次` —— 这个报错看起来像代码 bug，实际是服务没起。

### ⑥ 建向量库（最耗时的一步）

```powershell
uv run python -m scripts.build_index data/corpus/drugs_all.jsonl
```

1659 条 × 最长 600 字，约 **17 分钟**。输出 `657 药 / 1659 条` 即成功。

### ⑦ 验证

```powershell
uv run pytest -q                              # 应 175 passed（Ollama 没开则 171 passed + 4 skipped）
uv run python -m scripts.search "华法林的禁忌是什么"   # 应返回带出处的检索结果
uv run python -m scripts.run_eval             # 检索评测，应接近 42/42
```

三条都过，环境完整。

### ⑧ 启动

```powershell
uv run uvicorn app.main:app --reload
```

浏览器打开 <http://127.0.0.1:8000/> → 自动跳到演示界面。

---

## 方式二：Docker 部署

前置：Docker Desktop（依赖 WSL2，安装器会引导，可能要求重启）。

### ①–⑤ 同上（拉代码、配 `.env` 仍需要；uv sync 可跳过）

`.env` 里的 `localhost` **不用改**——compose 会自动覆盖成 `host.docker.internal`。

### ⑥ 构建并启动

```powershell
cd docker
docker compose up --build
```

首次构建约 2 分钟。看到 `Engine running`、容器状态 `Up (healthy)` 即可。

### ⑦ 验证（顺序很重要）

```powershell
curl http://127.0.0.1:8000/health          # {"status":"ok","env":"docker",...}
curl -X POST http://127.0.0.1:8000/api/report/parse -F "file=@../data/samples/report_demo.pdf"
curl -X POST http://127.0.0.1:8000/api/chat -H "Content-Type: application/json" ^
     -d "{\"question\":\"华法林的禁忌是什么\"}"
```

> ⚠️ **第三条必须测**。`/health` 和解析接口都不依赖模型，就算 `LLM_BASE_URL` 没配对它们也全绿——只有真问一次问题才能暴露模型连不通（503）。

**Docker 专属注意**：

| 事项 | 说明 |
| --- | --- |
| 容器访问宿主机服务 | 用 `host.docker.internal`（compose 已配好，不用手动改） |
| 重排模型缓存 | 已挂载到 `data/models/`，重启容器不重下 570MB |
| 首次问答慢 | 约 100s（模型加载），第二次约 11s，正常 |
| 向量库挂载 | `data/` 整体挂进容器，宿主机建好的库直接可用 |

---

## 常见问题速查

| 症状 | 原因 | 处理 |
| --- | --- | --- |
| `embedding 连续失败 3 次` | Ollama 没启动 | 打开 Ollama 应用，确认托盘有小羊驼 |
| `ModuleNotFoundError: No module named 'app'` | 用文件路径跑了脚本 | 用 `uv run python -m scripts.xxx` |
| `uv run pytest` 有 4 个 skipped | Ollama 未启动，集成测试跳过 | 不是故障，启动后变 passed |
| 检索能跑，问答 503 | `.env` 里 LLM 端点没配对 | 检查 `LLM_BASE_URL` / `LLM_API_KEY` |
| DeepSeek 报 404 | 用了旧模型名或带 `/v1` | 用 `deepseek-flash`，端点不带 `/v1` |
| 首次带 `--rerank` 卡住 | 在下载 570MB 重排模型 | 等约 10 分钟（走 hf-mirror） |
| `docker compose` 报 unknown command | compose 插件未注册 | 用独立的 `docker-compose` 命令 |
| 中文输出乱码 / git 当二进制 | 文件被写成 UTF-16 | 用 VS Code 存成 UTF-8 |

---

## 两台机器之间同步开发

```powershell
# 新电脑上改了代码
git add . && git commit -m "..." && git push

# 原电脑拉回来
git pull
```

注意：**向量库不随 git 走**。如果两边都改了语料（`drugs_all.jsonl`），拉完代码后要重新 `build_index`，否则检索的还是旧数据。
