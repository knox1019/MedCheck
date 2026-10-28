FROM python:3.11-slim

# PYTHONUNBUFFERED=1 很重要：不设的话 Python 会缓冲标准输出，
# 容器日志要等缓冲区满了才出现，排障时看不到实时输出。
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

# 用 pip 装 uv，而不是从 ghcr.io 拷二进制。
# 原因：从 ghcr.io 拉文件在国内经常超时，而 pyproject.toml 里已经
# 把清华 PyPI 源设为默认，走 pip 更稳。
RUN pip install --no-cache-dir uv

# 先只拷依赖清单。这一层会被缓存：
# 只要 pyproject/uv.lock 没变，改业务代码时不会重新装一遍依赖。
COPY pyproject.toml uv.lock README.md ./
# --no-cache 很关键：不加的话 uv 会在 /root/.cache/uv 里再存一份完整的
# 包解压副本，实测多出 543MB，和 .venv 完全重复。镜像构建是一次性的，
# 这个缓存没有任何复用价值。
RUN uv sync --frozen --no-dev --no-cache

# 再拷业务代码。代码改动只影响这之后的两层，构建很快。
COPY app ./app
# 演示界面（/ui）。不打进镜像的话容器里访问 /ui 会是 404。
COPY web ./web

EXPOSE 8000

# 健康检查用 python 而不是 curl —— slim 镜像里没有 curl，装了也是白占体积。
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')" || exit 1

# --host 0.0.0.0 必须加。
# uvicorn 默认只监听 127.0.0.1，那是「容器内部的回环地址」，
# 宿主机通过端口映射访问不到，表现为「容器起来了但浏览器打不开」。
CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
