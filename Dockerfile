# syntax=docker/dockerfile:1
# Multi-stage build for the custom langflow fork

# Stage 1: Build frontend
FROM node:22-alpine AS frontend-builder
WORKDIR /app/frontend

# 国内 npm 镜像加速
RUN npm config set registry https://registry.npmmirror.com

COPY src/frontend/package.json src/frontend/package-lock.json ./
RUN npm ci

COPY src/frontend/ ./
RUN npm run build

# Stage 2: Python dependencies + app
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS python-builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libaio-dev curl \
    && rm -rf /var/lib/apt/lists/*

ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

# 国内 PyPI 镜像加速
ENV UV_INDEX_URL=https://mirrors.cloud.tencent.com/pypi/simple

WORKDIR /app

# --- 一次性复制所有文件（避免 workspace 成员解析失败）---
COPY pyproject.toml uv.lock ./
COPY src/backend/base/pyproject.toml /app/src/backend/base/pyproject.toml
COPY src/backend/base/README.md /app/src/backend/base/README.md
COPY src/backend/base/langflow /app/src/backend/base/langflow
COPY src/lfx/pyproject.toml /app/src/lfx/pyproject.toml
COPY src/lfx/README.md /app/src/lfx/README.md
COPY src/lfx/src /app/src/lfx/src
COPY src/backend/langflow /app/src/backend/langflow

# 一次性安装所有依赖 + 构建 workspace 包
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable --package langflow

# 注入前端构建产物
COPY --from=frontend-builder /app/frontend/build /app/src/backend/base/langflow/frontend

# Stage 3: Runtime
FROM python:3.12-bookworm-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -m -u 1000 langflow

# 从 builder 复制 venv 和源码
COPY --from=python-builder --chown=1000 /app/.venv /app/.venv
COPY --from=python-builder --chown=1000 /app/src/backend/base/langflow /app/src/backend/base/langflow
COPY --from=python-builder --chown=1000 /app/src/backend/langflow /app/src/backend/langflow
COPY --from=python-builder --chown=1000 /app/src/lfx/src /app/src/lfx/src

ENV PATH="/app/.venv/bin:$PATH"
ENV LANGFLOW_HOST=0.0.0.0
ENV LANGFLOW_PORT=7860

USER langflow
WORKDIR /app/data

EXPOSE 7860

CMD ["python", "-m", "langflow", "run", "--host", "0.0.0.0", "--port", "7860"]
