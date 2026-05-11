# syntax=docker/dockerfile:1
# Multi-stage build for the custom langflow fork

# Stage 1: Build frontend
FROM node:22-alpine AS frontend-builder
WORKDIR /app/frontend

# 国内 npm 镜像加速
RUN npm config set registry https://registry.npmmirror.com

# Frontend auth mode is decided at build time by Vite.
# Default to manual-login mode for this deployment unless explicitly overridden.
ARG LANGFLOW_AUTO_LOGIN=false
ENV LANGFLOW_AUTO_LOGIN=${LANGFLOW_AUTO_LOGIN}

COPY src/frontend/package.json src/frontend/package-lock.json ./
RUN npm ci

COPY src/frontend/ ./
RUN npm run build

# Stage 2: Python dependencies + app
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS python-builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libaio-dev curl ffmpeg \
    && rm -rf /var/lib/apt/lists/*

ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

# 国内 PyPI 镜像加速
ENV UV_INDEX_URL=https://mirrors.cloud.tencent.com/pypi/simple

WORKDIR /app

# 先只复制依赖声明文件 + 空的源码占位（让 uv sync 层可缓存）
COPY pyproject.toml uv.lock README.md ./
COPY src/backend/base/pyproject.toml /app/src/backend/base/pyproject.toml
RUN mkdir -p /app/src/backend/base/langflow && touch /app/src/backend/base/langflow/__init__.py
COPY src/backend/base/README.md /app/src/backend/base/README.md
COPY src/lfx/pyproject.toml /app/src/lfx/pyproject.toml
RUN mkdir -p /app/src/lfx/src/lfx && touch /app/src/lfx/src/lfx/__init__.py
COPY src/lfx/README.md /app/src/lfx/README.md
RUN mkdir -p /app/src/backend/langflow && touch /app/src/backend/langflow/__init__.py

# 安装依赖（只要 pyproject.toml/uv.lock 不变，此层被缓存）
RUN uv sync --frozen --no-dev --no-editable --extra postgresql --package langflow

# 再复制真实源码（此层在依赖安装之后，源码变更只重建此层）
COPY scripts/build_component_index.py /app/scripts/build_component_index.py
COPY src/backend/base/langflow /app/src/backend/base/langflow
COPY src/lfx/src /app/src/lfx/src
COPY src/backend/langflow /app/src/backend/langflow

# 基于最新源码在镜像构建阶段生成组件索引，避免依赖本地开发环境
RUN PYTHONPATH="/app/src/backend/base:/app/src/backend:/app/src/lfx/src" \
    .venv/bin/python /app/scripts/build_component_index.py

# 注入前端构建产物
COPY --from=frontend-builder /app/frontend/build /app/src/backend/base/langflow/frontend

# Stage 3: Runtime
FROM python:3.12-slim-bookworm AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl libpq5 ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -m -u 1000 langflow

# 从 builder 复制 venv 和源码
COPY --from=python-builder --chown=1000 /app/.venv /app/.venv
COPY --from=python-builder --chown=1000 /app/src/backend/base/langflow /app/src/backend/base/langflow
COPY --from=python-builder --chown=1000 /app/src/backend/langflow /app/src/backend/langflow
COPY --from=python-builder --chown=1000 /app/src/lfx/src /app/src/lfx/src

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app/src/backend/base:/app/src/backend:/app/src/lfx/src"
ENV LANGFLOW_HOST=0.0.0.0
ENV LANGFLOW_PORT=7860

USER langflow
WORKDIR /app/data

EXPOSE 7860

CMD ["langflow", "run", "--host", "0.0.0.0", "--port", "7860"]
