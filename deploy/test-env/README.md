# Langflow 测试环境单机部署说明

本文档适用于单台测试服务器上的 Langflow 部署，部署思路参考 `new-api` 的“基础设施独立 + 应用多实例重建”模式，并结合当前项目的实际依赖做了调整。

推荐结构：

- 反向代理/Nginx 对外提供 HTTPS 和负载均衡
- Docker Compose 独立运行 PostgreSQL 和 Redis
- Langflow 应用容器单独发布和重建
- 多个 Langflow 实例共享同一个 PostgreSQL、同一个 `LANGFLOW_SECRET_KEY`、同一个应用数据目录
- 默认使用 Langflow 内置 AnyIO 任务后端，不单独启动 Celery worker

和 `new-api` 的主要区别：

- Langflow 主数据库推荐使用 PostgreSQL，不建议改成 MySQL
- 当前版本默认 `LANGFLOW_CELERY_ENABLED=False`，异步任务由 Web 容器内的 AnyIO 后端处理；只有显式启用 Celery 时才需要独立 worker
- 如果 Redis 仅用于 Docker 内部网络通信，可以不对公网暴露

## 目录规划

应用源码或发布目录：

```text
/www/wwwroot/langflow-canvas
```

Docker 基础设施和持久化数据目录：

```text
/www/dk_project/dk_compose/langflow-canvas
├── infra
│   ├── docker-compose.infra.yml
│   └── .env.infra
├── app
│   ├── docker-compose.app.yml
│   └── .env.app
├── postgres
├── redis
└── app-data
```

## 一、先部署 PostgreSQL 和 Redis

基础设施不跟随应用每次发布重启。

```bash
mkdir -p /www/dk_project/dk_compose/langflow-canvas/infra
mkdir -p /www/dk_project/dk_compose/langflow-canvas/postgres
mkdir -p /www/dk_project/dk_compose/langflow-canvas/redis
mkdir -p /www/dk_project/dk_compose/langflow-canvas/app-data
cd /www/dk_project/dk_compose/langflow-canvas/infra
```

将仓库中的 [docker-compose.infra.yml](./docker-compose.infra.yml) 放到该目录，并基于 [`.env.infra.example`](./.env.infra.example) 创建 `.env.infra`。

启动：

```bash
docker compose --env-file .env.infra -f docker-compose.infra.yml up -d
```

检查：

```bash
docker ps
docker exec -it langflow-postgres psql -U langflow -d langflow
docker exec -it langflow-redis redis-cli ping
```

建议：

- 测试环境优先使用 Docker 内部网络，不要把 Redis 暴露到公网
- PostgreSQL 如果需要外部连接，可设置 `POSTGRES_BIND_HOST=0.0.0.0`，并在云安全组和服务器防火墙里只放行固定来源 IP
- Redis 仍建议保持 `REDIS_BIND_HOST=127.0.0.1`，应用容器通过 Docker 网络访问 Redis，不需要对外开放
- 基础设施网络名保持固定，例如 `langflow-test-net`
- PostgreSQL 建议先固定使用 `postgres:15.4`，满足 Langflow 的 PostgreSQL 15+ 要求，并和仓库现有部署模板保持一致

## 二、部署 Langflow 应用和 Worker

在应用目录准备 compose 和环境变量：

```bash
mkdir -p /www/dk_project/dk_compose/langflow-canvas/app
cd /www/dk_project/dk_compose/langflow-canvas/app
```

将仓库中的 [docker-compose.app.yml](./docker-compose.app.yml) 放到该目录，并基于 [`.env.app.example`](./.env.app.example) 创建 `.env.app`。

应用关键配置：

- `APP_IMAGE`：Langflow 镜像地址，支持你们自己的 TCR 镜像
- `WORKER_IMAGE`：仅在 `LANGFLOW_CELERY_ENABLED=True` 时需要；Web 镜像和 Worker 镜像都必须包含 Celery 依赖
- `LANGFLOW_DATABASE_URL` 对应外部 PostgreSQL
- `BROKER_URL` 和 `RESULT_BACKEND` 指向 Redis
- `LANGFLOW_SECRET_KEY` 必须固定，多个实例必须一致
- `LANGFLOW_SUPERUSER` 和 `LANGFLOW_SUPERUSER_PASSWORD` 建议显式设置
- 多个 Langflow 实例共用同一个 `APP_DATA_DIR`
- 应用容器连接 PostgreSQL 时使用 Docker 网络内端口 `5432`；外部客户端连接测试机时使用宿主机映射端口，例如 `35432`

首次启动前，确保应用数据目录允许容器用户写入：

```bash
mkdir -p /www/dk_project/dk_compose/langflow-canvas/app-data
chmod -R 775 /www/dk_project/dk_compose/langflow-canvas/app-data
```

如果启动日志仍出现 `Permission denied: '/app/langflow/secret_key'`，先查容器用户 UID/GID：

```bash
docker compose --env-file .env.app -f docker-compose.app.yml run --rm --entrypoint id langflow-master
```

再按输出的 UID/GID 修正目录属主，例如：

```bash
chown -R 1000:1000 /www/dk_project/dk_compose/langflow-canvas/app-data
```

启动一个主实例和一个备实例：

```bash
docker compose --env-file .env.app -f docker-compose.app.yml up -d
```

默认会启动：

- `langflow-master`
- `langflow-slave1`

访问端口示例：

- `master`: `7860`
- `slave1`: `7861`

如果机器配置偏小，第一阶段可以只启动 `master`：

```bash
docker compose --env-file .env.app -f docker-compose.app.yml up -d langflow-master
```

`langflow-worker` 已放入 `worker` profile，但默认不要启动。当前官方 `langflowai/langflow:latest` 和 `langflowai/langflow-backend:latest` 镜像都没有 `celery` 模块，并且默认 `LANGFLOW_CELERY_ENABLED=False` 时 Web 实例不会向 Celery 投递任务。

只有同时满足下面条件时，才启用 worker：

- `APP_IMAGE` 包含 `celery` 依赖
- `WORKER_IMAGE` 包含 `celery` 依赖
- `.env.app` 设置 `LANGFLOW_CELERY_ENABLED=True`

启用 Celery worker：

```bash
docker compose --env-file .env.app -f docker-compose.app.yml --profile worker up -d
```

## 三、反向代理和负载均衡

如果你使用宝塔 Nginx，可以按下面方式转发：

- `/` 轮询到 `127.0.0.1:7860` 和 `127.0.0.1:7861`
- 健康检查走 `/health`
- PostgreSQL 如果已经绑定 `0.0.0.0`，必须限制来源 IP；Redis 不要直接对公网开放

如果暂时不做负载均衡，也可以先只代理到 `127.0.0.1:7860`。

## 四、发布方式建议

如果你们和 `new-api` 一样走镜像发布，建议沿用相同策略：

- 先在外部构建机 build/push 镜像
- 测试服务器只负责 `docker compose pull` 和 `docker compose up -d`
- `APP_IMAGE` 第一阶段固定一个测试标签，例如 `:test-latest`
- 后续再切换到 `:test-<git-sha>` 方便回滚

发布命令：

```bash
docker compose --env-file .env.app -f docker-compose.app.yml pull
docker compose --env-file .env.app -f docker-compose.app.yml up -d
```

查看状态：

```bash
docker compose --env-file .env.app -f docker-compose.app.yml ps
docker compose --env-file .env.app -f docker-compose.app.yml logs -f langflow-master
docker compose --env-file .env.app -f docker-compose.app.yml logs -f langflow-slave1
```

## 五、验收检查

先检查基础服务：

```bash
docker ps
```

再检查 Langflow：

```bash
curl http://127.0.0.1:7860/health
curl http://127.0.0.1:7861/health
```

返回类似下面内容即可：

```json
{"status":"ok"}
```

还应确认：

- 两个实例都能正常登录
- 新建 Flow 后两个实例都能看到相同数据
- 文件上传、变量、登录状态正常
- Worker 容器没有持续报错

## 六、已知约束

- 当前模板默认不启用 Celery worker；Redis 预留给后续启用 Celery 或其他缓存/队列能力
- 多实例共享 `APP_DATA_DIR` 时，需要确保它是同一台宿主机上的持久目录
- 如果后续转正式环境，建议重新评估 Nginx、TLS、备份、监控、日志归档和访问控制
