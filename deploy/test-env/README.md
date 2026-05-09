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
├── postgres
├── redis
├── app-data
└── app-logs
```

## 一、先部署 PostgreSQL 和 Redis

基础设施不跟随应用每次发布重启。

```bash
mkdir -p /www/dk_project/dk_compose/langflow-canvas/infra
mkdir -p /www/dk_project/dk_compose/langflow-canvas/postgres
mkdir -p /www/dk_project/dk_compose/langflow-canvas/redis
mkdir -p /www/dk_project/dk_compose/langflow-canvas/app-data
mkdir -p /www/dk_project/dk_compose/langflow-canvas/app-logs
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

## 二、Spug 配置中心

应用运行配置由 Spug 配置中心管理，发布时生成 `/www/wwwroot/langflow-canvas/.env.deploy`。应用镜像提前在外部构建并推送到腾讯云 TCR，目标服务器只负责拉取镜像和重建容器。

建议配置这些 key：

```env
COMPOSE_PROJECT_NAME=langflow-test
APP_IMAGE=ai-capability.tencentcloudcr.com/ai-capability-test/lang-flow-test:test-latest
WORKER_IMAGE=ai-capability.tencentcloudcr.com/ai-capability-test/lang-flow-test:test-latest
LANGFLOW_CELERY_ENABLED=False
DOCKER_NETWORK_NAME=langflow-test-net
TZ=Asia/Shanghai

MASTER_PORT=7862
SLAVE1_PORT=7861

POSTGRES_HOST=langflow-postgres
POSTGRES_PORT=5432
POSTGRES_USER=langflow
POSTGRES_PASSWORD=和.env.infra一致
POSTGRES_DB=langflow_build

REDIS_HOST=langflow-redis
REDIS_PORT=6379

APP_DATA_DIR=/www/dk_project/dk_compose/langflow-canvas/app-data
APP_LOG_DIR=/www/dk_project/dk_compose/langflow-canvas/app-logs

LANGFLOW_AUTO_LOGIN=False
LANGFLOW_SUPERUSER=admin@xiaoti.com
LANGFLOW_SUPERUSER_PASSWORD=LangFlow2024!Secure
LANGFLOW_SECRET_KEY=使用 Fernet.generate_key() 生成
LANGFLOW_LOG_LEVEL=info
LANGFLOW_OPEN_BROWSER=False
LFX_DEV=
LANGFLOW_DEVELOPER_API_ENABLED=
```

应用关键配置：

- `APP_IMAGE`：Langflow 镜像地址，支持你们自己的 TCR 镜像
- `WORKER_IMAGE`：仅在 `LANGFLOW_CELERY_ENABLED=True` 时需要；Web 镜像和 Worker 镜像都必须包含 Celery 依赖
- `LANGFLOW_DATABASE_URL` 对应外部 PostgreSQL
- `BROKER_URL` 和 `RESULT_BACKEND` 指向 Redis
- `LANGFLOW_SECRET_KEY` 必须固定，多个实例必须一致
- `LANGFLOW_SUPERUSER` 和 `LANGFLOW_SUPERUSER_PASSWORD` 建议显式设置
- 多个 Langflow 实例共用同一个 `APP_DATA_DIR`
- `APP_LOG_DIR` 会映射到容器内 `/app/logs`，用于保存 `LANGFLOW_LOG_FILE`
- 应用容器连接 PostgreSQL 时使用 Docker 网络内端口 `5432`；外部客户端连接测试机时使用宿主机映射端口，例如 `35432`
- `LFX_DEV` 和 `LANGFLOW_DEVELOPER_API_ENABLED` 默认留空，仅在临时排查组件动态加载或开发接口问题时使用，不建议长期在测试环境开启

Spug 发布目录建议为：

```text
/www/wwwroot/langflow-canvas
```

该目录下至少需要：

- `docker-compose.app.yml`
- `spug-release.sh`
- `.env.deploy`（由 Spug 发布前脚本生成）

首次启动前，确保应用数据目录允许容器用户写入：

```bash
mkdir -p /www/dk_project/dk_compose/langflow-canvas/app-data
mkdir -p /www/dk_project/dk_compose/langflow-canvas/app-logs
chmod -R 775 /www/dk_project/dk_compose/langflow-canvas/app-data
chmod -R 775 /www/dk_project/dk_compose/langflow-canvas/app-logs
```

如果启动日志仍出现 `Permission denied: '/app/langflow/secret_key'` 或 `Permission denied: '/app/logs/langflow.log'`，先查容器用户 UID/GID：

```bash
docker compose --env-file .env.app -f docker-compose.app.yml run --rm --entrypoint id langflow-master
```

再按输出的 UID/GID 修正目录属主，例如：

```bash
chown -R 1000:1000 /www/dk_project/dk_compose/langflow-canvas/app-data
chown -R 1000:1000 /www/dk_project/dk_compose/langflow-canvas/app-logs
```

`langflow-worker` 已放入 `worker` profile，但默认不要启动。当前 `LANGFLOW_CELERY_ENABLED=False` 时 Web 实例不会向 Celery 投递任务。

## 三、反向代理和负载均衡

如果你使用宝塔 Nginx，可以按下面方式转发：

- `/` 轮询到 `127.0.0.1:7860` 和 `127.0.0.1:7861`
- 健康检查走 `/health`
- PostgreSQL 如果已经绑定 `0.0.0.0`，必须限制来源 IP；Redis 不要直接对公网开放

如果暂时不做负载均衡，也可以先只代理到 `127.0.0.1:7860`。

## 四、发布方式建议

如果你们和 `new-api` 一样走镜像发布，建议沿用相同策略：

- 先在外部构建机 build/push 镜像
- 测试服务器只负责 `docker compose pull` 和滚动重建容器
- `APP_IMAGE` 第一阶段固定一个测试标签，例如 `:test-latest`
- 后续再切换到 `:test-<git-sha>` 方便回滚

当前仓库的 `build-push.sh` 已在 Docker build 阶段自动生成组件索引。

因此：

- 组件索引已改为在 Docker build 阶段自动生成，不依赖开发者本地 Python/uv 环境
- 改动组件定义时，不需要手工再跑一次脚本，只要走 `build-push.sh` 即可
- 不建议依赖 `LFX_DEV=1` 作为长期发布方案

Spug 发布后脚本：

```bash
cd "${SPUG_DST_DIR:-/www/wwwroot/langflow-canvas}"
bash spug-release.sh
```

查看状态：

```bash
cd /www/wwwroot/langflow-canvas
docker compose --env-file .env.deploy -f docker-compose.app.yml ps
docker compose --env-file .env.deploy -f docker-compose.app.yml logs -f langflow-master
docker compose --env-file .env.deploy -f docker-compose.app.yml logs -f langflow-slave1
```

宿主机文件日志：

```bash
tail -f /www/dk_project/dk_compose/langflow-canvas/app-logs/langflow.log
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
