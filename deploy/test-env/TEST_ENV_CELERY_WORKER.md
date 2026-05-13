# Langflow 测试环境 Celery Worker 方案

## 1. 适用场景

该方案适用于以下场景：

- 测试环境必须保持两个 Langflow Web 实例
- 工作流通过 `job_id` 异步执行并由客户端轮询结果
- 当前 `LANGFLOW_CELERY_ENABLED=False` 时已经出现 `JOB_FAILED`
- 文件日志中出现 `Job <job_id> was cancelled by system`

这类现象通常不是 Nginx 连通性问题，而是多实例下缺少共享任务执行层，导致异步任务生命周期不稳定。

## 2. 目标架构

切换后架构应为：

- `langflow-master`：Web/API 实例
- `langflow-slave1`：Web/API 实例
- `langflow-worker`：Celery worker
- `redis`：Broker + Result Backend
- `postgres`：主数据库

即：

代码请求
-> Nginx / 负载均衡
-> 两个 Langflow Web 实例
-> Redis 队列
-> Celery worker 执行任务
-> PostgreSQL / Redis 返回状态

## 3. 当前仓库现状

当前仓库已经具备一部分基础：

- [docker-compose.app.yml](./docker-compose.app.yml) 已定义 `langflow-worker`
- [docker-compose.infra.yml](./docker-compose.infra.yml) 已定义 `redis`
- [`.env.app.example`](./.env.app.example) 已预留 `WORKER_IMAGE`、`REDIS_HOST`、`REDIS_PORT`

但还存在两个关键现实：

1. 当前默认模板仍是 `LANGFLOW_CELERY_ENABLED=False`
2. 当前 [spug-release.sh](/Users/muzi/code/company/go/langflow-canvas/spug-release.sh:1) 默认只拉起 `langflow-master` 和 `langflow-slave1`，不会自动拉起 `langflow-worker`

所以要切换到 Celery 架构，不能只改一个环境变量。

## 4. 环境变量调整

测试环境 `.env.app` 至少调整为：

```env
COMPOSE_PROJECT_NAME=langflow-test
APP_IMAGE=ai-capability.tencentcloudcr.com/ai-capability-test/lang-flow-test:test-latest
WORKER_IMAGE=ai-capability.tencentcloudcr.com/ai-capability-test/lang-flow-test:test-latest
LANGFLOW_CELERY_ENABLED=True
DOCKER_NETWORK_NAME=langflow-test-net
TZ=Asia/Shanghai

MASTER_PORT=7862
SLAVE1_PORT=7861

POSTGRES_HOST=langflow-postgres
POSTGRES_PORT=5432
POSTGRES_USER=langflow
POSTGRES_PASSWORD=replace-with-postgres-password
POSTGRES_DB=langflow_build

REDIS_HOST=langflow-redis
REDIS_PORT=6379

APP_DATA_DIR=/www/dk_project/dk_compose/langflow-canvas/app-data
APP_LOG_DIR=/www/dk_project/dk_compose/langflow-canvas/app-logs

LANGFLOW_AUTO_LOGIN=False
LANGFLOW_SUPERUSER=replace-with-admin@example.com
LANGFLOW_SUPERUSER_PASSWORD=replace-with-admin-password
LANGFLOW_SECRET_KEY=replace-with-fernet-key
LANGFLOW_LOG_LEVEL=info
LANGFLOW_OPEN_BROWSER=False
LFX_DEV=
LANGFLOW_DEVELOPER_API_ENABLED=
```

说明：

- `LANGFLOW_CELERY_ENABLED=True` 是必须项
- `WORKER_IMAGE` 不能留空
- Web 镜像和 Worker 镜像都必须包含 Celery 依赖
- `LANGFLOW_SECRET_KEY` 仍需固定，两个 Web 实例与 worker 保持一致

## 5. Redis 要先可用

先确认基础设施中的 Redis 已正常启动：

```bash
cd /www/dk_project/dk_compose/langflow-canvas/infra
docker compose --env-file .env.infra -f docker-compose.infra.yml up -d
docker exec -it langflow-redis redis-cli ping
```

预期返回：

```text
PONG
```

## 6. 首次切换步骤

建议按这个顺序做：

1. 启动或确认 Redis 正常
2. 更新测试环境 `.env.app`
3. 先启动 `langflow-worker`
4. 再滚动发布 `langflow-slave1`
5. 最后发布 `langflow-master`

### 6.1 手工拉起 worker

在当前脚本尚未支持 worker 自动发布前，先手工执行：

```bash
cd /www/wwwroot/langflow-canvas
docker compose --profile worker --env-file .env.app -f deploy/test-env/docker-compose.app.yml up -d langflow-worker
```

查看状态：

```bash
cd /www/wwwroot/langflow-canvas
docker compose --profile worker --env-file .env.app -f deploy/test-env/docker-compose.app.yml ps
```

### 6.2 再发布两个 Web 实例

如果仍使用当前的 `spug-release.sh`，它只会重建：

- `langflow-master`
- `langflow-slave1`

因此可继续执行：

```bash
cd /www/wwwroot/langflow-canvas
bash spug-release.sh
```

但要注意：这并不会自动管理 `langflow-worker`。

## 7. 对 Spug 的建议

长期来看，Spug 发布脚本应支持：

- 当 `LANGFLOW_CELERY_ENABLED=True` 时自动 `pull langflow-worker`
- 当 `LANGFLOW_CELERY_ENABLED=True` 时自动 `up -d langflow-worker`
- 发布顺序优先保证 worker 已就绪，再发布 Web 实例

也就是说，后续建议把 [spug-release.sh](/Users/muzi/code/company/go/langflow-canvas/spug-release.sh:1) 改成：

- 先判断 `LANGFLOW_CELERY_ENABLED`
- 如果为 `True`，带上 `--profile worker`
- 自动管理 `langflow-worker`

## 8. 发布后验证

### 8.1 容器状态

```bash
cd /www/wwwroot/langflow-canvas
docker compose --profile worker --env-file .env.app -f deploy/test-env/docker-compose.app.yml ps
```

预期至少看到：

- `langflow-master`
- `langflow-slave1`
- `langflow-worker`

### 8.2 Web 健康检查

```bash
curl http://127.0.0.1:7862/health
curl http://127.0.0.1:7861/health
```

### 8.3 Worker 日志

```bash
cd /www/wwwroot/langflow-canvas
docker compose --profile worker --env-file .env.app -f deploy/test-env/docker-compose.app.yml logs --tail=200 langflow-worker
```

### 8.4 文件日志检查

```bash
tail -n 200 /www/dk_project/dk_compose/langflow-canvas/app-logs/langflow.log
```

切换成功后，之前那类：

```text
Job <job_id> was cancelled by system
```

如果根因确实是多实例无共享任务层，理论上应明显减少或消失。

## 9. 常见误区

### 9.1 只改 `LANGFLOW_CELERY_ENABLED=True`

不够。

如果只改这个变量，但没有：

- Redis
- worker
- 发布脚本支持 worker

那么测试环境仍然不完整。

### 9.2 只保留双 Web，不启 worker

不建议。

对于基于 `job_id` 的异步执行，多实例 Web 没有共享执行层时，容易出现任务状态不一致。

### 9.3 继续只看 `docker compose logs langflow-master`

不够。

排查时应同时看：

- `langflow-master`
- `langflow-slave1`
- `langflow-worker`
- `/www/dk_project/dk_compose/langflow-canvas/app-logs/langflow.log`

## 10. 最小结论

如果测试环境必须双实例，那么推荐基线应为：

- 双 Web 实例
- Redis
- Celery worker
- `LANGFLOW_CELERY_ENABLED=True`

否则当前这种多实例 + AnyIO 本地任务后端的组合，异步工作流执行存在稳定性风险。
