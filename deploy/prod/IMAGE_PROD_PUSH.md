# Langflow 正式环境 Spug 发布

## 1. 适用范围

- 正式环境域名：`https://lfp.gbotai.cn`
- 发布分支：`main`
- 镜像仓库：`ai-capability.tencentcloudcr.com/ai-capability-prod/lang-flow-prod`
- 当前标签策略：`prod-latest`
- 适用场景：正式环境首次上线完成后的后续发布

## 2. 当前发布架构

代码提交到发布分支
-> 本地构建 Docker 镜像并推送到 TCR
-> Spug 下发 `.env.app`
-> Spug 在目标服务器执行发布脚本

- 目标服务器不本地构建镜像
- 目标服务器只负责：
  - 拉取镜像
  - 重建容器
  - 健康检查

## 3. 发布前准备

### 3.1 确认代码分支正确

发布前确认当前代码已经在 `main`，并包含本次要发布的提交。

### 3.2 确认 TCR 登录正常

本地执行：

```bash
echo '<tcr-password>' | docker login ai-capability.tencentcloudcr.com --username 'tcr$langflow_local' --password-stdin
```

如需使用临时登录指令，可在 TCR 控制台复制执行。

### 3.3 确认正式环境镜像固定为以下值

Spug 配置中心或发布环境变量中保持：

```bash
APP_IMAGE=ai-capability.tencentcloudcr.com/ai-capability-prod/lang-flow-prod:prod-latest
```

说明：

- 当前阶段固定使用 `prod-latest`
- 每次发布不需要手工修改 `APP_IMAGE`

### 3.4 确认构建参数

发布前在本地 shell 中显式设置：

```bash
export LANGFLOW_AUTO_LOGIN=false
```

说明：

- `build-push.sh` 会读取当前环境变量 `LANGFLOW_AUTO_LOGIN`
- 该参数同时影响后端运行时和前端构建产物
- 该值应与 Spug 下发到服务器的 `.env.app` 中 `LANGFLOW_AUTO_LOGIN` 保持一致

## 4. 标准发布流程

### 4.1 本地构建并推送正式镜像

在项目根目录执行：

```bash
cd /Users/muzi/code/company/go/langflow-canvas
git checkout main
git pull --rebase origin main

export IMAGE_REPO=ai-capability.tencentcloudcr.com/ai-capability-prod/lang-flow-prod
export IMAGE_TAG=prod-latest
export EXTRA_TAG=prod-$(date +%Y%m%d-%H%M%S)
export LANGFLOW_AUTO_LOGIN=false
bash build-push.sh
```

`build-push.sh` 的作用：

- 登录 TCR
- 在 Docker build 阶段自动生成 `component_index.json`
- 构建正式环境镜像
- 按 `IMAGE_REPO` / `IMAGE_TAG` / `EXTRA_TAG` 推送镜像

脚本会推送：

- `ai-capability.tencentcloudcr.com/ai-capability-prod/lang-flow-prod:prod-latest`
- `ai-capability.tencentcloudcr.com/ai-capability-prod/lang-flow-prod:prod-时间戳`

### 4.2 在 Spug 中发布

Spug 发布阶段应完成：

- 下发正式环境 `.env.app`
- 在发布目录执行 `spug-release-prod.sh`
- 按发布顺序先发 `app-02`，验证后再发 `app-01`

Spug 执行命令的目标目录应为：

```bash
/www/wwwroot/langflow-canvas
```

执行命令示例：

```bash
cd /www/wwwroot/langflow-canvas
bash spug-release-prod.sh
```

## 5. 发布后验证

### 5.1 容器状态

在目标服务器执行：

```bash
cd /www/wwwroot/langflow-canvas
docker compose --env-file .env.app -f deploy/prod/docker-compose.app.yml ps
```

### 5.2 健康检查

在两台服务器分别执行：

```bash
curl http://127.0.0.1:7860/health
```

### 5.3 Nginx 入口检查

在 `app-01` 执行：

```bash
curl -H "Host: lfp.gbotai.cn" http://127.0.0.1/health
curl https://lfp.gbotai.cn/health
```

## 6. 常见问题

### 6.1 改了组件字段但页面没变化

如果修改的是以下内容：

- `src/lfx/src/lfx/components/**`
- 组件 `inputs`
- `display_name`
- `info`
- `advanced`
- `field_order`

则必须重新生成组件索引后再构建镜像。当前 `build-push.sh` 已在 Docker build 阶段自动处理这一步，不要跳过脚本直接手工构建旧逻辑镜像。

### 6.2 容器启动时报 `Permission denied: '/app/logs/202605/langflow-2026-05-14.log'`

说明宿主机日志目录对容器用户不可写。执行：

```bash
mkdir -p /www/dk_project/dk_compose/langflow-canvas/app-logs
chown -R 1000:1000 /www/dk_project/dk_compose/langflow-canvas/app-logs
chmod -R 775 /www/dk_project/dk_compose/langflow-canvas/app-logs
```

### 6.3 发布脚本提示 `Missing env file: /www/wwwroot/langflow-canvas/.env.app`

说明 Spug 发布目录下缺少 `.env.app`，或者 Spug 没有按预期下发环境文件。

需要确认：

- Spug 发布目录是否为 `/www/wwwroot/langflow-canvas`
- `.env.app` 是否已经下发到发布目录根路径
- `spug-release-prod.sh` 是否位于发布目录根路径
