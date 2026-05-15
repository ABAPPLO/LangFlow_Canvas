# Langflow 测试环境镜像推送

## 1. 适用范围

- 测试环境域名：`https://lft.gbotai.cn`
- 发布分支：`test`
- 镜像仓库：`ai-capability.tencentcloudcr.com/ai-capability-test/lang-flow-test`
- 当前标签策略：`test-latest`
- 当前发布模式：本地/构建机构建镜像并推送 TCR，Spug 只负责拉镜像并滚动发布

## 2. 当前发布架构

代码提交到发布分支  
-> 本地构建 Docker 镜像并推送到 TCR  
-> Spug 发布

- 目标服务器不本地构建镜像
- 目标服务器只负责：
  - 拉取镜像
  - 重建容器
  - 健康检查

## 3. 发布前准备

### 3.1 确认代码分支正确

发布前确认当前代码已经在 `test`，并包含本次要发布的提交。

### 3.2 确认 TCR 登录正常

本地执行：

```bash
echo "ibAePU4C2oSdDuZac9J7jL0DWHZ9InoW" | docker login ai-capability.tencentcloudcr.com --username 'tcr$langflow_local' --password-stdin
```

如需使用临时登录指令，可在 TCR 控制台复制执行。

### 3.3 确认部署镜像固定为以下值

Spug 配置中心中保持：

```bash
APP_IMAGE=ai-capability.tencentcloudcr.com/ai-capability-test/lang-flow-test:test-latest
```

说明：

- 当前阶段固定使用 `test-latest`
- 每次发布不需要手工修改 `APP_IMAGE`

### 3.4 确认构建参数

发布前在本地 shell 中显式设置：

```bash
export LANGFLOW_AUTO_LOGIN=false
```

说明：

- `build-push.sh` 会读取当前环境变量 `LANGFLOW_AUTO_LOGIN`
- 该参数同时影响后端运行时和前端构建产物
- 该值与 Spug 配置中心中的 `LANGFLOW_AUTO_LOGIN` 必须保持一致

## 4. 标准发布流程

### 4.1 本地构建并推送测试镜像

在项目根目录执行：

```bash
cd /Users/muzi/code/company/go/langflow-canvas
git checkout test
git pull --rebase origin test

export IMAGE_REPO=ai-capability.tencentcloudcr.com/ai-capability-test/lang-flow-test
export IMAGE_TAG=test-latest
export EXTRA_TAG=test-$(date +%Y%m%d-%H%M%S)
export LANGFLOW_AUTO_LOGIN=false
bash build-push.sh
```

`build-push.sh` 的作用：

- 登录 TCR
- 在 Docker build 阶段自动生成 `component_index.json`
- 构建测试环境镜像
- 按 `IMAGE_REPO` / `IMAGE_TAG` / `EXTRA_TAG` 推送镜像

脚本会推送：

- `ai-capability.tencentcloudcr.com/ai-capability-test/lang-flow-test:test-latest`
- `ai-capability.tencentcloudcr.com/ai-capability-test/lang-flow-test:test-时间戳`

推送成功后，再到 Spug 点击发布。

### 4.2 触发 Spug 发布

在Spug-Web中触发发布，Spug会在目标服务器执行（自动）：
```bash
cd /www/wwwroot/langflow-canvas
ENV_FILE=.env.app \
COMPOSE_FILE=deploy/test-env/docker-compose.app.yml \
bash spug-release.sh
```

## 5. 发布后验证

### 5.1 容器状态
如果需要在服务器发布后验证容器状态，需要在服务器执行：

```bash
cd /www/wwwroot/langflow-canvas
docker compose --env-file .env.app -f deploy/test-env/docker-compose.app.yml ps
```

### 5.2 健康检查
如果需要在服务器发布后验证容器健康检查，需要在服务器执行：
```bash
curl http://127.0.0.1:7862/health
curl http://127.0.0.1:7861/health
```

### 5.3 页面访问验证

浏览器访问：

```text
https://lft.gbotai.cn
```

确认页面可以正常打开，登录后无 `/auto_login` 或 `/refresh` 循环。

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

### 6.2 是否需要长期设置 `LFX_DEV=1`

不需要。

- `LFX_DEV=1` 仅用于临时排查组件索引是否导致页面字段未刷新
- 正常测试环境发布应保持默认模式，通过构建阶段自动生成 `component_index.json` 解决问题

### 6.3 容器启动时报 `Permission denied: '/app/logs/202605/langflow-master-2026-05-14.log'`

说明宿主机日志目录对容器用户不可写。执行：

```bash
mkdir -p /www/dk_project/dk_compose/langflow-canvas/app-logs
chown -R 1000:1000 /www/dk_project/dk_compose/langflow-canvas/app-logs
chmod -R 775 /www/dk_project/dk_compose/langflow-canvas/app-logs
```
