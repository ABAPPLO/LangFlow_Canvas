# Langflow 正式环境部署指南

本文档适用于 **2 台全新腾讯云服务器** 的正式环境部署：

- `app-01`：Nginx 入口 + Langflow 应用实例
- `app-02`：Langflow 应用实例
- PostgreSQL：腾讯云托管版，内网访问
- 当前阶段：**不启用 Redis / Celery**

目录策略：

- **首次手动部署目录**：`/www/dk_project/dk_compose/langflow-canvas/app`
- **后续 Spug 发布目录**：`/www/wwwroot/langflow-canvas`

镜像说明：

- 当前 Docker 镜像已内置 `ffmpeg` / `ffprobe`
- 如果工作流中的 Python/代码节点直接调用 `ffmpeg`，无需再在服务器宿主机额外安装

当前推荐规格：

- 每台应用机：`8核16G`
- 公网：按流量计费，`20M` 或 `100M` 上限均可
- 系统：`Ubuntu 22.04 LTS`

## 1. 最终架构

```text
Internet
   |
lfp.gbotai.cn
   |
app-01 (Nginx : 80/443)
   |-----------------------> 127.0.0.1:7860   (Langflow on app-01)
   \-----------------------> app-02:7860      (Langflow on app-02, private IP)

PostgreSQL (TencentDB, private IP)
```

## 2. 前置条件

在开始部署前，先准备好：

1. 两台云服务器已开机并可 SSH 登录
2. 两台机器在同一个 VPC / 子网内
3. PostgreSQL 已购买完成，并拿到：
   - `POSTGRES_HOST`
   - `POSTGRES_PORT`
   - `POSTGRES_USER`
   - `POSTGRES_PASSWORD`
   - `POSTGRES_DB`
4. 域名 `lfp.gbotai.cn` 已解析到 `app-01` 公网 IP
5. 已具备腾讯云 TCR 拉取镜像的账号

## 3. 安全组建议

### app-01

- 放行公网：
  - `22`：仅办公 IP
  - `80`
  - `443`
- 放行内网：
  - `7860`：允许本机和 `app-02` 联调时访问即可

### app-02

- 放行公网：
  - `22`：仅办公 IP
- 放行内网：
  - `7860`：只允许 `app-01` 访问

### PostgreSQL

- 不对公网开放
- 只允许 `app-01`、`app-02` 所在安全组或内网 IP 访问 `5432`

## 4. 两台服务器初始化

以下步骤 **app-01 和 app-02 都执行**。

### 4.1 安装基础软件

```bash
apt-get update
apt-get install -y ca-certificates curl gnupg lsb-release git jq netcat-openbsd postgresql-client
```

### 4.2 安装 Docker

```bash
apt-get update
apt-get install -y docker.io docker-compose-v2 || apt-get install -y docker.io docker-compose
systemctl enable docker
systemctl start docker

docker --version
docker compose version || docker-compose --version
```

### 4.3 配置 Docker 镜像加速

在两台机器都执行：

```bash
mkdir -p /etc/docker

cat >/etc/docker/daemon.json <<'EOF'
{
  "registry-mirrors": [
    "https://docker.1ms.run",
    "https://mirror.ccs.tencentyun.com"
  ],
  "exec-opts": ["native.cgroupdriver=systemd"],
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "100m",
    "max-file": "3"
  }
}
EOF

systemctl daemon-reload
systemctl restart docker

docker info | grep -A 5 "Registry Mirrors"
```

预期输出应包含：

```text
Registry Mirrors:
 https://docker.1ms.run/
 https://mirror.ccs.tencentyun.com/
```

### 4.4 创建目录

```bash
mkdir -p /www/wwwroot/langflow-canvas
mkdir -p /www/dk_project/dk_compose/langflow-canvas/app
mkdir -p /www/dk_project/dk_compose/langflow-canvas/app-data
mkdir -p /www/dk_project/dk_compose/langflow-canvas/app-logs

chown -R 1000:1000 /www/dk_project/dk_compose/langflow-canvas/app-data
chown -R 1000:1000 /www/dk_project/dk_compose/langflow-canvas/app-logs
chmod -R 775 /www/dk_project/dk_compose/langflow-canvas/app-data
chmod -R 775 /www/dk_project/dk_compose/langflow-canvas/app-logs
```

### 4.5 手动准备部署目录

当前正式环境服务器 **不依赖 git 拉代码**。

首次部署时，手动把以下文件同步到两台服务器的 `/www/dk_project/dk_compose/langflow-canvas/app`：

- `spug-release-prod.sh`
- `deploy/prod/docker-compose.app.yml`
- `deploy/prod/.env.app.example`

至少先保证目录结构存在：

```bash
mkdir -p /www/dk_project/dk_compose/langflow-canvas/app/deploy/prod
```

如果当前不方便上传文件，也可以在服务器上手动创建：

```bash
cd /www/dk_project/dk_compose/langflow-canvas/app

mkdir -p /www/dk_project/dk_compose/langflow-canvas/app/deploy/prod
touch spug-release-prod.sh
touch deploy/prod/docker-compose.app.yml
touch .env.app
touch deploy/prod/.env.app.example

chmod 755 /www/dk_project/dk_compose/langflow-canvas/app/deploy/prod
chmod 755 spug-release-prod.sh
chmod 644 .env.app
chmod 644 deploy/prod/.env.app.example
chmod 644 deploy/prod/docker-compose.app.yml
```

创建完成后，使用编辑器或 `cat > 文件名` 的方式填入文件内容。例如：

```bash
cd /www/dk_project/dk_compose/langflow-canvas/app
cat > spug-release-prod.sh
```

粘贴脚本内容后按 `Ctrl + D` 保存，然后继续：

```bash
cd /www/dk_project/dk_compose/langflow-canvas/app/deploy/prod
cat > docker-compose.app.yml
```

粘贴 compose 内容后按 `Ctrl + D` 保存，然后回到项目根目录创建应用环境文件：

```bash
cd /www/dk_project/dk_compose/langflow-canvas/app
cat > .env.app
```

粘贴环境变量内容后按 `Ctrl + D` 保存。

### 4.6 登录镜像仓库

```bash
echo '<tcr-password>' | docker login ai-capability.tencentcloudcr.com --username 'tcr$langflow_local' --password-stdin
```

## 5. 验证 PostgreSQL 连通性

在两台服务器上都执行一次：

```bash
nc -vz 10.0.0.13 5432
```

如果能通，再验证账号密码：

```bash
PGPASSWORD='replace-with-prod-password' psql \
  -h 10.0.0.13 \
  -p 5432 \
  -U langflow \
  -d langflow_prod \
  -c '\conninfo'
```

## 6. 正式环境配置文件

当前正式环境使用：

- [docker-compose.app.yml](./docker-compose.app.yml)
- [`.env.app.example`](./.env.app.example)
- [IMAGE_PROD_PUSH.md](./IMAGE_PROD_PUSH.md)
- [spug-release-prod.sh](/Users/muzi/code/company/go/langflow-canvas/spug-release-prod.sh:1)

### 6.1 app-01 的 `.env.app`

在 `app-01` 创建：

```bash
cd /www/dk_project/dk_compose/langflow-canvas/app
cp deploy/prod/.env.app.example .env.app
```

编辑为：

```env
COMPOSE_PROJECT_NAME=langflow-prod
APP_IMAGE=ai-capability.tencentcloudcr.com/ai-capability-prod/lang-flow-prod:prod-latest
LANGFLOW_CELERY_ENABLED=False
DOCKER_NETWORK_NAME=langflow-prod-net
TZ=Asia/Shanghai

APP_PORT=7860

POSTGRES_HOST=10.0.0.13
POSTGRES_PORT=5432
POSTGRES_USER=langflow
POSTGRES_PASSWORD=replace-with-prod-password
POSTGRES_DB=langflow_prod

APP_DATA_DIR=/www/dk_project/dk_compose/langflow-canvas/app-data
APP_LOG_DIR=/www/dk_project/dk_compose/langflow-canvas/app-logs

LANGFLOW_AUTO_LOGIN=False
LANGFLOW_SUPERUSER=admin@gbotai.cn
LANGFLOW_SUPERUSER_PASSWORD=replace-with-prod-password
LANGFLOW_SECRET_KEY=replace-with-fixed-fernet-key
LANGFLOW_LOG_LEVEL=info
LANGFLOW_OPEN_BROWSER=False

LFX_DEV=false
LANGFLOW_DEVELOPER_API_ENABLED=false
```

### 6.2 app-02 的 `.env.app`

`app-02` 也创建同样的 `.env.app`，内容与 `app-01` 保持一致。

要求：

- `LANGFLOW_SECRET_KEY` 两台必须一致
- 数据库连接信息两台必须一致
- 当前阶段不填写 `BROKER_URL`、`RESULT_BACKEND`

## 7. 创建 Docker 网络

两台机器都执行：

```bash
docker network create langflow-prod-net || true
```

## 8. 首次发布应用

### 8.1 本地构建并推送镜像

在你的开发机执行：

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

说明：

- `build-push.sh` 已改为通用脚本
- 正式环境通过 `IMAGE_REPO` / `IMAGE_TAG` / `EXTRA_TAG` 控制推送目标
- 首次正式上线前，请确认你已经按正式环境仓库完成镜像构建和推送
- 如果近期新增了依赖 `ffmpeg` 的工作流代码，必须重新构建并推送新镜像

### 8.2 在 app-01 发布

```bash
cd /www/dk_project/dk_compose/langflow-canvas/app
bash spug-release-prod.sh
```

### 8.3 在 app-02 发布

```bash
cd /www/dk_project/dk_compose/langflow-canvas/app
bash spug-release-prod.sh
```

### 8.4 验证应用端口

在两台机器分别执行：

```bash
curl http://127.0.0.1:7860/health
```

成功应返回：

```json
{"status":"ok"}
```

## 9. 配置 Nginx 负载均衡

只在 `app-01` 执行。

### 9.1 安装 Nginx

```bash
apt-get install -y nginx
systemctl enable nginx
systemctl start nginx
```

### 9.2 写入配置

创建 `/etc/nginx/conf.d/langflow.conf`：

```nginx
upstream langflow_backend {
    server 127.0.0.1:7860 max_fails=3 fail_timeout=30s;
    server 10.0.0.16:7860 max_fails=3 fail_timeout=30s;
    keepalive 32;
}

server {
    listen 80;
    server_name lfp.gbotai.cn;

    client_max_body_size 100m;

    access_log /var/log/nginx/langflow.access.log;
    error_log /var/log/nginx/langflow.error.log warn;

    location /health {
        proxy_pass http://langflow_backend/health;
        proxy_http_version 1.1;
        proxy_set_header Connection "";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        access_log off;
    }

    location / {
        proxy_pass http://langflow_backend;
        proxy_http_version 1.1;
        proxy_set_header Connection "";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_connect_timeout 60s;
        proxy_send_timeout 600s;
        proxy_read_timeout 600s;
        send_timeout 600s;
    }
}
```

把上面示例中的 `app-02` 内网 IP 改成你的实际值。

### 9.3 重载 Nginx

```bash
nginx -t
systemctl reload nginx
```

本机先验证一次 `Host` 路由是否正确：

```bash
curl -H "Host: lfp.gbotai.cn" http://127.0.0.1/health
```

### 9.4 使用 certbot 申请 HTTPS 证书

确认：

- `lfp.gbotai.cn` 已解析到 `app-01` 公网 IP
- 安全组已放行 `80`、`443`

安装 certbot：

```bash
apt-get update
apt-get install -y certbot python3-certbot-nginx
```

申请证书：

```bash
certbot --nginx -d lfp.gbotai.cn
```

Certbot 在不同环境下的交互提示可能不同：

- 有些情况下会询问是否启用 `HTTP -> HTTPS` 自动跳转
- 有些情况下不会出现该选项，但仍会自动生成 `443 ssl` 与 `301` 跳转配置

执行完成后，以最终 Nginx 配置为准。可使用以下命令确认：

```bash
nginx -T | grep -n "return 301\\|listen 443\\|ssl_certificate\\|server_name lfp.gbotai.cn"
```

### 9.5 验证证书和自动续期

```bash
nginx -t
systemctl reload nginx
systemctl status certbot.timer
certbot renew --dry-run
```

## 10. 验证正式环境

### 10.1 访问域名

浏览器访问：

```text
https://lfp.gbotai.cn
```

### 10.2 验证 Nginx 到后端

在 `app-01` 执行：

```bash
curl -I https://lfp.gbotai.cn
```

### 10.3 查看应用状态

两台机器分别执行：

```bash
cd /www/dk_project/dk_compose/langflow-canvas/app
docker compose --env-file .env.app -f deploy/prod/docker-compose.app.yml ps
docker compose --env-file .env.app -f deploy/prod/docker-compose.app.yml logs --tail=100 langflow-app
```

## 11. 日常发布

### 11.1 开发机推镜像

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

### 11.2 通过 Spug 发布

正式环境后续发布建议通过 Spug 完成，不再依赖服务器 `git pull`。

推荐方式：

- 发布前：Spug 生成 `.env.app`
- 发布后：Spug 执行 `spug-release-prod.sh`
- Spug 发布目录使用：`/www/wwwroot/langflow-canvas`

### 11.3 手工滚动发布（临时）

先发 `app-02`：

```bash
cd /www/dk_project/dk_compose/langflow-canvas/app
bash spug-release-prod.sh
```

确认 `app-02` 正常后，再发 `app-01`：

```bash
cd /www/dk_project/dk_compose/langflow-canvas/app
bash spug-release-prod.sh
```

这样能减少入口机更新时的影响。

## 12. 配置原则

- `LFX_DEV=false`
- `LANGFLOW_DEVELOPER_API_ENABLED=false`
- `LANGFLOW_SECRET_KEY` 两台必须一致
- PostgreSQL 仅通过内网访问
- 当前阶段不启用 Redis / Celery
- 不对公网开放 `5432`、`7860`

## 13. Redis / Celery 预留

当前正式环境默认：

- `LANGFLOW_CELERY_ENABLED=False`
- 不启动 worker
- 不强依赖 Redis

如果后续启用 Celery：

1. 购买腾讯云 Redis（内网访问）
2. 在 `.env.app` 中补充：

```env
LANGFLOW_CELERY_ENABLED=True
BROKER_URL=redis://redis-private-host:6379/0
RESULT_BACKEND=redis://redis-private-host:6379/0
```

3. 再补 worker 服务和正式环境发布脚本逻辑
