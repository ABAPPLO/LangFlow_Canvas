#!/bin/bash
set -e

REGISTRY="ai-capability.tencentcloudcr.com/ai-capability-test/lang-flow-test"
TAG="$(date +%Y%m%d-%H%M%S)"

echo "=== 登录 TCR ==="
docker login ai-capability.tencentcloudcr.com

echo "=== 构建镜像（tag: ${TAG}） ==="
docker build \
  -t ${REGISTRY}:latest \
  -t ${REGISTRY}:${TAG} \
  .

echo "=== 推送镜像 ==="
docker push ${REGISTRY}:latest
docker push ${REGISTRY}:${TAG}

echo "=== 完成 ==="
echo "镜像已推送: ${REGISTRY}:${TAG}"
echo "镜像已推送: ${REGISTRY}:latest"
