#!/bin/bash
set -e

REGISTRY="ai-capability.tencentcloudcr.com/ai-capability-test/lang-flow-test"
TAG="test-$(date +%Y%m%d-%H%M%S)"
LANGFLOW_AUTO_LOGIN_VALUE="${LANGFLOW_AUTO_LOGIN:-false}"

echo "=== 登录 TCR ==="
docker login ai-capability.tencentcloudcr.com

echo "=== 构建镜像（tag: ${TAG}） ==="
docker buildx build \
  --platform linux/amd64 \
  --build-arg LANGFLOW_AUTO_LOGIN=${LANGFLOW_AUTO_LOGIN_VALUE} \
  -t ${REGISTRY}:test-latest \
  -t ${REGISTRY}:${TAG} \
  --push \
  .

echo "=== 完成 ==="
echo "镜像已推送: ${REGISTRY}:${TAG}"
echo "镜像已推送: ${REGISTRY}:test-latest"
