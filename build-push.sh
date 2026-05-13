#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing command: $1"
    exit 1
  fi
}

require_cmd docker

IMAGE_REPO="${IMAGE_REPO:-ai-capability.tencentcloudcr.com/ai-capability-test/lang-flow-test}"
IMAGE_TAG="${IMAGE_TAG:-test-latest}"
TARGET_PLATFORM="${TARGET_PLATFORM:-linux/amd64}"
LANGFLOW_AUTO_LOGIN_VALUE="${LANGFLOW_AUTO_LOGIN:-false}"
DEFAULT_EXTRA_TAG="test-$(date +%Y%m%d-%H%M%S)"
EXTRA_TAG="${EXTRA_TAG:-$DEFAULT_EXTRA_TAG}"

if [[ -z "$IMAGE_REPO" ]]; then
  echo "Missing IMAGE_REPO"
  exit 1
fi

if [[ -z "$IMAGE_TAG" ]]; then
  echo "Missing IMAGE_TAG"
  exit 1
fi

echo "=== 登录 TCR ==="
docker login ai-capability.tencentcloudcr.com

BUILD_ARGS=(
  --platform "$TARGET_PLATFORM"
  --build-arg "LANGFLOW_AUTO_LOGIN=${LANGFLOW_AUTO_LOGIN_VALUE}"
  -t "${IMAGE_REPO}:${IMAGE_TAG}"
)

if [[ -n "$EXTRA_TAG" ]]; then
  BUILD_ARGS+=(-t "${IMAGE_REPO}:${EXTRA_TAG}")
fi

echo "=== 构建镜像 ==="
echo "ROOT_DIR=$ROOT_DIR"
echo "IMAGE_REPO=$IMAGE_REPO"
echo "IMAGE_TAG=$IMAGE_TAG"
echo "EXTRA_TAG=${EXTRA_TAG:-<none>}"
echo "TARGET_PLATFORM=$TARGET_PLATFORM"
echo "LANGFLOW_AUTO_LOGIN=$LANGFLOW_AUTO_LOGIN_VALUE"

cd "$ROOT_DIR"
docker buildx build "${BUILD_ARGS[@]}" --push .

echo "=== 完成 ==="
echo "镜像已推送: ${IMAGE_REPO}:${IMAGE_TAG}"
if [[ -n "$EXTRA_TAG" ]]; then
  echo "镜像已推送: ${IMAGE_REPO}:${EXTRA_TAG}"
fi
