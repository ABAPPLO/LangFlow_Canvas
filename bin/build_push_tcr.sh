#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing command: $1"
    exit 1
  fi
}

require_cmd docker
require_cmd git

IMAGE_REPO="${IMAGE_REPO:-}"
TARGET_PLATFORM="${TARGET_PLATFORM:-linux/amd64}"
SHORT_SHA="$(git -C "$ROOT_DIR" rev-parse --short HEAD)"
IMAGE_TAG="${IMAGE_TAG:-test-latest}"
IMAGE="${IMAGE_REPO}:${IMAGE_TAG}"

if [[ -z "$IMAGE_REPO" ]]; then
  echo "Missing IMAGE_REPO"
  echo "Example:"
  echo "  IMAGE_REPO=ai-capability.tencentcloudcr.com/ai-capability-test/lang-flow-test bin/build_push_tcr.sh"
  exit 1
fi

if ! docker buildx version >/dev/null 2>&1; then
  echo "docker buildx is required"
  exit 1
fi

if ! docker buildx inspect >/dev/null 2>&1; then
  docker buildx create --use >/dev/null
fi

BUILD_ARGS=(
  --platform "$TARGET_PLATFORM"
  -t "$IMAGE"
)

if [[ -n "${EXTRA_TAG:-}" ]]; then
  BUILD_ARGS+=(-t "${IMAGE_REPO}:${EXTRA_TAG}")
fi

echo "Building and pushing image..."
echo "ROOT_DIR=$ROOT_DIR"
echo "IMAGE=$IMAGE"
echo "TARGET_PLATFORM=$TARGET_PLATFORM"
if [[ "$IMAGE_TAG" == "test-latest" ]]; then
  echo "TIP=当前默认覆盖 test-latest，适合先把测试发布链路跑通"
fi

cd "$ROOT_DIR"
docker buildx build "${BUILD_ARGS[@]}" --push .

echo
echo "Push completed."
echo "Use this APP_IMAGE in Spug:"
echo "APP_IMAGE=${IMAGE}"
