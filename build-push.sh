#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

pause_on_exit() {
  local status=$?
  trap - EXIT

  if [[ "${CI:-}" == "true" || "${BUILD_PUSH_NO_PAUSE:-}" == "1" ]]; then
    exit "$status"
  fi

  echo
  if [[ "$status" -eq 0 ]]; then
    echo "=== 脚本执行成功，按 Enter 关闭窗口 ==="
  else
    echo "=== 脚本执行失败，退出码: ${status}，按 Enter 关闭窗口 ==="
  fi
  read -r _ || true
  exit "$status"
}

trap pause_on_exit EXIT

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
BUILD_CACHE="${BUILD_CACHE:-auto}"
CACHE_REF="${CACHE_REF:-${IMAGE_REPO}:buildcache}"
BUILD_RETRIES="${BUILD_RETRIES:-3}"
BUILD_RETRY_DELAY="${BUILD_RETRY_DELAY:-15}"

if [[ -z "$IMAGE_REPO" ]]; then
  echo "Missing IMAGE_REPO"
  exit 1
fi

if [[ -z "$IMAGE_TAG" ]]; then
  echo "Missing IMAGE_TAG"
  exit 1
fi

if [[ ! "$BUILD_RETRIES" =~ ^[1-9][0-9]*$ ]]; then
  echo "Invalid BUILD_RETRIES: $BUILD_RETRIES"
  exit 1
fi

if [[ ! "$BUILD_RETRY_DELAY" =~ ^[0-9]+$ ]]; then
  echo "Invalid BUILD_RETRY_DELAY: $BUILD_RETRY_DELAY"
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

detect_buildx_driver() {
  local driver
  driver="$(docker buildx inspect 2>/dev/null | awk -F': *' '/^Driver:/ {print $2; exit}' || true)"
  echo "$driver"
}

BUILDX_DRIVER="$(detect_buildx_driver)"
ENABLE_REGISTRY_CACHE=0

case "$BUILD_CACHE" in
  0|false|no)
    ENABLE_REGISTRY_CACHE=0
    ;;
  auto)
    if [[ -n "$BUILDX_DRIVER" && "$BUILDX_DRIVER" != "docker" ]]; then
      ENABLE_REGISTRY_CACHE=1
    fi
    ;;
  1|true|yes|registry)
    if [[ -z "$BUILDX_DRIVER" || "$BUILDX_DRIVER" == "docker" ]]; then
      echo "=== 当前 buildx driver=docker，不启用 registry cache ==="
      echo "=== 如需 registry cache，请切换 docker-container driver 或开启 Docker Desktop containerd image store ==="
    else
      ENABLE_REGISTRY_CACHE=1
    fi
    ;;
  force)
    ENABLE_REGISTRY_CACHE=1
    ;;
  *)
    echo "Invalid BUILD_CACHE: $BUILD_CACHE"
    echo "Allowed values: auto, 0, 1, registry, force"
    exit 1
    ;;
esac

if [[ "$ENABLE_REGISTRY_CACHE" == "1" ]]; then
  BUILD_ARGS+=(
    --cache-from "type=registry,ref=${CACHE_REF}"
    --cache-to "type=registry,ref=${CACHE_REF},mode=max"
  )
fi

echo "=== 构建镜像 ==="
echo "ROOT_DIR=$ROOT_DIR"
echo "IMAGE_REPO=$IMAGE_REPO"
echo "IMAGE_TAG=$IMAGE_TAG"
echo "EXTRA_TAG=${EXTRA_TAG:-<none>}"
echo "TARGET_PLATFORM=$TARGET_PLATFORM"
echo "LANGFLOW_AUTO_LOGIN=$LANGFLOW_AUTO_LOGIN_VALUE"
echo "BUILD_CACHE=$BUILD_CACHE"
echo "BUILDX_DRIVER=${BUILDX_DRIVER:-unknown}"
echo "REGISTRY_CACHE=$ENABLE_REGISTRY_CACHE"
echo "BUILD_RETRIES=$BUILD_RETRIES"
echo "BUILD_RETRY_DELAY=$BUILD_RETRY_DELAY"
if [[ "$ENABLE_REGISTRY_CACHE" == "1" ]]; then
  echo "CACHE_REF=$CACHE_REF"
fi

cd "$ROOT_DIR"

run_build_with_retries() {
  local attempt=1
  local status=0

  while true; do
    if docker buildx build "${BUILD_ARGS[@]}" --push .; then
      return 0
    else
      status=$?
    fi

    if (( attempt >= BUILD_RETRIES )); then
      return "$status"
    fi

    echo "=== 构建/推送失败，${BUILD_RETRY_DELAY}s 后重试 ($attempt/$BUILD_RETRIES) ==="
    sleep "$BUILD_RETRY_DELAY"
    attempt=$((attempt + 1))
  done
}

run_build_with_retries

echo "=== 完成 ==="
echo "镜像已推送: ${IMAGE_REPO}:${IMAGE_TAG}"
if [[ -n "$EXTRA_TAG" ]]; then
  echo "镜像已推送: ${IMAGE_REPO}:${EXTRA_TAG}"
fi
