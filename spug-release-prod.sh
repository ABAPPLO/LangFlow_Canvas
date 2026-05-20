#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="${COMPOSE_FILE:-$ROOT_DIR/deploy/prod/docker-compose.app.yml}"
DEFAULT_ENV_FILE="$ROOT_DIR/.env.app"
ENV_FILE="${ENV_FILE:-$DEFAULT_ENV_FILE}"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Missing env file: $ENV_FILE"
  exit 1
fi

if [[ ! -f "$COMPOSE_FILE" ]]; then
  echo "Missing compose file: $COMPOSE_FILE"
  exit 1
fi

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing command: $1"
    exit 1
  fi
}

read_env() {
  local key="$1"
  local default_value="${2:-}"
  local value
  value="$(grep -E "^[[:space:]]*${key}[[:space:]]*=" "$ENV_FILE" | tail -n 1 | sed -E "s/^[[:space:]]*${key}[[:space:]]*=[[:space:]]*//" || true)"
  value="${value%\"}"
  value="${value#\"}"
  value="${value%\'}"
  value="${value#\'}"
  if [[ -z "$value" ]]; then
    printf '%s' "$default_value"
  else
    printf '%s' "$value"
  fi
}

require_cmd docker
require_cmd curl
require_cmd grep

APP_PORT="$(read_env APP_PORT 7860)"
APP_DATA_DIR="$(read_env APP_DATA_DIR /www/dk_project/dk_compose/langflow-canvas/app-data)"
APP_LOG_DIR="$(read_env APP_LOG_DIR /www/dk_project/dk_compose/langflow-canvas/app-logs)"
DOCKER_NETWORK_NAME="$(read_env DOCKER_NETWORK_NAME langflow-prod-net)"
APP_IMAGE="$(read_env APP_IMAGE)"

if [[ -z "$APP_IMAGE" ]]; then
  echo "Missing APP_IMAGE in $ENV_FILE"
  exit 1
fi

compose() {
  docker compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE" "$@"
}

wait_health() {
  local port="$1"
  local i

  for i in $(seq 1 120); do
    if curl -fsS --max-time 5 "http://127.0.0.1:${port}/health" | grep -q '"status"[[:space:]]*:[[:space:]]*"ok"'; then
      echo "langflow-app is healthy on port $port"
      return 0
    fi
    sleep 2
  done

  echo "langflow-app health check failed on port $port"
  compose ps
  compose logs --tail=200 langflow-app || true
  exit 1
}

if ! docker network inspect "$DOCKER_NETWORK_NAME" >/dev/null 2>&1; then
  echo "Missing Docker network: $DOCKER_NETWORK_NAME"
  exit 1
fi

mkdir -p "$APP_DATA_DIR" "$APP_LOG_DIR"

cd "$ROOT_DIR"

echo "Pulling image: $APP_IMAGE"
compose pull langflow-app

echo "Recreating langflow-app..."
compose up -d --no-deps --force-recreate langflow-app
wait_health "$APP_PORT"

compose ps

if [[ "${PRUNE_IMAGES:-0}" == "1" ]]; then
  docker image prune -f
fi
