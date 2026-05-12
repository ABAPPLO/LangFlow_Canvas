#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="${COMPOSE_FILE:-$ROOT_DIR/docker-compose.app.yml}"
DEFAULT_ENV_FILE="$ROOT_DIR/.env.deploy"

if [[ ! -f "$DEFAULT_ENV_FILE" && -f "$ROOT_DIR/.env.app" ]]; then
  DEFAULT_ENV_FILE="$ROOT_DIR/.env.app"
fi

if [[ ! -f "$COMPOSE_FILE" && -f "$ROOT_DIR/deploy/test-env/docker-compose.app.yml" ]]; then
  COMPOSE_FILE="$ROOT_DIR/deploy/test-env/docker-compose.app.yml"
fi

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

MASTER_PORT="$(read_env MASTER_PORT 7862)"
SLAVE1_PORT="$(read_env SLAVE1_PORT 7861)"
APP_DATA_DIR="$(read_env APP_DATA_DIR /www/dk_project/dk_compose/langflow-canvas/app-data)"
APP_LOG_DIR="$(read_env APP_LOG_DIR /www/dk_project/dk_compose/langflow-canvas/app-logs)"
DOCKER_NETWORK_NAME="$(read_env DOCKER_NETWORK_NAME langflow-test-net)"
APP_IMAGE="$(read_env APP_IMAGE)"

if [[ -z "$APP_IMAGE" ]]; then
  echo "Missing APP_IMAGE in $ENV_FILE"
  exit 1
fi

compose() {
  docker compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE" "$@"
}

wait_health() {
  local name="$1"
  local port="$2"
  local i
  local response=""
  local curl_output=""
  local curl_status=0

  for i in $(seq 1 60); do
    curl_status=0
    curl_output="$(curl -fsS --max-time 5 "http://127.0.0.1:${port}/health" 2>&1)" || curl_status=$?
    if [[ "$curl_status" -eq 0 ]] && printf '%s' "$curl_output" | grep -q '"status"[[:space:]]*:[[:space:]]*"ok"'; then
      echo "$name is healthy on port $port"
      return 0
    fi
    response="$curl_output"
    sleep 2
  done

  echo "$name health check failed on port $port"
  if [[ -n "$response" ]]; then
    echo "Last health check output: $response"
  fi
  compose ps
  compose logs --tail 100 "$name" || true
  exit 1
}

if ! docker network inspect "$DOCKER_NETWORK_NAME" >/dev/null 2>&1; then
  echo "Missing Docker network: $DOCKER_NETWORK_NAME"
  echo "Start infra first: docker compose --env-file .env.infra -f docker-compose.infra.yml up -d"
  exit 1
fi

mkdir -p "$APP_DATA_DIR" "$APP_LOG_DIR"

cd "$ROOT_DIR"

echo "Pulling image: $APP_IMAGE"
compose pull langflow-master langflow-slave1

if [[ -z "$(compose ps -q langflow-master 2>/dev/null)" ]]; then
  echo "First deployment: starting master first."
  compose up -d --remove-orphans langflow-master
  wait_health langflow-master "$MASTER_PORT"

  compose up -d --remove-orphans langflow-slave1
  wait_health langflow-slave1 "$SLAVE1_PORT"

  compose ps
  exit 0
fi

if [[ "${MIGRATION_FIRST:-1}" == "1" ]]; then
  echo "MIGRATION_FIRST=1: recreating master before slave."
  compose up -d --no-deps --force-recreate langflow-master
  wait_health langflow-master "$MASTER_PORT"
fi

echo "Recreating slave..."
compose up -d --no-deps --force-recreate langflow-slave1
wait_health langflow-slave1 "$SLAVE1_PORT"

if [[ "${MIGRATION_FIRST:-1}" != "1" ]]; then
  echo "Recreating master..."
  compose up -d --no-deps --force-recreate langflow-master
  wait_health langflow-master "$MASTER_PORT"
fi

compose ps

if [[ "${PRUNE_IMAGES:-0}" == "1" ]]; then
  docker image prune -f
fi
