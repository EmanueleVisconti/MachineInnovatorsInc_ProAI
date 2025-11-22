#!/usr/bin/env bash
set -euo pipefail

DC="docker compose"
AIRFLOW_PORT="${AIRFLOW_PORT:-8080}"
MLFLOW_PORT="${MLFLOW_PORT:-5000}"
APP_PORT="${APP_PORT:-8000}"
PROM_PORT="${PROM_PORT:-9090}"
PUSHGW_PORT="${PUSHGW_PORT:-9091}"
GRAFANA_PORT="${GRAFANA_PORT:-3000}"

free_port() {
  local p="$1"
  { fuser -k "${p}/tcp"; } >/dev/null 2>&1 || true
}

wait_http() {
  local url="$1" timeout="${2:-120}" start ts
  start=$(date +%s || /bin/date +%s)
  while true; do
    if curl -fsS "$url" >/dev/null 2>&1; then return 0; fi
    ts=$(date +%s || /bin/date +%s)
    if (( ts - start > timeout )); then
      echo "TIMEOUT waiting $url" >&2
      return 1
    fi
    sleep 1
  done
}

case "${1:-up}" in
  up)
    echo "==> validating docker-compose.yml"
    $DC config >/dev/null

    echo "==> freeing ports"
    for p in "$MLFLOW_PORT" "$AIRFLOW_PORT" "$APP_PORT" "$PROM_PORT" "$PUSHGW_PORT" "$GRAFANA_PORT"; do
      free_port "$p"
    done

    echo "==> building images (if needed)"
    $DC build app >/dev/null || true

    echo "==> starting MLflow"
    $DC up -d mlflow
    wait_http "http://localhost:${MLFLOW_PORT}/" 120
    echo "✓ MLflow ready"

    echo "==> initializing Airflow (idempotent)"
    $DC run --rm airflow-init bash -lc '
      mkdir -p /opt/airflow/{logs,dags,plugins,logs/scheduler} &&
      airflow db migrate &&
      airflow users create \
        --username admin --password admin \
        --firstname a --lastname b \
        --role Admin --email admin@example.com || true
    '

    echo "==> starting Airflow"
    $DC up -d airflow

    echo "==> starting app + monitoring stack"
    $DC up -d app pushgateway prometheus grafana

    echo "==> status"
    $DC ps
    echo ""
    echo "Airflow UI:   http://localhost:${AIRFLOW_PORT}"
    echo "MLflow UI:    http://localhost:${MLFLOW_PORT}"
    echo "App (FastAPI):http://localhost:${APP_PORT}/docs"
    echo "Prometheus:   http://localhost:${PROM_PORT}"
    echo "Pushgateway:  http://localhost:${PUSHGW_PORT}"
    echo "Grafana:      http://localhost:${GRAFANA_PORT}"
    ;;

  down)
    $DC down -v
    ;;

  restart-airflow)
    $DC stop airflow || true
    free_port "$AIRFLOW_PORT"
    $DC up -d airflow
    ;;

  logs)
    $DC logs -f "$2"
    ;;

  *)
    echo "Usage: $0 {up|down|restart-airflow|logs <service>}"
    exit 1
    ;;
esac
