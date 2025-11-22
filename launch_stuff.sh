#!/usr/bin/env bash
set -euo pipefail

# ========= Config =========
PROJECT_NAME="sentiment"
SERVICES=("mlflow" "airflow" "airflow-init")
COMPOSE_FILE="docker-compose.yml"

MLFLOW_PORT=5000
AIRFLOW_PORT=8080

# ========= Helpers =========
msg()  { printf "\n\033[1;36m[dev]\033[0m %s\n" "$*"; }
err()  { printf "\n\033[1;31m[dev:ERR]\033[0m %s\n" "$*" >&2; }
ok()   { printf "\033[1;32mOK\033[0m\n"; }

exists() { command -v "$1" >/dev/null 2>&1; }

free_port() {
  local port="$1"
  if exists fuser; then
    sudo fuser -k "${port}/tcp" 2>/dev/null || true
  elif exists lsof; then
    lsof -ti tcp:"$port" | xargs -r kill -9 || true
  fi
}

wait_http() {
  local url="$1" name="$2" tries="${3:-60}"
  for i in $(seq 1 "$tries"); do
    if curl -fsS "$url" >/dev/null 2>&1; then
      msg "$name è UP su $url"; return 0
    fi
    sleep 1
  done
  return 1
}

codespace_url() {
  local port="$1"
  if [[ -n "${CODESPACE_NAME:-}" ]]; then
    # GitHub Codespaces URL pubblico per la porta
    echo "https://${CODESPACE_NAME}-${port}.app.github.dev/"
  else
    echo "http://localhost:${port}/"
  fi
}

ensure_dirs() {
  mkdir -p mlruns artifacts data/raw data/incoming
  # seed opzionale della reference se non esiste
  if [[ -f "data/holdout.csv" && ! -f "data/raw/reference.csv" ]]; then
    cp -f data/holdout.csv data/raw/reference.csv
  fi
}

compose() {
  DOCKER_BUILDKIT=1 COMPOSE_PROJECT_NAME="$PROJECT_NAME" docker compose -f "$COMPOSE_FILE" "$@"
}

check_prereqs() {
  exists docker || { err "Docker non trovato."; exit 1; }
  exists docker compose || { err "Docker Compose V2 non trovato."; exit 1; }
  [[ -f "$COMPOSE_FILE" ]] || { err "Manca $COMPOSE_FILE"; exit 1; }
}

print_urls() {
  local mlflow_url airflow_url
  mlflow_url="$(codespace_url "$MLFLOW_PORT")"
  airflow_url="$(codespace_url "$AIRFLOW_PORT")"
  msg "MLflow UI:  $mlflow_url"
  msg "Airflow UI: $airflow_url"
}

# ========= Commands =========
cmd_up() {
  check_prereqs
  ensure_dirs

  msg "Validazione compose…"
  compose config >/dev/null && ok || { err "compose config KO"; exit 1; }

  msg "Pulizia porte (MLflow ${MLFLOW_PORT})…"
  free_port "$MLFLOW_PORT"

  msg "Avvio MLflow…"
  compose up -d mlflow
  wait_http "$(codespace_url "$MLFLOW_PORT")" "MLflow" 60 || { err "MLflow non risponde"; exit 1; }

  msg "Inizializzazione Airflow (db migrate + utente admin)…"
  compose run --rm airflow-init

  msg "Avvio Airflow…"
  compose up -d airflow

  msg "Attesa webserver Airflow…"
  # la home reindirizza con 302, proviamo un endpoint noto
  wait_http "$(codespace_url "$AIRFLOW_PORT")" "Airflow" 90 || { err "Airflow non risponde"; exit 1; }

  print_urls
  msg "Stack avviato ✅"
}

cmd_down() {
  check_prereqs
  msg "Stop e rimozione stack + volumi…"
  compose down -v || true
  ok
}

cmd_restart() {
  cmd_down
  cmd_up
}

cmd_logs() {
  check_prereqs
  msg "Log seguiti (mlflow + airflow)…  (Ctrl+C per uscire)"
  compose logs -f mlflow airflow
}

cmd_status() {
  check_prereqs
  msg "Stato servizi:"
  compose ps
  print_urls
}

cmd_seed() {
  # comodo se vuoi generare un "current" al volo
  ensure_dirs
  if [[ -f data/holdout.csv ]]; then
    ts=$(date +%F-%H%M%S)
    cp -f data/holdout.csv "data/incoming/${ts}.csv"
    msg "Creato batch current: data/incoming/${ts}.csv"
  else
    err "Manca data/holdout.csv per creare un current di esempio."
    exit 1
  fi
}

cmd_help() {
  cat <<EOF

Uso: $(basename "$0") <comando>

Comandi:
  up        - avvia stack (MLflow + Airflow) con controlli e attese
  down      - ferma e rimuove stack e volumi
  restart   - down + up
  logs      - segue i log di mlflow e airflow
  status    - mostra stato servizi e URL
  seed      - copia data/holdout.csv in data/incoming/<timestamp>.csv

Note:
- URL (locali o Codespaces) stampati automaticamente.
- Se Airflow non ha ancora DAG visibile, assicurati che 'airflow/dags' sia montato correttamente.
- Utente Airflow di default: admin / admin

EOF
}

# ========= Entry =========
case "${1:-}" in
  up)       shift; cmd_up "$@";;
  down)     shift; cmd_down "$@";;
  restart)  shift; cmd_restart "$@";;
  logs)     shift; cmd_logs "$@";;
  status)   shift; cmd_status "$@";;
  seed)     shift; cmd_seed "$@";;
  help|-h|--help|"") cmd_help;;
  *) err "Comando sconosciuto: $1"; cmd_help; exit 1;;
esac
