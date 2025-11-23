#!/bin/bash
set -euo pipefail

echo ""
echo "======================================="
echo "      MachineInnovators – Launcher"
echo "======================================="
echo ""

# ------------------------------------------------------------
# 1) Verifica che docker-compose.yml esista
# ------------------------------------------------------------
if [ ! -f docker-compose.yml ]; then
    echo "❌ ERRORE: docker-compose.yml non trovato!"
    exit 1
fi
echo "✓ docker-compose.yml trovato"

# ------------------------------------------------------------
# 2) Libera porte comuni se bloccate
# ------------------------------------------------------------
FREE_PORT() {
    local PORT="$1"
    if lsof -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
        echo "→ libero porta $PORT"
        kill -9 $(lsof -t -i:"$PORT") 2>/dev/null || true
    fi
}

echo ""
echo "→ Libero eventuali porte bloccate…"
FREE_PORT 5000   # MLflow
FREE_PORT 8080   # Airflow
FREE_PORT 8000   # App
FREE_PORT 9090   # Prometheus
FREE_PORT 9091   # Pushgateway

echo "✓ Porte liberate"

# ------------------------------------------------------------
# 3) Ricostruisci l’immagine Airflow se necessario
# ------------------------------------------------------------
echo ""
echo "→ Ricostruisco immagine Docker personalizzata di Airflow…"
docker compose build airflow --no-cache
echo "✓ Immagine ricostruita"

# ------------------------------------------------------------
# 4) Avvia MLflow e aspetta che risponda
# ------------------------------------------------------------
echo ""
echo "→ Avvio MLflow…"
docker compose up -d mlflow

echo "→ Attendo MLflow (max 120s)…"
timeout 120 bash -c '
    until curl -fsS http://localhost:5000/ >/dev/null 2>&1; do
        sleep 2
    done
'

echo "✓ MLflow è pronto"

# ------------------------------------------------------------
# 5) Esegui Airflow Init (idempotente)
# ------------------------------------------------------------
echo ""
echo "→ Inizializzo Airflow…"
docker compose run --rm airflow-init || true
echo "✓ Airflow DB pronto"

# ------------------------------------------------------------
# 6) Avvia Airflow Webserver + Scheduler
# ------------------------------------------------------------
echo ""
echo "→ Avvio Airflow…"
docker compose up -d airflow

echo "→ Attendo Airflow (max 120s)…"
timeout 120 bash -c '
    until curl -fsS http://localhost:8080/health >/dev/null 2>&1; do
        sleep 2
    done
'

echo "✓ Airflow è pronto"

# ------------------------------------------------------------
# 7) Avvia App + PushGW + Prometheus + Grafana
# ------------------------------------------------------------
echo ""
echo "→ Avvio App + Monitoraggio…"
docker compose up -d app pushgateway prometheus grafana

echo ""
echo "✓ Tutti i servizi sono attivi!"
echo "---------------------------------------"
echo "MLflow:       https://$(hostname)-5000.app.github.dev"
echo "Airflow:      https://$(hostname)-8080.app.github.dev"
echo "App Serving:  https://$(hostname)-8000.app.github.dev/docs"
echo "Prometheus:   https://$(hostname)-9090.app.github.dev"
echo "PushGateway:  https://$(hostname)-9091.app.github.dev"
echo "Grafana:      https://$(hostname)-3000.app.github.dev"
echo "---------------------------------------"
