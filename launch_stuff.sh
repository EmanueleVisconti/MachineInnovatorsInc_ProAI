#!/usr/bin/env bash
set -euo pipefail

echo
echo "======================================="
echo "  MachineInnovators – Full Launcher"
echo "======================================="
echo

# 0) Validazione compose
if [[ ! -f docker-compose.yml ]]; then
  echo "ERRORE: docker-compose.yml non trovato nella cartella corrente."
  exit 1
fi

# 1) Libera porte (ancora, per sicurezza)
echo "→ Libero porte potenzialmente bloccate…"
for p in 5000 8080 8000 9090 9091; do
  if command -v fuser >/dev/null 2>&1; then
    sudo fuser -k "${p}/tcp" 2>/dev/null || true
  fi
done
echo "✓ Porte liberate"

# 2) Ferma stack se esiste
echo
echo "→ Fermando eventuali container…"
docker compose down -v || true
echo "✓ Containers fermati"

# 3) Build immagini custom (Airflow + app)
echo
echo "→ Build immagini (Airflow + App)…"
docker compose build airflow-init airflow app
echo "✓ Build completata"

# 4) Avvia MLflow
echo
echo "→ Avvio MLflow…"
docker compose up -d mlflow
echo "✓ MLflow avviato"

# 5) Avvia init Airflow (db migrate + utente)
echo
echo "→ Inizializzo Airflow (db + utente admin)…"
docker compose run --rm airflow-init
echo "✓ Airflow init completato"

# 6) Avvia Airflow, app, Prometheus, Pushgateway, Grafana
echo
echo "→ Avvio Airflow + App + Monitoring…"
docker compose up -d airflow app prometheus grafana
echo "✓ Stack avviato"

echo
echo "→ Container attivi:"
docker ps --format "table {{.Names}}\t{{.Ports}}\t{{.Status}}"

echo
echo "Fatto. Airflow dovrebbe essere su porta 8080, MLflow su 5000, app su 8000, Grafana su 3000."
