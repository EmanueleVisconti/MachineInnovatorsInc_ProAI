# chiudi tutto e libera la 5000 se bloccata
docker compose down -v
sudo fuser -k 5000/tcp 2>/dev/null || true

# 1) avvia solo mlflow
docker compose up -d mlflow
docker compose logs -f mlflow   # interrompi con Ctrl+C quando è ok

# 2) inizializza Airflow (one-off, esce da solo se va a buon fine)
docker compose run --rm airflow-init

# 3) avvia Airflow (webserver + scheduler)
docker compose up -d airflow
docker compose logs -f airflow  # aspetta "Listening at: http://0.0.0.0:8080"
