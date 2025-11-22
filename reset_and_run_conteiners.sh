# 0) Mostra chi usa la 5000 (solo per visibilità)
sudo lsof -i :5000 -sTCP:LISTEN -nP || true
docker ps --format "table {{.ID}}\t{{.Names}}\t{{.Ports}}" | (sed -n '1p;/:5000->/p')

# 1) Chiudi tutto lo stack compose e rimuovi le risorse collegate
docker compose down -v

# 2) Uccidi qualunque processo/servizio che tiene la 5000 occupata
sudo fuser -k 5000/tcp 2>/dev/null || true

# 3) (Extra safety) ferma eventuali container sciolti che mappano 5000
for id in $(docker ps -q --filter publish=5000); do docker stop "$id"; done

# 4) Verifica che la 5000 sia libera
sudo lsof -i :5000 -sTCP:LISTEN -nP || echo "OK: 5000 libera"

# 5) Avvia SOLO mlflow, attendi “ready”
docker compose up -d mlflow
docker compose logs -f mlflow

# 6) Verifica che il servizio MLflow stia rispondendo
docker compose up -d airflow
docker compose logs -f airflow