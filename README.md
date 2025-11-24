# MLOps Sentiment – Milestone 1

Service FastAPI che serve `cardiffnlp/twitter-roberta-base-sentiment-latest` con endpoint `/predict`, `/health`, `/metrics`.

## Avvio locale

### app
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn src.serving.app:app --reload --port 8000
```

### docker (solo app)
```bash
docker build -t machineinnovators_inc_proai -f docker/Dockerfile.app .
docker run --rm -p 8000:8000 machineinnovators_inc_proai
```

### docker compose (stack con monitoring)
```bash
docker compose up --build
```
- FastAPI: http://localhost:8000
- Prometheus: http://localhost:9090 (job `app` + `pushgateway`)
- Pushgateway: http://localhost:9091
- Grafana: http://localhost:3000 (admin/admin). Dashboard preconfigurata `MLOps – Sentiment App`.

Le metriche esportate dalla app sono `app_requests_total`, `app_errors_total`, `app_request_latency_seconds` e `data_drift_flag`. Il DAG Airflow (task drift) invia `data_drift_flag` al Pushgateway, che viene scrappato da Prometheus e visualizzato in Grafana.
