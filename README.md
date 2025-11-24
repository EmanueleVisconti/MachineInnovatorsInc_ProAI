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

Le metriche esportate dalla app sono `app_requests_total`, `app_errors_total`, `app_request_latency_seconds` e `data_drift_flag`.
Il DAG Airflow (task drift) invia `data_drift_flag` al Pushgateway, che viene scrappato da Prometheus e visualizzato in Grafana.

## Checklist di test rapida

### 0) Pulizia (opzionale, se vuoi ripartire da zero)
```bash
docker compose down --volumes --remove-orphans
docker system prune -f  # opzionale per pulire immagini non usate
```

### 1) Avvio stack completo
```bash
docker compose up --build
```
Attendi che i log mostrino tutti i servizi in `running` (app, mlflow, airflow, prometheus, pushgateway, grafana).

### 2) Test endpoint FastAPI
- Healthcheck:
```bash
curl -f http://localhost:8000/health
```
- Inferenza (sostituisci il testo a piacere):
```bash
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"text": "I love this product"}'
```

### 3) Metriche e Prometheus
- Endpoint grezzo delle metriche esposte dalla app:
```bash
curl http://localhost:8000/metrics | head
```
- Interfaccia web Prometheus: http://localhost:9090
  1. **Status → Targets**: assicurati che i target `app` (8000) e `pushgateway` (9091) siano in stato `UP`. In alternativa, da terminale:
     ```bash
     curl "http://localhost:9090/api/v1/targets" | jq '.data.activeTargets[] | {job: .labels.job, health: .health, endpoint: .discoveredLabels.__address__}'
     ```
  2. **Graph**: inserisci la query `app_requests_total` e premi **Execute**. Se non vedi dati, manda una richiesta a `/predict` (punto 2) e ripremi **Execute**.
  3. Per verificare la metrica di drift via API REST:
     ```bash
     curl "http://localhost:9090/api/v1/query?query=data_drift_flag"
     ```

### 4) Grafana
- GUI: http://localhost:3000 (user/pass `admin` / `admin`).
- Dashboard preprovisionata: `MLOps – Sentiment App`.
- Dopo aver fatto almeno una richiesta a `/predict`, i pannelli su richieste/latency devono aggiornarsi.
- Il pannello `data_drift_flag` si aggiorna quando il DAG Airflow invia la metrica al Pushgateway (vedi punto 6).

### 5) MLflow
- GUI: http://localhost:5000
- I run vengono creati dal DAG (train/evaluate) e sono salvati in `./mlruns`. Puoi verificare da UI che esperimenti e versioni del modello `Sentiment` siano presenti.

### 6) Airflow (DAG `retrain_sentiment`)
- GUI: http://localhost:8080 (user/pass `admin` / `admin`). Attiva il DAG `retrain_sentiment` e avvia un run manuale.
- Esecuzione rapida da terminale (senza passare dalla UI):
```bash
docker compose exec airflow airflow dags test retrain_sentiment 2025-01-01
```
  - Il task `drift` genera la metrica `data_drift_flag` verso Pushgateway.
  - I task `train` e `evaluate_and_promote` registrano ed eseguono il modello in MLflow.

**Come forzare il ramo di retrain anche senza drift**
- Dalla UI, quando fai "Trigger DAG" aggiungi nel JSON di configurazione: `{ "force_retrain": true }`.
- In alternativa (più persistente), imposta la Variable Airflow `force_retrain=true` da Admin → Variables.
  - Entrambi i metodi fanno sì che il task `branch` scelga sempre `train` → `evaluate_and_promote` invece di fermarsi a `finish`.

**Eseguire un singolo task dalla UI**
- Nella griglia del DAG, clicca sul task (es. `train`) → **Run** → "Run with upstream" se vuoi includere le dipendenze oppure "Run" per eseguire solo quel task con i valori XCom già presenti.
- Per riprovare un task fallito senza rilanciare tutto il DAG, puoi cliccare sul cerchio del task → "Clear" (per ripartire) o "Run" (per schedulare subito).

Tip: se vuoi vedere il valore pubblicato in tempo reale dalla UI Prometheus, dopo aver lanciato il DAG vai su **Graph**, inserisci `data_drift_flag`, premi **Execute** e poi **(Graph)** in alto per visualizzare il punto.

### 7) Verifica metrica di drift
- Dopo aver eseguito il DAG, controlla in Prometheus (UI o con curl):
```bash
curl "http://localhost:9090/api/v1/query?query=data_drift_flag"
```
- In Grafana, il pannello `Data Drift Flag` dovrebbe mostrare il valore appena pubblicato.
