# MLOps Sentiment – Milestone 1

Service FastAPI che serve `cardiffnlp/twitter-roberta-base-sentiment-latest` con endpoint `/predict`, `/health`, `/metrics`.

## Notebook Colab (consegna)
- [`notebooks/colab_delivery.ipynb`](notebooks/colab_delivery.ipynb): notebook pronto per Google Colab con i passaggi per clonare il repo, installare le dipendenze, addestrare una versione del modello, valutarla/promuoverla con MLflow file-based e testare l'inferenza.
- Aprilo da GitHub con “Open in Colab” oppure caricalo su Drive; è autoconclusivo e usa i file già presenti nel repository (`data/holdout.csv`, ecc.).

## Documentazione di progetto
- [docs/codebase_overview.md](docs/codebase_overview.md): mappa delle componenti con ruoli, flussi e riferimenti file-per-file.

## Avvio locale

> Nota sui tempi: lo stack richiede ~60–90 secondi per partire completamente
> (Airflow + MLflow + Grafana). La prima inferenza può impiegare qualche
> secondo per il warm-up del modello Hugging Face; le successive sono quasi
> immediate perché la pipeline resta in memoria.

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
- FastAPI: http://localhost:8000 (homepage risponde `{"message": "working!"}` se tutto è ok)
- Prometheus: http://localhost:9090 (job `app` + scrape del Pushgateway)
- Pushgateway: http://localhost:9091 (UI dedicata al buffer delle metriche pushate)
- Grafana: http://localhost:3000 (admin/admin). Dashboard preconfigurata `MLOps – Sentiment App`.

Le metriche esportate dalla app sono `app_requests_total`, `app_errors_total`,
`app_request_latency_seconds` e `data_drift_flag`. Il DAG Airflow (task drift)
invia `data_drift_flag` al Pushgateway, che viene scrappato da Prometheus e
visualizzato in Grafana. Una spiegazione dettagliata dei pannelli e delle
statistiche (p50/p90 della latenza, ecc.) è in
[`docs/metrics_guide.md`](docs/metrics_guide.md).

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
- L'interfaccia celeste su http://localhost:9091 è il Pushgateway (buffer delle metriche pushate dal DAG), non un secondo Prometheus: è normale che mostri una UI diversa e minimale.

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
## Dev/Smoke mode per retraining (demo)
Per risparmiare risorse in fase di sviluppo e demo, è disponibile una modalità "dev/smoke" per il retraining:

- Esecuzione: dalla GUI Airflow, quando fai "Trigger DAG" passa nel JSON: `{ "dev_smoke": true }` per far eseguire una versione rapida del training (script `train_smoke`).
- Configurazione: puoi impostare la Variable Airflow `force_dev_smoke=true` per abilitare permanentemente la scelta di `train_smoke`.
- Comportamento: `train_smoke` addestra un piccolo modello sklearn su una cima del CSV (head N rows) e registra una versione con suffisso `-dev` (es. `Sentiment-dev`) su MLflow, così **non** verrà automaticamente promossa a `Production`.
- Parametri runtime: imposta `SMOKE_N_SAMPLES` env var per cambiare il numero di righe campionate. Default: `1`.

Nota: la modalità dev è pensata per test del flusso e demo; i modelli con suffisso `-dev` non sostituiscono la versione reale del modello di produzione.
  - Il task `drift` genera la metrica `data_drift_flag` verso Pushgateway.
  - I task `train` e `evaluate_and_promote` registrano ed eseguono il modello in MLflow.

**Come forzare il ramo di retrain anche senza drift**
- Dalla UI, quando fai "Trigger DAG" aggiungi nel JSON di configurazione: `{ "force_retrain": true }`.
- In alternativa (più persistente), imposta la Variable Airflow `force_retrain=true` da Admin → Variables.
  - Entrambi i metodi fanno sì che il task `branch` scelga sempre `train` → `evaluate_and_promote` invece di fermarsi a `finish`.

Tip: se vuoi vedere il valore pubblicato in tempo reale dalla UI Prometheus, dopo aver lanciato il DAG vai su **Graph**, inserisci `data_drift_flag`, premi **Execute** e poi **(Graph)** in alto per visualizzare il punto.

### 7) Verifica metrica di drift
- Dopo aver eseguito il DAG, controlla in Prometheus (UI o con curl):
```bash
curl "http://localhost:9090/api/v1/query?query=data_drift_flag"
```
- In Grafana, il pannello `Data Drift Flag` dovrebbe mostrare il valore appena pubblicato.

### 8) Simulare un data drift per innescare il retraining
- È già pronto un batch “estremo” in `data/incoming/drift_example.csv` con testi lunghi/off-topic che dovrebbero far rilevare drift.
- Attiva e triggera il DAG `retrain_sentiment`: il task `ingest` userà quel batch e il ramo `train -> evaluate_and_promote` verrà eseguito.
- Dettagli e alternative (creare un batch personalizzato) in `docs/data_drift_simulation.md`.

### 9) Capire come scegliamo e swappiamo il modello
- Il flusso completo di training, valutazione (macro-F1 su `data/holdout.csv`) e promozione a `Production` è spiegato in `docs/training_and_promotion.md`.
