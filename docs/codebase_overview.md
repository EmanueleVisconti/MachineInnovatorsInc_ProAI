# Panoramica della codebase

Questa nota sintetizza cosa fa ogni blocco (serving, training, monitoraggio) e
come interagiscono. Per ogni sezione sono indicati i file chiave e i punti dove
le componenti si agganciano tra loro.

## Architettura di alto livello

```
[Dataset (data/raw, data/incoming)]
      │
      ▼
Airflow DAG `retrain_sentiment`
  ├─ ingest → prepara `raw/current.csv`
  ├─ drift → calcola drift su lunghezza/etichette → push `data_drift_flag` → Pushgateway → Prometheus → Grafana
  └─ branch → (train → evaluate_and_promote → MLflow Registry) oppure finish
                                    │
                                    ▼
                      FastAPI serving (carica Production da MLflow o modello HF)
```

## Stack e deploy locale
- `docker-compose.yml` avvia MLflow, Airflow (init + webserver/scheduler), app FastAPI, Prometheus, Pushgateway e Grafana con i mount di `src/`, `data/` e `artifacts/` condivisi tra DAG e app.【F:docker-compose.yml†L3-L145】
- Prometheus scrappa l'app (porta 8000) e il Pushgateway (9091); Grafana è già provisionata con datasource Prometheus e la dashboard "MLOps – Sentiment App".【F:docker/prometheus.yml†L1-L12】【F:docker/grafana-datasource.yml†L1-L9】【F:docker/grafana-provisioning.yml†L1-L10】

## Servizio di inference FastAPI
- Gli endpoint `/predict`, `/health`, `/` e `/metrics` sono definiti in `src/serving/app.py`. Ogni richiesta incrementa `app_requests_total`, misura la latenza in `app_request_latency_seconds` e, in caso di eccezione, `app_errors_total`. Lo startup inizializza `data_drift_flag` a 0; la homepage (`/`) risponde `{"message": "working!"}` per un check rapido.【F:src/serving/app.py†L20-L85】
- `src/serving/load_model.py` prova prima a caricare il modello `Production` dal Registry MLflow (URI in `MODEL_URI`). Se non è disponibile, usa la pipeline Hugging Face `cardiffnlp/twitter-roberta-base-sentiment-latest`; in mancanza di rete cade su uno stub che restituisce `neutral` per evitare crash. La funzione `predict_fn` normalizza le etichette (`LABEL_0`→`negative`, ecc.).【F:src/serving/load_model.py†L12-L83】

## Pipeline di training, valutazione e registry MLflow
- `src/models/train_roberta.py` allena/logga il wrapper `HFTextClassifier` e lo registra come nuova versione del modello (nome configurabile via env `REGISTERED_MODEL_NAME`, default `Sentiment`).【F:src/models/train_roberta.py†L1-L44】
- `src/models/evaluate.py` confronta il modello candidato con l'eventuale `Production` esistente calcolando la macro-F1 su `eval_csv` (default `data/holdout.csv`); promuove automaticamente la versione migliore o il primo modello disponibile.【F:src/models/evaluate.py†L1-L77】
- Utility comuni per trovare il modello `Production`, promuovere versioni e creare esperimenti sono in `src/utils/mlflow_utils.py`. La serving app usa la stessa URI `MODEL_URI` per recuperare il `Production` al boot.【F:src/utils/mlflow_utils.py†L1-L28】

## Monitoraggio e data drift
- `src/monitoring/drift_report.py` confronta riferimento (`data/raw/reference.csv`) e batch corrente (`data/raw/current.csv`): calcola shift sulla mediana della lunghezza del testo e drift della distribuzione delle etichette (TV distance). Se uno dei due supera soglia, restituisce exit code 1 e scrive `drift_report.json`/`html` con i dettagli.【F:src/monitoring/drift_report.py†L1-L109】
- `src/monitoring/push_metrics.py` pubblica il valore di drift (0/1) sul Pushgateway con job `retrain_sentiment` e instance `airflow`; Prometheus lo scrappa e il gauge è visibile in Grafana.【F:src/monitoring/push_metrics.py†L1-L32】

## DAG Airflow `retrain_sentiment`
- Task sequence in `airflow/dags/retrain_sentiment_dag.py`: `ingest` copia il primo CSV in `data/incoming/` su `raw/current.csv` (altrimenti riusa l'holdout), `drift` chiama `src.monitoring.drift_report` e push della metrica, `branch` sceglie se andare a `train`/`evaluate_and_promote` in base a drift, timer di 7 giorni o flag `force_retrain`, `finish` chiude senza azioni. I task di training/eval loggano in MLflow e possono promuovere una nuova `Production`.【F:airflow/dags/retrain_sentiment_dag.py†L1-L181】

## Dataset e artefatti
- Dataset inclusi: `data/holdout.csv` per la valutazione, `data/raw/reference.csv` come baseline, `data/incoming/drift_example.csv` per simulare drift (usato da `ingest` se presente). I report di drift vengono scritti in `artifacts/` e rimangono disponibili localmente e nei volumi dei container.【F:airflow/dags/retrain_sentiment_dag.py†L10-L66】

## Notebook di consegna e test
- Notebook Colab per demo end-to-end in `notebooks/colab_delivery.ipynb`, linkato dal README.【F:README.md†L5-L7】
- I test unitari in `tests/` coprono preprocess, serving (incluso fallback) e rilevazione drift; esecuzione con `pytest`.【F:tests/test_serving.py†L1-L78】【F:tests/test_drift_report.py†L1-L33】

