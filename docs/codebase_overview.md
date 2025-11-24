# Panoramica della codebase

Questa nota orienta chi legge sul ruolo di ogni cartella/file chiave, come sono collegati i componenti (serving, training, monitoring) e dove osservare i punti di integrazione.

## Architettura di alto livello

```
[Dataset (data/raw, data/incoming)]
      │
      ▼
Airflow DAG `retrain_sentiment`
  ├─ ingest → copia batch corrente
  ├─ drift (Evidently) → push `data_drift_flag` → Pushgateway → Prometheus → Grafana
  └─ branch → (train → evaluate_and_promote → MLflow Registry) oppure finish
                                    │
                                    ▼
                         FastAPI serving (carica Production da MLflow o modello HF)
```

## Stack e deploy locale
- `docker-compose.yml` avvia MLflow, Airflow (init + webserver/scheduler), app FastAPI, Prometheus, Pushgateway e Grafana con i mount di `src/`, `data/` e `artifacts/` già pronti per il DAG e la serving app.【F:docker-compose.yml†L3-L145】
- Prometheus scrappa l'app (porta 8000) e il Pushgateway (9091); la dashboard Grafana è provisionata e punta al datasource Prometheus.【F:docker/prometheus.yml†L1-L12】【F:docker/grafana-datasource.yml†L1-L9】【F:docker/grafana-provisioning.yml†L1-L10】

## Servizio di inference FastAPI
- Endpoint `/predict`, `/health`, `/metrics` definiti in `src/serving/app.py`; esporta le metriche `app_requests_total`, `app_errors_total`, `app_request_latency_seconds` e il gauge `data_drift_flag` inizializzato a 0 allo startup.【F:src/serving/app.py†L20-L78】
- `src/serving/load_model.py` carica il modello Production da MLflow se `MODEL_URI` è valorizzato, altrimenti usa la pipeline Hugging Face `cardiffnlp/twitter-roberta-base-sentiment-latest`; include fallback stub per evitare crash in assenza di download HF.【F:src/serving/load_model.py†L12-L83】

## Pipeline di training, valutazione e registry MLflow
- `src/models/train_roberta.py` logga e registra nel Registry MLflow un wrapper `HFTextClassifier` che incapsula tokenizer, modello e pipeline di Hugging Face, annotando il modello base usato.【F:src/models/train_roberta.py†L1-L44】
- `src/models/evaluate.py` carica il nuovo modello e, se presente, quello Production; calcola la macro-F1 su `eval_csv` (es. `data/holdout.csv`) e promuove a `Production` se migliora o se non esiste un modello precedente.【F:src/models/evaluate.py†L1-L77】
- Utility condivise per Registry e tracking URI in `src/utils/mlflow_utils.py` (creazione esperimenti, promozione di versione, lookup del modello Production).【F:src/utils/mlflow_utils.py†L1-L28】

## Monitoraggio e data drift
- `src/monitoring/drift_report.py` confronta dataset di riferimento e corrente calcolando drift su lunghezza del testo e predizione del modello; salva report HTML/JSON e ritorna exit code 1 se drift rilevato.【F:src/monitoring/drift_report.py†L1-L89】
- `src/monitoring/push_metrics.py` pubblica `data_drift_flag` sul Pushgateway (job `retrain_sentiment`, instance `airflow`), da cui Prometheus e Grafana leggono la metrica.【F:src/monitoring/push_metrics.py†L1-L32】

## DAG Airflow `retrain_sentiment`
- Task `ingest` copia il primo batch in `data/incoming/` (altrimenti riusa l'holdout) come `raw/current.csv`; `drift` lancia Evidently e push della metrica; `branch` forza il ramo di retrain se drift==1, se è passato >=7 giorni dall'ultima esecuzione del branch, o se `force_retrain` è impostato via conf/Variable; `train` logga una nuova versione nel Registry; `evaluate_and_promote` valuta su `holdout.csv` e può promuovere a Production; `finish` è un no-op.【F:airflow/dags/retrain_sentiment_dag.py†L1-L181】

## Dataset e artefatti
- Dataset di supporto inclusi: `data/holdout.csv` per la valutazione, batch demo in `data/raw/reference.csv` e `data/raw/current.csv`, e batch per simulare drift in `data/incoming/drift_example.csv` usato dal task `ingest`. Il DAG scrive i report Evidently in `artifacts/` (montato sia in Airflow sia disponibile localmente).【F:airflow/dags/retrain_sentiment_dag.py†L10-L66】

## Notebook di consegna e test
- Notebook Colab per la demo completa in `notebooks/colab_delivery.ipynb`, linkato dal README nella sezione dedicata alla consegna.【F:README.md†L5-L7】
- Test unitari principali in `tests/` coprono preprocess, serving e pipeline di fallback; sono lanciabili con `pytest`. (Vedi `tests/` per ulteriori dettagli sui casi coperti.)

