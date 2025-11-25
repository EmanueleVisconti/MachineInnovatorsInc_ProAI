# Flusso di training, valutazione e promozione del modello

Questo documento riassume come il DAG `retrain_sentiment` addestra, valuta e decide se sostituire il modello in produzione, utile per comprendere e riprodurre lo **swap** del modello.

## Componenti coinvolti
- **Training**: `python -m src.models.train_roberta --experiment sentiment`
  - Registra il modello HuggingFace `cardiffnlp/twitter-roberta-base-sentiment-latest` in MLflow come modello registrato `Sentiment`.
- **Valutazione/promozione**: `python -m src.models.evaluate --new_model_uri <uri> --eval_csv data/holdout.csv --min_improvement 0.0`
  - Confronta il nuovo modello con quello in stage `Production` su `data/holdout.csv` usando macro-F1.
  - Se il nuovo modello è **>=** del precedente (soglia `min_improvement`), viene promosso a `Production` (archiviando la versione precedente).
- **Serving**: `src.serving.load_model.predict_fn`
  - Se esiste `MODEL_URI` (es. `models:/Sentiment/Production`), serve la versione in produzione; altrimenti usa il modello HF di base.

## Sequenza nel DAG `retrain_sentiment`
1. `ingest`: prepara `data/raw/current.csv` copiando il primo batch in `data/incoming/` (o l'holdout come fallback demo).
2. `drift`: esegue `src.monitoring.drift_report` per confrontare `data/raw/reference.csv` vs `data/raw/current.csv`.
   - Genera un report Evidently e restituisce **0** (no drift) o **1** (drift rilevato).
   - Pusha `data_drift_flag` al Pushgateway (Grafana mostra il valore).
3. `branch`:
   - Se `force_retrain=true` (run config o Variable) **oppure** è passato ≥7 giorni dall'ultimo run **oppure** `drift=1`, segue il ramo `train -> evaluate_and_promote`; altrimenti termina in `finish`.
4. `train`: allena e registra una nuova versione MLflow del modello `Sentiment` e pubblica la URI in XCom.
5. `evaluate_and_promote`: carica la URI della nuova versione (o l'ultima registrata se manca XCom), valuta su `data/holdout.csv` e, se migliore, la promuove a `Production`.

### Nota Dev/Smoke mode
Per testing e demo è disponibile una modalità `dev_smoke` che addestra un small-model sklearn su una porzione (head) del CSV e registra il modello con suffisso `-dev` (ad es. `Sentiment-dev`). La modalità dev è pensata solo per test del flusso; i modelli `-dev` non vengono promossi in `Production` automaticamente.

## Come verificare lo swap di modello
1. Apri MLflow UI (http://localhost:5000) e vai in **Models → Sentiment**: noterai versioni con stage `None` o `Production`.
2. Dopo l'esecuzione del task `evaluate_and_promote`:
   - Se il nuovo modello è migliore, la versione più recente sarà in stage `Production` (le precedenti archiviate).
   - Se non migliora, il nuovo modello resta in stage `None` e la produzione non cambia.
3. Il servizio FastAPI, se lanciato con `MODEL_URI=models:/Sentiment/Production`, servirà automaticamente la versione promossa (altrimenti userà il modello HF di base).

## Esecuzione manuale (fuori da Airflow)
Se vuoi riprodurre il ciclo completo senza DAG:
```bash
# 1) Training
MLFLOW_TRACKING_URI=http://localhost:5000 python -m src.models.train_roberta --experiment sentiment

# 2) Identifica l'ultima versione registrata
#    (oppure prendi la URI dal log del training)

# 3) Valutazione + eventuale promozione
MLFLOW_TRACKING_URI=http://localhost:5000 python -m src.models.evaluate \
  --new_model_uri "models:/Sentiment/None" \
  --eval_csv data/holdout.csv \
  --min_improvement 0.0
```
Il comando di valutazione carica anche il modello in produzione (se esiste) e, se il nuovo non è peggiore, lo promuove automaticamente.
