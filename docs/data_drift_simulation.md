# Come simulare un data drift e forzare il retraining

Questa guida mostra come presentare rapidamente un caso di **drift rilevato** che instrada il DAG `retrain_sentiment` nei task `train` → `evaluate_and_promote`.

## Opzione A – usare il batch già pronto
1. C'è un batch "estremo" in `data/incoming/drift_example.csv` con testi molto lunghi e distribuzioni di etichette diverse dal reference; la logica di drift controlla sia la mediana della lunghezza sia la distanza tra distribuzioni di label.
2. Avvia lo stack:
   ```bash
   docker compose up --build
   ```
3. Dalla UI Airflow (http://localhost:8080) attiva e triggera il DAG `retrain_sentiment` **senza** configurazione extra.
   - Il task `ingest` copierà automaticamente `data/incoming/drift_example.csv` in `data/raw/current.csv`.
   - Il task `drift` dovrebbe restituire `1` (drift rilevato) e pushare `data_drift_flag{job="retrain_sentiment",instance="airflow"}=1` sul Pushgateway.
4. Verifica:
   - In Prometheus: `curl "http://localhost:9090/api/v1/query?query=data_drift_flag"`
   - In Grafana: pannello **Data Drift Flag** nella dashboard `MLOps – Sentiment App`.
   - In Airflow: il ramo `train -> evaluate_and_promote` sarà eseguito (stato verde) invece di `finish`.

## Opzione B – creare un batch personalizzato
Se vuoi dimostrare un drift "costruito" al volo (lunghezze molto diverse o sbilanciamento forte delle etichette):
```bash
cat > data/incoming/custom_drift.csv <<'TXT'
text,label
"Very long, spammy, off-topic paragraph repeated many times to change length distribution and sentiment predictions.",negative
"Another huge block of text with lots of neutral wording and filler to make the sample unlike the short reference tweets.",neutral
"Super positive and overly enthusiastic essay that repeats compliments to skew the predicted labels toward positive.",positive
TXT
```
Poi rilancia il DAG `retrain_sentiment` (o usa `docker compose exec airflow airflow dags test retrain_sentiment 2025-01-02`). Il file in `data/incoming/` con ordine alfabetico più basso viene usato da `ingest`.

## Nota sul force retrain
Se per qualunque motivo Evidently non restituisse drift (0), puoi comunque forzare il ramo di retraining:
- Trigger DAG con configurazione JSON `{ "force_retrain": true }`, **oppure**
- Imposta la Variable Airflow `force_retrain=true` da Admin → Variables.
