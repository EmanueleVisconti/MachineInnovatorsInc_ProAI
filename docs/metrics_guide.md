# Guida alle metriche esposte

## Endpoint FastAPI `/metrics`
- **`app_requests_total`**: counter delle richieste a `/predict` (1 per successo o errore). Utile per vedere il volume nel tempo.
- **`app_errors_total`**: counter degli errori applicativi durante l'inferenza. Se cresce in parallelo alle richieste, la qualità del servizio sta degradando.
- **`app_request_latency_seconds`**: istogramma della latenza end-to-end di `/predict`.
  - In Grafana i pannelli p50/p90 usano `histogram_quantile(0.5|0.9, rate(app_request_latency_seconds_bucket[5m]))` per stimare la mediana e il 90° percentile negli ultimi 5 minuti.
  - Perché p50/p90: la media nasconde code lunghe; p50 mostra la risposta tipica, p90 cattura le code lente che impattano l'esperienza utente.
- **`data_drift_flag`**: gauge inizializzato a 0 all'avvio app. Il DAG di retrain aggiorna la stessa metrica via Pushgateway quando rileva drift; la dashboard mostra sia il valore pushato da Airflow sia quello dell'app.

## Flusso Prometheus/Pushgateway/Grafana
1. L'app espone le metriche Prometheus su `/metrics` (scrape job `app`).
2. Il DAG Airflow pubblica `data_drift_flag` sul Pushgateway (porta 9091, UI celeste). Prometheus ha un job dedicato per scrappare il Pushgateway e fondere i dati con il resto.
3. Grafana usa il datasource Prometheus preconfigurato e la dashboard `MLOps – Sentiment App` per visualizzare volume richieste, errori, latenze p50/p90 e `data_drift_flag`.

## Come leggere i pannelli principali
- **Request volume**: deriva da `app_requests_total`; un aumento senza corrispondente crescita di errori è un buon segno di stabilità.
- **Error rate**: basato su `app_errors_total` confrontato con le richieste; se il pannello supera lo 0–1% indagare i log dell'app.
- **Latency p50/p90**: se il p90 cresce mentre il p50 resta basso, alcune richieste sono molto lente (es. cold start, rete). Dopo il warm-up la p50/p90 dovrebbe stabilizzarsi sotto pochi secondi.
- **Data drift flag**: valore 0/1 aggiornato dai run del DAG; 1 indica che la finestra corrente differisce dalla baseline e che è stato scelto il ramo di retraining.

## Verifiche rapide da terminale
- Ultimo valore di drift (Prometheus REST):
  ```bash
  curl "http://localhost:9090/api/v1/query?query=data_drift_flag"
  ```
- Sample delle metriche dell'app:
  ```bash
  curl http://localhost:8000/metrics | head
  ```
