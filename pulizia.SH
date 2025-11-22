# 1) Stop container che occupano risorse
docker stop mlflow airflow app 2>/dev/null || true
docker rm -f mlflow airflow app 2>/dev/null || true

# 2) Docker prune aggressivo (immagini/layer/reti/volumi non usati)
docker system prune -af --volumes
docker builder prune -af

# 3) Cache Python/HF/Torch
pip cache purge
rm -rf ~/.cache/huggingface ~/.cache/torch ~/.cache/pip/*

# 4) (Opzionale ma consigliato) pulisci MLflow locale se non ti serve lo storico
rm -rf mlruns/* && mkdir -p mlruns

# 5) Ricrea la venv SENZA CUDA (torch CPU-only)
deactivate 2>/dev/null || true
rm -rf .venv
python -m venv .venv && source .venv/bin/activate

# rimuovi torch dal requirements (verrà installato a parte in CPU)
sed -i '/^torch==/d' requirements.txt

# installa tutto il resto senza cache
pip install --no-cache-dir -r requirements.txt

# ora installa torch CPU (ruote ufficiali)
pip install --no-cache-dir \
  torch==2.4.1+cpu torchvision==0.19.1+cpu torchaudio==2.4.1+cpu \
  --index-url https://download.pytorch.org/whl/cpu

# 6) ricontrolla spazio
df -h


###########################

# 1) verifica sintassi
docker compose config   # deve stampare il compose renderizzato, senza errori

# 2) ricrea i servizi puliti
docker compose down -v
docker compose up -d mlflow airflow

# 3) controlla nomi e porte (ora prevedibili: sentiment-mlflow-1, sentiment-airflow-1)
docker compose ps

# 4) segui i log di Airflow finché non vedi "Listening at: http://0.0.0.0:8080"
docker compose logs -f airflow



