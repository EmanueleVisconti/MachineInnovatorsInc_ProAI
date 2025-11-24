FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TRANSFORMERS_NO_TF=1

WORKDIR /app

# dipendenze di sistema minime + curl per test da container
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# copia sorgenti
COPY src /app/src
COPY requirements.txt /app/requirements.txt

# lib python principali
RUN pip install --no-cache-dir \
      fastapi==0.115.5 uvicorn==0.32.0 \
      transformers==4.45.2 mlflow==2.16.0 \
      scikit-learn==1.5.2 prometheus-client==0.20.0

# PyTorch CPU dai wheel ufficiali
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
      torch==2.4.1+cpu torchvision==0.19.1+cpu torchaudio==2.4.1+cpu

# opzionale: se hai file extra in requirements.txt li installa (non è un errore se è vuoto)
RUN if [ -s requirements.txt ]; then pip install --no-cache-dir -r requirements.txt; fi

EXPOSE 8000

CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
