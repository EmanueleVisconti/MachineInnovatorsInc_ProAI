FROM apache/airflow:2.9.2

# -------------------------------------------------------------------
# Extra strumenti utili nel container (curl, git…)
# -------------------------------------------------------------------
USER root

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    wget \
    git \
 && rm -rf /var/lib/apt/lists/*

# -------------------------------------------------------------------
# Librerie Python aggiuntive (come avevamo quando tutto girava)
# -------------------------------------------------------------------
USER airflow

RUN pip install --no-cache-dir \
      mlflow==2.16.0 \
      transformers==4.45.2 \
      scikit-learn==1.5.2 \
      Evidently==0.4.36 \
      prometheus-client==0.20.0 \
 && pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
      torch==2.4.1+cpu \
      torchvision==0.19.1+cpu \
      torchaudio==2.4.1+cpu

