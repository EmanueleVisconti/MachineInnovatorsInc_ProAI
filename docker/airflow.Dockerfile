FROM apache/airflow:2.9.2

USER airflow
ARG AIRFLOW_VERSION=2.9.2
ARG PYTHON_VERSION=3.11
ARG CONSTRAINT_URL=https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${PYTHON_VERSION}.txt

# Pacchetti che non confliggono con le constraint
RUN pip install --no-cache-dir --constraint "${CONSTRAINT_URL}" \
    Evidently==0.4.36 \
    transformers==4.45.2 \
    mlflow==2.16.0

# Torch CPU fuori dalle constraint (ruote ufficiali)
RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    torch==2.4.1+cpu torchvision==0.19.1+cpu torchaudio==2.4.1+cpu
