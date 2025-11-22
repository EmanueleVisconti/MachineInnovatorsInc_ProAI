# airflow/dags/retrain_sentiment_dag.py
from datetime import datetime, timedelta
import os, shutil, glob, subprocess, json
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.utils import timezone as tz


DATA_DIR = "/opt/airflow/data"
ART_DIR = "/opt/airflow/artifacts"
HOLDOUT = os.path.join(DATA_DIR, "holdout.csv")
REF = os.path.join(DATA_DIR, "raw", "reference.csv")
CUR = os.path.join(DATA_DIR, "raw", "current.csv")

MLFLOW = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MODEL_NAME = os.environ.get("REGISTERED_MODEL_NAME", "Sentiment")


def ingest():
    os.makedirs(os.path.dirname(CUR), exist_ok=True)
    incoming = sorted(glob.glob(os.path.join(DATA_DIR, "incoming", "*.csv")))
    if incoming:
        shutil.copy(incoming[0], CUR)
    else:
        # fallback: riusa l'holdout come batch current per demo
        shutil.copy(HOLDOUT, CUR)


def compute_drift():
    os.makedirs(ART_DIR, exist_ok=True)
    cmd = [
        "python",
        "-m",
        "src.monitoring.drift_report",
        "--reference",
        REF,
        "--current",
        CUR,
        "--out",
        ART_DIR,
    ]
    res = subprocess.run(cmd, check=False)
    return res.returncode  # 0=no drift, 1=drift


def branch_callable(**context):
    ti = context["ti"]
    # Forza retrain se è passato >=7 giorni dall'ultimo eseguito di QUESTO task (branch)
    time_gate = False
    last = ti.get_previous_ti()  # può essere None al primo run
    if last and last.end_date:
        # tz.utcnow() => aware; last.end_date è già aware (pendulum)
        time_gate = (tz.utcnow() - last.end_date) > timedelta(days=7)

    drift_code = ti.xcom_pull(task_ids="drift", key="return_value")
    return "train" if (drift_code == 1 or time_gate) else "finish"


def train(ti=None):
    env = os.environ.copy()
    env["MLFLOW_TRACKING_URI"] = MLFLOW
    # 1) esegui il training (registra nuova versione nel Registry)
    subprocess.check_call(
        ["python", "-m", "src.models.train_roberta", "--experiment", "sentiment"],
        env=env
    )
    # 2) risali alla ULTIMA versione registrata e mettila in XCom
    from mlflow.tracking import MlflowClient
    client = MlflowClient(tracking_uri=MLFLOW)
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    if not versions:
        raise RuntimeError(f"Nessuna versione trovata per il modello '{MODEL_NAME}'")
    latest = max(versions, key=lambda v: int(v.version))
    new_uri = f"models:/{MODEL_NAME}/{int(latest.version)}"
    if ti:
        ti.xcom_push(key="new_uri", value=new_uri)


def evaluate_and_promote(ti=None):
    # 1) prova a leggere la URI dal train
    new_uri = None
    if ti:
        new_uri = ti.xcom_pull(task_ids="train", key="new_uri")

    # 2) fallback robusto: prendi comunque l'ultima versione registrata
    if not new_uri:
        from mlflow.tracking import MlflowClient
        client = MlflowClient(tracking_uri=MLFLOW)
        versions = client.search_model_versions(f"name='{MODEL_NAME}'")
        if not versions:
            raise RuntimeError(f"Nessuna versione trovata per il modello '{MODEL_NAME}'")
        latest = max(versions, key=lambda v: int(v.version))
        new_uri = f"models:/{MODEL_NAME}/{int(latest.version)}"

    print(f"[evaluate_and_promote] new_model_uri = {new_uri}")

    env = os.environ.copy()
    env["MLFLOW_TRACKING_URI"] = MLFLOW
    subprocess.check_call([
        "python", "-m", "src.models.evaluate",
        "--new_model_uri", new_uri,
        "--eval_csv", HOLDOUT,
        "--min_improvement", "0.0",
    ], env=env)



def _noop():
    pass


with DAG(
    dag_id="retrain_sentiment",
    schedule_interval="@daily",
    start_date=datetime(2025, 1, 1),
    catchup=False,
    default_args={"retries": 0},
) as dag:
    t_ingest = PythonOperator(task_id="ingest", python_callable=ingest)
    t_drift = PythonOperator(task_id="drift", python_callable=compute_drift)
    t_branch = BranchPythonOperator(
        task_id="branch",
        python_callable=branch_callable,
    )
    t_train = PythonOperator(task_id="train", python_callable=train)
    t_eval = PythonOperator(
        task_id="evaluate_and_promote", python_callable=evaluate_and_promote
    )
    t_finish = PythonOperator(task_id="finish", python_callable=_noop)

    t_ingest >> t_drift >> t_branch
    t_branch >> t_train >> t_eval
    t_branch >> t_finish
