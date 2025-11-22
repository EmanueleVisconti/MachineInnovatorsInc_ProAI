# airflow/dags/retrain_sentiment_dag.py
from datetime import datetime, timedelta
import os, shutil, glob, subprocess, json
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator

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


def branch_callable(ti, **_):
    # tempo: retrain forzato ogni 7 giorni
    last = ti.get_previous_ti()
    time_gate = False
    if last and last.end_date:
        time_gate = (datetime.utcnow() - last.end_date) > timedelta(days=7)
    drift_code = ti.xcom_pull(task_ids="drift", key="return_value")
    return "train" if (drift_code == 1 or time_gate) else "finish"


def train():
    env = os.environ.copy()
    env["MLFLOW_TRACKING_URI"] = MLFLOW
    subprocess.check_call(
        ["python", "-m", "src.models.train_roberta", "--experiment", "sentiment"],
        env=env,
    )


def evaluate_and_promote():
    # Recupera ultimo run_id dal tracking server via MLflow client sarebbe ideale;
    # per semplicità usiamo la notazione models:/MODEL_NAME/None (version appena registrata)
    new_uri = f"models:/{MODEL_NAME}/None"
    env = os.environ.copy()
    env["MLFLOW_TRACKING_URI"] = MLFLOW
    subprocess.check_call(
        [
            "python",
            "-m",
            "src.models.evaluate",
            "--new_model_uri",
            new_uri,
            "--eval_csv",
            HOLDOUT,
            "--min_improvement",
            "0.0",
        ],
        env=env,
    )


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
        task_id="branch", python_callable=branch_callable, provide_context=True
    )
    t_train = PythonOperator(task_id="train", python_callable=train)
    t_eval = PythonOperator(
        task_id="evaluate_and_promote", python_callable=evaluate_and_promote
    )
    t_finish = PythonOperator(task_id="finish", python_callable=_noop)

    t_ingest >> t_drift >> t_branch
    t_branch >> t_train >> t_eval
    t_branch >> t_finish
