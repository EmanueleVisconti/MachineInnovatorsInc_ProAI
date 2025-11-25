# airflow/dags/retrain_sentiment_dag.py
from datetime import datetime, timedelta
import os, shutil, glob, subprocess, json
from airflow import DAG
from airflow.models import Variable
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
    incoming = glob.glob(os.path.join(DATA_DIR, "incoming", "*.csv"))
    if incoming:
        latest_csv = max(incoming, key=os.path.getmtime)
        shutil.copy(latest_csv, CUR)
        print(f"[ingest] Copiato batch {latest_csv} -> {CUR}")
    else:
        # fallback: riusa l'holdout come batch current per demo
        shutil.copy(HOLDOUT, CUR)
        print(f"[ingest] Nessun incoming: fallback {HOLDOUT} -> {CUR}")


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
    proc = subprocess.run(cmd, text=True)
    code = (
        proc.returncode if proc.returncode in (0, 1) else 1
    )  # 0=no drift, 1=drift (default: prudente)
    # push metrica
    try:
        subprocess.check_call(
            [
                "python",
                "-m",
                "src.monitoring.push_metrics",
                "--gateway",
                "http://pushgateway:9091",
                "--job",
                "retrain_sentiment",
                "--instance",
                "airflow",
                "--drift",
                str(code),
            ]
        )
    except Exception as e:
        print("[drift] pushgateway WARN:", e)
    return code


def branch_callable(**context):
    ti = context["ti"]
    dag_run = context.get("dag_run")

    # Override manuale: dag_run.conf o Variable di Airflow "force_retrain"
    conf_force = False
    if dag_run and dag_run.conf:
        conf_force = dag_run.conf.get("force_retrain", False) is True

    var_force_raw = Variable.get("force_retrain", default_var="false")
    var_force = str(var_force_raw).lower() in {"1", "true", "yes", "y"}

    # Forza retrain se è passato >=7 giorni dall'ultimo eseguito di QUESTO task (branch)
    time_gate = False
    last = ti.get_previous_ti()  # può essere None al primo run
    if last and last.end_date:
        # tz.utcnow() => aware; last.end_date è già aware (pendulum)
        time_gate = (tz.utcnow() - last.end_date) > timedelta(days=7)

    drift_code = ti.xcom_pull(task_ids="drift", key="return_value")

    # support dev smoke flag (dag_run.conf or Airflow Variable)
    conf_dev_smoke = False
    if dag_run and dag_run.conf:
        conf_dev_smoke = dag_run.conf.get("dev_smoke", False) is True

    var_dev_raw = Variable.get("force_dev_smoke", default_var="false")
    var_dev = str(var_dev_raw).lower() in {"1", "true", "yes", "y"}

    if conf_dev_smoke or var_dev:
        return "train_smoke"

    if conf_force or var_force:
        return "train"

    return "train" if (drift_code == 1 or time_gate) else "finish"


def train(ti=None):
    env = os.environ.copy()
    env["MLFLOW_TRACKING_URI"] = MLFLOW
    # 1) esegui il training (registra nuova versione nel Registry)
    subprocess.check_call(
        ["python", "-m", "src.models.train_roberta", "--experiment", "sentiment"],
        env=env,
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


def train_smoke(ti=None):
    env = os.environ.copy()
    env["MLFLOW_TRACKING_URI"] = MLFLOW
    dev_suffix = os.environ.get("REGISTERED_MODEL_DEV_SUFFIX", "-dev")
    # esegui il training dev (script leggero)
    subprocess.check_call(
        [
            "python",
            "-m",
            "src.models.train_smoke",
            "--experiment",
            "sentiment",
            "--n_samples",
            os.environ.get("SMOKE_N_SAMPLES", "1"),
        ],
        env=env,
    )
    # recupera la versione dev appena registrata
    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=MLFLOW)
    dev_model_name = f"{MODEL_NAME}{dev_suffix}"
    versions = client.search_model_versions(f"name='{dev_model_name}'")
    if not versions:
        raise RuntimeError(f"Nessuna versione trovata per il modello '{dev_model_name}'")
    latest = max(versions, key=lambda v: int(v.version))
    new_uri = f"models:/{dev_model_name}/{int(latest.version)}"
    if ti:
        ti.xcom_push(key="new_uri", value=new_uri)


def evaluate_and_promote(ti=None):
    # 1) prova a leggere la URI dal train
    new_uri = None
    if ti:
        new_uri = ti.xcom_pull(task_ids="train", key="new_uri")
        if not new_uri:
            # try the dev smoke task too
            new_uri = ti.xcom_pull(task_ids="train_smoke", key="new_uri")

    # 2) fallback robusto: prendi comunque l'ultima versione registrata
    if not new_uri:
        from mlflow.tracking import MlflowClient

        client = MlflowClient(tracking_uri=MLFLOW)
        versions = client.search_model_versions(f"name='{MODEL_NAME}'")
        if not versions:
            raise RuntimeError(
                f"Nessuna versione trovata per il modello '{MODEL_NAME}'"
            )
        latest = max(versions, key=lambda v: int(v.version))
        new_uri = f"models:/{MODEL_NAME}/{int(latest.version)}"

    print(f"[evaluate_and_promote] new_model_uri = {new_uri}")

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
        task_id="branch",
        python_callable=branch_callable,
    )
    t_train = PythonOperator(task_id="train", python_callable=train)
    t_train_smoke = PythonOperator(task_id="train_smoke", python_callable=train_smoke)
    t_eval = PythonOperator(
        task_id="evaluate_and_promote", python_callable=evaluate_and_promote
    )
    t_finish = PythonOperator(task_id="finish", python_callable=_noop)

    t_ingest >> t_drift >> t_branch
    t_branch >> t_train >> t_eval
    t_branch >> t_train_smoke >> t_eval
    t_branch >> t_finish
