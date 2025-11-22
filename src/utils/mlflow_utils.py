import os
import mlflow

# Imposta tracking URI (default locale)
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(MLFLOW_URI)

REGISTERED_NAME = os.getenv("REGISTERED_MODEL_NAME", "Sentiment")


def get_or_create_experiment(name: str) -> str:
    exp = mlflow.get_experiment_by_name(name)
    return exp.experiment_id if exp else mlflow.create_experiment(name)


def promote_to_stage(model_name: str, version: int, stage: str = "Production") -> None:
    client = mlflow.tracking.MlflowClient()
    client.transition_model_version_stage(
        name=model_name, version=version, stage=stage, archive_existing_versions=True
    )


def get_production_model_uri(model_name: str) -> str | None:
    client = mlflow.tracking.MlflowClient()
    versions = client.get_latest_versions(model_name, stages=["Production"]) or []
    if not versions:
        return None
    return f"models:/{model_name}/{versions[0].current_stage}"
