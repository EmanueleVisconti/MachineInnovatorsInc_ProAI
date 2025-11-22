import argparse
import mlflow
import mlflow.pyfunc
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TextClassificationPipeline,
)
from src.utils.mlflow_utils import get_or_create_experiment, REGISTERED_NAME

MODEL_ID = "cardiffnlp/twitter-roberta-base-sentiment-latest"


class HFTextClassifier(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
        self.pipe = TextClassificationPipeline(
            model=self.model, tokenizer=self.tokenizer, return_all_scores=False
        )

    def predict(self, context, model_input):
        outputs = []
        for text in model_input:
            res = self.pipe(text, truncation=True)
            first = res[0] if isinstance(res, list) else res
            if isinstance(first, list):
                first = first[0]
            outputs.append({"label": first["label"], "score": float(first["score"])})
        return outputs


def main(experiment: str = "sentiment"):
    exp_id = get_or_create_experiment(experiment)
    mlflow.set_experiment(experiment)
    with mlflow.start_run(experiment_id=exp_id) as run:
        mlflow.log_param("base_model", MODEL_ID)
        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=HFTextClassifier(),
            registered_model_name=REGISTERED_NAME,
        )
        print(f"Run logged: {run.info.run_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="sentiment")
    args = parser.parse_args()
    main(args.experiment)
