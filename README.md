# MLOps Sentiment – Milestone 1


Service FastAPI che serve `cardiffnlp/twitter-roberta-base-sentiment-latest` con endpoint `/predict`, `/health`, `/metrics`.


## Avvio locale


```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn src.serving.app:app --reload --port 8000