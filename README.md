# MLOps Sentiment – Milestone 1


Service FastAPI che serve `cardiffnlp/twitter-roberta-base-sentiment-latest` con endpoint `/predict`, `/health`, `/metrics`.


## Avvio locale

### app
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn src.serving.app:app --reload --port 8000
```
### docker
```bash
docker build -t machineinnovators_inc_proai -f docker/Dockerfile.app .
docker run --rm -p 8000:8000 machineinnovators_inc_proai

```