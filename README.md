# pandas_reports

This project fetches test reports from an Allure API and runs RAG analysis on them.

## Environment variables

- `ALLURE_API` – base URL of the Allure API (default `http://allure-report-bcc-qa:8080/api`).
- `ALLURE_TOKEN` – if set, a bearer token used for authentication when contacting the API.
- `ALLURE_USER` and `ALLURE_PASS` – credentials for HTTP basic authentication. These are used only when a token is not provided.

- `CHUNKS_PATH` – path to a chunk file or directory containing chunk files used
  by `embeddings.py` and `save_embeddings_to_qdrant.py` (default `chunks`).
- `EMBEDDINGS_PATH` – file path where embeddings are stored or loaded
  (default `embeddings.npy`).
- `EMBEDDINGS_DIR` – directory where per-report embedding files are created
  by `embeddings.py` (default `embeddings`).
- `MODEL_PATH` – location of the SentenceTransformer model used for embedding
  generation (default `local_models/intfloat/multilingual-e5-small`).

When authentication variables are provided, requests made by `main.py` and `utils.py` automatically attach the appropriate `Authorization` header or basic auth parameters.

## Logging

All entry-point scripts configure Python's logging module. By default the level
is `INFO`. Set the `LOG_LEVEL` environment variable to control verbosity, e.g.

```bash
LOG_LEVEL=DEBUG python embeddings.py
```
