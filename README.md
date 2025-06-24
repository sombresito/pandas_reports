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
- `QDRANT_URL` – base URL of the Qdrant service used by
  `rag_pipeline.get_client` and `save_embeddings_to_qdrant.py`
  (default `http://localhost:6333`).
- `QDRANT_TIMEOUT` – request timeout for the Qdrant client (default `10`).
- `OLLAMA_URL` – base URL of the Ollama API (default `http://localhost:11434/api/generate`).

When authentication variables are provided, requests made by `main.py` and `utils.py` automatically attach the appropriate `Authorization` header or basic auth parameters.

## Launching Qdrant

To run Qdrant locally you can start the official Docker container:

```bash
docker run -p 6333:6333 qdrant/qdrant
```

This exposes the service on port `6333` so that it can be reached at
`http://localhost:6333`. Set the `QDRANT_URL` environment variable to that
address (or another hostname/port if you change the mapping).

## Logging

All entry-point scripts configure Python's logging module. By default the level
is `INFO`. Set the `LOG_LEVEL` environment variable to control verbosity, e.g.

```bash
LOG_LEVEL=DEBUG python embeddings.py
```

## Troubleshooting

If the application fails to connect to Qdrant, ensure that `QDRANT_URL` points to
the running service. Connection errors from `rag_pipeline.get_client` will be logged before raising a
`QdrantConnectionError`.
Increase `QDRANT_TIMEOUT` if requests to Qdrant repeatedly time out.
Verify that `OLLAMA_URL` points to the running Ollama API when connection
errors occur.
If the client reports an incompatible version error, initialize it with
`check_compatibility=False` to bypass the check.
