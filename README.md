# pandas_reports

This project fetches test reports from an Allure API and runs RAG analysis on them.

## Environment variables

- `ALLURE_API` – base URL of the Allure API (default `http://allure-report-bcc-qa:8080/api`).
- `ALLURE_TOKEN` – if set, a bearer token used for authentication when contacting the API.
- `ALLURE_USER` and `ALLURE_PASS` – credentials for HTTP basic authentication. These are used only when a token is not provided.

When authentication variables are provided, requests made by `main.py` and `utils.py` automatically attach the appropriate `Authorization` header or basic auth parameters.
