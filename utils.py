import os
import logging
try:
    from fastapi import HTTPException
except Exception:  # pragma: no cover - fastapi not installed during tests
    class HTTPException(Exception):
        def __init__(self, status_code: int, detail: str):
            self.status_code = status_code
            self.detail = detail
            super().__init__(detail)
from pandas_chunking import chunk_json_to_jsonl
from rag_pipeline import run_rag_analysis
import requests

def _auth_kwargs():
    """Return request kwargs for optional Allure authentication.

    If ``ALLURE_TOKEN`` is set, a bearer token header is used. Otherwise, if
    ``ALLURE_USER`` and ``ALLURE_PASS`` are provided, basic authentication is
    configured. When none of these variables are present, an empty ``dict`` is
    returned.
    """
    token = os.getenv("ALLURE_TOKEN")
    if token:
        return {"headers": {"Authorization": f"Bearer {token}"}}

    user = os.getenv("ALLURE_USER")
    password = os.getenv("ALLURE_PASS")
    if user and password:
        return {"auth": (user, password)}

    return {}

ALLURE_API = os.getenv("ALLURE_API", "http://allure-report-bcc-qa:8080/api")

logger = logging.getLogger(__name__)

def extract_team_name(report_json):
    """Return the first ``parentSuite`` label value found in the report data.

    The Allure API may return either a list of test dictionaries or a nested
    dictionary structure. This function detects the root type and walks through
    the appropriate containers until the first ``parentSuite`` label is found.

    Parameters
    ----------
    report_json : Any
        Parsed JSON data returned by the Allure API.

    Returns
    -------
    str | None
        Value of the ``parentSuite`` label or ``None`` if it is not present.
    """

    def _search(node):
        if isinstance(node, dict):
            for label in node.get("labels", []):
                if label.get("name") == "parentSuite":
                    return label.get("value")
            for value in node.values():
                if isinstance(value, (dict, list)):
                    result = _search(value)
                    if result:
                        return result
        elif isinstance(node, list):
            for item in node:
                result = _search(item)
                if result:
                    return result
        return None

    return _search(report_json)

def chunk_and_save_json(json_data, uuid, team_name):
    base_dir = os.path.join("chunks", team_name)
    os.makedirs(base_dir, exist_ok=True)

    # Чанкуем
    output_path = os.path.join(base_dir, f"{uuid}.jsonl")
    chunk_json_to_jsonl(json_data, output_path, uuid)

    # Удаляем старые отчёты (оставляем 3)
    files = [os.path.join(base_dir, f) for f in os.listdir(base_dir)]
    files.sort(key=os.path.getmtime, reverse=True)
    while len(files) > 3:
        oldest = files.pop()  # last element is the oldest
        os.remove(oldest)

def analyze_and_post(uuid, team_name):
    result = run_rag_analysis(team_name)

    # Отправка анализа
    url = f"{ALLURE_API}/analysis/report/{uuid}"
    auth_kwargs = _auth_kwargs()
    try:
        resp = requests.post(url, json=result, timeout=10, **auth_kwargs)
    except requests.RequestException as e:
        logger.error("Failed to post analysis for %s: %s", uuid, e)
        raise HTTPException(status_code=500, detail=f"Failed to post analysis: {e}") from e

    # Check response status explicitly so analyze_report can handle errors
    if not 200 <= resp.status_code < 300:
        msg = f"Unexpected status {resp.status_code}: {resp.text}"
        logger.error("Failed to post analysis for %s: %s", uuid, msg)
        raise HTTPException(status_code=500, detail=msg)

