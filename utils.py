import os
import logging
from fastapi import HTTPException
from pandas_chunking import chunk_json_to_jsonl
from rag_pipeline import run_rag_analysis, RagAnalysisError
import requests

logger = logging.getLogger(__name__)

# Базовый URL Allure API, по умолчанию взят из переменной окружения
ALLURE_API = os.getenv("ALLURE_API", "http://allure-report-bcc-qa:8080/api")


def _auth_kwargs():
    """
    Возвращает kwargs для запросов к Allure: либо заголовок Bearer,
    либо basic-auth, либо пустой dict
    """
    token = os.getenv("ALLURE_TOKEN")
    if token:
        return {"headers": {"Authorization": f"Bearer {token}"}}

    user = os.getenv("ALLURE_USER")
    password = os.getenv("ALLURE_PASS")
    if user and password:
        return {"auth": (user, password)}

    return {}


def extract_team_name(report_json):
    """
    Извлекает первое значение метки parentSuite из JSON-отчёта.
    """
    def _search(node):
        if isinstance(node, dict):
            for label in node.get("labels", []):
                if label.get("name") == "parentSuite":
                    return label.get("value")
            for value in node.values():
                if isinstance(value, (dict, list)):
                    found = _search(value)
                    if found:
                        return found
        elif isinstance(node, list):
            for item in node:
                found = _search(item)
                if found:
                    return found
        return None

    return _search(report_json)


def chunk_and_save_json(json_data, uuid, team_name):
    """
    Разбивает JSON-отчёт на чанки, сохраняет в <chunks>/<team_name>/<uuid>.jsonl
    и удаляет старые файлы, оставляя не более 3.
    """
    base_dir = os.path.join("chunks", team_name)
    os.makedirs(base_dir, exist_ok=True)

    output_path = os.path.join(base_dir, f"{uuid}.jsonl")
    df = chunk_json_to_jsonl(json_data, output_path, uuid)

    # Оставляем только последние 3 отчёта
    files = [os.path.join(base_dir, f) for f in os.listdir(base_dir)]
    files.sort(key=os.path.getmtime, reverse=True)
    while len(files) > 3:
        os.remove(files.pop())

    return output_path, df


def analyze_and_post(uuid: str, team_name: str):
    """
    Выполняет RAG-анализ и отправляет результат на Allure-сервер.

    Вход: POST /uuid/analyze с JSON {"uuid": "..."}
    GET Allure-отчёта: GET {ALLURE_API}/report/{uuid}/test-cases/aggregate
    Выход: POST {ALLURE_API}/analysis/report/{uuid} с JSON вида:
      [{"rule": "auto-analysis", "message": "<текст анализа>"}]
    """
    # 1. Генерируем анализ
    try:
        analysis_result = run_rag_analysis(team_name)
        analysis_text = analysis_result.get("analysis", "")
    except RagAnalysisError as e:
        logger.error("RAG analysis failed for %s: %s", uuid, e)
        raise HTTPException(
            status_code=500,
            detail=f"RAG analysis error: {e}"
        ) from e

    # Формируем полезную нагрузку по спецификации
    payload = [
        {
            "rule": "auto-analysis",
            "message": analysis_text
        }
    ]

    # 2. Отправляем анализ на Allure-сервер
    url = f"{ALLURE_API}/analysis/report/{uuid}"
    auth_kwargs = _auth_kwargs()
    try:
        resp = requests.post(url, json=payload, timeout=10, **auth_kwargs)
        resp.raise_for_status()
    except requests.RequestException as e:
        logger.error("Failed to post analysis for %s: %s", uuid, e)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to post analysis: {e}"
        ) from e

    logger.info("Analysis posted successfully for %s", uuid)
