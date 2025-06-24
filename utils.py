import os
from pandas_chunking import chunk_json_to_jsonl
from rag_pipeline import run_rag_analysis
import requests

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
    os.makedirs(f"chunks/{team_name}", exist_ok=True)

    # Чанкуем
    output_path = f"chunks/{team_name}/{uuid}.jsonl"
    chunk_json_to_jsonl(json_data, output_path, uuid)

    # Удаляем старые отчёты (оставляем 3)
    files = sorted(os.listdir(f"chunks/{team_name}"))
    if len(files) > 3:
        os.remove(f"chunks/{team_name}/{files[0]}")

def analyze_and_post(uuid, team_name):
    result = run_rag_analysis(team_name)

    # Отправка анализа
    url = f"http://allure-report-bcc-qa:8080/api/analysis/report/{uuid}"
    requests.post(url, json=result)
