import os
import json
import shutil
from pandas_chunking import chunk_json_to_jsonl
from rag_pipeline import run_rag_analysis
import requests

def extract_team_name(report_json):
    for test in report_json:
        for label in test.get("labels", []):
            if label.get("name") == "parentSuite":
                return label.get("value")
    return None

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
