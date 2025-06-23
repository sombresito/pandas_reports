import json
from typing import Any, Dict, List, Optional

import pandas as pd


def extract_test_cases(node: Any) -> List[Dict[str, Any]]:
    """Recursively extract test cases from an allure report tree."""
    raw: List[Dict[str, Any]] = []
    if isinstance(node, dict):
        if "uid" in node:
            raw.append(node)
        for child in node.get("children", []):
            raw.extend(extract_test_cases(child))
    elif isinstance(node, list):
        for item in node:
            raw.extend(extract_test_cases(item))
    return raw


def chunk_json_to_jsonl(json_data: Any, output_path: str, report_uuid: Optional[str] = None) -> pd.DataFrame:
    """Convert raw allure JSON report into a JSONL file with additional columns.

    Parameters
    ----------
    json_data : Any
        Structure loaded from the allure API.
    output_path : str
        Where to save the resulting ``.jsonl`` file.
    report_uuid : Optional[str]
        Identifier of the processed report. When ``None`` the value is attempted
        to be taken from ``json_data`` (``json_data['uuid']``) otherwise the
        literal ``"unknown"`` is used.
    """

    if report_uuid is None:
        if isinstance(json_data, dict) and "uuid" in json_data:
            report_uuid = str(json_data["uuid"])
        else:
            report_uuid = "unknown"

    raw = extract_test_cases(json_data)

    df = pd.DataFrame(
        [
            {
                "uid": t.get("uid"),
                "name": t.get("name"),
                "parentSuite": next((l["value"] for l in t.get("labels", []) if l["name"] == "parentSuite"), "unknown"),
                "suite": next((l["value"] for l in t.get("labels", []) if l["name"] == "suite"), "unknown"),
                "owner": next((l["value"] for l in t.get("labels", []) if l["name"] == "owner"), "unknown"),
                "severity": next((l["value"] for l in t.get("labels", []) if l["name"] == "severity"), "unknown"),
                "feature": next((l["value"] for l in t.get("labels", []) if l["name"] == "feature"), "unknown"),
                "host": next((l["value"] for l in t.get("labels", []) if l["name"] == "host"), "unknown"),
                "status": t.get("status"),
                "statusMessage": t.get("statusMessage", "").strip(),
                "statusTrace": t.get("statusTrace", "").strip(),
            }
            for t in raw
        ]
    )

    df["rag_text"] = df.apply(
        lambda row: (
            f"Название теста: {row['name']}\n"
            f"Команда: {row['parentSuite']}\n"
            f"Модуль: {row['suite']}\n"
            f"Владелец: {row['owner']}\n"
            f"Серьёзность: {row['severity']}\n"
            f"Фича: {row['feature']}\n"
            f"Хост: {row['host']}\n"
            f"Статус: {row['status']}\n"
            f"Сообщение: {row['statusMessage']}\n"
            f"Трейс: {row['statusTrace']}"
        ),
        axis=1,
    )

    df["report_uuid"] = report_uuid

    df.to_json(output_path, orient="records", lines=True, force_ascii=False)
    return df


if __name__ == "__main__":
    with open("reports/response_1750420712331.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    chunk_json_to_jsonl(data, "output_chunks.jsonl")
