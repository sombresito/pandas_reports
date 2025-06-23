import pandas as pd
import json

def extract_test_cases(node):
    raw = []
    if isinstance(node, dict):
        if "uid" in node:  # это тест-кейс
            raw.append(node)
        for child in node.get("children", []):
            raw.extend(extract_test_cases(child))
    elif isinstance(node, list):
        for item in node:
            raw.extend(extract_test_cases(item))
    return raw

# Загружаем JSON
with open("reports/response_1750420712331.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Извлекаем все тест-кейсы
raw = extract_test_cases(data)

# Преобразуем в DataFrame
df = pd.DataFrame([{
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
    "statusTrace": t.get("statusTrace", "").strip()
} for t in raw])

# Добавим колонку с rag-текстом
df["rag_text"] = df.apply(lambda row: f"""Название теста: {row['name']}
Команда: {row['parentSuite']}
Модуль: {row['suite']}
Владелец: {row['owner']}
Серьёзность: {row['severity']}
Фича: {row['feature']}
Хост: {row['host']}
Статус: {row['status']}
Сообщение: {row['statusMessage']}
Трейс: {row['statusTrace']}""", axis=1)

# Сохраняем в формате .jsonl
report_uuid = df["rag_text"]
output_path = "output_chunks.jsonl"
df.to_json(output_path, orient="records", lines=True, force_ascii=False)
print(f"[OK] Файл сохранён: {output_path}")
