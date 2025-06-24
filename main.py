from fastapi import FastAPI, HTTPException, Request
import requests
import os
from utils import extract_team_name, chunk_and_save_json, analyze_and_post
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

ALLURE_API = os.getenv("ALLURE_API", "http://allure-report-bcc-qa:8080/api")

@app.post("/uuid/analyze")
async def analyze_report(request: Request):
    body = await request.json()
    uuid = body.get("uuid")
    if not uuid:
        raise HTTPException(status_code=400, detail="UUID not provided.")

    # 1. Получаем JSON отчёт
    url = f"{ALLURE_API}/report/{uuid}/test-cases/aggregate"
    resp = requests.get(url)
    if resp.status_code != 200:
        raise HTTPException(status_code=500, detail=f"Failed to fetch report: {resp.text}")
    
    report_data = resp.json()

    # 2. Получаем название команды
    team_name = extract_team_name(report_data)
    if not team_name:
        raise HTTPException(status_code=400, detail="Team name (parentSuite) not found.")

    # 3. Чанкуем и сохраняем
    chunk_and_save_json(report_data, uuid, team_name)

    # 4. Анализ и отправка результата
    try:
        analyze_and_post(uuid, team_name)
    except Exception as e:
        return {"result": "partial", "error": str(e)}

    return {"result": "ok", "team": team_name}
