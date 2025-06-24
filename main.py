from fastapi import FastAPI, HTTPException, Request
import requests
import os
import logging
from utils import extract_team_name, chunk_and_save_json, analyze_and_post, _auth_kwargs
from dotenv import load_dotenv

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO))
logger = logging.getLogger(__name__)

load_dotenv()

app = FastAPI()

ALLURE_API = os.getenv("ALLURE_API", "http://allure-report-bcc-qa:8080/api")

@app.post("/uuid/analyze")
async def analyze_report(request: Request):
    body = await request.json()
    uuid = body.get("uuid")
    if not uuid:
        raise HTTPException(status_code=400, detail="UUID not provided.")
    logger.info("Analyze request received for UUID %s", uuid)

    # 1. Получаем JSON отчёт
    url = f"{ALLURE_API}/report/{uuid}/test-cases/aggregate"
    auth_kwargs = _auth_kwargs()
    try:
        resp = requests.get(url, timeout=10, **auth_kwargs)
        resp.raise_for_status()
        logger.info("Fetched report data from Allure")
    except requests.RequestException as e:
        logger.error("Failed to fetch report for %s: %s", uuid, e)
        raise HTTPException(status_code=500, detail=f"Failed to fetch report: {e}") from e

    report_data = resp.json()

    # 2. Получаем название команды
    team_name = extract_team_name(report_data)
    if not team_name:
        logger.error("Team name not found in report %s", uuid)
        raise HTTPException(status_code=400, detail="Team name (parentSuite) not found.")
    logger.info("Team name extracted: %s", team_name)

    # 3. Чанкуем и сохраняем
    chunk_and_save_json(report_data, uuid, team_name)
    logger.info("Chunks saved for %s", uuid)

    # 4. Анализ и отправка результата
    try:
        analyze_and_post(uuid, team_name)
        logger.info("Analysis posted for %s", uuid)
    except Exception as e:
        logger.error("Analysis failed for %s: %s", uuid, e)
        return {"result": "partial", "error": str(e)}

    return {"result": "ok", "team": team_name}
