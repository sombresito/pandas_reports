import os
import numpy as np
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, Match

# Настройки
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
COLLECTION_NAME = "allure_chunks"
VECTOR_SIZE = 384

# Загружаем данные
CHUNKS_PATH = os.getenv("CHUNKS_PATH", "output_chunks.jsonl")
EMBEDDINGS_PATH = os.getenv("EMBEDDINGS_PATH", "embeddings.npy")
df = pd.read_json(CHUNKS_PATH, lines=True)
embeddings = np.load(EMBEDDINGS_PATH)

# Получаем название команды и UUID отчёта из первой строки
first_row = df.iloc[0]
team = first_row["parentSuite"]
report_uuid = first_row["report_uuid"]

# Подключаемся к Qdrant
client = qdrant_client

# Создаём коллекцию, если не существует
if not client.collection_exists(collection_name=COLLECTION_NAME):
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )
    print(f"[INFO] Коллекция '{COLLECTION_NAME}' создана")

# Удалим самые старые отчёты, если их уже 3 для этой команды
search_filter = Filter(
    must=[FieldCondition(key="team", match=Match(value=team))]
)
existing = client.scroll(
    collection_name=COLLECTION_NAME,
    scroll_filter=search_filter,
    limit=1000,
    with_payload=True
)[0]

# Группируем по UUID и удаляем, если больше 2 старых
from collections import defaultdict

by_uuid = defaultdict(list)
for point in existing:
    report_id = point.payload.get("report_uuid", "unknown")
    by_uuid[report_id].append(point)

# Оставим 2 самых свежих
if len(by_uuid) >= 3:
    # Сортировка по UUID (или можно по timestamp, если есть)
    sorted_uuids = sorted(by_uuid.items(), key=lambda x: x[0])  # замените на дату при наличии
    for old_uuid, old_points in sorted_uuids[:-2]:
        ids_to_delete = [p.id for p in old_points]
        client.delete(collection_name=COLLECTION_NAME, points_selector={"points": ids_to_delete})
        print(f"[INFO] Удалён старый отчёт: {old_uuid}")

# Генерация уникальных точек
points = [
    PointStruct(
        id=f"{team}_{report_uuid}_{i}",
        vector=embeddings[i],
        payload={
            "rag_text": df.loc[i, "rag_text"],
            "name": df.loc[i, "name"],
            "status": df.loc[i, "status"],
            "suite": df.loc[i, "suite"],
            "uid": df.loc[i, "uid"],
            "team": team,
            "report_uuid": report_uuid
        }
    )
    for i in range(len(df))
]

# Загрузка
client.upsert(collection_name=COLLECTION_NAME, points=points)
print(f"[OK] Загружено {len(points)} точек для команды '{team}' и UUID {report_uuid}")
