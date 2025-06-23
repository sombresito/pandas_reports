from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd

# Загружаем модель (если уже скачана, путь можно указать локальный)
model = SentenceTransformer("local_models/intfloat/multilingual-e5-small")

df = pd.read_json("output_chunks.jsonl", lines=True)

# Получаем список текстов для эмбеддинга
texts = df["rag_text"].tolist()

# Генерируем эмбеддинги
embeddings = model.encode(texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True)

# Проверка формы
print(f"[INFO] Сгенерировано эмбеддингов: {embeddings.shape}")

np.save("embeddings.npy", embeddings)
