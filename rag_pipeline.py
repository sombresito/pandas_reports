import requests
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, Filter, FieldCondition, Match
import json

# ==== Настройки ====
MODEL_PATH = "local_models/intfloat/multilingual-e5-small"
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "allure_chunks"
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen3:0.6b"


# ==== Инициализация ====
model = SentenceTransformer(MODEL_PATH)
client = QdrantClient(url=QDRANT_URL)


# ==== Поиск релевантных векторов ====
def search_similar_chunks(query: str, top_k: int = 5):
    query_embedding = model.encode(query, convert_to_numpy=True).tolist()
    results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_embedding,
        limit=top_k,
    )
    return [hit.payload["rag_text"] for hit in results]


# ==== Генерация ответа через Ollama ====
def generate_answer_with_ollama(chunks, question):
    context = "\n\n".join(chunks)
    prompt = f"Вот информация из отчёта:\n{context}\n\nВопрос: {question}\nОтвет:"
    
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": OLLAMA_MODEL,   # или другая твоя модель
            "prompt": prompt,
            "stream": True
        },
        stream=True,
        timeout=300
    )

    answer = ""
    for line in response.iter_lines():
        if line:
            data = json.loads(line)
            answer += data.get("response", "")

    return answer.strip()


def run_rag_analysis(team_name: str) -> dict:
    """Generate a short analysis for a team's latest report using RAG."""
    search_filter = Filter(
        must=[FieldCondition(key="team", match=Match(value=team_name))]
    )
    points, _ = client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=search_filter,
        with_payload=True,
        limit=1000,
    )
    chunks = [p.payload.get("rag_text", "") for p in points]

    question = "Кратко проанализируй результаты отчёта."
    answer = generate_answer_with_ollama(chunks, question) if chunks else ""
    return {"team": team_name, "analysis": answer}


# ==== Основная функция ====
def ask(question: str):
    print(f"[Q] {question}")
    chunks = search_similar_chunks(question, top_k=5)
    answer = generate_answer_with_ollama(chunks, question)
    print(f"\n[A] {answer}")


# ==== Пример запуска ====
if __name__ == "__main__":
    ask("Какие тесты завершились с ошибкой?")
