"""Generate embeddings for Allure chunk files."""

from __future__ import annotations

from pathlib import Path
import os

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# Paths can be overridden via environment variables
MODEL_PATH = os.getenv("MODEL_PATH", "local_models/intfloat/multilingual-e5-small")
CHUNKS_PATH = os.getenv("CHUNKS_PATH", "chunks")
EMBEDDINGS_PATH = os.getenv("EMBEDDINGS_PATH", "embeddings.npy")


def load_chunks(path: str | os.PathLike) -> pd.DataFrame:
    """Return a DataFrame with chunk data from ``path``.

    ``path`` may point to a single ``.jsonl`` file or a directory containing
    multiple chunk files.
    """
    p = Path(path)
    if p.is_dir():
        files = sorted(p.glob("**/*.jsonl"))
        if not files:
            raise FileNotFoundError(f"No .jsonl files found in {path}")
        dfs = [pd.read_json(f, lines=True) for f in files]
        return pd.concat(dfs, ignore_index=True)
    return pd.read_json(p, lines=True)


def create_embeddings(df: pd.DataFrame, model_path: str = MODEL_PATH) -> np.ndarray:
    """Generate embeddings for the ``rag_text`` column in ``df``."""
    model = SentenceTransformer(model_path)
    texts = df["rag_text"].tolist()
    return model.encode(
        texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True
    )


if __name__ == "__main__":
    df = load_chunks(CHUNKS_PATH)
    embeddings = create_embeddings(df)
    print(f"[INFO] Сгенерировано эмбеддингов: {embeddings.shape}")
    np.save(EMBEDDINGS_PATH, embeddings)
