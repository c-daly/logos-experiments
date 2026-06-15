"""Candidate embedding models + a cached embed() for the bake-off (lx39, #39).

OpenAI arms hit the API (OPENAI_API_KEY, exported); sentence-transformers arms
run locally on CPU. Vectors are cached to .cache/<name>.npy so re-scoring is
free and models are only embedded once.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

CACHE = Path(__file__).resolve().parent / ".cache"

# (name, kind, model_id, dim, cost_note)
MODELS = [
    ("openai-3-large", "openai", "text-embedding-3-large", 3072, "API $0.13/Mtok"),
    ("openai-3-small", "openai", "text-embedding-3-small", 1536, "API $0.02/Mtok"),
    ("st-minilm-l6", "st", "sentence-transformers/all-MiniLM-L6-v2", 384, "local/free"),
    ("st-mpnet-base", "st", "sentence-transformers/all-mpnet-base-v2", 768, "local/free"),
    ("st-bge-base", "st", "BAAI/bge-base-en-v1.5", 768, "local/free"),
]


def _openai(texts: list[str], model: str, dim: int, chunk: int = 1000) -> np.ndarray:
    import httpx

    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise SystemExit("OPENAI_API_KEY is not set (needed for the openai arms)")
    out: list[list[float]] = []
    for i in range(0, len(texts), chunk):
        batch = texts[i : i + chunk]
        resp = httpx.post(
            "https://api.openai.com/v1/embeddings",
            json={"model": model, "input": batch, "dimensions": dim},
            headers={"Authorization": f"Bearer {key}"},
            timeout=180,
        )
        resp.raise_for_status()
        out.extend(
            it["embedding"] for it in sorted(resp.json()["data"], key=lambda x: x["index"])
        )
    return np.asarray(out, dtype="float32")


def _st(texts: list[str], model: str) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    enc = SentenceTransformer(model)
    return enc.encode(
        texts, batch_size=64, show_progress_bar=False, normalize_embeddings=False
    ).astype("float32")


def _fingerprint(texts: list[str]) -> str:
    """Content hash of the exact text list, so a regenerated sample with the
    same node count but different nodes invalidates the cache (otherwise stale
    vectors would be silently scored against new labels)."""
    h = hashlib.sha1("\n".join(texts).encode("utf-8")).hexdigest()
    return f"{len(texts)}:{h}"


def embed(name: str, kind: str, model: str, dim: int, texts: list[str]) -> np.ndarray:
    CACHE.mkdir(exist_ok=True)
    path = CACHE / f"{name}.npy"
    meta = CACHE / f"{name}.meta"
    fp = _fingerprint(texts)
    if path.exists() and meta.exists():
        try:
            if meta.read_text() == fp:
                cached = np.load(path)
                if cached.shape[0] == len(texts):
                    return cached
        except Exception:
            pass  # corrupt/mismatched cache -> re-embed
    vecs = _openai(texts, model, dim) if kind == "openai" else _st(texts, model)
    np.save(path, vecs)
    meta.write_text(fp)
    return vecs
