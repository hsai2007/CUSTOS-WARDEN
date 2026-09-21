"""Shared sentence-embedding utility (sentence-transformers), used by the
M2 rule-5 proof and the M3 classifier's cosine fast path."""
from __future__ import annotations

import numpy as np

_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(_MODEL_NAME)
    return _model


def embed(texts: list[str]) -> np.ndarray:
    model = _get_model()
    return np.asarray(model.encode(list(texts), normalize_embeddings=True))


def cosine_sim(a: str, b: str) -> float:
    vecs = embed([a, b])
    return float(np.dot(vecs[0], vecs[1]))
