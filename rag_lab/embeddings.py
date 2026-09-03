"""
Stage 3: Embeddings.

Turns text into a vector that represents its semantic meaning, so
"personal time off" and "casual leave" end up close together in
vector space even though they share no words.

Uses sentence-transformers/all-MiniLM-L6-v2 (384-dim) when available.
The first call downloads the model from Hugging Face — that needs
network access once; after that it's cached locally by the library.

Falls back to a deterministic hashing-based bag-of-words vector when
sentence-transformers or network access isn't available, purely so
the rest of the pipeline (chunking, FAISS, retrieval, reranking,
evaluation) can still be run and developed offline. This fallback
captures crude word-overlap only, NOT semantic meaning — swap it out
for the real model before drawing any conclusion from the semantic-
vs-TF-IDF comparison in evaluation.py.
"""
import hashlib

import numpy as np

_FALLBACK_DIM = 384
_model = None
_model_load_failed = False


def _get_model():
    global _model, _model_load_failed
    if _model is not None or _model_load_failed:
        return _model
    try:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    except Exception:
        _model_load_failed = True
        _model = None
    return _model


def embed_texts(texts: list[str]) -> np.ndarray:
    model = _get_model()
    if model is not None:
        return np.asarray(model.encode(texts, normalize_embeddings=True), dtype="float32")
    return np.asarray([_hash_embedding(t) for t in texts], dtype="float32")


def _hash_embedding(text: str, dim: int = _FALLBACK_DIM) -> np.ndarray:
    vector = np.zeros(dim, dtype="float32")
    for word in text.lower().split():
        bucket = int(hashlib.md5(word.encode()).hexdigest(), 16) % dim
        vector[bucket] += 1.0
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 0 else vector


def embedding_dim() -> int:
    model = _get_model()
    return model.get_sentence_embedding_dimension() if model is not None else _FALLBACK_DIM


def using_real_model() -> bool:
    return _get_model() is not None


if __name__ == "__main__":
    # Run directly to inspect an actual embedding vector:
    # python -m rag_lab.embeddings
    text = "Casual Leave entitlement"
    vector = embed_texts([text])[0]
    print(f"Using real sentence-transformer model: {using_real_model()}")
    print(f"'{text}' -> vector of dim {vector.shape[0]}")
    print(vector[:8], "...")
