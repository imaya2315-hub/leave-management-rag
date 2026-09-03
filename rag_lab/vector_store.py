"""
Stage 4: Vector database.

Stores chunk embeddings in a FAISS index and retrieves the closest
ones to a query embedding by similarity search. Metadata (which
document/chunk each vector came from) is kept in a parallel Python
list, since FAISS itself only stores vectors.
"""
import pickle
from pathlib import Path

import faiss
import numpy as np


class VectorStore:
    def __init__(self, dim: int):
        self.dim = dim
        # Inner product on normalized vectors == cosine similarity.
        self.index = faiss.IndexFlatIP(dim)
        self.metadata: list[dict] = []  # metadata[i] describes the vector at index row i

    def add(self, embeddings: np.ndarray, metadata: list[dict]) -> None:
        if len(metadata) != embeddings.shape[0]:
            raise ValueError("One metadata entry is required per embedding row")
        self.index.add(embeddings)
        self.metadata.extend(metadata)

    def search(self, query_embedding: np.ndarray, k: int = 5) -> list[tuple[dict, float]]:
        query_embedding = query_embedding.reshape(1, -1)
        scores, indices = self.index.search(query_embedding, k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append((self.metadata[idx], float(score)))
        return results

    def save(self, path: str) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(Path(path) / "index.faiss"))
        with open(Path(path) / "metadata.pkl", "wb") as f:
            pickle.dump(self.metadata, f)

    @classmethod
    def load(cls, path: str, dim: int) -> "VectorStore":
        store = cls(dim)
        store.index = faiss.read_index(str(Path(path) / "index.faiss"))
        with open(Path(path) / "metadata.pkl", "rb") as f:
            store.metadata = pickle.load(f)
        return store


if __name__ == "__main__":
    # Run directly for a minimal, visible demo of the mechanics:
    # python -m rag_lab.vector_store
    from rag_lab.chunking import chunk_documents
    from rag_lab.embeddings import embed_texts, embedding_dim
    from rag_lab.ingestion import load_documents

    docs = load_documents("rag_lab/documents")
    chunks = chunk_documents(docs)

    store = VectorStore(embedding_dim())
    store.add(embed_texts([c["content"] for c in chunks]), chunks)

    query_embedding = embed_texts(["How many casual leave days do I get?"])[0]
    for chunk, score in store.search(query_embedding, k=3):
        print(f"{score:.3f}  [{chunk['title']}] {chunk['content'][:80]}...")
