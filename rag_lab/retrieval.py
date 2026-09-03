"""
Stage 5: Retrieval.

Two independent strategies over the same chunks, kept side by side on
purpose so they can be compared directly in evaluation.py:

- tfidf_retrieve: keyword-overlap retrieval (sklearn TF-IDF + cosine).
  Fast, needs no model, but misses a query that shares no words with
  the answer (e.g. "personal time off" vs. "Casual Leave").
- SemanticRetriever: embedding-based retrieval (sentence-transformers +
  FAISS). Slower and needs a model, but retrieves on meaning, not
  just word overlap.
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from rag_lab.embeddings import embed_texts, embedding_dim
from rag_lab.vector_store import VectorStore


def tfidf_retrieve(chunks: list[dict], query: str, top_k: int = 5) -> list[tuple[dict, float]]:
    if not chunks:
        return []

    corpus = [c["content"] for c in chunks]
    vectorizer = TfidfVectorizer(stop_words="english")
    try:
        matrix = vectorizer.fit_transform(corpus + [query])
    except ValueError:
        return []

    scores = cosine_similarity(matrix[-1], matrix[:-1]).flatten()
    ranked = sorted(zip(chunks, scores), key=lambda pair: pair[1], reverse=True)[:top_k]
    return [(chunk, float(score)) for chunk, score in ranked]


class SemanticRetriever:
    """Embeds all chunks once at construction time, then retrieves by cosine similarity via FAISS."""

    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        self.store = VectorStore(embedding_dim())
        if chunks:
            embeddings = embed_texts([c["content"] for c in chunks])
            self.store.add(embeddings, chunks)

    def retrieve(self, query: str, top_k: int = 5) -> list[tuple[dict, float]]:
        if not self.chunks:
            return []
        query_embedding = embed_texts([query])[0]
        return self.store.search(query_embedding, k=top_k)


if __name__ == "__main__":
    # Run directly to compare the two retrievers on one query:
    # python -m rag_lab.retrieval
    from rag_lab.chunking import chunk_documents
    from rag_lab.ingestion import load_documents

    query = "How much personal time off can I take?"  # deliberately avoids the word "Casual"
    docs = load_documents("rag_lab/documents")
    chunks = chunk_documents(docs)

    print(f"Query: {query!r}\n")

    print("TF-IDF results:")
    for chunk, score in tfidf_retrieve(chunks, query, top_k=3):
        print(f"  {score:.3f}  [{chunk['title']}] {chunk['content'][:70]}...")

    print("\nSemantic results:")
    retriever = SemanticRetriever(chunks)
    for chunk, score in retriever.retrieve(query, top_k=3):
        print(f"  {score:.3f}  [{chunk['title']}] {chunk['content'][:70]}...")
