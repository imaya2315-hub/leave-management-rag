"""
Lightweight retrieval over the knowledge_chunks table (the "R" in RAG).

Uses TF-IDF + cosine similarity rather than an embedding model or an
external vector database. For a company policy corpus (a handful of
documents, re-ingested rarely), that's the right amount of
infrastructure: it needs no API key, no GPU, and no extra service to
run, and it's rebuilt from the DB on every call so it's never stale.

If the knowledge base grows large enough for TF-IDF's keyword-overlap
matching to fall short of semantic queries, swap this module for an
embedding-based retriever (e.g. sentence-transformers + pgvector) —
callers only depend on `retrieve_relevant_chunks`'s signature, not on
how it's implemented.
"""
from sqlalchemy.orm import Session
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app import crud


def retrieve_relevant_chunks(
    db: Session, query: str, top_k: int = 4, min_score: float = 0.05
) -> list[tuple[str, str, float]]:
    """
    Returns up to top_k (document_title, chunk_content, score) tuples,
    most relevant first, for chunks whose similarity to the query
    clears min_score. Returns an empty list if the knowledge base is
    empty or nothing clears the threshold — callers should treat that
    as "no matching policy found," not an error.
    """
    chunks = crud.knowledge.get_all_chunks(db)
    if not chunks:
        return []

    corpus = [chunk.content for chunk in chunks]

    vectorizer = TfidfVectorizer(stop_words="english")
    try:
        matrix = vectorizer.fit_transform(corpus + [query])
    except ValueError:
        # Happens if the corpus + query contain only stopwords/symbols.
        return []

    query_vector = matrix[-1]
    chunk_vectors = matrix[:-1]
    scores = cosine_similarity(query_vector, chunk_vectors).flatten()

    ranked = sorted(zip(chunks, scores), key=lambda pair: pair[1], reverse=True)

    results = []
    for chunk, score in ranked[:top_k]:
        if score < min_score:
            continue
        results.append((chunk.document.title, chunk.content, float(score)))

    return results
