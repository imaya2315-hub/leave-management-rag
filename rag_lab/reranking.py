"""
Stage 7: Reranking.

The retriever's bi-encoder similarity (query embedded once, chunks
embedded once, compared by a single dot product) is fast but
approximate. A cross-encoder scores each (query, chunk) pair jointly
through one forward pass, which is far more accurate — but too slow
to run over an entire corpus. The standard pattern is: retrieve a
wide net of candidates cheaply, then rerank only those with the
expensive, accurate model.

    Retriever -> top 10 candidates -> Cross-encoder -> best 3

Falls back to keeping the retriever's own ordering if the
cross-encoder can't be loaded (e.g. no network to Hugging Face), so
the pipeline stays runnable offline — reranking then becomes a no-op
rather than a failure.
"""
_cross_encoder = None
_load_failed = False


def _get_cross_encoder():
    global _cross_encoder, _load_failed
    if _cross_encoder is not None or _load_failed:
        return _cross_encoder
    try:
        from sentence_transformers import CrossEncoder
        _cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    except Exception:
        _load_failed = True
    return _cross_encoder


def using_cross_encoder() -> bool:
    return _get_cross_encoder() is not None


def rerank(
    query: str, candidates: list[tuple[dict, float]], top_n: int = 3
) -> list[tuple[dict, float]]:
    """candidates: [(chunk_dict, retriever_score), ...] -> reranked [(chunk_dict, cross_encoder_score), ...]."""
    if not candidates:
        return []

    encoder = _get_cross_encoder()
    if encoder is None:
        return candidates[:top_n]

    pairs = [(query, chunk["content"]) for chunk, _ in candidates]
    scores = encoder.predict(pairs)
    reranked = sorted(
        zip([chunk for chunk, _ in candidates], scores), key=lambda pair: pair[1], reverse=True
    )
    return [(chunk, float(score)) for chunk, score in reranked[:top_n]]


if __name__ == "__main__":
    # Run directly to compare pre- and post-rerank ordering:
    # python -m rag_lab.reranking
    from rag_lab.chunking import chunk_documents
    from rag_lab.ingestion import load_documents
    from rag_lab.retrieval import SemanticRetriever

    query = "Can I cancel my leave after it's approved?"
    docs = load_documents("rag_lab/documents")
    chunks = chunk_documents(docs)
    retriever = SemanticRetriever(chunks)

    top_10 = retriever.retrieve(query, top_k=10)
    print("Before reranking (top 3 of 10):")
    for chunk, score in top_10[:3]:
        print(f"  {score:.3f}  [{chunk['title']}] {chunk['content'][:70]}...")

    reranked = rerank(query, top_10, top_n=3)
    print("\nAfter reranking:")
    for chunk, score in reranked:
        print(f"  {score:.3f}  [{chunk['title']}] {chunk['content'][:70]}...")
