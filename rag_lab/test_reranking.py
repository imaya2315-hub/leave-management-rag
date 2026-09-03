"""
Regression test for the reranking stage: reranking should never make
retrieval quality *worse* than semantic-only, on the shared evaluation
set in evaluation.py.

This intentionally does NOT hardcode specific percentages (e.g.
"91.7%" or "100%") — those numbers depend on which embedding/
cross-encoder models are actually loaded (see rag_lab/embeddings.py
and rag_lab/reranking.py's offline fallbacks), which varies by
environment and network access. The invariant that should always hold
regardless of which models are loaded is: reranking >= semantic-only.

Run with: pytest rag_lab/test_reranking.py -v
"""
from rag_lab.evaluation import evaluate


def test_reranking_meets_or_beats_semantic_top1():
    results = evaluate()
    assert results["reranked_top1"] >= results["semantic_top1"], (
        f"Reranked Top-1 ({results['reranked_top1']:.1%}) should be >= "
        f"Semantic Top-1 ({results['semantic_top1']:.1%})"
    )


def test_reranking_meets_or_beats_semantic_mrr():
    results = evaluate()
    assert results["reranked_mrr"] >= results["semantic_mrr"], (
        f"Reranked MRR ({results['reranked_mrr']:.3f}) should be >= "
        f"Semantic MRR ({results['semantic_mrr']:.3f})"
    )


def test_evaluation_set_is_fully_covered():
    """Sanity check: every question in the eval set should retrieve at least one chunk."""
    from rag_lab.chunking import chunk_documents
    from rag_lab.evaluation import EVAL_SET
    from rag_lab.ingestion import load_documents
    from rag_lab.retrieval import SemanticRetriever

    documents = load_documents("rag_lab/documents")
    chunks = chunk_documents(documents)
    retriever = SemanticRetriever(chunks)

    for item in EVAL_SET:
        results = retriever.retrieve(item["question"], top_k=5)
        assert results, f"No chunks retrieved at all for: {item['question']!r}"
