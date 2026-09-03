"""
Orchestrates the full policy-question pipeline:

    User question
        -> Query decomposition
        -> Semantic retrieval (per sub-question)
        -> Candidate merging / deduplication
        -> Cross-encoder reranking
        -> Grounded context
        -> Groq LLM
        -> Concise answer + sources

This is the single entry point Streamlit (or anything else) should
call for a policy question — it owns the RAG pipeline's internal
wiring so callers don't have to know about decomposition, retrieval,
or reranking individually.

This module has NO knowledge of the FastAPI backend, live employee
data, or leave actions — see streamlit_app.py's route_message/
handle_action/handle_live_data for that split. Keeping this boundary
is what keeps the LLM from ever being able to perform a backend
action itself.
"""
from rag_lab.chunking import chunk_documents
from rag_lab.decomposition import decompose_query
from rag_lab.generation import generate_answer
from rag_lab.ingestion import load_documents
from rag_lab.reranking import rerank, using_cross_encoder
from rag_lab.retrieval import SemanticRetriever

DOCUMENTS_FOLDER = "rag_lab/documents"
RETRIEVE_PER_SUBQUESTION = 10
RERANK_TOP_N = 3

# FAISS/cosine search always returns its k nearest neighbors, even
# when none of them are actually relevant — there's no "no match"
# result. These cutoffs are what turn "nearest anyway" into "not
# relevant enough to answer from," which is what makes the
# work-from-home-style refusal actually fire instead of always
# grounding on whatever was closest.
#
# The two are on different scales: a cross-encoder's raw score is an
# unbounded logit (roughly centered on 0 for "relevant" vs "not"),
# while the fallback path (no cross-encoder loaded) uses the
# retriever's own cosine similarity, so it needs a different cutoff.
# Both are approximate — tune them against your real embedding model
# if refusals are firing on relevant questions or not firing on
# irrelevant ones.
RELEVANCE_THRESHOLD_CROSS_ENCODER = 0.0
RELEVANCE_THRESHOLD_COSINE = 0.2

_retriever = None


def get_retriever() -> SemanticRetriever:
    """Lazily builds the retriever once and reuses it across calls."""
    global _retriever
    if _retriever is None:
        documents = load_documents(DOCUMENTS_FOLDER)
        chunks = chunk_documents(documents)
        _retriever = SemanticRetriever(chunks)
    return _retriever


def answer_policy_question(question: str) -> tuple[str, list[str]]:
    """
    Returns (answer, source_titles). source_titles is deduplicated and
    sorted for stable display; it's empty when nothing relevant was
    retrieved, which generate_answer turns into the standard
    "not enough information" refusal.
    """
    retriever = get_retriever()
    sub_questions = decompose_query(question)

    candidates = _merge_and_dedupe(retriever, sub_questions)
    top_chunks = rerank(question, candidates, top_n=RERANK_TOP_N)

    threshold = RELEVANCE_THRESHOLD_CROSS_ENCODER if using_cross_encoder() else RELEVANCE_THRESHOLD_COSINE
    relevant_chunks = [(chunk, score) for chunk, score in top_chunks if score >= threshold]

    context = "\n\n".join(f"[{chunk['title']}] {chunk['content']}" for chunk, _ in relevant_chunks)
    answer = generate_answer(question, context)
    sources = sorted({chunk["title"] for chunk, _ in relevant_chunks})

    return answer, sources


def _merge_and_dedupe(
    retriever: SemanticRetriever, sub_questions: list[str]
) -> list[tuple[dict, float]]:
    """
    Retrieves candidates for every sub-question and merges them into
    one list, keeping each (document, chunk_index) at most once. When
    a chunk is retrieved for more than one sub-question, its highest
    score is kept — that's the strongest evidence it's actually
    relevant to the original compound question.
    """
    best_by_key: dict[tuple[str, int], tuple[dict, float]] = {}
    for sub_question in sub_questions:
        for chunk, score in retriever.retrieve(sub_question, top_k=RETRIEVE_PER_SUBQUESTION):
            key = (chunk["title"], chunk["chunk_index"])
            if key not in best_by_key or score > best_by_key[key][1]:
                best_by_key[key] = (chunk, score)

    return sorted(best_by_key.values(), key=lambda pair: pair[1], reverse=True)


if __name__ == "__main__":
    # Run directly to see the whole pipeline end to end:
    # python -m rag_lab.agent
    for question in [
        "How many Casual Leave days can I take?",
        "Can unused Casual Leave be carried forward?",
        "What is the company's work-from-home policy?",
    ]:
        answer, sources = answer_policy_question(question)
        print(f"Q: {question}")
        print(f"A: {answer}")
        print(f"Sources: {', '.join(sources) if sources else '(none)'}\n")
