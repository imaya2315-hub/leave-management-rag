"""
Policy RAG orchestration.

Pipeline:

    User question
        -> Query decomposition
        -> Semantic retrieval per sub-question
        -> Cross-encoder reranking per sub-question
        -> Relevance filtering
        -> Grounded generation per sub-question
        -> Combined answer + sources

This module handles policy questions only.
It does not access FastAPI, employee data, or leave actions.
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

RELEVANCE_THRESHOLD_CROSS_ENCODER = 0.0
RELEVANCE_THRESHOLD_COSINE = 0.2

NO_CONTEXT_ANSWER = (
    "The provided policy context does not contain enough "
    "information to answer that question."
)

_retriever = None


def get_retriever() -> SemanticRetriever:
    """
    Lazily load documents and build the semantic retriever once.
    """
    global _retriever

    if _retriever is None:
        documents = load_documents(DOCUMENTS_FOLDER)
        chunks = chunk_documents(documents)
        _retriever = SemanticRetriever(chunks)

    return _retriever


def _is_relevant(score: float) -> bool:
    """
    Apply the appropriate relevance threshold.
    """

    if using_cross_encoder():
        return score >= RELEVANCE_THRESHOLD_CROSS_ENCODER

    return score >= RELEVANCE_THRESHOLD_COSINE


def _fallback_split_question(question: str) -> list[str]:
    """
    Deterministic fallback when LLM decomposition does not correctly
    separate multiple questions.

    Example:

        A? B? C?

    becomes:

        A?
        B?
        C?
    """

    parts = []

    for part in question.split("?"):
        cleaned = part.strip()

        if cleaned:
            parts.append(cleaned + "?")

    return parts


def _get_sub_questions(question: str) -> list[str]:
    """
    Decompose the query.

    The LLM decomposition is used first. If it produces only one
    question for a clearly multi-question input, fall back to
    deterministic question-mark splitting.
    """

    try:
        sub_questions = decompose_query(question)
    except Exception:
        sub_questions = []

    if sub_questions:
        cleaned = [
            str(q).strip()
            for q in sub_questions
            if q and str(q).strip()
        ]
    else:
        cleaned = []

    # Deterministic safety fallback.
    fallback = _fallback_split_question(question)

    if len(fallback) > 1 and len(cleaned) <= 1:
        return fallback

    return cleaned or [question]


def answer_policy_question(
    question: str,
) -> tuple[str, list[dict]]:
    """
    Answer one or more policy questions independently.

    Every sub-question gets:
        retrieval -> reranking -> relevance check -> generation

    This prevents:
        - one unsupported question from causing a global refusal
        - one question consuming another question's context
        - multi-question output being truncated after the first answer
    """

    question = (question or "").strip()

    if not question:
        return (
            "I couldn't identify a question. Please try again.",
            [],
        )

    retriever = get_retriever()

    # ------------------------------------------------------------
    # 1. Get independent questions
    # ------------------------------------------------------------
    sub_questions = _get_sub_questions(question)

    answers = []
    sources = []
    seen_sources = set()

    # ------------------------------------------------------------
    # 2. Process every question independently
    # ------------------------------------------------------------
    for index, sub_question in enumerate(
        sub_questions,
        start=1,
    ):

        # --------------------------------------------------------
        # Semantic retrieval
        # --------------------------------------------------------
        retrieved = retriever.retrieve(
            sub_question,
            top_k=RETRIEVE_PER_SUBQUESTION,
        )

        # --------------------------------------------------------
        # No candidates
        # --------------------------------------------------------
        if not retrieved:
            answer = NO_CONTEXT_ANSWER

            answers.append(
                f"{index}. {answer}"
            )

            continue

        # --------------------------------------------------------
        # Cross-encoder reranking
        # --------------------------------------------------------
        reranked = rerank(
            sub_question,
            retrieved,
            top_n=RERANK_TOP_N,
        )

        # --------------------------------------------------------
        # Relevance filtering
        # --------------------------------------------------------
        relevant_chunks = []

        for chunk, score in reranked:

            if not _is_relevant(score):
                continue

            relevant_chunks.append(
                (chunk, score)
            )

        # --------------------------------------------------------
        # No relevant policy evidence
        # --------------------------------------------------------
        if not relevant_chunks:
            answer = NO_CONTEXT_ANSWER

            answers.append(
                f"{index}. {answer}"
            )

            continue

        # --------------------------------------------------------
        # Build context for THIS question only
        # --------------------------------------------------------
        context_parts = []

        for chunk, score in relevant_chunks:

            context_parts.append(
                f"[{chunk['title']}]\n"
                f"{chunk['content']}"
            )

            source_key = (
                chunk["title"],
                chunk["chunk_index"],
            )

            if source_key not in seen_sources:
                seen_sources.add(source_key)
                sources.append(chunk)

        context = "\n\n".join(context_parts)

        # --------------------------------------------------------
        # Generate THIS answer only
        # --------------------------------------------------------
        answer = generate_answer(
            sub_question,
            context,
        )

        if not answer or not answer.strip():
            answer = NO_CONTEXT_ANSWER

        answers.append(
            f"{index}. {answer.strip()}"
        )

    # ------------------------------------------------------------
    # 3. Combine independently generated answers
    # ------------------------------------------------------------
    final_answer = "\n\n".join(answers)

    return final_answer, sources


def _merge_and_dedupe(
    retriever: SemanticRetriever,
    sub_questions: list[str],
) -> list[tuple[dict, float]]:
    """
    Retrieve candidates for every sub-question and merge duplicates.

    Kept for compatibility with existing tests/code.
    """

    best_by_key: dict[
        tuple[str, int],
        tuple[dict, float],
    ] = {}

    for sub_question in sub_questions:

        sub_question = (
            sub_question or ""
        ).strip()

        if not sub_question:
            continue

        candidates = retriever.retrieve(
            sub_question,
            top_k=RETRIEVE_PER_SUBQUESTION,
        )

        for chunk, score in candidates:

            key = (
                chunk["title"],
                chunk["chunk_index"],
            )

            if (
                key not in best_by_key
                or score > best_by_key[key][1]
            ):
                best_by_key[key] = (
                    chunk,
                    score,
                )

    return sorted(
        best_by_key.values(),
        key=lambda pair: pair[1],
        reverse=True,
    )