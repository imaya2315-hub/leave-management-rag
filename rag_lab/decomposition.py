"""
Stage 6: Query decomposition.

One retrieval call against one embedding works well for a single-fact
question. A compound question ("what is X, how many days, and are
weekends counted") blurs into one vector that isn't close to any single
answer chunk. Splitting it into independent, standalone sub-questions
lets each be retrieved on its own and the results merged.

Uses an LLM when one is configured (see generation.get_llm_client);
otherwise falls back to a punctuation/conjunction heuristic split, so
decomposition can still be demonstrated offline — just less reliably
than the LLM path.
"""
import re

from rag_lab.generation import MODEL_NAME, get_llm_client

_CONJUNCTION_SPLIT = re.compile(
    r"\?|;| and (?=how|what|are|is|do|does|can|will)", re.IGNORECASE
)


def decompose_query(question: str) -> list[str]:
    kind, client = get_llm_client()
    if client is not None:
        return _decompose_with_llm(question, kind, client)
    return _decompose_heuristic(question)


def _decompose_with_llm(question: str, kind: str, client) -> list[str]:
    prompt = (
        "Split the following question into the minimal list of independent, "
        "standalone sub-questions it contains. Each sub-question must be "
        "understandable on its own, with no pronouns referring back to the "
        "original question. If it is already a single question, return it "
        "unchanged. Reply with exactly one sub-question per line, no "
        "numbering, no extra commentary.\n\n"
        f"Question: {question}"
    )

    if kind == "groq":
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
        )
        text = response.choices[0].message.content
    else:  # anthropic
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(block.text for block in response.content if block.type == "text")

    return _clean_sub_questions(text, fallback=question)


def _clean_sub_questions(text: str, fallback: str) -> list[str]:
    """
    Deterministic cleanup of the LLM's raw output: strips numbering/
    bullets, drops blank lines, and removes near-duplicate lines
    (case/whitespace/trailing-punctuation-insensitive) while
    preserving first-seen order.
    """
    seen = set()
    sub_questions = []
    for line in text.splitlines():
        cleaned = line.strip().lstrip("-•*0123456789. )").strip()
        if not cleaned:
            continue
        dedupe_key = cleaned.lower().rstrip("?. ")
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        sub_questions.append(cleaned)
    return sub_questions or [fallback]


def _decompose_heuristic(question: str) -> list[str]:
    """No LLM configured: rough split on '?', ';', and 'and' before a question word."""
    parts = [p.strip() for p in _CONJUNCTION_SPLIT.split(question) if p.strip()]
    return parts if len(parts) > 1 else [question]


if __name__ == "__main__":
    # Run directly to see a compound question get split:
    # python -m rag_lab.decomposition
    question = "What is Casual Leave, how many days can I take, and are weekends counted?"
    for i, sub_q in enumerate(decompose_query(question), start=1):
        print(f"Q{i} -> {sub_q}")
