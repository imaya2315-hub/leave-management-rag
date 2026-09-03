"""
Stage 8: Generation (and the shared LLM client used by decomposition too).

Takes retrieved context + the user's question and produces one
concise, grounded answer — grounded meaning the model is instructed
to answer only from what was retrieved, not from its own training
data, which is what prevents hallucination. This module never dumps
the raw retrieved chunks back to the user; callers show sources
separately (see rag_lab/agent.py).

Prefers Groq (GROQ_API_KEY), then Anthropic (ANTHROPIC_API_KEY), then
falls back to a short extractive answer, so the whole lab runs end to
end with zero API keys.
"""
import os
import re
MODEL_NAME = "openai/gpt-oss-20b"

NO_CONTEXT_ANSWER = "The provided policy context does not contain enough information to answer that question."

SYSTEM_PROMPT = (
    "You are a leave-policy assistant. Answer ONLY using the provided "
    "context, in 1-3 short sentences — never repeat or quote the context "
    "verbatim, summarize it into a direct answer instead. If the context "
    "doesn't contain the answer, reply with exactly this sentence and "
    "nothing else: \"" + NO_CONTEXT_ANSWER + "\" "
    "Never guess or invent a policy detail that isn't in the context."
)


def get_llm_client():
    """
    Returns (kind, client): kind is 'groq' or 'anthropic', or
    (None, None) if no provider is configured. decomposition.py
    depends on this exact (kind, client) tuple shape — don't change it
    without updating that caller too.
    """
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        try:
            from groq import Groq
            return "groq", Groq(api_key=groq_key)
        except ImportError:
            pass

    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    if anthropic_key:
        try:
            import anthropic
            return "anthropic", anthropic.Anthropic(api_key=anthropic_key)
        except ImportError:
            pass

    return None, None


def generate_answer(question: str, context: str) -> str:
    if not context:
        return NO_CONTEXT_ANSWER

    kind, client = get_llm_client()
    if client is None:
        return _fallback_answer(context, question)

    user_content = f"Question: {question}\n\nContext:\n{context}"

    if kind == "groq":
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            max_tokens=200,
        )
        return response.choices[0].message.content.strip()

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=200,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_content}],
    )
    return "".join(block.text for block in response.content if block.type == "text").strip()


def _fallback_answer(context: str, question: str = "") -> str:
    """
    No LLM configured: select the sentence most relevant to the question
    instead of blindly returning the first sentence of the first chunk.
    """
    if not context:
        return NO_CONTEXT_ANSWER

    clean_context = re.sub(r"(?m)^\[[^\]]+\]\s*", "", context)

    sentences = [
        s.strip()
        for s in re.split(
            r"(?<=[.!?])\s+",
            clean_context.replace("\n", " "),
        )
        if s.strip()
    ]

    if not sentences:
        return NO_CONTEXT_ANSWER

    question_lower = question.lower()
    words = re.findall(r"\b[a-zA-Z]{3,}\b", question_lower)

    stop_words = {
        "what", "is", "are", "the", "how", "many", "much", "can", "i",
        "do", "does", "did", "my", "me", "to", "for", "of", "and", "or",
        "in", "on", "a", "an", "this", "that", "when", "where", "who",
        "why", "should", "would", "could", "will", "be", "take", "days"
    }

    keywords = [w for w in words if w not in stop_words]

    def sentence_score(sentence: str) -> int:
        lower = sentence.lower()
        score = sum(2 for word in keywords if word in lower)

        if "casual leave" in question_lower and "casual leave" in lower:
            score += 6
        if "annual leave" in question_lower and "annual leave" in lower:
            score += 6
        if "sick leave" in question_lower and "sick leave" in lower:
            score += 6

        if (
            ("how many" in question_lower or "how much" in question_lower)
            and re.search(r"\b\d+\b", sentence)
        ):
            score += 5

        if "carry forward" in question_lower and "carry forward" in lower:
            score += 8
        if "eligible" in question_lower and "eligible" in lower:
            score += 6
        if "weekend" in question_lower and "weekend" in lower:
            score += 6
        if "approval" in question_lower and "approval" in lower:
            score += 5

        return score

    ranked = sorted(sentences, key=sentence_score, reverse=True)
    best = ranked[0]

    if sentence_score(best) == 0:
        return NO_CONTEXT_ANSWER

    return best if best.endswith((".", "?", "!")) else best + "."


if __name__ == "__main__":
    # Run directly to see grounding + hallucination refusal in action:
    # python -m rag_lab.generation
    print("With relevant context:")
    print(generate_answer(
        "How many casual leave days do I get?",
        "[Leave Types And Eligibility] Every employee is entitled to 5 Casual Leave days per calendar year.",
    ))

    print("\nWith no context (should refuse, not invent an answer):")
    print(generate_answer("What is the capital of France?", ""))
