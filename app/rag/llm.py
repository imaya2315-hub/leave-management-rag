"""
Generates the assistant's final, natural-language answer from
already-retrieved context. This module never decides WHAT is true —
it only phrases what `retriever.py` (policy text) and `leave_service`
(live balances/eligibility) already determined. That separation is
what keeps the assistant from hallucinating a balance or a policy
figure: the numbers it's allowed to mention are handed to it, not
recalled from the model's own memory.

If ANTHROPIC_API_KEY is configured, real generation is used for a
fluent answer. Otherwise, falls back to a deterministic, template-based
composition of the same context — so the assistant is fully usable
with zero external dependencies, API keys, or cost.
"""
from app.core.config import settings

_client = None
if settings.ANTHROPIC_API_KEY:
    try:
        import anthropic
        _client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)
    except ImportError:
        # anthropic package not installed — fall back silently.
        _client = None


SYSTEM_PROMPT = (
    "You are the Leave Management Assistant for this company. Answer the "
    "employee's question using ONLY the policy context and employee data "
    "provided below — never invent a leave balance, policy figure, or "
    "approval status that isn't in the provided context. Be concise "
    "(2-4 sentences). If the context doesn't contain enough information "
    "to answer, say so plainly and suggest they check with HR instead of "
    "guessing."
)


def generate_answer(question: str, policy_context: str, employee_context: str) -> str:
    if _client is not None:
        return _generate_with_anthropic(question, policy_context, employee_context)
    return _fallback_answer(policy_context, employee_context)


def _generate_with_anthropic(question: str, policy_context: str, employee_context: str) -> str:
    message = _client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=400,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": (
                    f"Employee question: {question}\n\n"
                    f"Relevant policy context:\n{policy_context or '(no matching policy found)'}\n\n"
                    f"Employee's current data:\n{employee_context or '(not applicable to this question)'}"
                ),
            }
        ],
    )
    text_blocks = [block.text for block in message.content if block.type == "text"]
    return "".join(text_blocks).strip()


def _fallback_answer(policy_context: str, employee_context: str) -> str:
    """
    Extractive fallback: composes the retrieved context directly,
    without an LLM. Less fluent than a generated answer, but every
    word in it is traceable to a source, and it works with no API key.
    """
    parts = []
    if policy_context:
        parts.append(f"Based on the applicable policy:\n{policy_context}")
    if employee_context:
        parts.append(f"Your current status:\n{employee_context}")
    if not parts:
        return (
            "I couldn't find a policy that covers this question. Please "
            "check with HR or try rephrasing your question."
        )
    return "\n\n".join(parts)
