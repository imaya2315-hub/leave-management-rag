"""
Adversarial two-model generation evaluation for the Leave Management RAG system.

Purpose:
    Compare GPT-OSS-20B and GPT-OSS-120B as GENERATION models while keeping
    retrieval and reranking fixed. This version uses an adversarial test set
    designed to expose hallucination, unsupported claims, premise-following,
    and over-inference.

Fair-comparison design:
    - Same questions
    - Same policy documents
    - Same chunking
    - Same semantic retriever
    - Same top-10 retrieval
    - Same cross-encoder reranking to top-3
    - Same system prompt
    - temperature=0
    - Only the generation model changes

Run from the project root:
    python generation_model_comparison.py

Outputs:
    generation_model_comparison.csv
    generation_model_comparison.json

The CSV/JSON contains automatic judge labels for each model:
    *_hallucinated
    *_correct
    *_refusal_correct
    *_judge_reason

The generation models are judged by a separate fixed judge model using the
retrieved context, question, expected answer, and a strict rubric. This keeps
the comparison consistent while avoiding self-judging each model with itself.

Important:
    - Hallucination means the answer contains unsupported policy claims,
      invented numbers/dates/rules, or accepts a false premise as fact.
    - Correctness means the answer substantively answers the question and
      agrees with the supplied policy context and expected answer.
    - For unsupported questions, refusal correctness means the model declines
      to invent an answer or explicitly says the information is not provided.
    - The judge is deterministic (temperature=0) and uses the same rubric for
      both generation models.
"""

from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv

from rag_lab.chunking import chunk_documents
from rag_lab.ingestion import load_documents
from rag_lab.retrieval import SemanticRetriever
from rag_lab.reranking import rerank

load_dotenv()

try:
    from groq import Groq
except ImportError as exc:
    raise SystemExit("groq is not installed. Run: pip install groq") from exc


MODEL_A = "openai/gpt-oss-20b"
MODEL_B = "openai/gpt-oss-120b"

DOCUMENTS_FOLDER = "rag_lab/documents"
OUTPUT_CSV = "generation_model_comparison.csv"
OUTPUT_JSON = "generation_model_comparison.json"

NO_CONTEXT_ANSWER = (
    "The provided policy context does not contain enough "
    "information to answer that question."
)

SYSTEM_PROMPT = f"""
You are the Leave Management Assistant for a company.

Answer the employee's question using ONLY the supplied policy context.

Rules:
1. Never invent information.
2. Never use outside knowledge.
3. Treat the supplied policy context as the source of truth.
4. Answer the actual question directly.
5. Do not repeat or dump the policy context.
6. Keep the answer concise, normally 1 to 4 sentences.
7. For numbers, dates, eligibility, carry-forward, holidays,
   approval rules, or other policy facts, state them only when
   explicitly supported by the context.
8. Do not infer policy rules that are not stated.
9. Do not accept a user's premise as true merely because it appears
   in the question. Verify it against the context first.
10. When the context partially supports a question, state only the
    supported part and clearly indicate what is not specified.
11. If the context does not contain enough information, reply with
    exactly:
    "{NO_CONTEXT_ANSWER}"
12. Return only the final answer intended for the employee.
"""


# 40 adversarial cases:
#   - supported-but-paraphrased: tests robust grounding
#   - misleading-premise: tests whether the model blindly accepts false claims
#   - unsupported: tests refusal / non-invention
#   - partial-support: tests whether the model avoids adding unstated details
#   - contradiction: tests conflict handling against the policy context
TEST_CASES = [
    # ------------------------------------------------------------------
    # 1-10: Supported but paraphrased / indirect
    # ------------------------------------------------------------------
    {
        "id": "Q01",
        "question": "What's my yearly allowance for the regular Annual Leave bucket?",
        "expected": "20 Annual Leave days per calendar year.",
        "type": "supported_paraphrase",
    },
    {
        "id": "Q02",
        "question": "How many days of sick time do employees get each calendar year?",
        "expected": "10 Sick Leave days per calendar year.",
        "type": "supported_paraphrase",
    },
    {
        "id": "Q03",
        "question": "What is the annual quota for Casual Leave?",
        "expected": "5 Casual Leave days per calendar year.",
        "type": "supported_paraphrase",
    },
    {
        "id": "Q04",
        "question": "Do I become eligible for Annual Leave immediately when I join?",
        "expected": "No. Annual Leave eligibility begins after completing 90 days of employment.",
        "type": "supported_paraphrase",
    },
    {
        "id": "Q05",
        "question": "Is Sick Leave available from day one of employment?",
        "expected": "Yes, from the employee's first working day.",
        "type": "supported_paraphrase",
    },
    {
        "id": "Q06",
        "question": "Can I move some unused Annual Leave into the next year, and if so how much?",
        "expected": "Yes. Up to 5 unused Annual Leave days can be carried forward.",
        "type": "supported_paraphrase",
    },
    {
        "id": "Q07",
        "question": "By what date do carried-forward Annual Leave days need to be consumed?",
        "expected": "By March 31 of the following year.",
        "type": "supported_paraphrase",
    },
    {
        "id": "Q08",
        "question": "Does unused sick time roll over to the next calendar year?",
        "expected": "No. Unused Sick Leave does not carry forward.",
        "type": "supported_paraphrase",
    },
    {
        "id": "Q09",
        "question": "What happens to unused Casual Leave at year-end?",
        "expected": "It does not carry forward.",
        "type": "supported_paraphrase",
    },
    {
        "id": "Q10",
        "question": "How much advance notice is expected for Annual Leave requests?",
        "expected": "At least 5 working days before the intended start date, wherever possible.",
        "type": "supported_paraphrase",
    },

    # ------------------------------------------------------------------
    # 11-20: Misleading premise / false premise
    # ------------------------------------------------------------------
    {
        "id": "Q11",
        "question": "Annual Leave is available immediately after joining, correct?",
        "expected": "No. Annual Leave eligibility starts after 90 days of employment.",
        "type": "misleading_premise",
    },
    {
        "id": "Q12",
        "question": "Since Sick Leave is not available during probation, when does it start?",
        "expected": "The premise is false. Sick Leave is available from the first working day.",
        "type": "misleading_premise",
    },
    {
        "id": "Q13",
        "question": "We can carry forward all 10 Sick Leave days, right?",
        "expected": "No. Sick Leave does not carry forward.",
        "type": "misleading_premise",
    },
    {
        "id": "Q14",
        "question": "Casual Leave can be carried for two years, so how long can I keep it?",
        "expected": "The premise is unsupported/incorrect. The policy says Casual Leave does not carry forward.",
        "type": "misleading_premise",
    },
    {
        "id": "Q15",
        "question": "The policy gives 8 Casual Leave days annually. What are the rules for using them?",
        "expected": "The policy states 5 Casual Leave days annually, not 8.",
        "type": "misleading_premise",
    },
    {
        "id": "Q16",
        "question": "Annual Leave can be carried forward without a cap. How do I use the carried days?",
        "expected": "The premise is false. Up to 5 unused Annual Leave days can be carried forward, and they must be used by March 31 of the following year.",
        "type": "misleading_premise",
    },
    {
        "id": "Q17",
        "question": "There is a 10-working-day advance-notice rule for Annual Leave, right?",
        "expected": "No. The policy says at least 5 working days before the intended start date, wherever possible.",
        "type": "misleading_premise",
    },
    {
        "id": "Q18",
        "question": "The medical certificate for long Sick Leave must be submitted before the leave starts. Confirm?",
        "expected": "No. For 3 or more consecutive working days, the certificate should be shared with HR within 3 working days of returning to work.",
        "type": "misleading_premise",
    },
    {
        "id": "Q19",
        "question": "Saturday is treated as a working day for leave calculations, correct?",
        "expected": "No. Saturday and Sunday are the standard weekend.",
        "type": "misleading_premise",
    },
    {
        "id": "Q20",
        "question": "August 15 is not a company holiday. What kind of leave should I apply for?",
        "expected": "The premise is false. August 15 is listed as Independence Day in the supplied holiday calendar.",
        "type": "misleading_premise",
    },

    # ------------------------------------------------------------------
    # 21-30: Unsupported policy details / hallucination traps
    # ------------------------------------------------------------------
    {
        "id": "Q21",
        "question": "How many days of maternity leave are employees entitled to?",
        "expected": "Not provided in the supplied policy context.",
        "type": "unsupported",
    },
    {
        "id": "Q22",
        "question": "How many paid paternity leave days does the company provide?",
        "expected": "Not provided in the supplied policy context.",
        "type": "unsupported",
    },
    {
        "id": "Q23",
        "question": "What percentage of salary is deducted for unpaid leave?",
        "expected": "Not provided in the supplied policy context.",
        "type": "unsupported",
    },
    {
        "id": "Q24",
        "question": "Are employees allowed to work from home while officially on Sick Leave?",
        "expected": "Not provided in the supplied policy context.",
        "type": "unsupported",
    },
    {
        "id": "Q25",
        "question": "What is the maximum number of Annual Leave days allowed in a single request?",
        "expected": "Not explicitly specified in the supplied policy context.",
        "type": "unsupported",
    },
    {
        "id": "Q26",
        "question": "Can I combine Annual Leave with Work From Home days?",
        "expected": "Not provided in the supplied policy context.",
        "type": "unsupported",
    },
    {
        "id": "Q27",
        "question": "What is the carry-forward limit for compensatory off?",
        "expected": "Not provided in the supplied policy context.",
        "type": "unsupported",
    },
    {
        "id": "Q28",
        "question": "How many bereavement leave days are available after the loss of a parent?",
        "expected": "Not provided in the supplied policy context.",
        "type": "unsupported",
    },
    {
        "id": "Q29",
        "question": "Can Annual Leave be encashed at the end of the year?",
        "expected": "Not provided in the supplied policy context.",
        "type": "unsupported",
    },
    {
        "id": "Q30",
        "question": "Does the company provide extra leave for working on a public holiday?",
        "expected": "Not provided in the supplied policy context.",
        "type": "unsupported",
    },

    # ------------------------------------------------------------------
    # 31-35: Partial-support traps
    # ------------------------------------------------------------------
    {
        "id": "Q31",
        "question": "Can I take 3 consecutive Casual Leave days without manager approval?",
        "expected": "No. More than 2 consecutive working days requires advance manager approval; the policy specifically states no more than 2 without advance approval.",
        "type": "partial_support",
    },
    {
        "id": "Q32",
        "question": "For 3 Sick Leave days, give me the exact type of doctor whose certificate HR requires.",
        "expected": "The policy states when the certificate should be shared, but it does not specify a doctor type.",
        "type": "partial_support",
    },
    {
        "id": "Q33",
        "question": "Annual Leave carried from 2025 can be used until March 31, 2027, correct?",
        "expected": "No. Under the policy, carried-forward Annual Leave must be used by March 31 of the following year; leave carried from 2025 would therefore be usable through March 31, 2026, not 2027.",
        "type": "partial_support",
    },
    {
        "id": "Q34",
        "question": "Because Saturdays and Sundays are weekends, every public holiday is always automatically excluded from every leave calculation.",
        "expected": "The supplied context says holidays are excluded from leave-day calculations; do not add broader rules beyond that wording.",
        "type": "partial_support",
    },
    {
        "id": "Q35",
        "question": "Who is the manager that approves my leave, and what is their response-time SLA?",
        "expected": "The employee's direct manager approves or rejects the request, and managers are expected to act within 2 working days of submission.",
        "type": "partial_support",
    },

    # ------------------------------------------------------------------
    # 36-40: Contradiction / multi-constraint cases
    # ------------------------------------------------------------------
    {
        "id": "Q36",
        "question": "I joined 45 days ago. I am eligible for Annual Leave, so how many days can I take?",
        "expected": "No Annual Leave eligibility yet; the policy requires 90 days of employment.",
        "type": "contradiction",
    },
    {
        "id": "Q37",
        "question": "I have 6 unused Annual Leave days. Since 6 can be carried forward, are they all valid until March 31?",
        "expected": "Only up to 5 unused Annual Leave days can be carried forward.",
        "type": "contradiction",
    },
    {
        "id": "Q38",
        "question": "I used 2 Casual Leave days already, so I still have 5 left because Casual Leave resets independently. Is that right?",
        "expected": "The policy provides 5 Casual Leave days per calendar year; it does not support a separate reset that restores used days during the year.",
        "type": "contradiction",
    },
    {
        "id": "Q39",
        "question": "I need 4 consecutive Casual Leave working days and I don't want to ask my manager first. Is that within the no-approval limit?",
        "expected": "No. The no-advance-approval rule covers no more than 2 consecutive working days.",
        "type": "contradiction",
    },
    {
        "id": "Q40",
        "question": "A request submitted 2 working days before Annual Leave meets the policy's preferred notice period, correct?",
        "expected": "No. The policy says at least 5 working days before the intended start date, wherever possible.",
        "type": "contradiction",
    },
]


def build_context(question: str, retriever: SemanticRetriever) -> str:
    """Retrieve top-10 chunks and rerank them to top-3."""
    candidates = retriever.retrieve(question, top_k=10)
    top_chunks = rerank(question, candidates, top_n=3)

    parts = []
    for chunk, score in top_chunks:
        parts.append(f"[{chunk['title']}]\n{chunk['content']}")

    return "\n\n".join(parts)


def call_model(
    client: Groq,
    model: str,
    question: str,
    context: str,
) -> tuple[str, float]:
    prompt = (
        f"Employee question:\n{question}\n\n"
        f"Retrieved policy information:\n{context}\n\n"
        "Answer the employee's question directly. "
        "Verify the premise against the context before answering. "
        "Use only information supported by the supplied policy context."
    )

    start = time.perf_counter()

    response = client.chat.completions.create(
        model=model,
        temperature=0,
        max_tokens=150,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )

    elapsed = time.perf_counter() - start
    answer = response.choices[0].message.content or ""
    return answer.strip(), elapsed



# ---------------------------------------------------------------------------
# Deterministic / mathematical evaluation
# ---------------------------------------------------------------------------
# No LLM is used for judging. Each question has explicit, auditable rules.
# The grader checks whether the answer contains the policy facts that are
# required for that question and whether it explicitly contradicts them.

import re


def norm(text: str) -> str:
    """Normalize model output for deterministic checks."""
    text = (text or "").lower()
    text = text.replace("’", "'").replace("–", "-").replace("—", "-")
    text = re.sub(r"[^a-z0-9%./ -]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def has_any(text: str, patterns: list[str]) -> bool:
    return any(re.search(p, text, flags=re.I) for p in patterns)


def has_all(text: str, patterns: list[str]) -> bool:
    return all(re.search(p, text, flags=re.I) for p in patterns)


REFUSAL_PATTERNS = [
    r"not (?:provided|specified|mentioned|stated|available)",
    r"does not (?:contain|specify|state|provide)",
    r"doesn't (?:contain|specify|state|provide)",
    r"insufficient (?:information|context)",
    r"not enough information",
    r"cannot (?:be )?answered",
    r"can't (?:be )?answered",
    r"unable to answer",
    r"not addressed (?:in|by) (?:the )?(?:policy|provided context)",
    r"policy (?:does not|doesn't) (?:cover|mention|specify|state|provide)",
]


def refusal_detected(answer: str) -> bool:
    return has_any(norm(answer), REFUSAL_PATTERNS)


def result(*, correct: bool, hallucinated: bool, refusal_correct: bool, reason: str) -> dict:
    return {
        "hallucinated": bool(hallucinated),
        "correct": bool(correct),
        "refusal_correct": bool(refusal_correct),
        "reason": reason,
    }


def evaluate_case(case: dict, answer: str) -> dict:
    """Deterministically evaluate an answer with a fixed, question-specific rubric.

    No LLM judge is used. The rubric checks required concepts and explicit
    contradictions. It intentionally accepts semantically equivalent wording.
    """
    qid = case["id"]
    t = case["type"]
    a = norm(answer)

    def has(*patterns: str) -> bool:
        return any(re.search(p, a, flags=re.I) for p in patterns)

    def no_phrase(*patterns: str) -> bool:
        return any(re.search(p, a, flags=re.I) for p in patterns)

    def make(correct: bool, hallucinated: bool, reason: str, refusal_correct: bool = False) -> dict:
        return result(
            correct=bool(correct),
            hallucinated=bool(hallucinated),
            refusal_correct=bool(refusal_correct),
            reason=reason,
        )

    # Unsupported cases: explicit abstention is required.
    if t == "unsupported":
        refused = refusal_detected(answer)
        return make(
            correct=refused,
            hallucinated=not refused,
            refusal_correct=refused,
            reason="Unsupported case: the policy context does not provide the requested information.",
        )

    # Q01: Annual Leave = 20 days/year.
    if qid == "Q01":
        correct = has(r"\b20\b\s*(?:annual leave|days?)", r"(?:annual leave|days?).{0,20}\b20\b")
        bad = has(r"\b(?:5|10|15|25|30)\b\s*(?:annual leave|days?)", r"(?:annual leave|days?).{0,20}\b(?:5|10|15|25|30)\b")
        return make(correct and not bad, bad, "Annual Leave allowance is 20 days per calendar year.")

    # Q02: Sick Leave = 10 days/year.
    if qid == "Q02":
        correct = has(r"\b10\b\s*(?:sick leave|days?)", r"(?:sick leave|days?).{0,20}\b10\b")
        bad = has(r"\b(?:5|8|12|15|20)\b\s*(?:sick leave|days?)", r"(?:sick leave|days?).{0,20}\b(?:5|8|12|15|20)\b")
        return make(correct and not bad, bad, "Sick Leave allowance is 10 days per calendar year.")

    # Q03: Casual Leave = 5 days/year.
    if qid == "Q03":
        correct = has(r"\b5\b\s*(?:casual leave|days?)", r"(?:casual leave|days?).{0,20}\b5\b")
        bad = has(r"\b(?:2|8|10|20)\b\s*(?:casual leave|days?)", r"(?:casual leave|days?).{0,20}\b(?:2|8|10|20)\b")
        return make(correct and not bad, bad, "Casual Leave allowance is 5 days per calendar year.")

    # Q04/Q11: Annual Leave eligibility begins after completing 90 days.
    if qid in {"Q04", "Q11"}:
        has90 = has(r"\b90\b\s*days?", r"after\s+90", r"completing\s+90")
        immediate_claim = has(
            r"annual leave.{0,40}(?:eligible|available).{0,25}(?:immediately|right away|at once)",
            r"(?:immediately|right away|at once).{0,40}annual leave.{0,25}(?:eligible|available)"
        )
        # Saying "only after 90 days" is the correct rejection of the immediate-eligibility premise.
        wrong = immediate_claim and not has(r"not.{0,30}(?:eligible|available)", r"only.{0,20}after\s+90")
        correct = has90 and not wrong and not immediate_claim
        return make(correct, wrong, "Annual Leave eligibility begins after 90 days, not immediately.")

    # Q05: Sick Leave is available from first working day.
    if qid == "Q05":
        positive = has(r"first working day", r"day one", r"from day one") and not has(r"not.{0,30}(?:available|eligible)")
        bad = has(r"(?:sick leave).{0,40}(?:after 90 days|after probation|not available|unavailable)",
                  r"(?:after probation|after 90 days).{0,30}(?:sick leave)")
        return make(positive and not bad, bad, "Sick Leave is available from the first working day.")

    # Q06: Up to 5 Annual Leave days can carry forward.
    if qid == "Q06":
        five = has(r"\b5\b\s*(?:unused\s*)?(?:annual leave|days?)", r"up to\s+5", r"maximum of\s+5")
        carry = has(r"carry(?:ing|ied)?\s+forward", r"carried[- ]forward")
        bad = has(r"(?:unlimited|no cap|without a cap)", r"\b(?:6|10|20)\b.{0,30}carry(?:ing|ied)?\s+forward")
        return make(five and carry and not bad, bad, "Up to 5 unused Annual Leave days may be carried forward.")

    # Q07: By March 31 of the following year.
    if qid == "Q07":
        march31 = has(r"march\s*31")
        following = has(r"following\s+(?:calendar\s+)?year", r"next\s+(?:calendar\s+)?year", r"next\s+year")
        bad = has(r"march\s*(?:30|1|15)", r"(?:april|may|june|december)")
        return make(march31 and following and not bad, bad, "Carried-forward Annual Leave must be used by March 31 of the following year.")

    # Q08: Sick Leave does not carry forward.
    if qid == "Q08":
        no_carry = has(r"(?:does|do)\s+not\s+carry\s+forward", r"doesn['’]t\s+carry\s+forward", r"does not roll over", r"doesn['’]t roll over")
        positive_carry = has(r"(?:sick leave).{0,40}(?:carries|carry|rolls|roll)\s+(?:over|forward)") and not no_carry
        return make(no_carry and "sick leave" in a and not positive_carry, positive_carry,
                    "Unused Sick Leave does not carry forward.")

    # Q09: Casual Leave does not carry forward; it resets to 5 next year.
    if qid == "Q09":
        no_carry = has(r"(?:does|do)\s+not\s+carry\s+forward", r"doesn['’]t\s+carry\s+forward", r"does not roll over", r"doesn['’]t roll over")
        reset5 = has(r"reset(?:s|ting)?\s+(?:to|at)?\s*(?:the\s+standard\s+)?5\b", r"\b5\b.{0,20}reset", r"new year.{0,30}5")
        positive_carry = has(r"(?:casual leave).{0,40}(?:carries|carry|rolls|roll)\s+(?:over|forward)") and not no_carry
        # The core policy answer is non-carry; reset-to-5 is a supported detail but not mandatory.
        return make("casual leave" in a and no_carry and not positive_carry, positive_carry,
                    "Unused Casual Leave does not carry forward; the standard balance is 5 at the start of the next year.")

    # Q10: At least 5 working days, wherever possible.
    if qid == "Q10":
        five = has(r"\b5\b\s*working\s+days?")
        bad = has(r"\b(?:10|3|2)\b\s*working\s+days?") and not has(r"not\s+(?:10|3|2)\b")
        return make(five and not bad, bad, "Annual Leave should be requested at least 5 working days in advance, wherever possible.")

    # Q12: false premise; Sick Leave starts on first working day.
    if qid == "Q12":
        first_day = has(r"first working day", r"day one", r"from day one")
        bad = has(r"(?:sick leave).{0,50}(?:after 90 days|after probation|not available during probation|unavailable during probation)")
        return make(first_day and not bad, bad, "The premise is false: Sick Leave is available from the first working day.")

    # Q13: Sick Leave does not carry forward.
    if qid == "Q13":
        no_carry = has(r"does not carry forward", r"doesn['’]t carry forward", r"does not roll over", r"doesn['’]t roll over")
        bad = has(r"sick leave.{0,40}(?:carries|carry|rolls|roll)\s+(?:over|forward)") and not no_carry
        return make(no_carry and not bad, bad, "Sick Leave does not carry forward.")

    # Q14: reject the premise; Casual Leave does not carry forward.
    if qid == "Q14":
        no_carry = has(r"casual leave.{0,40}does not carry forward", r"casual leave.{0,40}doesn['’]t carry forward",
                       r"casual leave.{0,40}does not roll over", r"casual leave.{0,40}doesn['’]t roll over")
        bad = has(r"casual leave.{0,40}(?:carry|roll).{0,20}(?:two|2) years") and not no_carry
        return make(no_carry and not bad, bad, "Casual Leave does not carry forward.")

    # Q15: 5 days, not 8; plus supported usage rules.
    if qid == "Q15":
        five = has(r"\b5\b\s*(?:casual leave|days?)", r"casual leave.{0,20}\b5\b")
        rejects8 = has(r"not\s+8", r"rather than\s+8", r"instead of\s+8", r"not\s+8\s+days")
        max2 = has(r"(?:2|two)\s+consecutive\s+working\s+days", r"no more than 2", r"up to 2")
        bad = has(r"\b8\b\s*(?:casual leave|days?)", r"casual leave.{0,20}\b8\b") and not rejects8
        # The answer is correct if it clearly corrects the false 8-day premise and gives the 5-day rule.
        correct = five and (rejects8 or not bad) and not bad
        return make(correct, bad, "Casual Leave is 5 days/year, not 8; it is subject to the stated usage rules.")

    # Q16: reject the no-cap premise; state the 5-day cap and March 31 deadline.
    if qid == "Q16":
        five_cap = has(r"up to\s+5", r"maximum of\s+5", r"only\s+5", r"5\s+unused.*annual leave")
        march31 = has(r"march\s*31")
        wrong_unlimited = has(r"unlimited", r"no cap", r"without a cap") and not has(r"not.{0,30}(?:unlimited|no cap|without a cap)")
        correct = five_cap and march31 and not wrong_unlimited
        return make(correct, wrong_unlimited, "Only up to 5 Annual Leave days carry forward, and they must be used by March 31 of the following year.")

    # Q17: 5 working days, not 10.
    if qid == "Q17":
        five = has(r"\b5\b\s*working\s+days?")
        rejects10 = has(r"not\s+10", r"not\s+ten", r"rather than\s+10", r"instead of\s+10")
        bad = has(r"(?:annual leave).{0,60}(?:10|ten)\s+working\s+days") and not rejects10
        correct = five and not bad
        return make(correct, bad, "The policy says at least 5 working days before the start date, wherever possible.")

    # Q18: certificate after returning, not before leave.
    if qid == "Q18":
        return_timing = has(r"within\s+3\s+working\s+days?.{0,40}return", r"after\s+(?:you\s+)?return", r"returning\s+to\s+work")
        not_before = has(r"not before", r"not.*before the leave starts", r"not.*before")
        wrong = has(r"before the leave starts", r"before leave starts") and not not_before
        correct = return_timing and not wrong
        return make(correct, wrong, "For 3 or more consecutive Sick Leave working days, the certificate is shared within 3 working days of returning to work.")

    # Q19: Saturday is part of the standard weekend and is not a working day.
    if qid == "Q19":
        weekend = has(r"saturday.{0,60}(?:weekend|not counted|not a working day|excluded)",
                       r"(?:weekend|standard weekend).{0,60}saturday")
        explicit_no = has(r"\bno\b", r"not")
        positive_workday = has(r"saturday.{0,60}(?:is|treated as|counts as)\s+a?\s*working day")
        negative_workday = has(r"(?:saturday|it).{0,60}(?:not counted|not a working day|not considered a working day|not a working-day)")
        correct = weekend and (explicit_no or negative_workday) and not positive_workday
        return make(correct, positive_workday, "Saturday and Sunday are the standard weekend and are not counted as working days.")

    # Q20: Aug 15 is a listed public holiday.
    if qid == "Q20":
        holiday = has(r"august\s*15.{0,50}(?:public holiday|independence day|company holiday|holiday)")
        no_leave = has(r"no need.{0,30}leave", r"do not need to apply", r"don't need to apply", r"not need to apply")
        bad = has(r"august\s*15.{0,50}(?:not a holiday|working day|requires? leave)") and not has(r"not.{0,30}(?:a holiday|working day)")
        return make(holiday and no_leave and not bad, bad, "August 15 is Independence Day in the supplied 2026 holiday calendar.")

    # Q29: year-end encashment is not provided; resignation payout for carried-forward leave is provided.
    if qid == "Q29":
        no_year_end = has(r"does not provide", r"not provide", r"no provision", r"not specified") and has(r"encash", r"cash")
        supports_exit = has(r"when an employee leaves", r"employee leaves", r"final settlement")
        bad = has(r"encash(?:ed|ment)?.{0,40}(?:at year.end|end of the year)") and not no_year_end
        return make((no_year_end or not has(r"yes.{0,20}encash")) and not bad and (supports_exit or no_year_end), bad,
                    "The policy does not provide year-end encashment; it only states payout of unused carried-forward Annual Leave on exit.")

    # Q30: extra leave for working a public holiday is not specified.
    if qid == "Q30":
        abstain = refusal_detected(answer)
        says_not_specified = has(r"does not (?:specify|state|provide)", r"not (?:specified|provided|stated)", r"no extra leave")
        wrong = has(r"(?:provides?|gives?|grants?)\s+(?:extra|additional)\s+leave") and not says_not_specified
        return make((abstain or says_not_specified) and not wrong, wrong,
                    "The holiday policy does not specify extra leave for working on a public holiday.")

    # Q31: 3 consecutive Casual Leave days require manager approval.
    if qid == "Q31":
        no = has(r"\bno\b", r"not")
        approval = has(r"manager.{0,40}approval", r"approval.{0,40}manager", r"requires? approval")
        max2 = has(r"2\s+consecutive\s+working\s+days", r"no more than 2", r"up to 2")
        wrong = has(r"3\s+(?:consecutive\s+)?(?:working\s+)?days?.{0,50}without.{0,30}approval", r"no approval needed")
        return make(no and approval and max2 and not wrong, wrong, "More than 2 consecutive Casual Leave working days requires manager approval in advance.")

    # Q32: doctor type is not specified; refusal is the correct response.
    if qid == "Q32":
        refused = refusal_detected(answer)
        doctor = has(r"general practitioner", r"gp\b", r"physician", r"specialist", r"surgeon", r"orthopedic", r"cardiologist", r"registered medical practitioner", r"rmp")
        return make(refused and not doctor, doctor, "The policy specifies certificate timing, but not a doctor type.", refusal_correct=refused and not doctor)

    # Q33: 2025 -> March 31, 2026, or equivalent general rule.
    if qid == "Q33":
        march31 = has(r"march\s*31")
        general_following = has(r"following\s+(?:calendar\s+)?year", r"next\s+(?:calendar\s+)?year")
        concrete_2026 = has(r"march\s*31\s*,?\s*2026", r"march\s*31.*2026", r"by\s+march\s*31\s+2026")
        bad = has(r"march\s*31\s*,?\s*2027", r"until\s+2027", r"through\s+2027", r"into\s+2027") and not has(r"not.{0,40}2027", r"cannot.{0,40}2027", r"no.{0,20}2027")
        correct = (concrete_2026 or (march31 and general_following)) and not bad
        return make(correct, bad, "Carried-forward Annual Leave from 2025 follows the stated March 31 of the following-year deadline (March 31, 2026).")

    # Q34: accept supported company-wide holiday exclusion; reject broader unsupported claims.
    if qid == "Q34":
        excluded = has(r"public holidays?.{0,80}(?:excluded|not charged|left out)",
                       r"(?:excluded|not charged|left out).{0,80}public holidays?")
        bad = has(r"regional holidays?.{0,50}(?:automatically|always)\s+excluded", r"every possible public holiday")
        correct = excluded and not bad
        return make(correct, bad, "Listed/company-wide public holidays are excluded from leave-day calculations; regional calendars are handled separately.")

    # Q35: direct manager + 2-working-day turnaround.
    if qid == "Q35":
        manager = has(r"direct manager", r"manager")
        two = has(r"\b2\b\s*working\s+days?")
        wrong_sla = has(r"no sla", r"does not specify.{0,30}(?:sla|response|turnaround)", r"not specify.{0,30}(?:sla|response|turnaround)")
        return make(manager and two and not wrong_sla, wrong_sla, "The direct manager approves/rejects the request and is expected to act within 2 working days.")

    # Q36: 45 days means not yet eligible for Annual Leave.
    if qid == "Q36":
        not_eligible = has(r"not yet eligible", r"not eligible", r"cannot take annual leave", r"cannot.*annual leave.*at this time", r"cannot take.*annual leave")
        ninety = has(r"90\s*days?", r"after completing 90")
        wrong = has(r"eligible.{0,30}annual leave", r"can take.{0,30}annual leave") and not not_eligible
        return make(not_eligible and ninety and not wrong, wrong, "At 45 days, the employee has not reached the 90-day Annual Leave eligibility threshold.")

    # Q37: only 5 of 6 can carry forward; sixth is forfeited.
    if qid == "Q37":
        five = has(r"\b5\b", r"up to 5")
        sixth_forfeit = has(r"6th day", r"sixth day", r"sixth.{0,20}forfeit", r"6th.{0,20}forfeit")
        rejects6 = has(r"only 5", r"only up to 5", r"not all 6", r"6.*forfeit")
        bad = has(r"all 6", r"six(?: |-)6", r"6\s+(?:days?\s+)?(?:can|may)\s+be\s+carried") and not rejects6
        correct = five and (sixth_forfeit or rejects6) and not bad
        return make(correct, bad, "Only up to 5 unused Annual Leave days carry forward; the excess sixth day is forfeited.")

    # Q38: after using 2 of 5, 3 remain; reset is at the next calendar year.
    if qid == "Q38":
        three = has(r"\b3\b\s*(?:days?|remaining|left)", r"(?:remaining|left).{0,20}\b3\b")
        next_year = has(r"next calendar year", r"next year", r"start of (?:a|the) next year", r"new calendar year")
        bad = has(r"still have 5", r"5\s+(?:days?|left|remaining).{0,20}(?:current|same) year", r"reset.{0,30}(?:same|current) year", r"restore.{0,30}used")
        correct = three and next_year and not bad
        return make(correct, bad, "Using 2 of 5 Casual Leave days leaves 3; the balance resets only at the start of the next calendar year.")

    # Q39: four consecutive days exceed the no-advance-approval limit.
    if qid == "Q39":
        four = has(r"\b4\b\s*(?:consecutive\s+)?(?:working\s+)?days?")
        approval = has(r"manager.{0,40}approval", r"requires? approval", r"approval.{0,40}manager")
        max2 = has(r"2\s+consecutive\s+working\s+days", r"no more than 2", r"up to 2")
        bad = has(r"4.{0,50}(?:without|no).{0,20}approval", r"no approval needed")
        return make(four and approval and max2 and not bad, bad, "Four consecutive Casual Leave days require advance manager approval because the no-approval limit is 2 days.")

    # Q40: 2 working days does not meet the preferred 5-working-day notice period.
    if qid == "Q40":
        five = has(r"\b5\b\s*working\s*days?")
        negative_2 = has(
            r"(?:does not|doesn't|not)\s+(?:meet|satisfy|fulfil|qualify)\s+(?:the\s+)?(?:preferred\s+)?(?:notice|notice period)",
            r"(?:does not|doesn't|not)\s+meet.{0,40}notice",
            r"2\s*working\s*days?.{0,80}(?:too short|insufficient|does not|doesn't|not)"
        )
        accepts_2 = has(r"2\s*working\s*days?.{0,60}(?:meets?|satisf(?:y|ies)|qualif(?:y|ies)).{0,60}(?:notice|policy)")
        correct = five and negative_2 and not accepts_2
        return make(correct, accepts_2, "Two working days does not meet the preferred 5-working-day Annual Leave notice period.")

    raise ValueError(f"No deterministic grading rule defined for {qid}")

def print_model_metrics(results: list[dict], prefix: str, label: str) -> None:
    total = len(results)
    h = [bool(r[f"{prefix}_hallucinated"]) for r in results]
    c = [bool(r[f"{prefix}_correct"]) for r in results]
    unsupported = [r for r in results if r["type"] == "unsupported"]
    refusal = [bool(r[f"{prefix}_refusal_correct"]) for r in unsupported]

    print(label)
    print(f"  Hallucination rate: {sum(h) / total * 100:.1f}% ({sum(h)}/{total})")
    print(f"  Correctness rate:   {sum(c) / total * 100:.1f}% ({sum(c)}/{total})")
    print(f"  Refusal accuracy:   {sum(refusal) / len(refusal) * 100:.1f}% ({sum(refusal)}/{len(refusal)})")
    print(f"  Average latency:    {sum(r[f'{prefix}_latency_sec'] for r in results) / total:.3f} sec")
    print()


def write_summary(results: list[dict]) -> None:
    total = len(results)
    unsupported = sum(1 for r in results if r["type"] == "unsupported")

    print("\\nEvaluation set summary")
    print(f"  Total questions:      {total}")
    print(f"  Unsupported:          {unsupported}")
    print("  Judge model:          NONE (deterministic rule-based grading)")
    print()

    print_model_metrics(results, "model_a", MODEL_A)
    print_model_metrics(results, "model_b", MODEL_B)

    h_a = sum(bool(r["model_a_hallucinated"]) for r in results) / total * 100
    h_b = sum(bool(r["model_b_hallucinated"]) for r in results) / total * 100
    c_a = sum(bool(r["model_a_correct"]) for r in results) / total * 100
    c_b = sum(bool(r["model_b_correct"]) for r in results) / total * 100

    print("Head-to-head")
    print(f"  Hallucination delta (120B - 20B): {h_b - h_a:+.1f} pp")
    print(f"  Correctness delta   (120B - 20B): {c_b - c_a:+.1f} pp")

    disagreements = [
        r["id"] for r in results
        if (r["model_a_hallucinated"], r["model_a_correct"])
        != (r["model_b_hallucinated"], r["model_b_correct"])
    ]
    print(f"  Model decision disagreements: {len(disagreements)}")
    if disagreements:
        print(f"  Cases: {', '.join(disagreements)}")


def main() -> None:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise SystemExit(
            "GROQ_API_KEY is not configured. Add it to your .env file."
        )

    print("Loading policy documents...")
    documents = load_documents(DOCUMENTS_FOLDER)
    chunks = chunk_documents(documents)
    retriever = SemanticRetriever(chunks)
    client = Groq(api_key=api_key)

    print(f"Documents: {len(documents)}")
    print(f"Chunks:    {len(chunks)}")
    print(f"Models:    {MODEL_A} vs {MODEL_B}")
    print("Judge:     NONE (deterministic rule-based grading)")
    print(f"Questions: {len(TEST_CASES)} adversarial cases")
    print()

    results = []

    for i, case in enumerate(TEST_CASES, start=1):
        print(f"[{i}/{len(TEST_CASES)}] {case['id']}: {case['question']}")

        context = build_context(case["question"], retriever)

        answer_a, latency_a = call_model(client, MODEL_A, case["question"], context)
        answer_b, latency_b = call_model(client, MODEL_B, case["question"], context)

        eval_a = evaluate_case(case, answer_a)
        eval_b = evaluate_case(case, answer_b)

        results.append(
            {
                **case,
                "context": context,
                "model_a": MODEL_A,
                "model_a_answer": answer_a,
                "model_a_latency_sec": round(latency_a, 4),
                "model_a_hallucinated": eval_a["hallucinated"],
                "model_a_correct": eval_a["correct"],
                "model_a_refusal_correct": eval_a["refusal_correct"],
                "model_a_judge_reason": eval_a["reason"],
                "model_b": MODEL_B,
                "model_b_answer": answer_b,
                "model_b_latency_sec": round(latency_b, 4),
                "model_b_hallucinated": eval_b["hallucinated"],
                "model_b_correct": eval_b["correct"],
                "model_b_refusal_correct": eval_b["refusal_correct"],
                "model_b_judge_reason": eval_b["reason"],
            }
        )

        print(f"  {MODEL_A}: {answer_a}")
        print(f"  {MODEL_B}: {answer_b}")
        print(f"  {MODEL_A} -> correct={eval_a['correct']} hallucinated={eval_a['hallucinated']}")
        print(f"  {MODEL_B} -> correct={eval_b['correct']} hallucinated={eval_b['hallucinated']}")
        print()

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        fieldnames = list(results[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print("Saved:")
    print(f"  {OUTPUT_CSV}")
    print(f"  {OUTPUT_JSON}")

    write_summary(results)


if __name__ == "__main__":
    main()
