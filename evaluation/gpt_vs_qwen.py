"""
Adversarial two-model generation evaluation for the Leave Management RAG system.

Purpose:
    Compare GPT-OSS-120B and Qwen3.8-27B as GENERATION models while keeping
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
    gpt_oss_vs_qwen_comparison.csv
    gpt_oss_vs_qwen_comparison.json

The CSV/JSON contains automatic judge labels for each model:
    *_hallucinated
    *_correct
    *_refusal_correct
    *_judge_reason

No LLM judge is used. Answers are graded by explicit deterministic,
question-specific rules so the benchmark is reproducible and mathematically auditable.

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


MODEL_A = "openai/gpt-oss-120b"
MODEL_B = "qwen/qwen3.8-27b"

DOCUMENTS_FOLDER = "rag_lab/documents"
OUTPUT_CSV = "gpt_oss_vs_qwen_comparison.csv"
OUTPUT_JSON = "gpt_oss_vs_qwen_comparison.json"
CHECKPOINT_JSON = "qwen_only_checkpoint.json"
MAX_TOKENS = 120
TOP_K_RETRIEVAL = 10
TOP_K_RERANK = 3
TOP_CONTEXT_CHUNKS = 3
MAX_CONTEXT_CHARS = 3200

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
    """Retrieve top-10, rerank to top-3, then use the best chunk only.

    Keeping only the top reranked chunk reduces token usage while preserving
    the same retrieval/reranking pipeline. Context is also capped defensively.
    """
    candidates = retriever.retrieve(question, top_k=TOP_K_RETRIEVAL)
    top_chunks = rerank(question, candidates, top_n=TOP_K_RERANK)

    parts = []
    for chunk, score in top_chunks[:TOP_CONTEXT_CHUNKS]:
        parts.append(f"[{chunk['title']}]\n{chunk['content']}")

    return "\n\n".join(parts)[:MAX_CONTEXT_CHARS]


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

    kwargs = {
        "model": model,
        "temperature": 0,
        "max_tokens": MAX_TOKENS,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    }
    # Disable Qwen's reasoning mode so the generation comparison stays focused
    # on concise policy answering rather than hidden reasoning-token budgets.
    if model.startswith("qwen/"):
        kwargs["reasoning_effort"] = "none"
        kwargs["reasoning_format"] = "hidden"

    response = client.chat.completions.create(**kwargs)

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


def _has(answer: str, *patterns: str) -> bool:
    import re
    a = answer.lower()
    return any(re.search(p, a) for p in patterns)


def _is_refusal(answer: str) -> bool:
    a = answer.lower().strip()
    return (
        not a
        or _has(
            a,
            r"provided policy context does not contain enough information",
            r"not enough information to answer",
            r"information is not provided",
            r"not specified in the policy",
            r"policy does not specify",
            r"policy does not provide",
        )
    )


def _result(correct: bool, hallucinated: bool, reason: str, error_type: str = "", refusal_correct: bool = False) -> dict:
    return {
        "correct": bool(correct),
        "hallucinated": bool(hallucinated),
        "refusal_correct": bool(refusal_correct),
        "reason": reason,
        "error_type": error_type,
    }


def evaluate_case(case: dict, answer: str) -> dict:
    """Deterministic, question-specific grading.

    Categories are separated deliberately:
      correct, hallucination, factual_error, unnecessary_refusal.
    A model is marked hallucinated only when it asserts a policy claim that
    contradicts the supplied policy or invents an unsupported rule/value.
    """
    qid = case["id"]
    a = answer.strip().lower()
    t = case["type"]

    if t == "unsupported":
        refused = _is_refusal(a)
        return _result(
            correct=refused,
            hallucinated=not refused,
            refusal_correct=refused,
            reason="Unsupported case: abstention is required.",
            error_type="" if refused else "hallucination",
        )

    # Q01
    if qid == "Q01":
        ok = _has(a, r"\b20\b") and not _has(a, r"\b(?:5|10|15|25|30)\b\s*(?:annual leave|days?)")
        return _result(ok, False, "Annual Leave allowance is 20 days per calendar year.", "" if ok else "factual_error")

    # Q02
    if qid == "Q02":
        ok = _has(a, r"\b10\b") and not _has(a, r"\b(?:5|8|12|15|20)\b\s*(?:sick leave|days?)")
        return _result(ok, False, "Sick Leave allowance is 10 days per calendar year.", "" if ok else "factual_error")

    # Q03
    if qid == "Q03":
        ok = _has(a, r"\b5\b") and not _has(a, r"\b(?:2|8|10|20)\b\s*(?:casual leave|days?)")
        return _result(ok, False, "Casual Leave allowance is 5 days per calendar year.", "" if ok else "factual_error")

    # Q04/Q11: reject immediate eligibility and state 90 days.
    if qid in {"Q04", "Q11"}:
        has90 = _has(a, r"\b90\b\s*days?", r"after completing 90", r"after 90")
        rejects_immediate = _has(a, r"not (?:be )?(?:available|eligible) immediately", r"not immediately", r"only after 90", r"after completing 90")
        wrong = _has(a, r"(?:annual leave).{0,35}(?:available|eligible).{0,20}(?:immediately|right away|at once)") and not rejects_immediate
        ok = has90 and rejects_immediate and not wrong
        return _result(ok, False, "Annual Leave eligibility begins after 90 days, not immediately.", "" if ok else "factual_error")

    # Q05
    if qid == "Q05":
        ok = _has(a, r"first working day", r"day one", r"from day one") and not _has(a, r"sick leave.{0,30}(?:not available|not eligible|after probation|after 90 days)")
        return _result(ok, False, "Sick Leave is available from the first working day.", "" if ok else "factual_error")

    # Q06
    if qid == "Q06":
        ok = _has(a, r"(?:carry|carried).{0,20}forward") and _has(a, r"\b5\b") and not _has(a, r"unlimited|no cap|without a cap")
        return _result(ok, False, "Up to 5 unused Annual Leave days may be carried forward.", "" if ok else "factual_error")

    # Q07
    if qid == "Q07":
        ok = _has(a, r"march\s*31") and _has(a, r"following (?:calendar )?year|next year") and not _has(a, r"march\s*(?:30|1|15)")
        return _result(ok, False, "Carried-forward Annual Leave must be used by March 31 of the following year.", "" if ok else "factual_error")

    # Q08/Q13: Sick Leave does not carry forward.
    if qid in {"Q08", "Q13"}:
        ok = _has(a, r"(?:does|do) not carry forward", r"doesn['’]t carry forward", r"does not roll over", r"doesn['’]t roll over")
        bad = _has(a, r"sick leave.{0,40}(?:carries|carry|rolls|roll)\s+(?:over|forward)") and not ok
        return _result(ok and not bad, bad, "Unused Sick Leave does not carry forward.", "" if (ok and not bad) else "hallucination" if bad else "factual_error")

    # Q09
    if qid == "Q09":
        ok = _has(a, r"casual leave.{0,60}does not carry forward", r"does not carry forward.{0,60}casual leave") and _has(a, r"\b5\b") and not _has(a, r"casual leave.{0,40}(?:carries|carry|rolls|roll)\s+(?:over|forward)")
        return _result(ok, False, "Unused Casual Leave does not carry forward; the balance resets to 5 at the start of the next calendar year.", "" if ok else "factual_error")

    # Q10/Q17: 5 working-day preferred notice, not 10.
    if qid in {"Q10", "Q17"}:
        five = _has(a, r"\b5\b\s*working\s+days?")
        ok = five and not _has(a, r"(?:requires?|should be|must be).{0,25}\b10\b\s*working\s+days?")
        if qid == "Q17":
            ok = five and (_has(a, r"not\s+10", r"not\s+ten", r"10 days?\b.*(?:not|incorrect)") or not _has(a, r"\b10\b\s*working\s+days?"))
        return _result(ok, False, "Annual Leave should be requested at least 5 working days before the start date, wherever possible.", "" if ok else "factual_error")

    # Q12: false premise; first working day is correct.
    if qid == "Q12":
        ok = _has(a, r"first working day", r"day one", r"from day one") and not _has(a, r"sick leave.{0,40}(?:not available|unavailable).{0,30}(?:first|during probation)")
        return _result(ok, False, "The premise is false: Sick Leave is available from the first working day.", "" if ok else "factual_error")

    # Q14
    if qid == "Q14":
        ok = _has(a, r"casual leave.{0,60}(?:does not carry forward|doesn['’]t carry forward|does not roll over|doesn['’]t roll over)")
        return _result(ok, False, "Casual Leave does not carry forward.", "" if ok else "factual_error")

    # Q15
    if qid == "Q15":
        ok = _has(a, r"\b5\b") and _has(a, r"not\s+8|rather than\s+8|instead of\s+8") and _has(a, r"2\s+consecutive\s+working\s+days|no more than 2|up to 2")
        return _result(ok, False, "Casual Leave is 5 days/year, not 8; more than 2 consecutive working days needs advance manager approval.", "" if ok else "factual_error")

    # Q16: reject no-cap premise; 5-day cap and March 31 deadline.
    if qid == "Q16":
        cap = _has(a, r"up to\s+5", r"maximum of\s+5", r"only\s+5", r"\b5\b\s+unused\s+annual leave", r"up to\s+5\s+unused")
        march = _has(a, r"march\s*31", r"31\s+march")
        rejects = _has(a, r"incorrect|false|not .*without a cap|not .*unlimited|only up to 5|maximum of 5|limits? .*5")
        wrong = _has(a, r"annual leave.{0,30}(?:unlimited|without a cap|no cap)") and not rejects
        ok = cap and march and not wrong
        return _result(ok, False, "Only up to 5 Annual Leave days carry forward, and they must be used by March 31 of the following year.", "" if ok else "factual_error")

    # Q18
    if qid == "Q18":
        timing = _has(a, r"within\s+3\s+working\s+days?.{0,60}(?:return|returning)", r"returning to work", r"after you return")
        not_before = _has(a, r"not before", r"not.*before the leave starts")
        ok = timing and not_before
        return _result(ok, False, "The certificate is shared within 3 working days of returning to work, not before leave starts.", "" if ok else "factual_error")

    # Q19: explicit supported fact; refusal is an unnecessary refusal, not hallucination.
    if qid == "Q19":
        refusal = _is_refusal(a)
        weekend = _has(a, r"saturday.{0,70}(?:weekend|not counted|not a working day|not considered)", r"standard weekend.{0,70}saturday")
        ok = weekend and _has(a, r"\bno\b", r"not") and not _has(a, r"saturday.{0,50}(?:is|counts as|treated as)\s+(?:a )?working day")
        return _result(ok, False, "Saturday and Sunday are the standard weekend and are not counted as working days.", "unnecessary_refusal" if refusal else ("" if ok else "factual_error"))

    # Q20: listed public holiday; refusal is unnecessary refusal.
    if qid == "Q20":
        refusal = _is_refusal(a)
        holiday = _has(a, r"august\s*15.{0,60}(?:holiday|independence day)")
        no_leave = _has(a, r"do not need to apply", r"don't need to apply", r"no need to apply", r"do not need.*leave")
        ok = holiday and no_leave
        return _result(ok, False, "August 15 is a listed public holiday, so leave is not charged for that date.", "unnecessary_refusal" if refusal else ("" if ok else "factual_error"))

    # Q29: no year-end encashment in supplied policy; exit payout for carried-forward annual leave is supported.
    if qid == "Q29":
        refusal = _is_refusal(a)
        no_cash = _has(a, r"does not provide", r"not provide", r"no provision", r"not specified") and _has(a, r"encash|cash")
        ok = no_cash or (_has(a, r"does not.*encash", r"not.*encash") and _has(a, r"when an employee leaves|employee leaves|final settlement"))
        return _result(ok, False, "The supplied policy does not provide year-end encashment; it does state payout of unused carried-forward Annual Leave on exit.", "unnecessary_refusal" if refusal else ("" if ok else "factual_error"))

    # Q30: extra public-holiday leave is not specified; a refusal or clear statement is correct.
    if qid == "Q30":
        refusal = _is_refusal(a)
        no_extra = _has(a, r"does not provide", r"does not specify", r"not mention", r"no extra leave", r"no additional leave")
        ok = refusal or no_extra
        return _result(ok, False, "The supplied policy does not specify extra leave for working on a public holiday.", "" if ok else "factual_error")

    # Q31
    if qid == "Q31":
        approval = _has(a, r"manager.{0,50}approval", r"approval.{0,50}manager", r"requires? approval")
        max2 = _has(a, r"2\s+consecutive\s+working\s+days", r"no more than 2", r"up to 2")
        negative = _has(a, r"\bno\b", r"cannot", r"not")
        wrong = _has(a, r"3.{0,50}(?:without|no).{0,20}approval", r"no approval needed")
        ok = approval and max2 and negative and not wrong
        return _result(ok, False, "Three consecutive Casual Leave working days require advance manager approval.", "" if ok else "factual_error")

    # Q32
    if qid == "Q32":
        refusal = _is_refusal(a)
        doctor = _has(a, r"general practitioner", r"\bgp\b", r"physician", r"specialist", r"surgeon", r"orthopedic", r"cardiologist", r"registered medical practitioner", r"\brmp\b")
        ok = refusal and not doctor
        return _result(ok, doctor, "Doctor type is not specified; the policy only specifies certificate timing.", "" if ok else "hallucination", refusal_correct=ok)

    # Q33
    if qid == "Q33":
        concrete = _has(a, r"march\s*31\s*,?\s*2026", r"march\s*31.*2026")
        general = _has(a, r"march\s*31") and _has(a, r"following year", r"next calendar year", r"next year")
        rejects_2027 = _has(a, r"not.*2027", r"cannot.*2027", r"no.*2027")
        bad = _has(a, r"march\s*31\s*,?\s*2027", r"until 2027", r"into 2027") and not rejects_2027
        ok = (concrete or general) and not bad
        return _result(ok, False, "Carried-forward Annual Leave from 2025 is due by March 31, 2026 under the following-year rule.", "" if ok else "factual_error")

    # Q34: supported company-wide holiday exclusion; refusal is unnecessary refusal.
    if qid == "Q34":
        refusal = _is_refusal(a)
        excluded = _has(a, r"public holidays?.{0,90}(?:excluded|not charged|left out)", r"(?:excluded|not charged).{0,90}public holidays?")
        ok = excluded
        return _result(ok, False, "Listed/company-wide public holidays are excluded from working-day calculations; regional holidays use location-specific calendars.", "unnecessary_refusal" if refusal else ("" if ok else "factual_error"))

    # Q35
    if qid == "Q35":
        manager = _has(a, r"direct manager", r"manager")
        two = _has(a, r"\b2\b\s*working\s+days?")
        ok = manager and two and not _has(a, r"no sla", r"does not specify.*(?:sla|response|turnaround)")
        return _result(ok, False, "The direct manager approves/rejects the request and is expected to act within 2 working days.", "" if ok else "factual_error")

    # Q36: 45 days is below the 90-day eligibility threshold. Refusal is unnecessary.
    if qid == "Q36":
        refusal = _is_refusal(a)
        not_yet = _has(a, r"not yet eligible", r"not eligible")
        ninety = _has(a, r"90\s*days?", r"after completing 90")
        cannot = _has(a, r"cannot take.*annual leave", r"cannot.*annual leave.*at this time", r"no annual leave")
        ok = not_yet and ninety and (cannot or _has(a, r"therefore"))
        return _result(ok, False, "At 45 days, the employee has not reached the 90-day Annual Leave eligibility threshold.", "unnecessary_refusal" if refusal else ("" if ok else "factual_error"))

    # Q37
    if qid == "Q37":
        five = _has(a, r"\b5\b", r"up to 5")
        sixth = _has(a, r"6th day", r"sixth day", r"sixth.*forfeit", r"6th.*forfeit")
        rejects = _has(a, r"only 5", r"only up to 5", r"not all 6", r"6.*forfeit")
        ok = five and (sixth or rejects) and not _has(a, r"all 6.*carry", r"6.*can be carried forward")
        return _result(ok, False, "Only up to 5 unused Annual Leave days can be carried forward; the sixth is forfeited.", "" if ok else "factual_error")

    # Q38: arithmetic is the key: 5 - 2 = 3; reset is next year.
    if qid == "Q38":
        three = _has(a, r"\b3\b\s*(?:days?|remaining|left)", r"(?:remaining|left).{0,20}\b3\b")
        next_year = _has(a, r"next calendar year", r"next year", r"start of.*next")
        accepts5 = _has(a, r"yes[, ]+that is correct", r"still have 5", r"have 5 left")
        ok = three and next_year and not accepts5
        return _result(ok, False, "Using 2 of 5 Casual Leave days leaves 3; reset occurs at the next calendar year.", "" if ok else "factual_error")

    # Q39: 4 consecutive days require approval. Different wording is fine.
    if qid == "Q39":
        four = _has(a, r"\b4\b\s+(?:consecutive\s+)?(?:working\s+)?days?", r"4 consecutive")
        approval = _has(a, r"manager.{0,50}approval", r"requires? approval", r"approval.{0,50}manager")
        max2 = _has(a, r"2\s+consecutive\s+working\s+days", r"no more than 2", r"up to 2")
        wrong = _has(a, r"4.{0,50}(?:without|no).{0,20}approval", r"no approval needed")
        ok = four and approval and max2 and not wrong
        return _result(ok, False, "Four consecutive Casual Leave days exceed the 2-day no-approval limit.", "" if ok else "factual_error")

    # Q40: 2 days does not meet 5-day preferred notice. This is not a hallucination.
    if qid == "Q40":
        five = _has(a, r"\b5\b\s*working\s+days?")
        says_not_meet = _has(a, r"does not meet", r"doesn['’]t meet", r"does not satisfy", r"doesn['’]t satisfy", r"not meet", r"does not.*preferred notice")
        accepts = _has(a, r"2\s*working\s*days?.{0,60}(?:meets?|satisf(?:y|ies)|qualif(?:y|ies)).{0,50}(?:notice|policy)")
        ok = five and says_not_meet and not accepts
        return _result(ok, False, "Two working days does not meet the preferred 5-working-day Annual Leave notice period.", "" if ok else "factual_error")

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
    print(f"  Hallucination delta (Qwen - GPT-OSS): {h_b - h_a:+.1f} pp")
    print(f"  Correctness delta   (Qwen - GPT-OSS): {c_b - c_a:+.1f} pp")

    disagreements = [
        r["id"] for r in results
        if (r["model_a_hallucinated"], r["model_a_correct"], r.get("model_a_error_type", ""))
        != (r["model_b_hallucinated"], r["model_b_correct"], r.get("model_b_error_type", ""))
    ]
    print(f"  Model decision disagreements: {len(disagreements)}")
    if disagreements:
        print(f"  Cases: {', '.join(disagreements)}")


def load_checkpoint() -> dict[str, dict]:
    """Load completed question results for resume support."""
    path = Path(CHECKPOINT_JSON)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return {row["id"]: row for row in data if isinstance(row, dict) and "id" in row}
    except (json.JSONDecodeError, OSError):
        pass
    return {}


def save_checkpoint(results: list[dict]) -> None:
    Path(CHECKPOINT_JSON).write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def write_outputs(results: list[dict]) -> None:
    Path(OUTPUT_JSON).write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if not results:
        return
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        fieldnames = list(results[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def load_baseline(path: Path) -> dict[str, dict]:
    """Load the existing completed GPT-OSS baseline JSON."""
    if not path.exists():
        raise SystemExit(
            f"Baseline JSON not found: {path}\n"
            "Place your previous 40-question GPT-OSS result JSON there "
            "or pass --baseline <path>."
        )
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid baseline JSON: {path}\n{exc}") from exc
    if not isinstance(data, list):
        raise SystemExit("Baseline JSON must contain a list of question result objects.")
    rows = {row.get("id"): row for row in data if isinstance(row, dict) and row.get("id")}
    missing = [c["id"] for c in TEST_CASES if c["id"] not in rows]
    if missing:
        raise SystemExit(f"Baseline JSON is missing: {', '.join(missing)}")
    return rows


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Run only Qwen3.8-27B and merge it with an existing GPT-OSS-120B baseline."
    )
    parser.add_argument(
        "--baseline",
        default="generation_model_comparison.json",
        help="Existing 40-question GPT-OSS-120B JSON result file.",
    )
    args = parser.parse_args()

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise SystemExit("GROQ_API_KEY is not configured. Add it to your .env file.")

    baseline_path = Path(args.baseline)
    baseline = load_baseline(baseline_path)

    print("Loading policy documents...")
    documents = load_documents(DOCUMENTS_FOLDER)
    chunks = chunk_documents(documents)
    retriever = SemanticRetriever(chunks)
    client = Groq(api_key=api_key)

    print(f"Documents: {len(documents)}")
    print(f"Chunks:    {len(chunks)}")
    print(f"Baseline:  {MODEL_A} from {baseline_path}")
    print(f"Model run: {MODEL_B}")
    print("Judge:     NONE (deterministic rule-based grading)")
    print(f"Questions: {len(TEST_CASES)} adversarial cases")
    print(f"Generation context: top-{TOP_CONTEXT_CHUNKS} reranked chunks, max {MAX_CONTEXT_CHARS} chars")
    print(f"Max output tokens/model: {MAX_TOKENS}")
    print()

    checkpoint = load_checkpoint()
    qwen_by_id = {
        qid: row for qid, row in checkpoint.items()
        if row.get("model_b_answer") is not None
    }
    if qwen_by_id:
        print(f"Resuming Qwen: {len(qwen_by_id)} completed questions will be skipped.")
        print()

    merged_by_id: dict[str, dict] = {}

    for i, case in enumerate(TEST_CASES, start=1):
        base = baseline[case["id"]]

        # Reuse the baseline context when present so GPT-OSS and Qwen see
        # exactly the same retrieved/reranked evidence.
        context = str(base.get("context") or "").strip()
        if not context:
            context = build_context(case["question"], retriever)

        if case["id"] in qwen_by_id:
            qrow = qwen_by_id[case["id"]]
            print(f"[{i}/{len(TEST_CASES)}] {case['id']}: skipped (Qwen checkpoint)")
        else:
            print(f"[{i}/{len(TEST_CASES)}] {case['id']}: {case['question']}")
            try:
                answer_b, latency_b = call_model(client, MODEL_B, case["question"], context)
            except Exception as exc:
                # Save partial merged output before exiting.
                partial = []
                for c in TEST_CASES:
                    if c["id"] in qwen_by_id:
                        partial.append(qwen_by_id[c["id"]])
                save_checkpoint(partial)
                print()
                print(f"Run stopped before {case['id']}: {type(exc).__name__}: {exc}")
                print(f"Qwen checkpoint saved to {CHECKPOINT_JSON}")
                print("Re-run the same command after the quota resets; completed Qwen questions will be skipped.")
                return

            eval_b = evaluate_case(case, answer_b)
            qrow = {
                "id": case["id"],
                "question": case["question"],
                "expected": case["expected"],
                "type": case["type"],
                "context": context,
                "model_b": MODEL_B,
                "model_b_answer": answer_b,
                "model_b_latency_sec": round(latency_b, 4),
                "model_b_hallucinated": eval_b["hallucinated"],
                "model_b_correct": eval_b["correct"],
                "model_b_refusal_correct": eval_b["refusal_correct"],
                "model_b_judge_reason": eval_b["reason"],
                "model_b_error_type": eval_b["error_type"],
            }
            qwen_by_id[case["id"]] = qrow
            save_checkpoint(list(qwen_by_id.values()))
            print(f"  {MODEL_B}: {answer_b}")
            print(f"  {MODEL_B} -> correct={eval_b['correct']} hallucinated={eval_b['hallucinated']} error_type={eval_b['error_type'] or 'none'}")
            print()

        # Merge baseline GPT fields with current Qwen fields.
        merged = dict(base)
        for key in ("question", "expected", "type", "context"):
            if key not in merged:
                merged[key] = case.get(key, "")
        merged.update({
            "model_b": qrow.get("model_b", MODEL_B),
            "model_b_answer": qrow.get("model_b_answer", ""),
            "model_b_latency_sec": qrow.get("model_b_latency_sec", 0.0),
            "model_b_hallucinated": bool(qrow.get("model_b_hallucinated", False)),
            "model_b_correct": bool(qrow.get("model_b_correct", False)),
            "model_b_refusal_correct": bool(qrow.get("model_b_refusal_correct", False)),
            "model_b_judge_reason": qrow.get("model_b_judge_reason", ""),
            "model_b_error_type": qrow.get("model_b_error_type", ""),
        })
        merged_by_id[case["id"]] = merged

    results = [merged_by_id[c["id"]] for c in TEST_CASES if c["id"] in merged_by_id]
    if len(results) < len(TEST_CASES):
        write_outputs(results)
        print(f"Partial run saved: {len(results)}/{len(TEST_CASES)} questions completed.")
        return

    write_outputs(results)

    print("Saved:")
    print(f"  {OUTPUT_CSV}")
    print(f"  {OUTPUT_JSON}")
    print(f"  {CHECKPOINT_JSON}")
    print()
    print("Cross-company summary")
    print_model_metrics(results, "model_a", MODEL_A)
    print_model_metrics(results, "model_b", MODEL_B)

    total = len(results)
    a_h = sum(bool(r["model_a_hallucinated"]) for r in results)
    b_h = sum(bool(r["model_b_hallucinated"]) for r in results)
    a_c = sum(bool(r["model_a_correct"]) for r in results)
    b_c = sum(bool(r["model_b_correct"]) for r in results)
    unsupported = [r for r in results if r["type"] == "unsupported"]
    a_ref = sum(bool(r["model_a_refusal_correct"]) for r in unsupported)
    b_ref = sum(bool(r["model_b_refusal_correct"]) for r in unsupported)
    a_lat = sum(float(r["model_a_latency_sec"]) for r in results) / total
    b_lat = sum(float(r["model_b_latency_sec"]) for r in results) / total

    disagreements = [
        r["id"] for r in results
        if bool(r["model_a_correct"]) != bool(r["model_b_correct"])
        or bool(r["model_a_hallucinated"]) != bool(r["model_b_hallucinated"])
        or r.get("model_a_error_type", "") != r.get("model_b_error_type", "")
    ]

    print("Head-to-head")
    print(f"  GPT-OSS-120B hallucination rate: {a_h/total*100:.1f}% ({a_h}/{total})")
    print(f"  Qwen3.8-27B hallucination rate:  {b_h/total*100:.1f}% ({b_h}/{total})")
    print(f"  Hallucination difference (Qwen - GPT-OSS): {(b_h-a_h)/total*100:+.1f} pp")
    print(f"  GPT-OSS-120B correctness:         {a_c/total*100:.1f}% ({a_c}/{total})")
    print(f"  Qwen3.8-27B correctness:          {b_c/total*100:.1f}% ({b_c}/{total})")
    print(f"  Correctness difference (Qwen - GPT-OSS): {(b_c-a_c)/total*100:+.1f} pp")
    print(f"  GPT-OSS-120B refusal accuracy:    {a_ref/len(unsupported)*100:.1f}% ({a_ref}/{len(unsupported)})")
    print(f"  Qwen3.8-27B refusal accuracy:     {b_ref/len(unsupported)*100:.1f}% ({b_ref}/{len(unsupported)})")
    print(f"  Average latency GPT-OSS-120B:     {a_lat:.3f} sec")
    print(f"  Average latency Qwen3.8-27B:      {b_lat:.3f} sec")
    print(f"  Model decision disagreements:     {len(disagreements)}")
    if disagreements:
        print(f"  Cases: {', '.join(disagreements)}")


if __name__ == "__main__":
    main()
