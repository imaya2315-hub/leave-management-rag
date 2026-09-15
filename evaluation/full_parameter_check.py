"""
Full 40-question validation with improved deterministic grading.

Models/configurations:

1. GPT-OSS-20B
   temperature = 0.0
   max_tokens  = 200
   top_p       = 1.0

2. Qwen3.8-27B
   temperature = 0.2
   max_tokens  = 120
   top_p       = 0.8

Important:
    - Same 40 questions
    - Same RAG retrieval
    - Same top-10 retrieval
    - Same top-3 reranking
    - Same prompt
    - No LLM judge
    - Improved deterministic evaluator
    - Checkpoint after every question
"""

from __future__ import annotations

import csv
import json
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

from rag_lab.ingestion import load_documents
from rag_lab.chunking import chunk_documents
from rag_lab.retrieval import SemanticRetriever

from generation_model_comparison import (
    TEST_CASES,
    build_context,
)

load_dotenv()


# ============================================================
# FILES
# ============================================================

OUTPUT_JSON = "full_parameter_validation_v2.json"
OUTPUT_CSV = "full_parameter_validation_v2.csv"
CHECKPOINT_JSON = "full_parameter_validation_v2_checkpoint.json"


# ============================================================
# CONFIGURATIONS
# ============================================================

CONFIGURATIONS = [
    {
        "name": "gpt_oss_20b_longer_output",
        "model": "openai/gpt-oss-20b",
        "temperature": 0.0,
        "max_tokens": 200,
        "top_p": 1.0,
    },
    {
        "name": "qwen_temp02_topp08",
        "model": "qwen/qwen3.8-27b",
        "temperature": 0.2,
        "max_tokens": 120,
        "top_p": 0.8,
    },
]


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are the Leave Management Assistant for a company.

Answer the employee's question using ONLY the supplied policy context.

Rules:
1. Never invent information.
2. Never use outside knowledge.
3. Treat the supplied policy context as the source of truth.
4. Answer the actual question directly.
5. Do not repeat or dump the policy context.
6. Keep the answer concise, normally 1 to 4 sentences.
7. State numbers, dates, eligibility, carry-forward,
   holidays, approval rules, and other policy facts only
   when supported by the context.
8. Do not infer policy rules that are not stated.
9. Verify the user's premise against the supplied context.
10. For unsupported questions, explicitly state that
    the information is not provided in the policy.
11. Return only the final answer intended for the employee.
"""


# ============================================================
# TEXT HELPERS
# ============================================================

def norm(text: str) -> str:
    """Normalize text for deterministic evaluation."""

    text = (text or "").lower()

    text = (
        text.replace("’", "'")
        .replace("–", "-")
        .replace("—", "-")
        .replace("-", "-")
    )

    text = re.sub(r"[^a-z0-9%./ -]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def has(text: str, pattern: str) -> bool:
    return bool(re.search(pattern, text, flags=re.I))


def any_has(text: str, patterns: list[str]) -> bool:
    return any(has(text, p) for p in patterns)


def explicit_refusal(text: str) -> bool:
    """
    Detect refusal / abstention wording.

    This intentionally includes several formulations such as:
    "does not include"
    "does not provide"
    "not mentioned"
    "not specified"
    """

    t = norm(text)

    patterns = [
        r"\bnot provided\b",
        r"\bnot specified\b",
        r"\bnot mentioned\b",
        r"\bnot stated\b",
        r"\bnot available in (?:the )?(?:policy|context)\b",
        r"\bdoes not (?:provide|include|specify|mention|state|contain|address)\b",
        r"\bdoesn't (?:provide|include|specify|mention|state|contain|address)\b",
        r"\bdo not (?:provide|include|specify|mention|state|contain|address)\b",
        r"\bdoesn't specify\b",
        r"\bnot enough information\b",
        r"\binsufficient information\b",
        r"\bcannot be answered\b",
        r"\bcan't be answered\b",
        r"\bunable to answer\b",
        r"\bnot covered by (?:the )?policy\b",
        r"\bnot addressed by (?:the )?policy\b",
        r"\bthe policy does not\b.*\b(?:provide|include|specify|mention|state|contain|address)\b",
    ]

    return any_has(t, patterns)


def is_negative_assertion(text: str, fact_pattern: str) -> bool:
    """
    Detect whether a fact is explicitly negated.

    Example:
        "2 working days does not meet the 5-day requirement"

    should NOT be interpreted as saying that
    "2 working days meets the requirement".
    """

    patterns = [
        rf"\bnot\b[^.?!]{{0,80}}\b{fact_pattern}\b",
        rf"\bdoes not\b[^.?!]{{0,80}}\b{fact_pattern}\b",
        rf"\bdo not\b[^.?!]{{0,80}}\b{fact_pattern}\b",
        rf"\bdon't\b[^.?!]{{0,80}}\b{fact_pattern}\b",
        rf"\bincorrect\b[^.?!]{{0,80}}\b{fact_pattern}\b",
        rf"\bfalse\b[^.?!]{{0,80}}\b{fact_pattern}\b",
        rf"\bno\b[^.?!]{{0,80}}\b{fact_pattern}\b",
    ]

    return any_has(text, patterns)


# ============================================================
# EVALUATOR
# ============================================================

def result(
    *,
    correct: bool,
    hallucinated: bool,
    refusal_correct: bool,
    reason: str,
) -> dict:

    return {
        "correct": bool(correct),
        "hallucinated": bool(hallucinated),
        "refusal_correct": bool(refusal_correct),
        "reason": reason,
    }


def evaluate_case(case: dict, answer: str) -> dict:

    qid = case["id"]
    case_type = case["type"]

    a = norm(answer)

    # Empty response is a failure, but NOT hallucination.
    if not a:
        return result(
            correct=False,
            hallucinated=False,
            refusal_correct=False,
            reason="Model returned an empty answer.",
        )

    # ========================================================
    # UNSUPPORTED
    # ========================================================

    if case_type == "unsupported":

        refused = explicit_refusal(answer)

        return result(
            correct=refused,
            hallucinated=not refused,
            refusal_correct=refused,
            reason=(
                "Unsupported case: correct behavior is explicit abstention."
            ),
        )

    # ========================================================
    # Q01
    # ========================================================

    if qid == "Q01":

        correct = (
            has(a, r"\b20\b")
            and has(a, r"annual leave")
        )

        wrong = (
            has(a, r"\b(?:5|10|15|25|30)\b")
            and has(a, r"annual leave")
        )

        return result(
            correct=correct and not wrong,
            hallucinated=wrong,
            refusal_correct=False,
            reason="Annual Leave allowance is 20 days/year.",
        )

    # ========================================================
    # Q02
    # ========================================================

    if qid == "Q02":

        correct = (
            has(a, r"\b10\b")
            and has(a, r"sick leave")
        )

        wrong = (
            has(a, r"\b(?:5|8|12|15|20)\b")
            and has(a, r"sick leave")
        )

        return result(
            correct=correct and not wrong,
            hallucinated=wrong,
            refusal_correct=False,
            reason="Sick Leave allowance is 10 days/year.",
        )

    # ========================================================
    # Q03
    # ========================================================

    if qid == "Q03":

        correct = (
            has(a, r"\b5\b")
            and has(a, r"casual leave")
        )

        wrong = (
            has(a, r"\b(?:2|8|10|20)\b")
            and has(a, r"casual leave")
        )

        return result(
            correct=correct and not wrong,
            hallucinated=wrong,
            refusal_correct=False,
            reason="Casual Leave allowance is 5 days/year.",
        )

    # ========================================================
    # Q04 / Q11
    # ========================================================

    if qid in {"Q04", "Q11"}:

        has_90 = has(a, r"\b90\b\s*days?")

        denies_immediate = any_has(
            a,
            [
                r"\bnot\b.*\bimmediately\b",
                r"\bno\b.*\bimmediately\b",
                r"\bafter\s+90\s+days?\b",
                r"\bonly after\s+90\s+days?\b",
                r"\bmust complete\s+90\s+days?\b",
                r"\bnot eligible\b.*\b90\b",
            ],
        )

        wrong_immediate = any_has(
            a,
            [
                r"\beligible immediately\b",
                r"\bavailable immediately\b",
                r"\beligible right away\b",
                r"\bavailable right away\b",
            ],
        )

        return result(
            correct=has_90 and denies_immediate and not wrong_immediate,
            hallucinated=wrong_immediate,
            refusal_correct=False,
            reason="Annual Leave eligibility starts after 90 days.",
        )

    # ========================================================
    # Q05
    # ========================================================

    if qid == "Q05":

        correct = any_has(
            a,
            [
                r"\bfirst working day\b",
                r"\bfrom day one\b",
                r"\bday one\b",
            ],
        )

        wrong = any_has(
            a,
            [
                r"\bafter 90 days\b",
                r"\bafter probation\b",
                r"\bnot available\b.*\bsick leave\b",
                r"\bunavailable\b.*\bsick leave\b",
            ],
        )

        return result(
            correct=correct and not wrong,
            hallucinated=wrong,
            refusal_correct=False,
            reason="Sick Leave is available from the first working day.",
        )

    # ========================================================
    # Q06
    # ========================================================

    if qid == "Q06":

        carry = has(a, r"carry(?:ing|ied)? forward")

        five = (
            has(a, r"\b5\b")
            and any_has(a, [r"\bup to\b", r"\bmaximum\b", r"\bonly\b", r"\bat most\b"])
        )

        wrong = any_has(
            a,
            [
                r"\bunlimited\b",
                r"\bno cap\b",
                r"\bwithout a cap\b",
                r"\b6\b.*carry",
                r"\b10\b.*carry",
                r"\b20\b.*carry",
            ],
        )

        return result(
            correct=carry and five and not wrong,
            hallucinated=wrong,
            refusal_correct=False,
            reason="Up to 5 Annual Leave days may be carried forward.",
        )

    # ========================================================
    # Q07
    # ========================================================

    if qid == "Q07":

        correct = (
            has(a, r"march\s*31")
            and any_has(
                a,
                [
                    r"following year",
                    r"following calendar year",
                    r"next year",
                    r"next calendar year",
                ],
            )
        )

        wrong = any_has(
            a,
            [
                r"march\s*(?:1|15|30)\b",
                r"\bapril\b",
                r"\bjune\b",
                r"\bdecember\b",
            ],
        )

        return result(
            correct=correct and not wrong,
            hallucinated=wrong,
            refusal_correct=False,
            reason="Carried-forward Annual Leave expires March 31 of the following year.",
        )

    # ========================================================
    # Q08
    # ========================================================

    if qid == "Q08":

        correct = (
            has(a, r"sick leave")
            and any_has(
                a,
                [
                    r"does not carry forward",
                    r"doesn't carry forward",
                    r"do not carry forward",
                    r"does not roll over",
                    r"doesn't roll over",
                    r"does not carry over",
                ],
            )
        )

        positive_carry = (
            has(a, r"sick leave")
            and any_has(
                a,
                [
                    r"carry forward",
                    r"roll over",
                    r"carry over",
                ],
            )
            and not any_has(
                a,
                [
                    r"does not carry",
                    r"doesn't carry",
                    r"do not carry",
                    r"does not roll",
                    r"doesn't roll",
                ],
            )
        )

        return result(
            correct=correct and not positive_carry,
            hallucinated=positive_carry,
            refusal_correct=False,
            reason="Unused Sick Leave does not carry forward.",
        )

    # ========================================================
    # Q09
    # ========================================================

    if qid == "Q09":

        correct = (
            has(a, r"casual leave")
            and any_has(
                a,
                [
                    r"does not carry forward",
                    r"doesn't carry forward",
                    r"do not carry forward",
                    r"does not roll over",
                    r"doesn't roll over",
                ],
            )
        )

        positive_carry = (
            has(a, r"casual leave")
            and any_has(
                a,
                [
                    r"carry forward",
                    r"roll over",
                ],
            )
            and not any_has(
                a,
                [
                    r"does not carry",
                    r"doesn't carry",
                    r"do not carry",
                    r"does not roll",
                    r"doesn't roll",
                ],
            )
        )

        return result(
            correct=correct and not positive_carry,
            hallucinated=positive_carry,
            refusal_correct=False,
            reason="Unused Casual Leave does not carry forward.",
        )

    # ========================================================
    # Q10
    # ========================================================

    if qid == "Q10":

        correct = has(a, r"\b5\b\s*working days?")

        wrong = (
            has(a, r"\b(?:10|3|2)\b\s*working days?")
            and not has(a, r"\bnot\b.*\b(?:10|3|2)\b")
        )

        return result(
            correct=correct and not wrong,
            hallucinated=wrong,
            refusal_correct=False,
            reason="Annual Leave notice is 5 working days wherever possible.",
        )

    # ========================================================
    # Q12
    # ========================================================

    if qid == "Q12":

        correct = any_has(
            a,
            [
                r"first working day",
                r"from day one",
                r"available.*first working day",
            ],
        ) and any_has(
            a,
            [
                r"premise.*incorrect",
                r"premise.*false",
                r"not available during probation.*incorrect",
                r"not available during probation.*false",
                r"incorrect.*not available during probation",
                r"false.*not available during probation",
                r"does not start.*probation",
                r"not only after probation",
            ],
        )

        # A genuinely wrong claim would positively assert
        # that Sick Leave starts after probation.
        wrong = any_has(
            a,
            [
                r"sick leave starts after probation",
                r"sick leave is available only after probation",
                r"sick leave is not available during probation",
                r"sick leave starts after 90 days",
            ],
        )

        # But do NOT mark it wrong if the statement is explicitly
        # rejecting that premise.
        corrected_premise = any_has(
            a,
            [
                r"premise.*incorrect",
                r"premise.*false",
                r"incorrect.*not available during probation",
                r"false.*not available during probation",
                r"does not start.*after probation",
            ],
        )

        return result(
            correct=correct and not (wrong and not corrected_premise),
            hallucinated=wrong and not corrected_premise,
            refusal_correct=False,
            reason="Sick Leave is available from the first working day.",
        )

    # ========================================================
    # Q13
    # ========================================================

    if qid == "Q13":

        correct = (
            any_has(
                a,
                [
                    r"does not carry forward",
                    r"doesn't carry forward",
                    r"do not carry forward",
                ],
            )
        )

        wrong = any_has(
            a,
            [
                r"carry forward.*10",
                r"10.*carry forward",
            ],
        )

        return result(
            correct=correct and not wrong,
            hallucinated=wrong,
            refusal_correct=False,
            reason="Sick Leave does not carry forward.",
        )

    # ========================================================
    # Q14
    # ========================================================

    if qid == "Q14":

        correct = any_has(
            a,
            [
                r"does not carry forward",
                r"doesn't carry forward",
                r"does not roll over",
                r"doesn't roll over",
            ],
        )

        wrong = any_has(
            a,
            [
                r"2 years",
                r"two years",
                r"two-year",
                r"carry.*two years",
            ],
        )

        return result(
            correct=correct and not wrong,
            hallucinated=wrong,
            refusal_correct=False,
            reason="Casual Leave does not carry forward.",
        )

    # ========================================================
    # Q15
    # ========================================================

    if qid == "Q15":

        correct = (
            has(a, r"\b5\b")
            and has(a, r"casual leave")
            and (
                has(a, r"\bnot\s+8\b")
                or has(a, r"\binstead of\s+8\b")
                or has(a, r"\brather than\s+8\b")
                or has(a, r"not\s+8")
            )
        )

        wrong = (
            has(a, r"\b8\b")
            and has(a, r"casual leave")
            and not any_has(
                a,
                [
                    r"not\s+8",
                    r"instead of\s+8",
                    r"rather than\s+8",
                ],
            )
        )

        return result(
            correct=correct and not wrong,
            hallucinated=wrong,
            refusal_correct=False,
            reason="Casual Leave is 5 days/year, not 8.",
        )

    # ========================================================
    # Q16
    # ========================================================

    if qid == "Q16":

        has_cap = (
            has(a, r"\b5\b")
            and any_has(
                a,
                [
                    r"up to",
                    r"maximum",
                    r"only",
                    r"at most",
                ],
            )
        )

        has_deadline = has(a, r"march\s*31")

        wrong = any_has(
            a,
            [
                r"without a cap",
                r"no cap",
                r"unlimited",
            ],
        )

        return result(
            correct=has_cap and has_deadline and not wrong,
            hallucinated=wrong,
            refusal_correct=False,
            reason="Carry-forward is capped at 5 days and must be used by March 31.",
        )

    # ========================================================
    # Q17
    # ========================================================

    if qid == "Q17":

        has5 = has(a, r"\b5\b\s*working days?")

        rejects10 = any_has(
            a,
            [
                r"not\s+10",
                r"no.*10 working days",
                r"incorrect.*10",
            ],
        )

        wrong = (
            has(a, r"\b10\b\s*working days?")
            and not rejects10
        )

        return result(
            correct=has5 and not wrong,
            hallucinated=wrong,
            refusal_correct=False,
            reason="Annual Leave notice period is 5 working days, not 10.",
        )

    # ========================================================
    # Q18
    # ========================================================

    if qid == "Q18":

        timing = any_has(
            a,
            [
                r"within.*3 working days?.*return",
                r"3 working days?.*after.*return",
                r"within.*3 working days?.*returning",
            ],
        )

        not_before = any_has(
            a,
            [
                r"not before",
                r"not.*before.*leave",
                r"after returning",
                r"returning to work",
            ],
        )

        wrong = (
            any_has(
                a,
                [
                    r"before the leave starts",
                    r"before leave starts",
                    r"before the leave begins",
                ],
            )
            and not not_before
        )

        return result(
            correct=timing and not_before and not wrong,
            hallucinated=wrong,
            refusal_correct=False,
            reason="Certificate is shared within 3 working days of returning.",
        )

    # ========================================================
    # Q19
    # ========================================================

    if qid == "Q19":

        weekend = any_has(
            a,
            [
                r"saturday.*standard weekend",
                r"saturday.*weekend",
                r"sunday.*weekend",
                r"standard saturday and sunday weekend",
                r"saturdays.*not treated as working days",
                r"saturday.*not.*working",
            ],
        )

        # Only count a positive assertion that Saturday IS a working day.
        positive_working_day = any_has(
            a,
            [
                r"saturday is a working day",
                r"saturday is treated as a working day",
                r"saturday.*counted as a working day",
                r"saturday.*treated as working day",
            ],
        )

        # Negated forms are not hallucinations.
        negated = any_has(
            a,
            [
                r"saturday.*not treated as a working day",
                r"saturday.*not.*working day",
                r"saturday.*not counted as a working day",
                r"saturdays are not treated as working days",
            ],
        )

        hallucinated = positive_working_day and not negated

        return result(
            correct=weekend and not hallucinated,
            hallucinated=hallucinated,
            refusal_correct=False,
            reason="Saturday and Sunday are the standard weekend.",
        )

    # ========================================================
    # Q20
    # ========================================================

    if qid == "Q20":

        holiday = any_has(
            a,
            [
                r"august\s*15.*holiday",
                r"holiday.*august\s*15",
                r"independence day",
                r"august\s*15.*independence",
            ],
        )

        no_leave = any_has(
            a,
            [
                r"do not need.*leave",
                r"don't need.*leave",
                r"no need.*leave",
                r"not need.*leave",
            ],
        )

        wrong = any_has(
            a,
            [
                r"august\s*15.*not a holiday",
                r"not a holiday.*august\s*15",
            ],
        )

        return result(
            correct=holiday and no_leave and not wrong,
            hallucinated=wrong,
            refusal_correct=False,
            reason="August 15 is Independence Day and a listed holiday.",
        )

    # ========================================================
    # Q31
    # ========================================================

    if qid == "Q31":

        max2 = any_has(
            a,
            [
                r"up to 2",
                r"no more than 2",
                r"only 2",
                r"more than 2.*approval",
                r"3.*approval",
            ],
        )

        approval = has(a, r"approval|manager")

        wrong = any_has(
            a,
            [
                r"3.*without.*approval",
                r"3.*without.*manager",
                r"no approval needed",
                r"without manager approval",
            ],
        )

        # Explicit rejection should override a quoted premise.
        rejection = any_has(
            a,
            [
                r"cannot be taken.*without",
                r"requires.*approval",
                r"must.*approval",
            ],
        )

        return result(
            correct=max2 and approval and not (wrong and not rejection),
            hallucinated=wrong and not rejection,
            refusal_correct=False,
            reason="More than 2 consecutive Casual Leave days require advance approval.",
        )

    # ========================================================
    # Q32
    # ========================================================

    if qid == "Q32":

        certificate = has(a, r"certificate")

        no_type = any_has(
            a,
            [
                r"does not specify.*doctor",
                r"does not specify.*type",
                r"not specify.*doctor",
                r"not specify.*type",
                r"does not mention.*doctor",
                r"does not mention.*type",
                r"no particular type of doctor",
                r"not specified.*doctor",
            ],
        )

        doctor_type = any_has(
            a,
            [
                r"\bgp\b",
                r"general practitioner",
                r"specialist",
                r"physician",
                r"surgeon",
                r"internist",
                r"cardiologist",
                r"orthopedic",
            ],
        )

        return result(
            correct=certificate and no_type and not doctor_type,
            hallucinated=doctor_type,
            refusal_correct=certificate and no_type,
            reason="Policy specifies certificate timing but not doctor type.",
        )

    # ========================================================
    # Q33
    # ========================================================

    if qid == "Q33":

        correct_date = any_has(
            a,
            [
                r"march\s*31,?\s*2026",
                r"march\s*31\s+2026",
                r"following year",
            ],
        )

        rejects_2027 = any_has(
            a,
            [
                r"not 2027",
                r"cannot.*2027",
                r"not.*into 2027",
                r"not.*until.*2027",
            ],
        )

        wrong = any_has(
            a,
            [
                r"march\s*31,?\s*2027",
                r"until 2027",
                r"through 2027",
                r"carry.*into 2027",
            ],
        )

        return result(
            correct=(correct_date or rejects_2027) and not wrong,
            hallucinated=wrong,
            refusal_correct=False,
            reason="2025 carry-forward expires March 31, 2026.",
        )

    # ========================================================
    # Q34
    # ========================================================

    if qid == "Q34":

        excluded = any_has(
            a,
            [
                r"public holidays?.*excluded.*leave",
                r"public holidays?.*excluded from leave",
                r"holidays?.*excluded from leave-day calculations",
                r"not charged leave",
            ],
        )

        overbroad = any_has(
            a,
            [
                r"every public holiday is always",
                r"all public holidays.*always",
                r"every public holiday.*every leave",
            ],
        )

        qualified = any_has(
            a,
            [
                r"listed",
                r"company holiday calendar",
                r"company-wide",
                r"supplied policy",
                r"policy context",
            ],
        )

        return result(
            correct=excluded and (qualified or not overbroad),
            hallucinated=overbroad and not qualified,
            refusal_correct=False,
            reason="Listed/company-wide holidays are excluded from leave calculations.",
        )

    # ========================================================
    # Q35
    # ========================================================

    if qid == "Q35":

        manager = any_has(
            a,
            [
                r"direct manager",
                r"manager approves",
                r"manager.*approve",
                r"manager.*reject",
            ],
        )

        two_days = has(a, r"\b2\b\s*working days?")

        return result(
            correct=manager and two_days,
            hallucinated=False,
            refusal_correct=False,
            reason="Direct manager approval and 2-working-day expected response.",
        )

    # ========================================================
    # Q36
    # ========================================================

    if qid == "Q36":

        not_eligible = any_has(
            a,
            [
                r"not.*eligible",
                r"not yet eligible",
                r"cannot.*annual leave",
                r"cannot take.*annual leave",
                r"not.*reached.*90 days",
            ],
        )

        ninety = has(a, r"\b90\b\s*days?")

        positive_eligible = any_has(
            a,
            [
                r"\beligible for annual leave\b",
                r"\bcan take annual leave\b",
                r"\bcan take.*annual leave",
            ],
        )

        # Positive eligibility counts only if it is NOT directly negated.
        negated_positive = any_has(
            a,
            [
                r"not eligible",
                r"not.*eligible for annual leave",
                r"cannot.*annual leave",
                r"not yet eligible",
            ],
        )

        wrong = positive_eligible and not negated_positive

        return result(
            correct=not_eligible and ninety and not wrong,
            hallucinated=wrong,
            refusal_correct=False,
            reason="At 45 days, Annual Leave eligibility has not started.",
        )

    # ========================================================
    # Q37
    # ========================================================

    if qid == "Q37":

        five = has(a, r"\b5\b")

        excess_handled = any_has(
            a,
            [
                r"6th.*forfeit",
                r"sixth.*forfeit",
                r"only 5",
                r"only up to 5",
                r"excess.*forfeit",
                r"remaining.*forfeited",
            ],
        )

        wrong = any_has(
            a,
            [
                r"all 6",
                r"6.*carry forward",
                r"6.*carried forward",
            ],
        )

        return result(
            correct=five and excess_handled and not wrong,
            hallucinated=wrong,
            refusal_correct=False,
            reason="Only up to 5 Annual Leave days carry forward.",
        )

    # ========================================================
    # Q38
    # ========================================================

    if qid == "Q38":

        remaining_3 = any_has(
            a,
            [
                r"\b3\b.*remaining",
                r"\b3\b.*left",
                r"remaining.*\b3\b",
                r"left.*\b3\b",
            ],
        )

        next_year_reset = any_has(
            a,
            [
                r"next calendar year",
                r"start of the next year",
                r"start of next year",
                r"new calendar year",
            ],
        )

        wrong = any_has(
            a,
            [
                r"still have 5",
                r"reset.*during the year",
                r"restore.*used",
                r"back to 5.*same year",
                r"back to 5.*current year",
            ],
        )

        return result(
            correct=remaining_3 and next_year_reset and not wrong,
            hallucinated=wrong,
            refusal_correct=False,
            reason="Using 2 of 5 leaves 3; reset occurs next year.",
        )

    # ========================================================
    # Q39
    # ========================================================

    if qid == "Q39":

        four = has(a, r"\b4\b\s*(?:consecutive|working)?\s*days?")

        approval = any_has(
            a,
            [
                r"approval",
                r"manager",
                r"requires.*approval",
                r"must.*approval",
            ],
        )

        max2 = any_has(
            a,
            [
                r"up to 2",
                r"only 2",
                r"no more than 2",
                r"more than 2.*approval",
            ],
        )

        wrong = any_has(
            a,
            [
                r"4.*without.*approval",
                r"no approval needed",
                r"4.*without.*manager",
            ],
        )

        rejection = any_has(
            a,
            [
                r"no.*without",
                r"cannot.*without",
                r"must.*approval",
                r"requires.*approval",
            ],
        )

        return result(
            correct=four and approval and max2 and not (wrong and not rejection),
            hallucinated=wrong and not rejection,
            refusal_correct=False,
            reason="Four consecutive Casual Leave days require approval.",
        )

    # ========================================================
    # Q40
    # ========================================================

    if qid == "Q40":

        has5 = has(a, r"\b5\b\s*working days?")

        correctly_rejects_2 = any_has(
            a,
            [
                r"2 working days.*does not meet",
                r"2 working days.*doesn't meet",
                r"2 working days.*not meet",
                r"2 working days.*does not satisfy",
                r"2 working days.*not satisfy",
                r"2 working days.*is insufficient",
                r"2 working days.*falls short",
                r"not.*meet.*2 working days",
            ],
        )

        # A real incorrect answer must positively say that
        # 2 working days DOES meet the requirement.
        wrongly_accepts_2 = any_has(
            a,
            [
                r"\byes\b.*2 working days",
                r"2 working days.*meets the",
                r"2 working days.*satisfies the",
                r"2 working days.*is sufficient",
            ],
        )

        return result(
            correct=has5 and correctly_rejects_2 and not wrongly_accepts_2,
            hallucinated=wrongly_accepts_2,
            refusal_correct=False,
            reason="Two working days does not meet the preferred 5-day notice period.",
        )

    raise ValueError(
        f"No deterministic grading rule defined for {qid}"
    )


# ============================================================
# CHECKPOINT
# ============================================================

def load_checkpoint() -> list[dict]:

    path = Path(CHECKPOINT_JSON)

    if not path.exists():
        return []

    try:
        with path.open(
            "r",
            encoding="utf-8",
        ) as f:
            data = json.load(f)

        return data if isinstance(data, list) else []

    except (
        OSError,
        json.JSONDecodeError,
    ):
        print(
            "Warning: checkpoint unreadable. "
            "Starting fresh."
        )
        return []


def save_checkpoint(results: list[dict]) -> None:

    temp_path = CHECKPOINT_JSON + ".tmp"

    with open(
        temp_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            results,
            f,
            indent=2,
            ensure_ascii=False,
        )

    os.replace(
        temp_path,
        CHECKPOINT_JSON,
    )


# ============================================================
# MODEL CALL
# ============================================================

def call_model(
    client: Groq,
    config: dict,
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
        model=config["model"],
        temperature=config["temperature"],
        max_tokens=config["max_tokens"],
        top_p=config["top_p"],
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
    )

    elapsed = time.perf_counter() - start

    answer = response.choices[0].message.content or ""

    return answer.strip(), elapsed


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(rows: list[dict]) -> dict:

    total = len(rows)

    if total == 0:
        return {
            "total": 0,
            "correct": 0,
            "hallucinated": 0,
            "correctness_pct": 0.0,
            "hallucination_pct": 0.0,
            "refusal_accuracy_pct": 0.0,
            "avg_latency_sec": 0.0,
        }

    correct = sum(
        bool(r["correct"])
        for r in rows
    )

    hallucinated = sum(
        bool(r["hallucinated"])
        for r in rows
    )

    unsupported = [
        r
        for r in rows
        if r["type"] == "unsupported"
    ]

    refusal_correct = sum(
        bool(r["refusal_correct"])
        for r in unsupported
    )

    refusal_accuracy = (
        refusal_correct / len(unsupported) * 100
        if unsupported
        else 0.0
    )

    average_latency = (
        sum(float(r["latency_sec"]) for r in rows)
        / total
    )

    return {
        "total": total,
        "correct": correct,
        "hallucinated": hallucinated,
        "correctness_pct": correct / total * 100,
        "hallucination_pct": hallucinated / total * 100,
        "refusal_accuracy_pct": refusal_accuracy,
        "avg_latency_sec": average_latency,
    }


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(results: list[dict]) -> None:

    with open(
        OUTPUT_JSON,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            results,
            f,
            indent=2,
            ensure_ascii=False,
        )

    if not results:
        return

    with open(
        OUTPUT_CSV,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(results[0].keys()),
        )

        writer.writeheader()
        writer.writerows(results)


# ============================================================
# MAIN
# ============================================================

def main():

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise SystemExit(
            "GROQ_API_KEY is not configured in .env"
        )

    client = Groq(api_key=api_key)

    print("=" * 72)
    print("FULL PARAMETER VALIDATION V2")
    print("=" * 72)

    print(
        f"Questions: {len(TEST_CASES)}"
    )

    print(
        f"Configurations: "
        f"{len(CONFIGURATIONS)}"
    )

    print(
        f"Planned model calls: "
        f"{len(TEST_CASES) * len(CONFIGURATIONS)}"
    )

    print()

    # --------------------------------------------------------
    # Load RAG system once
    # --------------------------------------------------------

    print("Loading policy documents...")

    documents = load_documents(
        "rag_lab/documents"
    )

    chunks = chunk_documents(documents)

    retriever = SemanticRetriever(chunks)

    print(
        f"Documents: {len(documents)}"
    )

    print(
        f"Chunks: {len(chunks)}"
    )

    print()

    # --------------------------------------------------------
    # Load checkpoint
    # --------------------------------------------------------

    results = load_checkpoint()

    completed = {
        (
            row["question_id"],
            row["config_name"],
        )
        for row in results
    }

    if results:
        print(
            f"Resuming checkpoint: "
            f"{len(results)} completed"
        )

        print()

    # --------------------------------------------------------
    # Run
    # --------------------------------------------------------

    for config in CONFIGURATIONS:

        print("=" * 72)
        print(
            f"MODEL: {config['model']}"
        )
        print(
            f"CONFIG: {config['name']}"
        )
        print(
            f"temperature={config['temperature']} "
            f"| max_tokens={config['max_tokens']} "
            f"| top_p={config['top_p']}"
        )
        print("=" * 72)

        for index, case in enumerate(
            TEST_CASES,
            start=1,
        ):

            key = (
                case["id"],
                config["name"],
            )

            if key in completed:

                print(
                    f"[{index}/{len(TEST_CASES)}] "
                    f"{case['id']} -> SKIPPED"
                )

                continue

            print(
                f"[{index}/{len(TEST_CASES)}] "
                f"{case['id']}: "
                f"{case['question']}"
            )

            # ------------------------------------------------
            # Retrieve top 10 -> rerank top 3
            # ------------------------------------------------

            context = build_context(
                case["question"],
                retriever,
            )

            # ------------------------------------------------
            # Generate
            # ------------------------------------------------

            try:

                answer, latency = call_model(
                    client,
                    config,
                    case["question"],
                    context,
                )

            except Exception as exc:

                print(
                    f"\nAPI ERROR: {exc}"
                )

                save_checkpoint(results)

                raise

            # ------------------------------------------------
            # Evaluate
            # ------------------------------------------------

            evaluation = evaluate_case(
                case,
                answer,
            )

            row = {
                "question_id": case["id"],
                "question": case["question"],
                "expected": case["expected"],
                "type": case["type"],

                "model": config["model"],
                "config_name": config["name"],

                "temperature": config["temperature"],
                "max_tokens": config["max_tokens"],
                "top_p": config["top_p"],

                "answer": answer,

                "latency_sec": round(
                    latency,
                    4,
                ),

                "correct": evaluation["correct"],
                "hallucinated": evaluation["hallucinated"],
                "refusal_correct": evaluation[
                    "refusal_correct"
                ],
                "reason": evaluation["reason"],
            }

            results.append(row)

            # Save after EVERY call.
            save_checkpoint(results)

            print(
                f"  Correct:      "
                f"{evaluation['correct']}"
            )

            print(
                f"  Hallucinated: "
                f"{evaluation['hallucinated']}"
            )

            print(
                f"  Refusal:      "
                f"{evaluation['refusal_correct']}"
            )

            print(
                f"  Latency:      "
                f"{latency:.3f}s"
            )

            print()

    # --------------------------------------------------------
    # Save final
    # --------------------------------------------------------

    save_results(results)

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("FINAL SUMMARY")
    print("=" * 72)

    grouped: dict[
        tuple[str, str],
        list[dict]
    ] = {}

    for row in results:

        key = (
            row["model"],
            row["config_name"],
        )

        grouped.setdefault(
            key,
            [],
        ).append(row)

    summaries = []

    for config in CONFIGURATIONS:

        key = (
            config["model"],
            config["name"],
        )

        rows = grouped.get(
            key,
            [],
        )

        if not rows:
            continue

        metrics = calculate_metrics(rows)

        summary = {
            "model": config["model"],
            "config": config["name"],
            **metrics,
        }

        summaries.append(summary)

        print()
        print(config["model"])

        print(
            f"  Config: "
            f"{config['name']}"
        )

        print(
            f"  Correctness: "
            f"{metrics['correctness_pct']:.1f}% "
            f"({metrics['correct']}/{metrics['total']})"
        )

        print(
            f"  Hallucination: "
            f"{metrics['hallucination_pct']:.1f}% "
            f"({metrics['hallucinated']}/{metrics['total']})"
        )

        print(
            f"  Refusal accuracy: "
            f"{metrics['refusal_accuracy_pct']:.1f}%"
        )

        print(
            f"  Average latency: "
            f"{metrics['avg_latency_sec']:.3f}s"
        )

    # --------------------------------------------------------
    # Head-to-head
    # --------------------------------------------------------

    if len(summaries) == 2:

        a = summaries[0]
        b = summaries[1]

        print()
        print("=" * 72)
        print("HEAD-TO-HEAD")
        print("=" * 72)

        print(
            f"Correctness difference "
            f"({b['model']} - {a['model']}): "
            f"{b['correctness_pct'] - a['correctness_pct']:+.1f} pp"
        )

        print(
            f"Hallucination difference "
            f"({b['model']} - {a['model']}): "
            f"{b['hallucination_pct'] - a['hallucination_pct']:+.1f} pp"
        )

        print(
            f"Refusal accuracy difference "
            f"({b['model']} - {a['model']}): "
            f"{b['refusal_accuracy_pct'] - a['refusal_accuracy_pct']:+.1f} pp"
        )

        print(
            f"Latency difference "
            f"({b['model']} - {a['model']}): "
            f"{b['avg_latency_sec'] - a['avg_latency_sec']:+.3f}s"
        )

    # --------------------------------------------------------
    # Flagged cases
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("FLAGGED CASES")
    print("=" * 72)

    for config in CONFIGURATIONS:

        print()
        print(config["model"])

        for row in results:

            if row["config_name"] != config["name"]:
                continue

            if (
                row["hallucinated"]
                or not row["correct"]
            ):
                print(
                    f"  {row['question_id']}: "
                    f"correct={row['correct']} "
                    f"hallucinated={row['hallucinated']}"
                )

                print(
                    f"    Answer: {row['answer']}"
                )

                print(
                    f"    Reason: {row['reason']}"
                )

    print()
    print("=" * 72)
    print("SAVED")
    print("=" * 72)

    print(
        f"  {OUTPUT_JSON}"
    )

    print(
        f"  {OUTPUT_CSV}"
    )

    print(
        f"  {CHECKPOINT_JSON}"
    )


if __name__ == "__main__":
    main()