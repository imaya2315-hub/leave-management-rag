"""
Parameter + model sweep for the Leave Management RAG system.

Goal:
    Find a good generation configuration without using excessive API tokens.

Design:
    - Same policy documents
    - Same chunking
    - Same semantic retrieval
    - Same top-10 retrieval
    - Same top-3 reranking
    - Same questions
    - Same system prompt
    - Deterministic evaluation
    - Only generation parameters / model change

Run:
    python parameter_model_sweep.py

Outputs:
    parameter_model_sweep.csv
    parameter_model_sweep.json
"""

from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

# Reuse your existing RAG + evaluator
from generation_model_comparison import (
    TEST_CASES,
    build_context,
    evaluate_case,
)

load_dotenv()


# ============================================================
# MODELS
# ============================================================

# Use models that you have already tested.
# Keep GPT-OSS-120B commented today because of the daily quota
# we encountered during the earlier experiment.

MODELS = [
    "openai/gpt-oss-20b",
    "qwen/qwen3.8-27b",
]

# Add this later when the 120B quota is available again:
# "openai/gpt-oss-120b"


# ============================================================
# SMALL CALIBRATION SET
# ============================================================
#
# We deliberately use 12 questions rather than all 40.
# They cover different behavior types.

SELECTED_IDS = {
    "Q01",  # supported
    "Q03",  # supported
    "Q04",  # supported eligibility
    "Q06",  # carry-forward
    "Q11",  # misleading premise
    "Q12",  # misleading premise
    "Q16",  # misleading premise
    "Q21",  # unsupported
    "Q24",  # unsupported
    "Q31",  # partial support
    "Q36",  # contradiction
    "Q40",  # contradiction
}

TEST_SET = [
    case for case in TEST_CASES
    if case["id"] in SELECTED_IDS
]


# ============================================================
# GENERATION CONFIGURATIONS
# ============================================================
#
# Baseline first.
#
# Then change one parameter at a time as much as possible.

CONFIGS = [
    {
        "name": "baseline",
        "temperature": 0.0,
        "max_tokens": 120,
        "top_p": 1.0,
    },
    {
        "name": "short_output",
        "temperature": 0.0,
        "max_tokens": 80,
        "top_p": 1.0,
    },
    {
        "name": "longer_output",
        "temperature": 0.0,
        "max_tokens": 200,
        "top_p": 1.0,
    },
    {
        "name": "temp_02",
        "temperature": 0.2,
        "max_tokens": 120,
        "top_p": 1.0,
    },
    {
        "name": "top_p_08",
        "temperature": 0.0,
        "max_tokens": 120,
        "top_p": 0.8,
    },
    {
        "name": "temp02_top_p08",
        "temperature": 0.2,
        "max_tokens": 120,
        "top_p": 0.8,
    },
]


# ============================================================
# OUTPUT
# ============================================================

OUTPUT_JSON = "parameter_model_sweep.json"
OUTPUT_CSV = "parameter_model_sweep.csv"

CHECKPOINT_JSON = "parameter_model_sweep_checkpoint.json"


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
7. State numbers, dates, eligibility, holidays, approval rules,
   and other policy facts only when supported by the context.
8. Do not infer rules that are not stated.
9. Verify the user's premise against the supplied context.
10. If the context does not contain enough information,
    clearly say that it is not provided.
11. Return only the final answer intended for the employee.
"""


# ============================================================
# MODEL CALL
# ============================================================

def call_model(
    client: Groq,
    model: str,
    question: str,
    context: str,
    config: dict,
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
        temperature=config["temperature"],
        max_tokens=config["max_tokens"],
        top_p=config["top_p"],
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )

    elapsed = time.perf_counter() - start

    answer = response.choices[0].message.content or ""

    return answer.strip(), elapsed


# ============================================================
# CHECKPOINT HELPERS
# ============================================================

def load_checkpoint() -> list[dict]:
    path = Path(CHECKPOINT_JSON)

    if not path.exists():
        return []

    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            return data

    except Exception:
        pass

    return []


def save_checkpoint(results: list[dict]) -> None:
    with open(CHECKPOINT_JSON, "w", encoding="utf-8") as f:
        json.dump(
            results,
            f,
            indent=2,
            ensure_ascii=False,
        )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(results: list[dict]) -> dict:

    total = len(results)

    if total == 0:
        return {}

    correct = sum(
        bool(r["correct"])
        for r in results
    )

    hallucinated = sum(
        bool(r["hallucinated"])
        for r in results
    )

    unsupported = [
        r for r in results
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

    avg_latency = (
        sum(r["latency_sec"] for r in results)
        / total
    )

    return {
        "total": total,
        "correctness": correct / total * 100,
        "hallucination": hallucinated / total * 100,
        "refusal_accuracy": refusal_accuracy,
        "avg_latency_sec": avg_latency,
    }


# ============================================================
# MAIN
# ============================================================

def main():
    from rag_lab.ingestion import load_documents
    from rag_lab.chunking import chunk_documents
    from rag_lab.retrieval import SemanticRetriever

    documents = load_documents("rag_lab/documents")
    chunks = chunk_documents(documents)
    retriever = SemanticRetriever(chunks)

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise SystemExit(
            "GROQ_API_KEY is not configured in .env"
        )

    client = Groq(api_key=api_key)

    print("=" * 70)
    print("PARAMETER + MODEL SWEEP")
    print("=" * 70)

    print(f"Questions: {len(TEST_SET)}")
    print(f"Models:   {len(MODELS)}")
    print(f"Configs:  {len(CONFIGS)}")

    total_calls = (
        len(TEST_SET)
        * len(MODELS)
        * len(CONFIGS)
    )

    print(f"API calls planned: {total_calls}")
    print()

    # --------------------------------------------------------
    # Load checkpoint
    # --------------------------------------------------------

    results = load_checkpoint()

    completed = {
        (
            r["question_id"],
            r["model"],
            r["config_name"],
        )
        for r in results
    }

    if results:
        print(
            f"Resuming from checkpoint: "
            f"{len(results)} completed calls"
        )

    print()

    # --------------------------------------------------------
    # Run experiment
    # --------------------------------------------------------

    for config in CONFIGS:

        for model in MODELS:

            print(
                f"\nCONFIG={config['name']} "
                f"| MODEL={model}"
            )

            for case in TEST_SET:

                key = (
                    case["id"],
                    model,
                    config["name"],
                )

                if key in completed:
                    print(
                        f"Skipping {case['id']} "
                        f"(already completed)"
                    )
                    continue

                print(
                    f"Running {case['id']} "
                    f"| temp={config['temperature']} "
                    f"| max={config['max_tokens']} "
                    f"| top_p={config['top_p']}"
                )

                try:

                    # IMPORTANT:
                    # Build the context only once per question.
                    context = build_context(
                        case["question"],
                        retriever
                    )

                except AttributeError:
                    # Safer fallback:
                    #
                    # We recreate the retriever here.
                    #
                    from rag_lab.ingestion import load_documents
                    from rag_lab.chunking import chunk_documents
                    from rag_lab.retrieval import SemanticRetriever

                    documents = load_documents(
                        "rag_lab/documents"
                    )

                    chunks = chunk_documents(documents)

                    retriever = SemanticRetriever(chunks)

                    context = build_context(
                        case["question"],
                        retriever
                    )

                answer, latency = call_model(
                    client=client,
                    model=model,
                    question=case["question"],
                    context=context,
                    config=config,
                )

                evaluation = evaluate_case(
                    case,
                    answer,
                )

                row = {
                    "question_id": case["id"],
                    "question": case["question"],
                    "type": case["type"],
                    "model": model,
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

                save_checkpoint(results)

                print(
                    f"  correct={evaluation['correct']} "
                    f"hallucinated={evaluation['hallucinated']} "
                    f"latency={latency:.3f}s"
                )

    # --------------------------------------------------------
    # Save detailed results
    # --------------------------------------------------------

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

    if results:

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

    # --------------------------------------------------------
    # Print summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    groups = {}

    for r in results:

        key = (
            r["model"],
            r["config_name"],
        )

        groups.setdefault(key, []).append(r)

    summary_rows = []

    for (model, config_name), group in groups.items():

        metrics = calculate_metrics(group)

        summary_rows.append(
            {
                "model": model,
                "config": config_name,
                **metrics,
            }
        )

        print()
        print(model)
        print(f"  Config:            {config_name}")
        print(
            f"  Correctness:       "
            f"{metrics['correctness']:.1f}%"
        )
        print(
            f"  Hallucination:     "
            f"{metrics['hallucination']:.1f}%"
        )
        print(
            f"  Refusal accuracy:  "
            f"{metrics['refusal_accuracy']:.1f}%"
        )
        print(
            f"  Average latency:   "
            f"{metrics['avg_latency_sec']:.3f}s"
        )

    print("\nBest configurations by model:")

    for model in MODELS:

        model_rows = [
            x
            for x in summary_rows
            if x["model"] == model
        ]

        if not model_rows:
            continue

        # Prefer:
        # 1. higher correctness
        # 2. lower hallucination
        # 3. lower latency

        best = sorted(
            model_rows,
            key=lambda x: (
                -x["correctness"],
                x["hallucination"],
                x["avg_latency_sec"],
            ),
        )[0]

        print(
            f"\n{model}"
            f"\n  Best config: {best['config']}"
            f"\n  Correctness: {best['correctness']:.1f}%"
            f"\n  Hallucination: {best['hallucination']:.1f}%"
            f"\n  Refusal accuracy: "
            f"{best['refusal_accuracy']:.1f}%"
            f"\n  Avg latency: "
            f"{best['avg_latency_sec']:.3f}s"
        )

    print("\nSaved:")
    print(f"  {OUTPUT_JSON}")
    print(f"  {OUTPUT_CSV}")
    print(f"  {CHECKPOINT_JSON}")


if __name__ == "__main__":
    main()