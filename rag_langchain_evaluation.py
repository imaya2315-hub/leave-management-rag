"""
Custom Orchestration vs LangChain Orchestration Scientific Evaluation Suite.

Compares:
  Pipeline A (Custom Orchestration):
    Direct Python orchestration via rag_lab.agent / SemanticRetriever / rerank
  Pipeline B (LangChain Orchestration):
    LangChainRetrieverAdapter (BaseRetriever) + LangChain RAG pipeline

Evaluates on identical sets:
  - 12-Question Baseline (EVAL_SET)
  - 19-Question Hard Set (HARD_EVAL_SET)
  - 10-Question Benchmark (Q06_Q15_SET)

Metrics measured:
  - Precision@1, Precision@3, Precision@5
  - Recall@1, Recall@3, Recall@5
  - MRR (Mean Reciprocal Rank)
  - nDCG@1, nDCG@3, nDCG@5
  - Latency (ms per query)

Results are saved to:
  evaluation_langchain_results/langchain_vs_custom_summary.csv
  evaluation_langchain_results/langchain_vs_custom_details.json

NOTE:
  - Existing evaluation results in evaluation_results/ and evaluation_pinecone_results/
    are strictly preserved and not overwritten.
"""
from __future__ import annotations

import csv
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

from rag_lab.chunking import chunk_documents
from rag_lab.evaluation import EVAL_SET, HARD_EVAL_SET
from rag_lab.ingestion import load_documents
from rag_lab.langchain_integration import (
    LangChainRetrieverAdapter,
    create_policy_retriever,
)
from rag_lab.pinecone_store import PineconeRetriever, PineconeStore
from rag_lab.reranking import rerank
from rag_lab.retrieval import SemanticRetriever
from rag_lab.sparse import BM25SparseEncoder

ROOT_DIR = Path(__file__).resolve().parent
DOCUMENTS_FOLDER = ROOT_DIR / "rag_lab" / "documents"
OUTPUT_DIR = ROOT_DIR / "evaluation_langchain_results"

CHUNK_SIZE = 80
CHUNK_OVERLAP = 20
TOP_K = 5
RERANK_TOP_N = 3

Q06_Q15_SET = [
    {
        "question": "How much Annual Leave can be carried forward to the next year?",
        "expected_title": "Carry Forward Rules",
    },
    {
        "question": "By when must carried-forward Annual Leave be used?",
        "expected_title": "Carry Forward Rules",
    },
    {
        "question": "Does unused Sick Leave carry forward?",
        "expected_title": "Carry Forward Rules",
    },
    {
        "question": "Does unused Casual Leave carry forward?",
        "expected_title": "Carry Forward Rules",
    },
    {
        "question": "How much advance notice is expected for Annual Leave?",
        "expected_title": "Leave Types And Eligibility",
    },
    {
        "question": "Are Saturday and Sunday treated as working days for leave calculations?",
        "expected_title": "Holiday Calendar",
    },
    {
        "question": "Is August 15 a company holiday?",
        "expected_title": "Holiday Calendar",
    },
    {
        "question": "Annual Leave is available immediately after joining, correct?",
        "expected_title": "Leave Types And Eligibility",
    },
    {
        "question": "Annual Leave can be carried forward without any cap, right?",
        "expected_title": "Carry Forward Rules",
    },
    {
        "question": "August 15 is not a company holiday. What leave should I apply for?",
        "expected_title": "Holiday Calendar",
    },
]


def ndcg_at_k(binary_rel: list[int], k: int) -> float:
    sub = binary_rel[:k]
    dcg = sum(r / math.log2(idx + 1) for idx, r in enumerate(sub, start=1))
    ideal_rel = sorted(sub, reverse=True)
    idcg = sum(r / math.log2(idx + 1) for idx, r in enumerate(ideal_rel, start=1))
    return dcg / idcg if idcg > 0 else (1.0 if not any(binary_rel) else 0.0)


def compute_metrics(ranked_titles: list[str], expected_title: str) -> dict[str, float]:
    rel = [1 if t == expected_title else 0 for t in ranked_titles]

    # Precision@K
    p1 = float(rel[0]) if len(rel) >= 1 else 0.0
    p3 = sum(rel[:3]) / 3.0 if len(rel) >= 3 else sum(rel) / max(1, len(rel))
    p5 = sum(rel[:5]) / 5.0 if len(rel) >= 5 else sum(rel) / max(1, len(rel))

    # Recall@K (relative to relevant retrieved candidates, min 1)
    total_relevant = max(1, sum(rel))
    r1 = sum(rel[:1]) / total_relevant
    r3 = sum(rel[:3]) / total_relevant
    r5 = sum(rel[:5]) / total_relevant

    # MRR
    mrr = 0.0
    for idx, r in enumerate(rel, start=1):
        if r == 1:
            mrr = 1.0 / idx
            break

    # nDCG@K
    ndcg1 = ndcg_at_k(rel, 1)
    ndcg3 = ndcg_at_k(rel, 3)
    ndcg5 = ndcg_at_k(rel, 5)

    return {
        "p1": p1,
        "p3": p3,
        "p5": p5,
        "r1": r1,
        "r3": r3,
        "r5": r5,
        "mrr": mrr,
        "ndcg1": ndcg1,
        "ndcg3": ndcg3,
        "ndcg5": ndcg5,
    }


def print_table(title: str, headers: list[str], rows: list[list[Any]]) -> None:
    print("\n" + "=" * 90)
    print(f" {title.upper()}")
    print("=" * 90)
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(val)))

    header_line = " | ".join(f"{h:<{col_widths[i]}}" for i, h in enumerate(headers))
    separator_line = "-+-".join("-" * col_widths[i] for i in range(len(headers)))
    print(header_line)
    print(separator_line)
    for row in rows:
        print(" | ".join(f"{str(v):<{col_widths[i]}}" for i, v in enumerate(row)))
    print("=" * 90 + "\n")


def run_evaluation():
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("[Setup] Loading policy documents and chunking...")
    docs = load_documents(str(DOCUMENTS_FOLDER))
    chunks = chunk_documents(docs, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
    print(f"[Setup] Loaded {len(docs)} documents into {len(chunks)} chunks.")

    # 1. Custom Retrievers
    print("[Setup] Initializing Custom SemanticRetriever (FAISS)...")
    custom_faiss = SemanticRetriever(chunks)

    # 2. LangChain Adapters
    print("[Setup] Initializing LangChainRetrieverAdapter...")
    langchain_faiss = LangChainRetrieverAdapter(
        underlying_retriever=custom_faiss,
        top_k=TOP_K,
        rerank_top_n=RERANK_TOP_N,
        use_reranker=True,
    )

    eval_datasets = [
        ("12-Question Baseline", EVAL_SET),
        ("19-Question Hard Set", HARD_EVAL_SET),
        ("Q06-Q15 Set", Q06_Q15_SET),
    ]

    summary_rows = []
    all_details = []

    for set_name, questions in eval_datasets:
        # Pipeline A: Custom Orchestration (Semantic + Rerank)
        custom_metrics_list = []
        custom_latencies = []

        # Pipeline B: LangChain Orchestration (Adapter + Rerank)
        langchain_metrics_list = []
        langchain_latencies = []

        for item in questions:
            q = item["question"]
            expected = item["expected_title"]

            # --- Custom Execution ---
            t0 = time.perf_counter()
            retrieved = custom_faiss.retrieve(q, top_k=TOP_K)
            reranked = rerank(q, retrieved, top_n=RERANK_TOP_N)
            custom_lat = (time.perf_counter() - t0) * 1000.0  # ms
            custom_latencies.append(custom_lat)

            custom_titles = [c["title"] for c, _ in reranked]
            c_m = compute_metrics(custom_titles, expected)
            custom_metrics_list.append(c_m)

            # --- LangChain Execution ---
            t0 = time.perf_counter()
            lc_docs = langchain_faiss.invoke(q)
            lc_lat = (time.perf_counter() - t0) * 1000.0  # ms
            langchain_latencies.append(lc_lat)

            lc_titles = [doc.metadata.get("title", "") for doc in lc_docs]
            lc_m = compute_metrics(lc_titles, expected)
            langchain_metrics_list.append(lc_m)

            all_details.append({
                "dataset": set_name,
                "question": q,
                "expected_title": expected,
                "custom_titles": custom_titles,
                "langchain_titles": lc_titles,
                "custom_latency_ms": round(custom_lat, 2),
                "langchain_latency_ms": round(lc_lat, 2),
                "retrieval_parity": custom_titles == lc_titles,
            })

        # Aggregate metrics
        def mean_m(metric_list, key):
            return sum(m[key] for m in metric_list) / len(metric_list)

        summary_rows.append({
            "Dataset": set_name,
            "Pipeline": "Custom Orchestration",
            "P@1": round(mean_m(custom_metrics_list, "p1"), 4),
            "P@3": round(mean_m(custom_metrics_list, "p3"), 4),
            "P@5": round(mean_m(custom_metrics_list, "p5"), 4),
            "R@1": round(mean_m(custom_metrics_list, "r1"), 4),
            "R@3": round(mean_m(custom_metrics_list, "r3"), 4),
            "R@5": round(mean_m(custom_metrics_list, "r5"), 4),
            "MRR": round(mean_m(custom_metrics_list, "mrr"), 4),
            "nDCG@1": round(mean_m(custom_metrics_list, "ndcg1"), 4),
            "nDCG@3": round(mean_m(custom_metrics_list, "ndcg3"), 4),
            "nDCG@5": round(mean_m(custom_metrics_list, "ndcg5"), 4),
            "Avg Latency (ms)": round(sum(custom_latencies) / len(custom_latencies), 2),
        })

        summary_rows.append({
            "Dataset": set_name,
            "Pipeline": "LangChain Orchestration",
            "P@1": round(mean_m(langchain_metrics_list, "p1"), 4),
            "P@3": round(mean_m(langchain_metrics_list, "p3"), 4),
            "P@5": round(mean_m(langchain_metrics_list, "p5"), 4),
            "R@1": round(mean_m(langchain_metrics_list, "r1"), 4),
            "R@3": round(mean_m(langchain_metrics_list, "r3"), 4),
            "R@5": round(mean_m(langchain_metrics_list, "r5"), 4),
            "MRR": round(mean_m(langchain_metrics_list, "mrr"), 4),
            "nDCG@1": round(mean_m(langchain_metrics_list, "ndcg1"), 4),
            "nDCG@3": round(mean_m(langchain_metrics_list, "ndcg3"), 4),
            "nDCG@5": round(mean_m(langchain_metrics_list, "ndcg5"), 4),
            "Avg Latency (ms)": round(sum(langchain_latencies) / len(langchain_latencies), 2),
        })

    # Save CSV & JSON
    csv_path = OUTPUT_DIR / "langchain_vs_custom_summary.csv"
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    json_path = OUTPUT_DIR / "langchain_vs_custom_details.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_details, f, indent=2)

    # Print Table
    headers = ["Dataset", "Pipeline", "P@1", "P@3", "P@5", "R@1", "R@3", "R@5", "MRR", "nDCG@3", "nDCG@5", "Latency"]
    table_rows = [
        [
            r["Dataset"],
            r["Pipeline"],
            f"{r['P@1']:.2f}",
            f"{r['P@3']:.2f}",
            f"{r['P@5']:.2f}",
            f"{r['R@1']:.2f}",
            f"{r['R@3']:.2f}",
            f"{r['R@5']:.2f}",
            f"{r['MRR']:.4f}",
            f"{r['nDCG@3']:.4f}",
            f"{r['nDCG@5']:.4f}",
            f"{r['Avg Latency (ms)']} ms",
        ]
        for r in summary_rows
    ]
    print_table("Custom Orchestration vs LangChain Orchestration Comparison", headers, table_rows)
    print(f"[Done] Saved evaluation summary to: {csv_path}")
    print(f"[Done] Saved detailed comparisons to: {json_path}")


if __name__ == "__main__":
    run_evaluation()
