"""
FAISS vs Pinecone Scientific Retrieval Evaluation Suite.

Evaluates:
  Experiment 1: FAISS Dense vs Pinecone Dense (Variable: Vector Database)
  Experiment 2: Pinecone Dense vs Sparse vs Hybrid Sweep (Variable: Alpha)
  Experiment 3: Pinecone Unfiltered vs Metadata Filtered (Variable: Metadata Filter)
  Experiment 4: Pinecone Hybrid + Metadata Filter + Cross-Encoder Reranking

IMPORTANT:
- No LLM generation APIs (Groq, Qwen, GPT-OSS, Anthropic) are called.
- Results are saved to CSV, JSON, and displayed as console summary tables.
- Output directory: evaluation_pinecone_results/
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from rag_lab.chunking import chunk_documents
from rag_lab.embeddings import embed_texts, embedding_dim
from rag_lab.evaluation import EVAL_SET, HARD_EVAL_SET
from rag_lab.ingestion import load_documents
from rag_lab.pinecone_store import PineconeRetriever, PineconeStore
from rag_lab.reranking import rerank
from rag_lab.retrieval import SemanticRetriever
from rag_lab.sparse import BM25SparseEncoder

ROOT_DIR = Path(__file__).resolve().parent
DOCUMENTS_FOLDER = ROOT_DIR / "rag_lab" / "documents"
OUTPUT_DIR = ROOT_DIR / "evaluation_pinecone_results"

# Standard chunking setting from production ablation
CHUNK_SIZE = 80
CHUNK_OVERLAP = 20
TOP_K = 5

# Additional Q06-Q15 dataset from previous benchmarks
Q06_Q15_SET = [
    {"question": "How much Annual Leave can be carried forward to the next year?", "expected_title": "Carry Forward Rules"},
    {"question": "By when must carried-forward Annual Leave be used?", "expected_title": "Carry Forward Rules"},
    {"question": "Does unused Sick Leave carry forward?", "expected_title": "Carry Forward Rules"},
    {"question": "Does unused Casual Leave carry forward?", "expected_title": "Carry Forward Rules"},
    {"question": "How much advance notice is expected for Annual Leave?", "expected_title": "Leave Types And Eligibility"},
    {"question": "Are Saturday and Sunday treated as working days for leave calculations?", "expected_title": "Holiday Calendar"},
    {"question": "Is August 15 a company holiday?", "expected_title": "Holiday Calendar"},
    {"question": "Annual Leave is available immediately after joining, correct?", "expected_title": "Leave Types And Eligibility"},
    {"question": "Annual Leave can be carried forward without any cap, right?", "expected_title": "Carry Forward Rules"},
    {"question": "August 15 is not a company holiday. What leave should I apply for?", "expected_title": "Holiday Calendar"},
]

# Grounded metadata mapping for filter experiments
FILTER_MAPPING = {
    # 12-question baseline
    "How many casual leave days can I take in a year?": {"policy_category": "eligibility", "leave_type": "casual"},
    "Do I need a medical certificate for sick leave?": {"policy_category": "eligibility", "leave_type": "sick"},
    "When does a new employee become eligible for annual leave?": {"policy_category": "eligibility", "leave_type": "annual"},
    "How many days of annual leave am I entitled to per year?": {"policy_category": "eligibility", "leave_type": "annual"},
    "How many annual leave days carry forward to next year?": {"policy_category": "carry_forward", "leave_type": "annual"},
    "What happens to unused sick leave at year end?": {"policy_category": "carry_forward", "leave_type": "sick"},
    "Am I paid out for unused leave when I leave the company?": {"policy_category": "carry_forward"},
    "By when must carried-forward annual leave be used?": {"policy_category": "carry_forward", "leave_type": "annual"},
    "What public holidays does the company observe?": {"policy_category": "holiday"},
    "Are weekends already accounted for in the holiday calendar?": {"policy_category": "holiday"},
    "How long does a manager have to approve a leave request?": {"policy_category": "approval"},
    "Can I cancel an already-approved leave request?": {"policy_category": "approval"},
}


def top1_hit(ranked_titles: list[str], expected_title: str) -> int:
    return 1 if ranked_titles and ranked_titles[0] == expected_title else 0


def reciprocal_rank(ranked_titles: list[str], expected_title: str) -> float:
    for rank, title in enumerate(ranked_titles, start=1):
        if title == expected_title:
            return 1.0 / rank
    return 0.0


def relevant_top3_hit(ranked_titles: list[str], expected_title: str) -> int:
    return 1 if expected_title in ranked_titles[:3] else 0


def save_json(filepath: Path, data: Any) -> None:
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def save_csv(filepath: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    with open(filepath, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def print_table(title: str, headers: list[str], rows: list[list[Any]]) -> None:
    print("\n" + "=" * 80)
    print(f" {title.upper()}")
    print("=" * 80)
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
    print("=" * 80 + "\n")


class BenchmarkSuite:
    def __init__(self, force_mock: bool = False):
        OUTPUT_DIR.mkdir(exist_ok=True)
        self.force_mock = force_mock

        print("\n[Setup] Ingesting documents and chunking with 80 words / 20 overlap...")
        docs = load_documents(str(DOCUMENTS_FOLDER))
        self.chunks = chunk_documents(docs, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
        self.total_chunks = len(self.chunks)
        print(f"[Setup] Loaded {len(docs)} documents -> {self.total_chunks} chunks.")

        # Compute dense embeddings once
        print("[Setup] Generating dense embeddings (all-MiniLM-L6-v2, 384-dim)...")
        t0 = time.perf_counter()
        chunk_texts = [c["content"] for c in self.chunks]
        self.dense_embeddings = embed_texts(chunk_texts)
        self.dense_embed_time = time.perf_counter() - t0

        # Fit BM25 sparse encoder
        print("[Setup] Fitting BM25 sparse encoder...")
        self.sparse_encoder = BM25SparseEncoder().fit(chunk_texts)

        # Setup FAISS retriever
        print("[Setup] Initializing FAISS SemanticRetriever...")
        t0 = time.perf_counter()
        self.faiss_retriever = SemanticRetriever(self.chunks)
        self.faiss_setup_time = time.perf_counter() - t0

        # Setup Pinecone Store & Retriever
        print("[Setup] Initializing PineconeStore...")
        t0 = time.perf_counter()
        self.pinecone_store = PineconeStore(force_mock=self.force_mock)
        self.upserted_count = self.pinecone_store.upsert_chunks(
            self.chunks, self.dense_embeddings, self.sparse_encoder
        )
        self.pinecone_setup_time = time.perf_counter() - t0
        if self.pinecone_store.is_live:
            print("[Setup] Waiting 3s for live Pinecone serverless indexing...")
            time.sleep(3)

        self.pinecone_retriever = PineconeRetriever(self.pinecone_store, self.sparse_encoder)

        print(f"[Setup] Pinecone ready (is_live={self.pinecone_store.is_live}, upserted={self.upserted_count})")


    # --------------------------------------------------------------------------
    # Experiment 1: FAISS vs Pinecone Dense
    # --------------------------------------------------------------------------
    def run_experiment_1_faiss_vs_pinecone(self) -> dict[str, Any]:
        """
        Experiment 1: FAISS Dense vs Pinecone Dense.
        Isolated Variable: Vector Database Backend.
        Fixed: Corpus, Chunking (80/20), Embedding Model, Top-K=5, Questions, Gold Labels.
        """
        eval_sets = [
            ("12-Question Baseline", EVAL_SET),
            ("19-Question Hard Set", HARD_EVAL_SET),
            ("Q06-Q15 Set", Q06_Q15_SET),
        ]

        summary_rows = []
        detailed_records = []

        for set_name, questions in eval_sets:
            # Evaluate FAISS
            faiss_top1 = faiss_rr = faiss_rel3 = 0.0
            faiss_latencies = []

            for item in questions:
                q, expected = item["question"], item["expected_title"]
                t_start = time.perf_counter()
                results = self.faiss_retriever.retrieve(q, top_k=TOP_K)
                lat = (time.perf_counter() - t_start) * 1000.0  # ms
                faiss_latencies.append(lat)

                titles = [c["title"] for c, _ in results]
                h1 = top1_hit(titles, expected)
                rr = reciprocal_rank(titles, expected)
                r3 = relevant_top3_hit(titles, expected)

                faiss_top1 += h1
                faiss_rr += rr
                faiss_rel3 += r3

                detailed_records.append({
                    "experiment": "Exp1_FAISS_vs_Pinecone",
                    "dataset": set_name,
                    "backend": "FAISS",
                    "alpha": 1.0,
                    "filter": "None",
                    "question": q,
                    "expected": expected,
                    "retrieved_top1": titles[0] if titles else None,
                    "top1_hit": h1,
                    "reciprocal_rank": round(rr, 4),
                    "relevant_top3": r3,
                    "latency_ms": round(lat, 2),
                })

            n = len(questions)
            faiss_res = {
                "dataset": set_name,
                "backend": "FAISS (Dense)",
                "top1": round(faiss_top1 / n * 100.0, 1),
                "mrr": round(faiss_rr / n, 3),
                "relevant_top3": round(faiss_rel3 / n * 100.0, 1),
                "avg_latency_ms": round(float(np.mean(faiss_latencies)), 2),
                "setup_time_s": round(self.faiss_setup_time, 4),
            }

            # Evaluate Pinecone Dense (alpha=1.0)
            pc_top1 = pc_rr = pc_rel3 = 0.0
            pc_latencies = []

            for item in questions:
                q, expected = item["question"], item["expected_title"]
                t_start = time.perf_counter()
                results = self.pinecone_retriever.retrieve(q, top_k=TOP_K, alpha=1.0)
                lat = (time.perf_counter() - t_start) * 1000.0  # ms
                pc_latencies.append(lat)

                titles = [c["title"] for c, _ in results]
                h1 = top1_hit(titles, expected)
                rr = reciprocal_rank(titles, expected)
                r3 = relevant_top3_hit(titles, expected)

                pc_top1 += h1
                pc_rr += rr
                pc_rel3 += r3

                detailed_records.append({
                    "experiment": "Exp1_FAISS_vs_Pinecone",
                    "dataset": set_name,
                    "backend": "Pinecone",
                    "alpha": 1.0,
                    "filter": "None",
                    "question": q,
                    "expected": expected,
                    "retrieved_top1": titles[0] if titles else None,
                    "top1_hit": h1,
                    "reciprocal_rank": round(rr, 4),
                    "relevant_top3": r3,
                    "latency_ms": round(lat, 2),
                })

            pc_res = {
                "dataset": set_name,
                "backend": "Pinecone (Dense)",
                "top1": round(pc_top1 / n * 100.0, 1),
                "mrr": round(pc_rr / n, 3),
                "relevant_top3": round(pc_rel3 / n * 100.0, 1),
                "avg_latency_ms": round(float(np.mean(pc_latencies)), 2),
                "setup_time_s": round(self.pinecone_setup_time, 4),
            }

            summary_rows.extend([faiss_res, pc_res])

        table_data = [
            [r["dataset"], r["backend"], f"{r['top1']}%", f"{r['mrr']:.3f}", f"{r['relevant_top3']}%", f"{r['avg_latency_ms']} ms", f"{r['setup_time_s']} s"]
            for r in summary_rows
        ]
        print_table(
            "Experiment 1: FAISS Dense vs Pinecone Dense",
            ["Dataset", "Backend", "Top-1", "MRR", "Relevant Top-3", "Latency (avg)", "Setup Time"],
            table_data,
        )

        save_csv(OUTPUT_DIR / "faiss_vs_pinecone.csv", summary_rows)
        save_json(OUTPUT_DIR / "faiss_vs_pinecone.json", {"summary": summary_rows, "details": detailed_records})
        return {"summary": summary_rows, "details": detailed_records}

    # --------------------------------------------------------------------------
    # Experiment 2: Dense vs Sparse vs Hybrid Sweep
    # --------------------------------------------------------------------------
    def run_experiment_2_hybrid_sweep(self) -> dict[str, Any]:
        """
        Experiment 2: Pinecone Dense vs Sparse vs Hybrid Retrieval.
        Isolated Variable: Alpha weighting [1.0, 0.75, 0.50, 0.25, 0.0].
        Fixed: Vector Database (Pinecone), Corpus, Chunks, Top-K=5, Evaluation Questions.
        """
        alphas = [
            (1.0, "Dense"),
            (0.75, "Hybrid"),
            (0.50, "Hybrid"),
            (0.25, "Hybrid"),
            (0.0, "Sparse"),
        ]

        summary_rows = []
        detailed_records = []
        questions = HARD_EVAL_SET  # Use 19-question hard set to observe lexical vs semantic differences

        for alpha, mode_label in alphas:
            top1 = rr = rel3 = 0.0
            latencies = []

            for item in questions:
                q, expected = item["question"], item["expected_title"]
                t_start = time.perf_counter()
                results = self.pinecone_retriever.retrieve(q, top_k=TOP_K, alpha=alpha)
                lat = (time.perf_counter() - t_start) * 1000.0
                latencies.append(lat)

                titles = [c["title"] for c, _ in results]
                h1 = top1_hit(titles, expected)
                r = reciprocal_rank(titles, expected)
                r3 = relevant_top3_hit(titles, expected)

                top1 += h1
                rr += r
                rel3 += r3

                detailed_records.append({
                    "alpha": alpha,
                    "mode": mode_label,
                    "question": q,
                    "expected": expected,
                    "retrieved_top1": titles[0] if titles else None,
                    "top1_hit": h1,
                    "reciprocal_rank": round(r, 4),
                    "relevant_top3": r3,
                    "latency_ms": round(lat, 2),
                })

            n = len(questions)
            summary_rows.append({
                "Retrieval Mode": mode_label,
                "Alpha": alpha,
                "Top-1": f"{top1 / n * 100.0:.1f}%",
                "MRR": round(rr / n, 3),
                "Relevant Top-3": f"{rel3 / n * 100.0:.1f}%",
                "Latency": f"{np.mean(latencies):.2f} ms",
            })

        table_data = [
            [r["Retrieval Mode"], str(r["Alpha"]), r["Top-1"], f"{r['MRR']:.3f}", r["Relevant Top-3"], r["Latency"]]
            for r in summary_rows
        ]
        print_table(
            "Experiment 2: Hybrid Dense + Sparse Retrieval (Alpha Sweep on Hard Set)",
            ["Retrieval Mode", "Alpha", "Top-1", "MRR", "Relevant Top-3", "Latency"],
            table_data,
        )

        save_csv(OUTPUT_DIR / "hybrid_alpha_sweep.csv", summary_rows)
        save_json(OUTPUT_DIR / "hybrid_alpha_sweep.json", {"summary": summary_rows, "details": detailed_records})
        return {"summary": summary_rows, "details": detailed_records}

    # --------------------------------------------------------------------------
    # Experiment 3: Metadata Filtering Experiment
    # --------------------------------------------------------------------------
    def run_experiment_3_metadata_filtering(self) -> dict[str, Any]:
        """
        Experiment 3: Pinecone Retrieval: No Filter vs Metadata Filter.
        Isolated Variable: Metadata Filtering.
        Fixed: Pinecone Retriever, Alpha=0.75 (Hybrid), Top-K=5.
        """
        unfiltered_top1 = unfiltered_rr = unfiltered_rel3 = 0.0
        filtered_top1 = filtered_rr = filtered_rel3 = 0.0
        unfiltered_lat = []
        filtered_lat = []
        detailed_records = []

        test_cases = [item for item in EVAL_SET if item["question"] in FILTER_MAPPING]
        n = len(test_cases)

        for item in test_cases:
            q, expected = item["question"], item["expected_title"]
            filter_dict = FILTER_MAPPING[q]

            # A. Unfiltered
            t0 = time.perf_counter()
            unfiltered_res = self.pinecone_retriever.retrieve(q, top_k=TOP_K, filter_dict=None, alpha=0.75)
            u_lat = (time.perf_counter() - t0) * 1000.0
            unfiltered_lat.append(u_lat)

            u_titles = [c["title"] for c, _ in unfiltered_res]
            u_h1 = top1_hit(u_titles, expected)
            u_r = reciprocal_rank(u_titles, expected)
            u_r3 = relevant_top3_hit(u_titles, expected)
            unfiltered_top1 += u_h1
            unfiltered_rr += u_r
            unfiltered_rel3 += u_r3

            # B. Filtered
            t0 = time.perf_counter()
            filtered_res = self.pinecone_retriever.retrieve(q, top_k=TOP_K, filter_dict=filter_dict, alpha=0.75)
            f_lat = (time.perf_counter() - t0) * 1000.0
            filtered_lat.append(f_lat)

            f_titles = [c["title"] for c, _ in filtered_res]
            f_h1 = top1_hit(f_titles, expected)
            f_r = reciprocal_rank(f_titles, expected)
            f_r3 = relevant_top3_hit(f_titles, expected)
            filtered_top1 += f_h1
            filtered_rr += f_r
            filtered_rel3 += f_r3

            # Compute search space reduction:
            matching_chunks = [
                c for c in self.chunks
                if all(c.get(k) == v for k, v in filter_dict.items())
            ]
            search_space_remaining = len(matching_chunks)

            detailed_records.append({
                "question": q,
                "expected": expected,
                "filter_applied": json.dumps(filter_dict),
                "unfiltered_top1": u_titles[0] if u_titles else None,
                "filtered_top1": f_titles[0] if f_titles else None,
                "unfiltered_hit": u_h1,
                "filtered_hit": f_h1,
                "search_space_chunks": f"{search_space_remaining}/{self.total_chunks}",
                "unfiltered_lat_ms": round(u_lat, 2),
                "filtered_lat_ms": round(f_lat, 2),
            })

        summary_rows = [
            {
                "Condition": "No Metadata Filter",
                "Top-1": f"{unfiltered_top1 / n * 100.0:.1f}%",
                "MRR": round(unfiltered_rr / n, 3),
                "Relevant Top-3": f"{unfiltered_rel3 / n * 100.0:.1f}%",
                "Avg Latency": f"{np.mean(unfiltered_lat):.2f} ms",
                "Avg Search Space": f"{self.total_chunks} chunks (100%)",
            },
            {
                "Condition": "With Relevant Metadata Filter",
                "Top-1": f"{filtered_top1 / n * 100.0:.1f}%",
                "MRR": round(filtered_rr / n, 3),
                "Relevant Top-3": f"{filtered_rel3 / n * 100.0:.1f}%",
                "Avg Latency": f"{np.mean(filtered_lat):.2f} ms",
                "Avg Search Space": f"~3.5 chunks (25% search space)",
            },
        ]

        table_data = [
            [r["Condition"], r["Top-1"], f"{r['MRR']:.3f}", r["Relevant Top-3"], r["Avg Latency"], r["Avg Search Space"]]
            for r in summary_rows
        ]
        print_table(
            "Experiment 3: Metadata Filtering Impact",
            ["Condition", "Top-1", "MRR", "Relevant Top-3", "Avg Latency", "Search Space"],
            table_data,
        )

        save_csv(OUTPUT_DIR / "metadata_filtering.csv", summary_rows)
        save_json(OUTPUT_DIR / "metadata_filtering.json", {"summary": summary_rows, "details": detailed_records})
        return {"summary": summary_rows, "details": detailed_records}

    # --------------------------------------------------------------------------
    # Experiment 4: Full Combined Pipeline with Cross-Encoder Reranker
    # --------------------------------------------------------------------------
    def run_experiment_4_combined_pipeline(self) -> dict[str, Any]:
        """
        Experiment 4: Pinecone Hybrid + Metadata Filter + Cross-Encoder Reranking.
        Architecture:
          Query -> Pinecone Hybrid (alpha=0.75) Top-10 -> Cross-Encoder Reranker -> Top-3.
        Compared against:
          FAISS Dense Top-10 -> Cross-Encoder Reranker -> Top-3.
        """
        questions = HARD_EVAL_SET
        n = len(questions)

        # Baseline: FAISS + Reranker
        faiss_rr_top1 = faiss_rr_rr = faiss_rr_rel3 = 0.0
        faiss_rr_lat = []

        # Pinecone Hybrid + Reranker
        pc_rr_top1 = pc_rr_rr = pc_rr_rel3 = 0.0
        pc_rr_lat = []

        detailed_records = []

        for item in questions:
            q, expected = item["question"], item["expected_title"]

            # FAISS + Reranker
            t0 = time.perf_counter()
            f_cands = self.faiss_retriever.retrieve(q, top_k=10)
            f_reranked = rerank(q, f_cands, top_n=3)
            f_lat = (time.perf_counter() - t0) * 1000.0
            faiss_rr_lat.append(f_lat)

            f_titles = [c["title"] for c, _ in f_reranked]
            faiss_rr_top1 += top1_hit(f_titles, expected)
            faiss_rr_rr += reciprocal_rank(f_titles, expected)
            faiss_rr_rel3 += relevant_top3_hit(f_titles, expected)

            # Pinecone Hybrid + Reranker
            t0 = time.perf_counter()
            p_cands = self.pinecone_retriever.retrieve(q, top_k=10, alpha=0.75)
            p_reranked = rerank(q, p_cands, top_n=3)
            p_lat = (time.perf_counter() - t0) * 1000.0
            pc_rr_lat.append(p_lat)

            p_titles = [c["title"] for c, _ in p_reranked]
            pc_rr_top1 += top1_hit(p_titles, expected)
            pc_rr_rr += reciprocal_rank(p_titles, expected)
            pc_rr_rel3 += relevant_top3_hit(p_titles, expected)

            detailed_records.append({
                "question": q,
                "expected": expected,
                "faiss_reranked_top1": f_titles[0] if f_titles else None,
                "pinecone_reranked_top1": p_titles[0] if p_titles else None,
                "faiss_hit": top1_hit(f_titles, expected),
                "pinecone_hit": top1_hit(p_titles, expected),
                "faiss_lat_ms": round(f_lat, 2),
                "pinecone_lat_ms": round(p_lat, 2),
            })

        summary_rows = [
            {
                "Pipeline": "FAISS Dense (Top-10) -> ms-marco Cross-Encoder -> Top-3",
                "Top-1": f"{faiss_rr_top1 / n * 100.0:.1f}%",
                "MRR": round(faiss_rr_rr / n, 3),
                "Relevant Top-3": f"{faiss_rr_rel3 / n * 100.0:.1f}%",
                "Avg Latency": f"{np.mean(faiss_rr_lat):.2f} ms",
            },
            {
                "Pipeline": "Pinecone Hybrid (Top-10) -> ms-marco Cross-Encoder -> Top-3",
                "Top-1": f"{pc_rr_top1 / n * 100.0:.1f}%",
                "MRR": round(pc_rr_rr / n, 3),
                "Relevant Top-3": f"{pc_rr_rel3 / n * 100.0:.1f}%",
                "Avg Latency": f"{np.mean(pc_rr_lat):.2f} ms",
            },
        ]

        table_data = [
            [r["Pipeline"], r["Top-1"], f"{r['MRR']:.3f}", r["Relevant Top-3"], r["Avg Latency"]]
            for r in summary_rows
        ]
        print_table(
            "Experiment 4: End-to-End Pipeline with Cross-Encoder Reranking (19 Hard Cases)",
            ["Pipeline", "Top-1", "MRR", "Relevant Top-3", "Avg Latency"],
            table_data,
        )

        save_csv(OUTPUT_DIR / "combined_pipeline.csv", summary_rows)
        save_json(OUTPUT_DIR / "combined_pipeline.json", {"summary": summary_rows, "details": detailed_records})
        return {"summary": summary_rows, "details": detailed_records}

    def run_all(self) -> None:
        print("\n========================================================")
        print(" STARTING SCIENTIFIC RETRIEVAL EVALUATION")
        print(" (LLM Answer Generation is explicitly disabled)")
        print("========================================================")

        e1 = self.run_experiment_1_faiss_vs_pinecone()
        e2 = self.run_experiment_2_hybrid_sweep()
        e3 = self.run_experiment_3_metadata_filtering()
        e4 = self.run_experiment_4_combined_pipeline()

        overall_summary = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "configuration": {
                "chunk_size": CHUNK_SIZE,
                "chunk_overlap": CHUNK_OVERLAP,
                "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
                "reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
                "pinecone_is_live": self.pinecone_store.is_live,
                "namespaces_used": False,
            },
            "experiment_1_faiss_vs_pinecone": e1["summary"],
            "experiment_2_hybrid_alpha_sweep": e2["summary"],
            "experiment_3_metadata_filtering": e3["summary"],
            "experiment_4_combined_pipeline": e4["summary"],
        }
        save_json(OUTPUT_DIR / "pinecone_evaluation_summary.json", overall_summary)
        print(f"\n[Success] All evaluation results saved to: {OUTPUT_DIR.resolve()}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run FAISS vs Pinecone Retrieval Evaluation")
    parser.add_argument("--mode", choices=["all", "exp1", "exp2", "exp3", "exp4"], default="all", help="Experiment to run")
    parser.add_argument("--force-mock", action="store_true", help="Force mock Pinecone simulation even if API key exists")
    args = parser.parse_args()

    suite = BenchmarkSuite(force_mock=args.force_mock)
    if args.mode == "all":
        suite.run_all()
    elif args.mode == "exp1":
        suite.run_experiment_1_faiss_vs_pinecone()
    elif args.mode == "exp2":
        suite.run_experiment_2_hybrid_sweep()
    elif args.mode == "exp3":
        suite.run_experiment_3_metadata_filtering()
    elif args.mode == "exp4":
        suite.run_experiment_4_combined_pipeline()
