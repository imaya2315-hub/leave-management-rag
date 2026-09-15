from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

from rag_lab.ingestion import load_documents
from rag_lab.retrieval import SemanticRetriever
from rag_lab.reranking import rerank

ROOT = Path(__file__).resolve().parent
DOCS = ROOT / "rag_lab" / "documents"
OUT = ROOT / "evaluation_remaining_results"
OUT.mkdir(exist_ok=True)

# Fixed settings for remaining experiments.
CHUNK_SIZE = 80
CHUNK_OVERLAP = 10
TOP_K = 10
TOP_N = 3

EMBEDDING_MODELS = {
    "MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
    "MPNet-base-v2": "sentence-transformers/all-mpnet-base-v2",
}

TITLE_MAP = {
    "approval_escalation.txt": "Approval Escalation",
    "carry_forward_rules.txt": "Carry Forward Rules",
    "holiday_calendar.txt": "Holiday Calendar",
    "leave_types_and_eligibility.txt": "Leave Types And Eligibility",
}

DECOMP_CASES = [
    {"id": "D01", "q": "How many annual leave days do I get, can I carry them forward, and by when must carried-forward days be used?", "gold": ["Leave Types And Eligibility", "Carry Forward Rules"]},
    {"id": "D02", "q": "How many sick leave days do I get, when do they start, and when is a certificate required?", "gold": ["Leave Types And Eligibility"]},
    {"id": "D03", "q": "What is casual leave entitlement, does it carry forward, and what happens when I take more than two consecutive working days?", "gold": ["Leave Types And Eligibility", "Carry Forward Rules", "Approval Escalation"]},
]


def load_previous_cases() -> list[dict[str, Any]]:
    """Import CASES from the already-used evaluator so we do not duplicate or alter the benchmark."""
    path = ROOT / "rag_full_evaluation.py"
    if not path.exists():
        path = ROOT / "rag_full_evaluation_fixed.py"
    if not path.exists():
        raise FileNotFoundError("Could not find rag_full_evaluation.py or rag_full_evaluation_fixed.py")

    spec = importlib.util.spec_from_file_location("previous_eval", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    raw = getattr(mod, "CASES", None)
    if not raw:
        raise RuntimeError("CASES was not found in the existing evaluator")

    out = []
    for item in raw:
        if isinstance(item, dict):
            out.append({
                "id": item.get("id") or item.get("qid"),
                "q": item.get("question") or item.get("q"),
                "gold": item.get("gold") or item.get("gold_title") or item.get("title"),
            })
        else:
            # Existing evaluator format: (id, question, type, gold_title, groups)
            qid, q, _typ, gold, _groups = item
            out.append({"id": qid, "q": q, "gold": gold})
    return [x for x in out if x["id"] and x["q"] and x["gold"]]


def make_chunks(docs, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    step = size - overlap
    if step <= 0:
        raise ValueError("overlap must be smaller than chunk size")
    chunks = []
    for d in docs:
        words = d["content"].split()
        i = 0
        idx = 0
        while i < len(words):
            chunks.append({
                "title": d["title"],
                "filename": d.get("filename", d.get("source", "unknown")),
                "chunk_index": idx,
                "content": " ".join(words[i:i + size]),
            })
            idx += 1
            i += step
    return chunks


def top1(results, gold):
    return bool(results) and results[0][0]["title"] == gold


def mrr(results, gold):
    for rank, (chunk, _score) in enumerate(results, 1):
        if chunk["title"] == gold:
            return 1.0 / rank
    return 0.0


def save_csv(path: Path, rows: list[dict[str, Any]]):
    if not rows:
        return
    fields = sorted({k for r in rows for k in r})
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def save_json(path: Path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def print_rows(title, rows):
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)
    if not rows:
        return
    cols = list(rows[0])
    print(" | ".join(cols))
    print("-" * 90)
    for r in rows:
        print(" | ".join(str(r.get(c, "")) for c in cols))


# ------------------------------------------------------------------
# 1. Chunking ablation — retrieval only, no LLM calls.
# ------------------------------------------------------------------
def run_chunking(docs, cases):
    configs = [
        ("small_o0", 80, 0), ("small_o10", 80, 10), ("small_o20", 80, 20),
        ("medium_o0", 160, 0), ("medium_o20", 160, 20), ("medium_o40", 160, 40),
        ("large_o0", 320, 0), ("large_o40", 320, 40), ("large_o80", 320, 80),
    ]
    summary, detail = [], []
    for name, size, overlap in configs:
        chunks = make_chunks(docs, size, overlap)
        build0 = time.perf_counter()
        retriever = SemanticRetriever(chunks)
        build_s = time.perf_counter() - build0
        hits = rel = 0
        mrr_sum = lat_sum = ctx_sum = 0.0
        for c in cases:
            t0 = time.perf_counter()
            results = retriever.retrieve(c["q"], top_k=TOP_K)
            lat = time.perf_counter() - t0
            hit = top1(results, c["gold"])
            rel3 = c["gold"] in {x[0]["title"] for x in results[:TOP_N]}
            hits += int(hit); rel += int(rel3); mrr_sum += mrr(results, c["gold"]); lat_sum += lat
            ctx_sum += sum(len(x[0]["content"].split()) for x in results[:TOP_N])
            detail.append({"config": name, "id": c["id"], "top1": hit, "relevant_top3": rel3, "mrr": mrr(results, c["gold"]), "latency_s": lat})
        n = len(cases)
        summary.append({
            "config": name, "chunk_size": size, "overlap": overlap, "chunks": len(chunks),
            "top1_pct": round(100 * hits / n, 2), "relevant_top3_pct": round(100 * rel / n, 2),
            "mrr": round(mrr_sum / n, 4), "avg_context_words_top3": round(ctx_sum / n, 2),
            "avg_retrieval_latency_s": round(lat_sum / n, 4), "index_build_s": round(build_s, 4),
        })
    save_csv(OUT / "chunking_summary.csv", summary)
    save_csv(OUT / "chunking_detail.csv", detail)
    save_json(OUT / "chunking.json", {"summary": summary, "detail": detail})
    print_rows("Chunking ablation", summary)


# ------------------------------------------------------------------
# 2. Retrieval-model comparison — same chunks, different embeddings.
# ------------------------------------------------------------------
class EmbeddingRetriever:
    def __init__(self, chunks, model_name):
        self.chunks = chunks
        self.model_name = model_name
        t0 = time.perf_counter()
        self.model = SentenceTransformer(model_name)
        self.embeddings = self.model.encode(
            [c["content"] for c in chunks], normalize_embeddings=True, show_progress_bar=False
        )
        self.build_s = time.perf_counter() - t0

    def retrieve(self, q, top_k=TOP_K):
        t0 = time.perf_counter()
        qv = self.model.encode([q], normalize_embeddings=True, show_progress_bar=False)[0]
        scores = np.asarray(self.embeddings) @ np.asarray(qv)
        idx = np.argsort(-scores)[:top_k]
        return [(self.chunks[int(i)], float(scores[int(i)])) for i in idx], time.perf_counter() - t0


def run_retrieval_models(chunks, cases):
    summary, detail = [], []
    for label, model_name in EMBEDDING_MODELS.items():
        print(f"\nLoading retrieval model: {label}")
        r = EmbeddingRetriever(chunks, model_name)
        hits = 0; mrr_sum = lat_sum = 0.0
        for c in cases:
            results, lat = r.retrieve(c["q"])
            hit = top1(results, c["gold"])
            hits += int(hit); mrr_sum += mrr(results, c["gold"]); lat_sum += lat
            detail.append({"model": label, "id": c["id"], "top1": hit, "mrr": mrr(results, c["gold"]), "latency_s": lat})
        n = len(cases)
        summary.append({"model": label, "model_name": model_name, "top1_pct": round(100*hits/n,2), "mrr": round(mrr_sum/n,4), "avg_query_latency_s": round(lat_sum/n,4), "index_build_s": round(r.build_s,4)})
    save_csv(OUT / "retrieval_model_summary.csv", summary)
    save_csv(OUT / "retrieval_model_detail.csv", detail)
    save_json(OUT / "retrieval_model_comparison.json", {"summary": summary, "detail": detail})
    print_rows("Retrieval-model comparison", summary)


# ------------------------------------------------------------------
# 3. Reranker ablation — no generation.
# ------------------------------------------------------------------
def run_reranker(chunks, cases):
    retriever = SemanticRetriever(chunks)
    rows = []
    for variant in ("without_reranker", "with_reranker"):
        hits = 0; mrr_sum = lat_sum = 0.0
        for c in cases:
            t0 = time.perf_counter()
            candidates = retriever.retrieve(c["q"], top_k=TOP_K)
            results = candidates[:TOP_N] if variant == "without_reranker" else rerank(c["q"], candidates, top_n=TOP_N)
            lat = time.perf_counter() - t0
            hits += int(top1(results, c["gold"])); mrr_sum += mrr(results, c["gold"]); lat_sum += lat
        n = len(cases)
        rows.append({"variant": variant, "top1_pct": round(100*hits/n,2), "mrr": round(mrr_sum/n,4), "avg_latency_s": round(lat_sum/n,4)})
    if len(rows) == 2:
        rows.append({"variant": "delta_with_minus_without", "top1_pct": round(rows[1]["top1_pct"]-rows[0]["top1_pct"],2), "mrr": round(rows[1]["mrr"]-rows[0]["mrr"],4), "avg_latency_s": round(rows[1]["avg_latency_s"]-rows[0]["avg_latency_s"],4)})
    save_csv(OUT / "reranker_ablation_summary.csv", rows)
    save_json(OUT / "reranker_ablation.json", rows)
    print_rows("Reranker ablation", rows)


# ------------------------------------------------------------------
# 4. Query decomposition ablation — quota-safe retrieval study.
# ------------------------------------------------------------------
def deterministic_decompose(q: str) -> list[str]:
    parts = [p.strip(" -\n\t") for p in re.split(r"\?|;", q) if p.strip()]
    out = []
    for p in parts:
        out.extend([x.strip() for x in re.split(r"\s+and\s+(?=(?:can|could|when|how|what|does|do|are|is|will|by|which)\b)", p, flags=re.I) if x.strip()])
    return out if len(out) > 1 else [q]


def run_decomposition(chunks):
    retriever = SemanticRetriever(chunks)
    rows = []
    for mode in ("without_decomposition", "with_deterministic_decomposition"):
        all_intents = 0; recovered = 0; expected_total = 0; lat_sum = 0.0
        for c in DECOMP_CASES:
            t0 = time.perf_counter()
            if mode == "without_decomposition":
                results = retriever.retrieve(c["q"], top_k=TOP_K)[:TOP_N]
                observed = {x[0]["title"] for x in results}
                subs = [c["q"]]
            else:
                subs = deterministic_decompose(c["q"])
                observed = set()
                for sq in subs:
                    results = retriever.retrieve(sq, top_k=TOP_K)[:TOP_N]
                    observed.update(x[0]["title"] for x in results)
            lat_sum += time.perf_counter() - t0
            gold = set(c["gold"])
            got = len(gold & observed)
            recovered += got; expected_total += len(gold); all_intents += int(got == len(gold))
        n = len(DECOMP_CASES)
        rows.append({"mode": mode, "all_intents_recovered_pct": round(100*all_intents/n,2), "expected_titles_recovered_pct": round(100*recovered/expected_total,2), "avg_latency_s": round(lat_sum/n,4)})
    if len(rows) == 2:
        rows.append({"mode": "delta_with_minus_without", "all_intents_recovered_pct": round(rows[1]["all_intents_recovered_pct"]-rows[0]["all_intents_recovered_pct"],2), "expected_titles_recovered_pct": round(rows[1]["expected_titles_recovered_pct"]-rows[0]["expected_titles_recovered_pct"],2), "avg_latency_s": round(rows[1]["avg_latency_s"]-rows[0]["avg_latency_s"],4)})
    save_csv(OUT / "decomposition_ablation_summary.csv", rows)
    save_json(OUT / "decomposition_ablation.json", rows)
    print_rows("Query-decomposition ablation (retrieval only)", rows)


# ------------------------------------------------------------------
# 5. LLM cost: DO NOT rerun. Just save the already-observed values.
# ------------------------------------------------------------------
def save_existing_llm_cost():
    data = {
        "status": "reuse_previous_results_no_api_call",
        "note": "LLM quality/token results are already covered. No new Groq generation request is made here.",
        "previous_results": [
            {"model": "Model A", "accuracy_pct": 100.0, "avg_latency_s": 2.6172, "avg_input_tokens": 643.4, "avg_output_tokens": 83.6, "reported_cost_per_question": 0.0, "reported_cost_is_placeholder": True},
            {"model": "Model B", "accuracy_pct": 100.0, "avg_latency_s": 2.0354, "avg_input_tokens": 602.0, "avg_output_tokens": 21.0, "reported_cost_per_question": 0.0, "reported_cost_is_placeholder": True},
        ],
    }
    save_json(OUT / "llm_cost_reused.json", data)
    print("\nLLM cost/token evaluation: previous results reused; no API calls made.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-first", type=int, default=5, help="Skip the first N cases already evaluated. Default: 5")
    ap.add_argument("--only", choices=["all", "chunking", "retrieval", "reranker", "decomposition"], default="all")
    args = ap.parse_args()

    docs = load_documents(str(DOCS))
    cases = load_previous_cases()[args.skip_first:]
    if not cases:
        raise SystemExit("No cases remain after skipping the requested number.")

    chunks = make_chunks(docs)
    print(f"Using questions: {cases[0]['id']} through {cases[-1]['id']} ({len(cases)} cases)")
    print("No Groq generation calls will be made.")
    print(f"Fixed chunking for model/reranker/decomposition comparisons: {CHUNK_SIZE} words, {CHUNK_OVERLAP} overlap")

    if args.only in ("all", "chunking"):
        run_chunking(docs, cases)
    if args.only in ("all", "retrieval"):
        run_retrieval_models(chunks, cases)
    if args.only in ("all", "reranker"):
        run_reranker(chunks, cases)
    if args.only in ("all", "decomposition"):
        run_decomposition(chunks)

    save_existing_llm_cost()

    manifest = {
        "skipped_first": args.skip_first,
        "cases_used": [c["id"] for c in cases],
        "fixed_chunk_size": CHUNK_SIZE,
        "fixed_chunk_overlap": CHUNK_OVERLAP,
        "top_k": TOP_K,
        "top_n": TOP_N,
        "groq_generation_calls": 0,
        "llm_cost": "reused from previous evaluation; not rerun",
    }
    save_json(OUT / "manifest.json", manifest)
    print(f"\nDone. Results: {OUT}")


if __name__ == "__main__":
    main()
