from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import statistics
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from rag_lab.retrieval import SemanticRetriever
from rag_lab.reranking import rerank

load_dotenv()

DOCS = Path("rag_lab/documents")
OUT = Path("evaluation_results")
TOP_K = 10
TOP_N = 3
TEMP = 0.2
TOP_P = 0.8
MAX_TOKENS = 120
MODEL_A = os.getenv("MODEL_A", "openai/gpt-oss-120b")
MODEL_B = os.getenv("MODEL_B", "qwen/qwen3.8-27b")
NO_CONTEXT = "The provided policy context does not contain enough information to answer that question."
SYSTEM = f'''You are the Leave Management Assistant for a company.\nAnswer ONLY from the supplied policy context. Never invent facts or accept a false premise. If the answer is not in the context, reply exactly:\n"{NO_CONTEXT}"'''

TITLE_MAP = {
    "approval_escalation.txt": "Approval Escalation",
    "carry_forward_rules.txt": "Carry Forward Rules",
    "holiday_calendar.txt": "Holiday Calendar",
    "leave_types_and_eligibility.txt": "Leave Types And Eligibility",
}

CHUNK_CONFIGS = {
    "small_o0": (80, 0), "small_o10": (80, 10), "small_o20": (80, 20),
    "medium_o0": (160, 0), "medium_o20": (160, 20), "medium_o40": (160, 40),
    "large_o0": (320, 0), "large_o40": (320, 40), "large_o80": (320, 80),
}

CASES: list[dict[str, Any]] = [
    ("Q01", "How many Annual Leave days do employees get each year?", "supported", "Leave Types And Eligibility", [["20"], ["annual leave"]]),
    ("Q02", "How many Sick Leave days are provided per calendar year?", "supported", "Leave Types And Eligibility", [["10"], ["sick leave"]]),
    ("Q03", "What is the yearly allowance for Casual Leave?", "supported", "Leave Types And Eligibility", [["5"], ["casual leave"]]),
    ("Q04", "Can a new employee take Annual Leave immediately after joining?", "supported", "Leave Types And Eligibility", [["90"], ["not", "no", "only after", "after completing"]]),
    ("Q05", "Is Sick Leave available from the first working day?", "supported", "Leave Types And Eligibility", [["first working day"], ["yes", "available"]]),
    ("Q06", "How much Annual Leave can be carried forward to the next year?", "supported", "Carry Forward Rules", [["5"], ["carry forward", "carried forward"]]),
    ("Q07", "By when must carried-forward Annual Leave be used?", "supported", "Carry Forward Rules", [["march 31", "march 31st"], ["following year", "next year"]]),
    ("Q08", "Does unused Sick Leave carry forward?", "supported", "Carry Forward Rules", [["does not carry", "not carry", "no"], ["sick leave"]]),
    ("Q09", "Does unused Casual Leave carry forward?", "supported", "Carry Forward Rules", [["does not carry", "not carry", "no"], ["casual leave"]]),
    ("Q10", "How much advance notice is expected for Annual Leave?", "supported", "Leave Types And Eligibility", [["5"], ["working days"]]),
    ("Q11", "Are Saturday and Sunday treated as working days for leave calculations?", "supported", "Holiday Calendar", [["saturday"], ["sunday"], ["weekend"]]),
    ("Q12", "Is August 15 a company holiday?", "supported", "Holiday Calendar", [["august 15", "august 15th"], ["holiday", "independence day"]]),
    ("Q13", "Annual Leave is available immediately after joining, correct?", "misleading", "Leave Types And Eligibility", [["no", "not", "incorrect", "false"], ["90"], ["annual leave"]]),
    ("Q14", "Annual Leave can be carried forward without any cap, right?", "misleading", "Carry Forward Rules", [["no", "not", "incorrect", "false"], ["5"], ["carry forward", "carried forward"]]),
    ("Q15", "August 15 is not a company holiday. What leave should I apply for?", "misleading", "Holiday Calendar", [["august 15", "august 15th"], ["holiday", "independence day"]]),
    ("Q16", "How many days of maternity leave are employees entitled to?", "unsupported", None, [["does not contain", "not contain", "not provided", "not mentioned", "no information", "cannot answer", "not specified"]]),
    ("Q17", "How many paid paternity leave days does the company provide?", "unsupported", None, [["does not contain", "not contain", "not provided", "not mentioned", "no information", "cannot answer", "not specified"]]),
    ("Q18", "What percentage of salary is deducted for unpaid leave?", "unsupported", None, [["does not contain", "not contain", "not provided", "not mentioned", "no information", "cannot answer", "not specified"]]),
]


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9.%]+", " ", s.lower()).strip()


def load_docs() -> list[dict[str, str]]:
    if not DOCS.exists():
        raise FileNotFoundError(f"Missing {DOCS.resolve()}")
    out = []
    for p in sorted(DOCS.glob("*.txt")):
        txt = re.sub(r"\s+", " ", p.read_text(encoding="utf-8")).strip()
        if txt:
            out.append({"title": TITLE_MAP.get(p.name, p.stem.replace("_", " ").title()), "filename": p.name, "content": txt})
    if not out:
        raise RuntimeError("No policy .txt files found")
    return out


def make_chunks(docs, size, overlap):
    if size <= 0 or overlap < 0 or overlap >= size:
        raise ValueError("Require size > 0 and 0 <= overlap < size")
    chunks = []
    step = size - overlap
    for d in docs:
        words = d["content"].split()
        i = 0
        idx = 0
        while i < len(words):
            chunks.append({"title": d["title"], "filename": d["filename"], "chunk_index": idx, "content": " ".join(words[i:i+size])})
            idx += 1
            i += step
    return chunks


def build_retriever(chunks):
    t = time.perf_counter()
    r = SemanticRetriever(chunks)
    return r, time.perf_counter() - t


def rank(retriever, question):
    t = time.perf_counter()
    retrieved = retriever.retrieve(question, top_k=TOP_K)
    ret_lat = time.perf_counter() - t
    t = time.perf_counter()
    ranked = rerank(question, retrieved, top_n=TOP_N)
    rr_lat = time.perf_counter() - t
    context = "\n\n".join(f"[{c['title']}]\n{c['content']}" for c, _ in ranked)
    return ranked, context, ret_lat, rr_lat


def contains_any(text, opts):
    x = norm(text)
    return any(norm(o) in x for o in opts)


def refusal(text):
    x = norm(text)
    pats = ["does not contain", "not contain", "not provided", "not mentioned", "no information", "cannot answer", "not specified", "not available"]
    return any(p in x for p in pats)


def grade(case, answer):
    """
    Deterministic answer grader.

    Supports:
        (id, question, type, gold_title, groups)

    Returns:
        correct, hallucinated_proxy, reason
    """
    if isinstance(case, dict):
        typ = case.get("type", "")
        groups = case.get("must_include", []) or []
    else:
        _, _, typ, _, groups = case
        groups = groups or []

    # Never iterate a string as a list of individual characters.
    if isinstance(groups, str):
        groups = [[groups]]

    normalized_groups = []
    for group in groups:
        if isinstance(group, str):
            normalized_groups.append([group])
        elif isinstance(group, (list, tuple, set)):
            normalized_groups.append(list(group))
        else:
            normalized_groups.append([str(group)])

    if typ == "unsupported":
        ok = refusal(answer)
        nums = re.findall(r"\b\d+(?:\.\d+)?\b", norm(answer))
        hall = bool(nums) and not ok
        return (
            ok,
            hall,
            "Correct refusal."
            if ok
            else "Unsupported question was not refused.",
        )

    for group in normalized_groups:
        if not contains_any(answer, group):
            return (
                False,
                True,
                f"Missing expected fact: {group}",
            )

    if typ == "misleading":
        words = norm(answer).split()
        correction_words = {
            "no",
            "not",
            "incorrect",
            "false",
            "actually",
            "premise",
            "however",
        }
        if not any(word in words for word in correction_words):
            return (
                False,
                True,
                "False premise was not corrected.",
            )

    return True, False, "Matches deterministic rubric."


def faithfulness_proxy(answer, context):
    if not answer.strip() or refusal(answer):
        return 1.0 if answer.strip() else 0.0
    a_nums = set(re.findall(r"\b\d+(?:\.\d+)?\b", norm(answer)))
    c_nums = set(re.findall(r"\b\d+(?:\.\d+)?\b", norm(context)))
    if any(n not in c_nums for n in a_nums):
        return 0.0
    ctx = set(re.findall(r"\b[a-z]{4,}\b", norm(context)))
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer.strip()) if s.strip()]
    scores = []
    for s in sents:
        ws = set(re.findall(r"\b[a-z]{4,}\b", norm(s)))
        scores.append(1.0 if not ws else min((len(ws & ctx) / len(ws)) / 0.35, 1.0))
    return round(sum(scores) / len(scores), 4) if scores else 0.0


def est_tokens(text):
    return max(1, math.ceil(len(text) / 4))


def titles(ranked):
    return [c["title"] for c, _ in ranked]


def groq_client():
    from groq import Groq
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY is not set")
    return Groq(api_key=key)


def generate(client, model, question, context):
    t = time.perf_counter()
    r = client.chat.completions.create(
        model=model, temperature=TEMP, top_p=TOP_P, max_tokens=MAX_TOKENS,
        messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": f"Question: {question}\n\nContext:\n{context}"}],
    )
    lat = time.perf_counter() - t
    u = r.usage
    return (r.choices[0].message.content or "").strip(), lat, int(getattr(u, "prompt_tokens", 0) or 0), int(getattr(u, "completion_tokens", 0) or 0)


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows: return
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def chunking_eval(docs, model, limit, run_gen):
    cases = CASES[:limit] if limit else CASES
    client = groq_client() if run_gen else None
    summary, detail = [], []
    for name, (size, overlap) in CHUNK_CONFIGS.items():
        chunks = make_chunks(docs, size, overlap)
        retriever, build_lat = build_retriever(chunks)
        ret_ok, rel_pct, ans_ok, ctx_words, lats = [], [], [], [], []
        for qid, q, typ, gold, groups in cases:
            ranked, context, rlat, xlat = rank(retriever, q)
            top = titles(ranked)
            top1 = (top and gold is not None and top[0] == gold) if gold else None
            top3 = (gold in top) if gold else None
            rel = (sum(t == gold for t in top) / len(top) * 100) if gold and top else None
            mrr = (1 / (next((i+1 for i,t in enumerate(top) if t == gold), 0))) if gold and gold in top else 0.0
            answer, glat, corr, hall, reason = "", 0.0, None, None, ""
            if run_gen:
                answer, glat, _, _ = generate(client, model, q, context)
                corr, hall, reason = grade((qid, q, typ, gold, groups), answer)
                ans_ok.append(int(corr))
            if top1 is not None:
                ret_ok.append(int(top1)); rel_pct.append(rel or 0.0)
            cw = len(context.split()); ctx_words.append(cw); lats.append(rlat+xlat+glat)
            detail.append({"config":name,"chunk_size_words":size,"overlap_words":overlap,"num_chunks":len(chunks),"index_build_latency_sec":round(build_lat,4),"question_id":qid,"question":q,"type":typ,"gold_title":gold or "","retrieved_sources":" | ".join(top),"retrieval_top1_correct":top1,"retrieval_top3_correct":top3,"mrr":round(mrr,4),"relevant_chunk_percentage":round(rel,2) if rel is not None else None,"context_words":cw,"context_chars":len(context),"estimated_context_tokens":est_tokens(context),"retrieval_latency_sec":round(rlat,4),"rerank_latency_sec":round(xlat,4),"generation_latency_sec":round(glat,4),"total_latency_sec":round(rlat+xlat+glat,4),"answer":answer,"answer_correct":corr,"hallucinated_proxy":hall,"grader_reason":reason})
        summary.append({"config":name,"chunk_size_words":size,"overlap_words":overlap,"num_chunks":len(chunks),"retrieval_top1_accuracy":round(statistics.mean(ret_ok)*100,2) if ret_ok else None,"relevant_chunk_percentage":round(statistics.mean(rel_pct),2) if rel_pct else None,"answer_correctness":round(statistics.mean(ans_ok)*100,2) if ans_ok else None,"avg_context_words":round(statistics.mean(ctx_words),2),"avg_estimated_context_tokens":round(statistics.mean(est_tokens("x"*max(1,w*5)) for w in ctx_words),2),"avg_latency_sec":round(statistics.mean(lats),4)})
    return summary, detail


def model_eval(docs, config, models, prices, limit):
    size, overlap = CHUNK_CONFIGS[config]; chunks = make_chunks(docs,size,overlap); retriever,_ = build_retriever(chunks); client = groq_client(); cases=CASES[:limit] if limit else CASES
    detail=[]; summary=[]
    for label,model in models.items():
        vals={k:[] for k in ["corr","lat","in","out","cost"]}
        for qid,q,typ,gold,groups in cases:
            ranked,ctx,rlat,xlat=rank(retriever,q); ans,glat,pt,ot=generate(client,model,q,ctx); corr,hall,reason=grade((qid, q, typ, gold, groups), ans)
            p=prices.get(model,{"input":0.0,"output":0.0}); cost=pt*p["input"]/1e6+ot*p["output"]/1e6
            vals["corr"].append(int(corr)); vals["lat"].append(rlat+xlat+glat); vals["in"].append(pt); vals["out"].append(ot); vals["cost"].append(cost)
            detail.append({"model":label,"model_name":model,"chunk_config":config,"question_id":qid,"question":q,"answer":ans,"correct":corr,"hallucinated_proxy":hall,"grader_reason":reason,"input_tokens":pt,"output_tokens":ot,"cost":round(cost,8),"retrieved_sources":" | ".join(titles(ranked)),"total_latency_sec":round(rlat+xlat+glat,4)})
        summary.append({"model":label,"model_name":model,"chunk_config":config,"accuracy_percent":round(statistics.mean(vals["corr"])*100,2),"avg_latency_sec":round(statistics.mean(vals["lat"]),4),"avg_input_tokens":round(statistics.mean(vals["in"]),2),"avg_output_tokens":round(statistics.mean(vals["out"]),2),"avg_cost_per_question":round(statistics.mean(vals["cost"]),8),"total_cost":round(sum(vals["cost"]),8),"input_price_per_million":prices[model]["input"],"output_price_per_million":prices[model]["output"]})
    return summary,detail


def e2e_eval(docs, config, model, limit):
    size,overlap=CHUNK_CONFIGS[config]; chunks=make_chunks(docs,size,overlap); retriever,_=build_retriever(chunks); client=groq_client(); cases=CASES[:limit] if limit else CASES; rows=[]
    for qid,q,typ,gold,groups in cases:
        ranked,ctx,rlat,xlat=rank(retriever,q); ans,glat,pt,ot=generate(client,model,q,ctx); corr,hall,reason=grade((qid, q, typ, gold, groups), ans); ft=faithfulness_proxy(ans,ctx); stc=(gold in titles(ranked)) if gold else None; top1=(titles(ranked)[0]==gold) if gold and ranked else None
        rows.append({"model":model,"chunk_config":config,"question_id":qid,"question":q,"type":typ,"gold_title":gold or "","retrieved_sources":" | ".join(titles(ranked)),"retrieval_correct_top1":top1,"retrieval_correct_top3":stc,"answer_correct":corr,"hallucinated_proxy":hall,"faithfulness_proxy":ft,"unsupported_question_handling":corr if typ=="unsupported" else None,"citation_source_correct":stc,"context_words":len(ctx.split()),"context_chars":len(ctx),"estimated_context_tokens":est_tokens(ctx),"input_tokens":pt,"output_tokens":ot,"retrieval_latency_sec":round(rlat,4),"rerank_latency_sec":round(xlat,4),"generation_latency_sec":round(glat,4),"total_latency_sec":round(rlat+xlat+glat,4),"answer":ans,"grader_reason":reason})
    return rows


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--limit",type=int); ap.add_argument("--skip-generation",action="store_true"); ap.add_argument("--generation-model",default=MODEL_B); ap.add_argument("--cost-config",choices=CHUNK_CONFIGS,default="medium_o20"); ap.add_argument("--end-to-end-config",choices=CHUNK_CONFIGS,default="medium_o20"); ap.add_argument("--end-to-end-model",default=MODEL_B); ap.add_argument("--a-input-price",type=float,default=float(os.getenv("MODEL_A_INPUT_PRICE_PER_MILLION","0"))); ap.add_argument("--a-output-price",type=float,default=float(os.getenv("MODEL_A_OUTPUT_PRICE_PER_MILLION","0"))); ap.add_argument("--b-input-price",type=float,default=float(os.getenv("MODEL_B_INPUT_PRICE_PER_MILLION","0"))); ap.add_argument("--b-output-price",type=float,default=float(os.getenv("MODEL_B_OUTPUT_PRICE_PER_MILLION","0"))); ap.add_argument("--skip-cost",action="store_true"); ap.add_argument("--skip-end-to-end",action="store_true"); args=ap.parse_args()
    docs=load_docs(); OUT.mkdir(exist_ok=True)
    cs,cd=chunking_eval(docs,args.generation_model,args.limit,not args.skip_generation); write_csv(OUT/"chunking_results.csv",cd); write_json(OUT/"chunking_results.json",{"summary":cs,"detail":cd})
    summaries=[]; details=[]
    if not args.skip_cost:
        models={"Model A":MODEL_A,"Model B":MODEL_B}; prices={MODEL_A:{"input":args.a_input_price,"output":args.a_output_price},MODEL_B:{"input":args.b_input_price,"output":args.b_output_price}}
        summaries,details=model_eval(docs,args.cost_config,models,prices,args.limit); write_csv(OUT/"model_cost_quality_summary.csv",summaries); write_csv(OUT/"model_cost_quality_details.csv",details)
    e2e=[]
    if not args.skip_end_to_end:
        e2e=e2e_eval(docs,args.end_to_end_config,args.end_to_end_model,args.limit); write_csv(OUT/"end_to_end_results.csv",e2e)
    write_json(OUT/"full_rag_evaluation.json",{"settings":{"retrieval_top_k":TOP_K,"rerank_top_n":TOP_N,"temperature":TEMP,"top_p":TOP_P,"max_tokens":MAX_TOKENS,"models":{"model_a":MODEL_A,"model_b":MODEL_B}},"chunking":{"summary":cs,"detail":cd},"cost_quality":{"summary":summaries,"detail":details},"end_to_end":e2e})
    print("\nDone. Results:",OUT.resolve())
    print("\nChunking summary")
    for r in cs: print(f"{r['config']:12s} Top1={r['retrieval_top1_accuracy']}% Relevant={r['relevant_chunk_percentage']}% Answer={r['answer_correctness']}% Context={r['avg_context_words']}w Latency={r['avg_latency_sec']}s")
    if summaries:
        print("\nCost vs quality")
        for r in summaries: print(f"{r['model']:8s} Accuracy={r['accuracy_percent']}% Latency={r['avg_latency_sec']}s In={r['avg_input_tokens']} Out={r['avg_output_tokens']} Cost/Q={r['avg_cost_per_question']}")
    if e2e:
        supported=[r for r in e2e if r['type']!="unsupported"]; uns=[r for r in e2e if r['type']=="unsupported"]
        print("\nEnd-to-end")
        print("Retrieval correctness (Top1):",round(statistics.mean(int(r['retrieval_correct_top1']) for r in supported if r['retrieval_correct_top1'] is not None)*100,2) if supported else 0)
        print("Answer correctness:",round(statistics.mean(int(r['answer_correct']) for r in e2e)*100,2))
        print("Faithfulness proxy:",round(statistics.mean(r['faithfulness_proxy'] for r in e2e),4))
        print("Unsupported handling:",round(statistics.mean(int(r['unsupported_question_handling']) for r in uns)*100,2) if uns else 0)
        print("Citation/source correctness:",round(statistics.mean(int(r['citation_source_correct']) for r in supported if r['citation_source_correct'] is not None)*100,2) if supported else 0)
        print("Latency:",round(statistics.mean(r['total_latency_sec'] for r in e2e),4),"s")

if __name__=="__main__": main()
