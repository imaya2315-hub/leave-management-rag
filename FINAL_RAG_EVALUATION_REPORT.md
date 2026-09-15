# Leave Management RAG — Final Evaluation Report

## 1. Overview

This report summarizes the quantitative evaluation of the Leave Management System's Retrieval-Augmented Generation (RAG) pipeline.

### Final selected architecture

```text
Policy Documents
      ↓
Chunking: 80 words / 20-word overlap
      ↓
Embedding / Retrieval: all-MiniLM-L6-v2
      ↓
Semantic Retrieval: Top-10
      ↓
Reranking: cross-encoder/ms-marco-MiniLM-L-6-v2
      ↓
Top-3 grounded context
      ↓
Generation: Qwen 3.8-27B
      ↓
Answer + sources
```

The production architecture separates policy RAG from employee-specific operations. RAG is used for policy questions, while FastAPI remains authoritative for authentication, authorization, validation, balances, leave transactions, and manager actions.

---

# 2. Evaluation Summary

| Component | Final choice | Main evidence |
|---|---|---|
| Chunking | **80 words + 20-word overlap** | Highest new-ablation MRR = **0.950**; Relevant Top-3 = **100%** |
| Embedding / Retrieval | **all-MiniLM-L6-v2** | **90% Top-1**, MRR **0.925**, much lower index/query cost than MPNet |
| Reranker | **cross-encoder/ms-marco-MiniLM-L-6-v2** | Top-1 **90% → 100%**, MRR **0.900 → 1.000** |
| Query decomposition | **Retained in production architecture** | Deterministic ablation showed no improvement on the small 3-query test |
| Generation | **Qwen 3.8-27B** | Better quality/speed trade-off in the selected LLM benchmark |
| End-to-end | RAG + reranking + grounded generation | Earlier end-to-end run: **100% retrieval correctness**, **100% answer correctness**, faithfulness proxy **1.0**, citation/source correctness **100%** |

---

# 3. Retrieval Baseline

## 3.1 12-question baseline

| Metric | Semantic Retrieval | Reranked Retrieval |
|---|---:|---:|
| Top-1 accuracy | **100.0%** | **100.0%** |
| MRR | **1.000** | **1.000** |

This dataset was retained as the baseline correctness benchmark.

## 3.2 19-question hard retrieval set

The hard set contained paraphrased, indirect, and less lexically similar policy questions.

| Metric | Semantic Retrieval | Reranked Retrieval |
|---|---:|---:|
| Top-1 accuracy | **84.2% (16/19)** | **100.0% (19/19)** |
| MRR | **0.908** | **1.000** |
| Top-1 improvement | — | **+15.8 percentage points** |
| MRR improvement | — | **+0.092** |

The reranker corrected all three semantic Top-1 failures.

---

# 4. Chunking Ablation

This experiment used the remaining evaluation set **Q06–Q15 (10 questions)**.

Metrics:

- **Top-1**: correct policy source ranked first.
- **Relevant Top-3**: correct policy source appears in the top three.
- **MRR**: Mean Reciprocal Rank.
- **Avg Context Words**: average number of words in the top retrieved context.
- **Retrieval Latency**: average retrieval time.
- **Index Build**: time to construct the embedding index.

| Configuration | Chunk Size | Overlap | Chunks | Top-1 | Relevant Top-3 | MRR | Avg Context Words | Retrieval Latency (s) | Index Build (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| small_o0 | 80 | 0 | 12 | 90.0% | 90.0% | 0.925 | 202.4 | 0.0848 | 22.8291 |
| small_o10 | 80 | 10 | 13 | 90.0% | 90.0% | 0.925 | 217.8 | 0.0173 | 0.7601 |
| **small_o20** | **80** | **20** | **16** | **90.0%** | **100.0%** | **0.950** | **214.2** | **0.0172** | **0.5654** |
| medium_o0 | 160 | 0 | 7 | 90.0% | 100.0% | 0.9333 | 352.1 | 0.0326 | 0.8299 |
| medium_o20 | 160 | 20 | 7 | 90.0% | 100.0% | 0.9333 | 366.9 | 0.0423 | 1.6594 |
| medium_o40 | 160 | 40 | 9 | 90.0% | 100.0% | 0.9333 | 377.6 | 0.0282 | 2.5712 |
| large_o0 | 320 | 0 | 4 | 80.0% | 100.0% | 0.900 | 621.4 | 0.0261 | 0.4048 |
| large_o40 | 320 | 40 | 4 | 80.0% | 100.0% | 0.900 | 621.4 | 0.0328 | 0.5928 |
| large_o80 | 320 | 80 | 5 | 80.0% | 100.0% | 0.900 | 597.4 | 0.0424 | 1.0150 |

### Chunking conclusion

**80-word chunks with 20-word overlap** were selected for the final configuration because they achieved:

- Top-1 = **90%**
- Relevant Top-3 = **100%**
- MRR = **0.950**
- Average context = **214.2 words**
- Retrieval latency = **0.0172 s**

Large 320-word chunks reduced Top-1 to **80%**.

> Note: The earlier 5-question pilot favored 80/10. The broader Q06–Q15 evaluation favors 80/20, so the broader evaluation is used for the final selection.

---

# 5. Retrieval-Model Comparison

The comparison used the fixed chunking configuration of **80 words / 10-word overlap** in the remaining-evaluation script.

| Model | Model Name | Top-1 | MRR | Avg Query Latency (s) | Index Build (s) |
|---|---|---:|---:|---:|---:|
| **MiniLM-L6-v2** | sentence-transformers/all-MiniLM-L6-v2 | **90.0%** | 0.925 | **0.0333** | **11.1833** |
| MPNet-base-v2 | sentence-transformers/all-mpnet-base-v2 | **90.0%** | **0.950** | 0.1318 | 446.0288 |

### Retrieval-model conclusion

MPNet improved MRR from **0.925 → 0.950**, but Top-1 remained **90%**.

MiniLM was substantially more efficient:

- Query latency: **0.0333 s vs 0.1318 s**
- Index build: **11.1833 s vs 446.0288 s**

Therefore **all-MiniLM-L6-v2** was retained as the practical retrieval model.

> Note: The retrieval-model and reranker ablations used 80/10 as the fixed configuration because that was the fixed setting in the remaining-evaluation script. The chunking experiment subsequently identified 80/20 as the preferred final chunking configuration. If a strict final ablation is needed, these two experiments should be rerun once with 80/20 so every downstream experiment uses the same selected chunking baseline.

---

# 6. Reranker Ablation

| Variant | Top-1 | MRR | Average Latency (s) |
|---|---:|---:|---:|
| Without reranker | 90.0% | 0.900 | **0.0162** |
| **With reranker** | **100.0%** | **1.000** | 1.6253 |
| Improvement | **+10.0 pp** | **+0.100** | +1.6091 s |

### Reranker conclusion

The cross-encoder produced a clear retrieval-quality gain:

- Top-1: **90% → 100%**
- MRR: **0.900 → 1.000**

The additional latency was approximately **1.6091 seconds**.

This finding is consistent with the separate 19-question hard evaluation, where reranking improved Top-1 from **84.2% → 100%** and MRR from **0.908 → 1.000**.

---

# 7. Query-Decomposition Ablation

The remaining evaluation used three multi-intent test questions.

| Mode | All Intents Recovered | Expected Titles Recovered | Avg Latency (s) |
|---|---:|---:|---:|
| Without decomposition | 66.67% | 83.33% | 0.0189 |
| With deterministic decomposition | 66.67% | 83.33% | 0.0410 |
| Change | **0.00 pp** | **0.00 pp** | **+0.0221 s** |

### Decomposition conclusion

The deterministic decomposition ablation did **not** improve retrieval coverage on this small test:

- All intents recovered: **66.67% → 66.67%**
- Expected titles recovered: **83.33% → 83.33%**

It added only **0.0221 seconds** of measured latency.

The production system retains query decomposition because the application uses an LLM-assisted decomposition path when configured, while the above experiment specifically tested the quota-safe deterministic decomposition method.

---

# 8. LLM Quality Evaluation

## 8.1 GPT-OSS-20B vs Qwen 3.8-27B — 40-question benchmark

| Metric | GPT-OSS-20B | Qwen 3.8-27B |
|---|---:|---:|
| Answer correctness | **75%** | **85%** |
| Hallucination rate | **7.5%** | **7.5%** |
| Unsupported/refusal handling | **90%** | **100%** |
| Average latency | **4.189 s** | **3.751 s** |

### Manual validation note

Manual review indicated that the deterministic grader was conservative:

- GPT-OSS-20B: approximately **90% validated correctness**, approximately **2.5% true hallucination** on the reviewed run.
- Qwen 3.8-27B: the reviewed flagged answers were substantively correct, giving approximately **100% validated correctness** and **0% observed true hallucination** for that reviewed run.

The deterministic benchmark is retained as the reproducible automated result; manual validation is reported only as a qualification.

---

# 9. LLM Parameter Sweep

## GPT-OSS-20B

| Configuration | Correctness | Hallucination | Refusal | Latency |
|---|---:|---:|---:|---:|
| Baseline: temp 0, max 120, top_p 1 | 50.0% | 8.3% | 50.0% | 3.028 s |
| max 80 | 25.0% | 8.3% | 50.0% | 2.085 s |
| max 200 | **83.3%** | 16.7% | 50.0% | 2.041 s |
| temp 0.2 | 50.0% | 16.7% | 0.0% | 2.422 s |
| top_p 0.8 | 58.3% | 8.3% | 50.0% | 2.115 s |
| temp 0.2, top_p 0.8 | 50.0% | 8.3% | 50.0% | 4.338 s |

**Selected GPT-OSS-20B configuration:** temperature = **0**, max_tokens = **200**, top_p = **1**.

## Qwen 3.8-27B

| Configuration | Correctness | Hallucination | Refusal | Latency |
|---|---:|---:|---:|---:|
| Baseline | 75.0% | 25.0% | 100.0% | — |
| Short output | 75.0% | 25.0% | 100.0% | — |
| Longer output | 75.0% | 25.0% | 100.0% | — |
| temp 0.2 | 75.0% | 25.0% | 100.0% | — |
| top_p 0.8 | 75.0% | 25.0% | 100.0% | — |
| **temp 0.2, top_p 0.8** | **75.0%** | **16.7%** | **100.0%** | **1.899 s** |

**Selected Qwen configuration:** temperature = **0.2**, max_tokens = **120**, top_p = **0.8**.

---

# 10. LLM Token and Cost Evaluation

A separate LLM efficiency run recorded input/output token counts.

| Model | Accuracy | Avg Input Tokens | Avg Output Tokens | Avg Latency |
|---|---:|---:|---:|---:|
| GPT-OSS-120B | **100%** | 643.4 | 83.6 | 2.6172 s |
| Qwen 3.8-27B | **100%** | 602 | 21 | **2.0354 s** |

The earlier evaluator incorrectly produced `Cost/Q = 0.0`, so the monetary cost values below are **recalculated estimates** from the recorded token counts and the pricing assumptions used during analysis; they are not values emitted by the original evaluator.

| Model | Estimated Cost / Query |
|---|---:|
| GPT-OSS-120B | **$0.0001467** |
| Qwen 3.8-27B | **$0.0005656** |

### Token/cost interpretation

- Qwen used fewer average input tokens: **602 vs 643.4**
- Qwen generated substantially fewer output tokens: **21 vs 83.6**
- Qwen was faster: **2.0354 s vs 2.6172 s**
- Under the pricing assumptions used in this analysis, GPT-OSS-120B had the lower estimated monetary cost/query.

Therefore:

- **Qwen** = stronger speed/response-length trade-off.
- **GPT-OSS-120B** = stronger estimated cost efficiency.

---

# 11. End-to-End Evaluation

Earlier end-to-end evaluation results:

| Metric | Result |
|---|---:|
| Retrieval correctness (Top-1) | **100%** |
| Answer correctness | **100%** |
| Faithfulness proxy | **1.0** |
| Unsupported handling | **0** |
| Citation/source correctness | **100%** |
| Latency | **5.0012 s** |

The `Unsupported handling = 0` value is retained exactly as produced by the evaluator and should not be converted to a percentage without reviewing the evaluator's encoding.

---

# 12. Final Model Selection

## Retrieval

### Selected: `sentence-transformers/all-MiniLM-L6-v2`

Reason:

- Top-1 = **90%** in the 10-question retrieval-model comparison.
- MRR = **0.925**.
- Query latency = **0.0333 s**.
- Index build = **11.1833 s**.

MPNet achieved MRR **0.950**, but required **446.0288 s** to build the index and **0.1318 s** per query in the comparison.

## Reranking

### Selected: `cross-encoder/ms-marco-MiniLM-L-6-v2`

Reason:

- Top-1 = **100%**
- MRR = **1.000**
- +10 percentage points Top-1 over no reranker
- +0.10 MRR over no reranker

## Chunking

### Selected: 80 words / 20-word overlap

Reason:

- Top-1 = **90%**
- Relevant Top-3 = **100%**
- MRR = **0.950**
- Context = **214.2 words**
- Retrieval latency = **0.0172 s**

## Generation

### Selected: Qwen 3.8-27B

Reason:

- 40-question automated benchmark correctness = **85%**, compared with **75%** for GPT-OSS-20B.
- Same deterministic hallucination rate = **7.5%** in that benchmark.
- Refusal handling = **100%**, compared with **90%**.
- Average latency = **3.751 s**, compared with **4.189 s**.
- Separate token run: **100% accuracy** and **2.0354 s** average latency.

The cost trade-off is retained as a limitation: the estimated per-query cost was higher for Qwen under the pricing assumptions used.

---

# 13. Final Evaluation Conclusions

1. **Reranking is the strongest demonstrated architectural improvement.** It consistently improved retrieval quality, reaching 100% Top-1 and 1.000 MRR in both the hard benchmark and the new ablation.
2. **Smaller chunks performed better than large chunks** on the current policy corpus. The selected 80/20 configuration gave the best MRR and complete Relevant Top-3 coverage in the new chunking experiment.
3. **MiniLM provides the best practical retrieval trade-off** between ranking quality and computational cost.
4. **Query decomposition was not shown to improve retrieval in the deterministic ablation**, so it should not be presented as a proven retrieval-quality gain from this experiment.
5. **Qwen 3.8-27B is the preferred generation model for the final application** based on the combined quality and latency evidence, while GPT-OSS-120B remains the cheaper option under the cost assumptions used.
6. The previously completed end-to-end benchmark achieved **100% retrieval correctness, 100% answer correctness, 1.0 faithfulness proxy, and 100% citation/source correctness** on its evaluation set.

---

# 14. Reproducibility Notes

- Policy documents evaluated: **4**
  - Approval Escalation
  - Carry Forward Rules
  - Holiday Calendar
  - Leave Types And Eligibility
- Original policy corpus produced **9 retrievable chunks** in the initial configuration.
- Embedding dimension for all-MiniLM-L6-v2: **384**
- Semantic retrieval candidate count: **Top-10**
- Reranker output count: **Top-3**
- New remaining ablations: **Q06–Q15 (10 cases)**
- Query-decomposition ablation: **3 multi-intent cases**
- LLM parameter sweep: **12-question runs**
- Main LLM benchmark: **40 questions**
- Hard retrieval benchmark: **19 questions**
- Baseline retrieval benchmark: **12 questions**
- No new Groq generation calls were used in the remaining chunking/retrieval/reranker/decomposition ablations.

---

## Important consistency note before publication

The chunking experiment selected **80/20**, while the retrieval-model and reranker ablations were run with **80/10** as their fixed configuration. Therefore, the results are valid as separate experiments, but the downstream ablations are not all controlled against the same final chunking configuration. For a strict dissertation-grade final comparison, rerun the retrieval-model, reranker, and decomposition ablations once using **80/20** as the fixed chunking baseline.
