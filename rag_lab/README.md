# rag_lab — a standalone RAG learning pipeline

This folder is deliberately **separate** from `app/` (the FastAPI leave-management backend). Nothing here imports from `app/`, and nothing in `app/` imports from here. The point is to build and inspect each stage of a RAG pipeline on its own, then integrate it with the backend only at the very end, through `streamlit_app.py`.

```
Employee -> AI Agent / Router
              /            \
             v              v
        RAG Pipeline    FastAPI backend
        (rag_lab/)      (app/, over HTTP)
             |               |
      Policy knowledge   Live employee data
      (documents/)       + leave actions
```

## Stages, in order

| # | Module | What it does |
|---|--------|---------------|
| 1 | `ingestion.py` | Loads raw policy `.txt` files from `documents/` |
| 2 | `chunking.py` | Splits each document into overlapping word chunks |
| 3 | `embeddings.py` | Turns text into a semantic vector (`all-MiniLM-L6-v2`) |
| 4 | `vector_store.py` | Stores chunk vectors in FAISS, searches by similarity |
| 5 | `retrieval.py` | `tfidf_retrieve` (keyword) and `SemanticRetriever` (meaning) — side by side, so they can be compared |
| 6 | `decomposition.py` | Splits a compound question into independent sub-questions |
| 7 | `reranking.py` | Cross-encoder reranks a wide candidate set down to the best few |
| 8 | `generation.py` | Generates the final answer, grounded strictly in retrieved context |
| 9 | `evaluation.py` | Compares Recall@K across TF-IDF / Semantic / +Decomposition / +Reranking |

Every module has a `python -m rag_lab.<module>` block at the bottom — run any of them directly to see that stage's output in isolation, e.g.:

```bash
python -m rag_lab.chunking      # see what a real chunk looks like
python -m rag_lab.embeddings    # see an actual embedding vector
python -m rag_lab.retrieval     # compare TF-IDF vs semantic on one query
python -m rag_lab.reranking     # see reordering before/after reranking
python -m rag_lab.evaluation    # full Recall@K comparison table
```

## Offline fallbacks

`embeddings.py` and `reranking.py` need to reach Hugging Face on first use to download `all-MiniLM-L6-v2` and `cross-encoder/ms-marco-MiniLM-L-6-v2`. If that download isn't available, both fall back automatically — a deterministic hashing-based vector for embeddings, and the retriever's own ordering for reranking — so the pipeline still runs end to end. Those fallbacks are for keeping the pipeline runnable without network access; they don't produce real semantic results, so don't draw conclusions from `evaluation.py`'s numbers until the real models have loaded at least once (check with `python -m rag_lab.embeddings` — it prints `Using real sentence-transformer model: True/False`).

`generation.py` and `decomposition.py` use an LLM if `GROQ_API_KEY` or `ANTHROPIC_API_KEY` is set (Groq is tried first), otherwise fall back to an offline extractive answer / heuristic split respectively.

## Connecting it to the backend

`streamlit_app.py` (project root) is the integration point:

```bash
# terminal 1
uvicorn app.main:app --reload

# terminal 2
streamlit run streamlit_app.py
```

It routes each chat message to one of three places:
- **Policy question** → this pipeline (`rag_lab/`), never touches the backend
- **"my balance" / "my history"** → `GET /employees/me/` or `/leaves/me/` on the backend
- **"apply leave..."** → collects any missing leave type / dates across turns, then `POST /leaves/`

The router is a simple keyword heuristic (`route_message` in `streamlit_app.py`) — swap it for an LLM-based intent classifier if you want to demonstrate that as a further extension.
