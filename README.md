# Leave Management System — FastAPI + PostgreSQL + RAG + LLM Tool Calling

A layered leave-management application with:

- FastAPI backend
- PostgreSQL database
- JWT authentication
- Role-based authorization
- Streamlit conversational interface
- Policy-focused RAG
- Semantic retrieval and cross-encoder reranking
- LLM tool calling for intent selection
- Deterministic Python execution for state-changing operations

The system separates **policy reasoning**, **live employee data**, and **leave transactions**. The LLM does not directly modify the database. FastAPI remains the source of truth for authentication, authorization, validation, balances, approval rules, and database transactions.

---

## Architecture

```text
                         ┌─────────────────────────┐
                         │      Streamlit UI       │
                         │   streamlit_app.py      │
                         └────────────┬────────────┘
                                      │
                           Natural-language request
                                      │
                         ┌────────────▼────────────┐
                         │     LLM Tool Router     │
                         │       Groq API          │
                         └────────────┬────────────┘
                                      │
                              Select one tool
                                      │
          ┌───────────────────────────┼───────────────────────────┐
          │                           │                           │
          ▼                           ▼                           ▼
   Policy Question              Live Employee Data          Leave Actions
          │                           │                           │
          ▼                           ▼                           ▼
      RAG Pipeline                FastAPI API                FastAPI API
   Retrieve → Rerank             Auth + DB                 Auth + Rules
      → Generate                     │                           │
          │                          │                           │
          └──────────────────────────┴───────────────────────────┘
                                     │
                                PostgreSQL
```

### Core design rule

The LLM only selects a high-level operation and extracts obvious arguments.

Python executes the selected operation.

FastAPI remains authoritative for:

- authentication
- authorization
- leave validation
- date validation
- balance checks
- approval/rejection rules
- database transactions

This prevents a generated LLM response from bypassing backend business rules.

---

# Project Structure

```text
.
├── app/
│   ├── main.py
│   ├── api/
│   │   ├── deps.py
│   │   └── v1/
│   │       ├── router.py
│   │       ├── employees.py
│   │       ├── leaves.py
│   │       ├── auth.py
│   │       ├── oauth.py
│   │       ├── knowledge.py
│   │       └── assistant.py
│   ├── core/
│   │   ├── config.py
│   │   ├── security.py
│   │   └── database.py
│   ├── models/
│   │   ├── employee.py
│   │   ├── leave.py
│   │   └── knowledge.py
│   ├── schemas/
│   │   ├── employee.py
│   │   ├── leave.py
│   │   ├── knowledge.py
│   │   └── assistant.py
│   ├── services/
│   │   ├── employee_service.py
│   │   ├── leave_service.py
│   │   ├── knowledge_service.py
│   │   └── assistant_service.py
│   ├── crud/
│   │   ├── employee.py
│   │   ├── leave.py
│   │   └── knowledge.py
│   ├── rag/
│   │   ├── ingest.py
│   │   ├── retriever.py
│   │   ├── llm.py
│   │   └── policies/
│   └── middleware/
│       └── auth.py
│
├── rag_lab/
│   ├── agent.py
│   ├── chunking.py
│   ├── decomposition.py
│   ├── embeddings.py
│   ├── evaluation.py
│   ├── generation.py
│   ├── ingestion.py
│   ├── reranking.py
│   ├── retrieval.py
│   ├── vector_store.py
│   └── documents/
│
├── scripts/
│   └── seed_knowledge_base.py
│
├── tests/
├── alembic/
├── .env.example
├── requirements.txt
├── README.md
├── run.py
└── streamlit_app.py
```

`rag_lab/` contains the newer policy-RAG experimentation and evaluation pipeline used by the conversational assistant. `app/rag/` contains the backend knowledge-base/RAG implementation.

---

# Backend Architecture

The FastAPI backend follows:

```text
API Routes
    ↓
Services
    ↓
CRUD
    ↓
SQLAlchemy Models
    ↓
PostgreSQL
```

### API layer

Handles HTTP requests and responses.

### Service layer

Contains business rules such as:

- leave validation
- balance validation
- approval/rejection logic
- duplicate pending-request protection
- employee-related rules

### CRUD layer

Contains database read/write operations.

### Models and schemas

SQLAlchemy models represent database tables. Pydantic schemas validate request and response data.

---

# Authentication and Authorization

The application uses JWT authentication.

Login returns:

```json
{
  "access_token": "...",
  "refresh_token": "...",
  "token_type": "bearer"
}
```

Access tokens protect authenticated operations. Refresh tokens are used to obtain a new access token without a fresh login.

## Roles

The current workflow supports:

```text
employee
manager
admin
```

---

# Team-Based Approval Rules

Approval authorization is based on **team membership**, not department.

- An `employee` request can be approved or rejected only by the manager of that employee's team.
- A `manager` request can be approved or rejected only by an `admin`.
- An `admin` request cannot be approved through the current workflow.
- A user cannot approve or reject their own leave request.
- Only admins can create teams and assign or remove team members.
- A manager can manage only one team.
- Team membership controls approval authorization; department does not.

These rules are enforced by the backend rather than by the LLM.

## Initial admin setup

Because the team model was deployed after the application originally used only employee/manager roles, an existing trusted account can be promoted once directly in PostgreSQL:

```sql
UPDATE employees
SET role = 'admin'
WHERE id = <trusted_admin_employee_id>;
```

Restart FastAPI after the change and use that account for team administration and manager-leave approval.

---

# Leave Management

## Applying for leave

A leave application initially creates:

```text
status = Pending
```

The balance is validated, but is not deducted at submission time.

The balance is deducted after approval.

## Manager approval

```text
PUT /api/v1/leaves/{id}/approve
```

The backend:

1. verifies authorization
2. verifies that the request is pending
3. re-validates the leave
4. checks available balance
5. deducts the balance
6. changes the request to `Approved`

## Manager rejection

```text
PUT /api/v1/leaves/{id}/reject
```

The request becomes:

```text
Rejected
```

No balance is deducted.

## Cancellation

```text
PUT /api/v1/leaves/{id}/cancel
```

Cancellation can restore the balance when an already-approved request had previously deducted it.

---

# Duplicate Pending Requests

The leave service prevents a new identical pending request from being created when the same employee, leave type, dates, and pending status already exist.

Existing duplicate records are not automatically deleted. The conversational manager workflow can disambiguate existing duplicates by presenting numbered choices.

---

# Policy RAG

The policy assistant is designed for company-policy questions rather than employee-specific database actions.

Current policy documents:

```text
Leave Types and Eligibility
Carry Forward Rules
Holiday Calendar
Approval Escalation
```

Initial policy corpus:

```text
4 policy documents
9 retrievable chunks
```

## RAG Pipeline

```text
User Question
     ↓
Query Decomposition
     ↓
Semantic Retrieval
     ↓
Candidate Merge / Deduplication
     ↓
Cross-Encoder Reranking
     ↓
Top Retrieved Context
     ↓
Grounded LLM Generation
```

## Retrieval

Embedding model:

```text
sentence-transformers/all-MiniLM-L6-v2
```

Embedding dimension:

```text
384
```

Semantic retrieval uses FAISS and retrieves up to the top 10 candidates before reranking.

## Reranking

Current reranker:

```text
cross-encoder/ms-marco-MiniLM-L-6-v2
```

The reranker selects the best 3 chunks for grounded context.

## Generation

The conversational tool router and current generation layer use the configured Groq model from the application configuration.

---

# LLM Tool Calling

The conversational interface uses high-level tools such as:

```text
answer_policy_question
get_leave_balance
get_leave_history
get_pending_leave_requests
apply_leave
cancel_leave
approve_leave
reject_leave
```

The model selects the appropriate tool. Python then executes the operation deterministically.

For state-changing operations, the LLM does not call FastAPI directly and does not modify PostgreSQL directly.

---

# Natural-Language Leave Requests

Employees can use natural language such as:

```text
I need Monday off
```

or:

```text
I want to be away for three days starting next Tuesday
```

Missing information can be collected over multiple turns.

Example:

```text
User: I need Monday off
Assistant: What type of leave would you like to use?

User: Annual
Assistant: [submits the request]
```

Date normalization is handled by deterministic Python logic rather than trusting the LLM to invent the final transaction date.

---

# Multi-Turn Action State

Conversational actions use:

```python
st.session_state.pending_action
```

This stores unfinished operations while the assistant collects missing information.

For example:

```text
User: Apply leave
Assistant: Which leave type?
User: Casual
Assistant: What dates?
User: Next Monday
```

The follow-up messages remain attached to the original pending action instead of being interpreted as unrelated requests.

---

# Manager Approval by Employee Name

Managers can work with employee names instead of internal leave IDs.

Example:

```text
approve Avinash leave
```

If several requests match, the application asks for a distinguishing detail.

For genuinely identical requests, it can display:

```text
1. Annual leave — 2026-09-07
2. Annual leave — 2026-09-07
3. Annual leave — 2026-09-07
```

The manager can respond:

```text
2
```

The visible number is mapped internally to the corresponding database leave ID.

---

# Pending Leave Requests

Managers can retrieve pending requests using:

```text
GET /api/v1/leaves/pending/
```

The conversational interface can show employee name, leave type, dates, status, and other relevant details while keeping internal database identifiers as implementation details.

---

# Separation of Concerns

## Policy question

```text
User
 ↓
LLM tool selection
 ↓
RAG retrieval
 ↓
Reranking
 ↓
Grounded answer
```

## Live employee information

```text
User
 ↓
LLM tool selection
 ↓
FastAPI
 ↓
Authenticated employee data
```

## Leave action

```text
User
 ↓
LLM tool selection
 ↓
Deterministic Python
 ↓
FastAPI
 ↓
Business validation
 ↓
PostgreSQL transaction
```

---

# FastAPI Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Health check |
| POST | `/api/v1/employees/` | Register employee |
| GET | `/api/v1/employees/` | List employees |
| GET | `/api/v1/employees/{id}` | Get employee |
| PUT | `/api/v1/employees/{id}` | Update employee |
| DELETE | `/api/v1/employees/{id}` | Delete employee |
| POST | `/api/v1/login` | Login |
| POST | `/api/v1/refresh` | Refresh JWT |
| GET | `/api/v1/employees/me/` | Current employee profile |
| POST | `/api/v1/employees/{id}/leaves/` | Apply for leave |
| GET | `/api/v1/employees/{id}/leaves/` | Leave history |
| GET | `/api/v1/leaves/pending/` | Pending requests |
| PUT | `/api/v1/leaves/{id}/approve` | Approve leave |
| PUT | `/api/v1/leaves/{id}/reject` | Reject leave |
| PUT | `/api/v1/leaves/{id}/cancel` | Cancel leave |
| POST | `/api/v1/knowledge/documents` | Add policy document |
| GET | `/api/v1/knowledge/documents` | List policy documents |
| DELETE | `/api/v1/knowledge/documents/{id}` | Delete policy document |
| POST | `/api/v1/assistant/ask` | Assistant endpoint |
| POST | `/api/v1/oauth/clients` | Register automation client |
| POST | `/api/v1/oauth/token` | OAuth2 client-credentials token |

---

# OAuth2 for Automation

The backend supports OAuth2 Client Credentials for trusted automation clients.

```text
Register client
     ↓
client_id + client_secret
     ↓
OAuth token endpoint
     ↓
Service access token
     ↓
Protected automation endpoint
```

Service tokens are separate from employee JWT access/refresh tokens.

---

# Environment Variables

Create `.env` from `.env.example`.

Example:

```env
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/leave_db
SECRET_KEY=your-secret-key
GROQ_API_KEY=your-groq-api-key
```

Generate a secret key with:

```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

Never commit `.env`, database passwords, API keys, JWT secrets, or other credentials to GitHub.

---

# Setup

## Prerequisites

- Python 3.11+
- PostgreSQL
- Groq API key

## Create virtual environment

### Windows

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### macOS/Linux

```bash
python -m venv .venv
source .venv/bin/activate
```

## Install dependencies

```bash
pip install -r requirements.txt
```

## Create the database

```sql
CREATE DATABASE leave_db;
```

## Configure environment

Copy:

```text
.env.example
```

to:

```text
.env
```

and add the required credentials.

## Run migrations

```bash
alembic upgrade head
```

## Start FastAPI

```bash
uvicorn app.main:app --reload
```

API:

```text
http://127.0.0.1:8000
```

Swagger:

```text
http://127.0.0.1:8000/docs
```

## Start Streamlit

In another terminal:

```bash
streamlit run streamlit_app.py
```

---

# Seed Policy Knowledge

The backend sample knowledge base can be seeded with:

```bash
python scripts/seed_knowledge_base.py
```

The newer `rag_lab` policy corpus is used separately for retrieval/reranking experimentation.

---

# Testing

Run:

```bash
pytest -q
```

Useful API checks:

```text
GET /openapi.json
GET /api/v1/leaves/pending/
PUT /api/v1/leaves/{id}/approve
PUT /api/v1/leaves/{id}/reject
```

---

# RAG Evaluation Results

The evaluation was separated into baseline retrieval, hard retrieval, chunking, embedding-model, reranker, query-decomposition, and LLM quality/efficiency experiments.

## Retrieval baseline

### 12-question baseline

```text
Semantic Top-1       100.0%
Reranked Top-1       100.0%
Semantic MRR         1.000
Reranked MRR         1.000
```

### 19-question hard set

```text
Semantic Top-1       84.2%  (16/19)
Reranked Top-1       100.0%
Semantic MRR         0.908
Reranked MRR         1.000
Top-1 improvement    +15.8 percentage points
MRR improvement      +0.092
```

The reranker corrected all three semantic Top-1 failures.

---

# Chunking Ablation

The broader chunking ablation used Q06–Q15 (10 questions).

| Configuration | Size | Overlap | Top-1 | Relevant Top-3 | MRR | Avg Context Words | Retrieval Latency |
|---|---:|---:|---:|---:|---:|---:|---:|
| small_o0 | 80 | 0 | 90.0% | 90.0% | 0.925 | 202.4 | 0.0848 s |
| small_o10 | 80 | 10 | 90.0% | 90.0% | 0.925 | 217.8 | 0.0173 s |
| **small_o20** | **80** | **20** | **90.0%** | **100.0%** | **0.950** | **214.2** | **0.0172 s** |
| medium_o0 | 160 | 0 | 90.0% | 100.0% | 0.9333 | 352.1 | 0.0326 s |
| medium_o20 | 160 | 20 | 90.0% | 100.0% | 0.9333 | 366.9 | 0.0423 s |
| medium_o40 | 160 | 40 | 90.0% | 100.0% | 0.9333 | 377.6 | 0.0282 s |
| large_o0 | 320 | 0 | 80.0% | 100.0% | 0.900 | 621.4 | 0.0261 s |
| large_o40 | 320 | 40 | 80.0% | 100.0% | 0.900 | 621.4 | 0.0328 s |
| large_o80 | 320 | 80 | 80.0% | 100.0% | 0.900 | 597.4 | 0.0424 s |

### Selected chunking

```text
80-word chunks
20-word overlap
```

This configuration achieved the highest MRR in the new chunking ablation and 100% relevant Top-3 coverage.

---

# Retrieval-Model Comparison

| Model | Top-1 | MRR | Query Latency | Index Build |
|---|---:|---:|---:|---:|
| **all-MiniLM-L6-v2** | **90.0%** | 0.925 | **0.0333 s** | **11.1833 s** |
| all-mpnet-base-v2 | 90.0% | **0.950** | 0.1318 s | 446.0288 s |

MiniLM provides a stronger quality/efficiency trade-off on this corpus.

---

# Reranker Ablation

| Variant | Top-1 | MRR | Average Latency |
|---|---:|---:|---:|
| Without reranker | 90.0% | 0.900 | 0.0162 s |
| **With reranker** | **100.0%** | **1.000** | 1.6253 s |
| Improvement | **+10 pp** | **+0.10** | +1.6091 s |

The cross-encoder reranker is therefore retained.

---

# Query-Decomposition Ablation

Three multi-intent questions were tested using a quota-safe deterministic decomposition method.

| Mode | All Intents Recovered | Expected Titles Recovered | Latency |
|---|---:|---:|---:|
| Without decomposition | 66.67% | 83.33% | 0.0189 s |
| Deterministic decomposition | 66.67% | 83.33% | 0.0410 s |

The deterministic experiment showed no retrieval-coverage improvement. This result does not measure the production LLM-assisted decomposition path.

---

# LLM Evaluation

## 40-question benchmark

| Metric | GPT-OSS-20B | Qwen 3.8-27B |
|---|---:|---:|
| Answer correctness | 75% | **85%** |
| Hallucination rate | 7.5% | 7.5% |
| Refusal handling | 90% | **100%** |
| Average latency | 4.189 s | **3.751 s** |

Manual review suggested the deterministic grader was conservative for some flagged cases. Those manual observations should be treated as validation notes rather than replacement benchmark scores.

## Parameter selection

### GPT-OSS-20B

Selected:

```text
temperature = 0
max_tokens  = 200
top_p       = 1
```

Best measured correctness in the parameter sweep:

```text
83.3%
```

### Qwen 3.8-27B

Selected:

```text
temperature = 0.2
max_tokens  = 120
top_p       = 0.8
```

Selected parameter set reduced the observed hallucination rate in the sweep to:

```text
16.7%
```

while retaining:

```text
75.0% correctness
100.0% refusal handling
```

---

# LLM Token and Cost Efficiency

Previously recorded 40-question/token-run averages:

| Model | Accuracy | Avg Input Tokens | Avg Output Tokens | Avg Latency |
|---|---:|---:|---:|---:|
| GPT-OSS-120B | 100% | 643.4 | 83.6 | 2.6172 s |
| Qwen 3.8-27B | 100% | 602 | 21 | **2.0354 s** |

Estimated cost/query from the recorded token counts and pricing assumptions used during the evaluation:

```text
GPT-OSS-120B  ≈ $0.0001467/query
Qwen 3.8-27B  ≈ $0.0005656/query
```

This indicates a trade-off:

- Qwen: lower latency and shorter output
- GPT-OSS-120B: lower estimated API cost under the pricing assumptions used

---

# End-to-End Evaluation

Previous end-to-end results:

| Metric | Result |
|---|---:|
| Retrieval correctness | **100%** |
| Answer correctness | **100%** |
| Faithfulness proxy | **1.0** |
| Citation/source correctness | **100%** |
| Latency | **5.0012 s** |

`Unsupported handling = 0` was produced by the evaluator and should be interpreted according to that evaluator's encoding rather than assumed to be a percentage.

---

# Final Selected RAG Configuration

```text
Chunking
  80 words
  20-word overlap

Embedding / Retrieval
  sentence-transformers/all-MiniLM-L6-v2
  384-dimensional embeddings

Semantic Retrieval
  Top-10 candidates

Reranker
  cross-encoder/ms-marco-MiniLM-L-6-v2
  Top-3 context

Generation
  Qwen 3.8-27B
  temperature = 0.2
  top_p = 0.8
  max_tokens = 120
```

The generation model is selected for the final application based on the combined quality and latency evidence. GPT-OSS remains a lower-cost alternative under the pricing assumptions used in the cost evaluation.

---

# Key Engineering Decisions

### LLM for intent, deterministic code for execution

Natural-language understanding benefits from an LLM, while state-changing operations are executed by deterministic Python and FastAPI.

### Backend as source of truth

FastAPI decides whether a leave can be created, approved, rejected, or cancelled.

### RAG for policy, API for live data

Policy rules come from the knowledge base. Employee balances and leave records come from authenticated PostgreSQL-backed APIs.

### Reranking after semantic retrieval

Semantic retrieval creates a candidate set; the cross-encoder reranker improves the ordering of the final context.

### Deterministic date normalization

Dates are extracted and normalized by Python logic before transactions are sent to the backend.

### Multi-turn state

Pending actions are stored in Streamlit session state so clarification messages remain associated with the original operation.

### Hidden internal IDs

Users work with employee names, leave types, dates, and visible option numbers. Database IDs remain internal implementation details.

---

# Known Data-Cleanup Consideration

During testing, duplicate pending leave rows were created for some employees.

The application now prevents new identical pending requests, but existing duplicates may still require manual review and cleanup.

---

# Running the Project

```powershell
python -m venv .venv
.venv\Scripts\activate

pip install -r requirements.txt

alembic upgrade head

uvicorn app.main:app --reload
```

In a second terminal:

```powershell
.venv\Scripts\activate
streamlit run streamlit_app.py
```

Open:

```text
http://127.0.0.1:8000/docs
```

for FastAPI Swagger.

---

# Security

Do not commit:

```text
.env
API keys
database credentials
JWT secrets
client secrets
tokens
```

Use `.env.example` as the public configuration template.

---

# Status

The project currently combines:

```text
FastAPI backend
+
PostgreSQL
+
JWT authentication
+
Team-based authorization
+
Streamlit conversational UI
+
LLM tool calling
+
Policy RAG
+
Semantic retrieval
+
Cross-encoder reranking
+
Grounded generation
+
Multi-turn leave workflows
```

The evaluated RAG architecture demonstrates strong retrieval performance, with reranking consistently providing the largest measured improvement in ranking quality.
