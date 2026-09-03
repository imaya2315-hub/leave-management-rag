# Leave Management System — FastAPI + RAG + LLM Tool Calling

A layered leave-management application with a FastAPI backend, PostgreSQL database, JWT authentication, a policy-focused RAG pipeline, and a Streamlit conversational interface.

The current system separates **policy reasoning**, **live employee data**, and **leave actions**. The LLM is used for intent/tool selection, while the backend remains the authority for authentication, validation, balances, and database transactions.

---

## Current Architecture

```text
                         ┌─────────────────────────┐
                         │      Streamlit UI       │
                         │   streamlit_app.py      │
                         └────────────┬────────────┘
                                      │
                         Natural-language request
                                      │
                         ┌────────────▼────────────┐
                         │   LLM Tool Router       │
                         │   Groq / GPT-OSS-20B    │
                         └────────────┬────────────┘
                                      │
                         Select exactly one tool
                                      │
          ┌───────────────────────────┼───────────────────────────┐
          │                           │                           │
          ▼                           ▼                           ▼
   Policy Question              Live Data                 Leave Actions
          │                           │                           │
          ▼                           ▼                           ▼
      RAG Lab                    FastAPI API                FastAPI API
   Retrieve → Rerank             Auth + DB                 Auth + rules
      → Generate                     │                           │
          │                          │                           │
          └──────────────────────────┴───────────────────────────┘
                                     │
                                PostgreSQL
```

### Important design rule

The LLM does **not** directly modify the database.

It only selects a high-level tool and extracts obvious arguments. Python code then executes the selected operation, and FastAPI remains authoritative for:

- authentication and authorization
- leave validation
- date validation
- leave-balance checks
- approval/rejection rules
- database transactions

This prevents an LLM response from bypassing business rules.

---

## Project Structure

```text
.
├── app/
│   ├── main.py
│   │
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
│   │
│   ├── core/
│   │   ├── config.py
│   │   ├── security.py
│   │   └── database.py
│   │
│   ├── models/
│   │   ├── employee.py
│   │   ├── leave.py
│   │   └── knowledge.py
│   │
│   ├── schemas/
│   │   ├── employee.py
│   │   ├── leave.py
│   │   ├── knowledge.py
│   │   └── assistant.py
│   │
│   ├── services/
│   │   ├── employee_service.py
│   │   ├── leave_service.py
│   │   ├── knowledge_service.py
│   │   └── assistant_service.py
│   │
│   ├── crud/
│   │   ├── employee.py
│   │   ├── leave.py
│   │   └── knowledge.py
│   │
│   ├── rag/
│   │   ├── ingest.py
│   │   ├── retriever.py
│   │   ├── llm.py
│   │   └── policies/
│   │
│   └── middleware/
│       └── auth.py
│
├── rag_lab/
│   ├── agent.py
│   ├── ...
│   └── policies/
│
├── scripts/
│   └── seed_knowledge_base.py
│
├── tests/
│
├── alembic/
├── .env.example
├── requirements.txt
├── README.md
├── run.py
└── streamlit_app.py
```

`rag_lab/` contains the newer policy-RAG experimentation/evaluation pipeline used by the Streamlit assistant, while `app/rag/` contains the backend knowledge-base/RAG implementation already present in the API.

---

# 1. Backend Architecture

The FastAPI backend follows a layered structure:

```text
API routes
    ↓
Services
    ↓
CRUD
    ↓
SQLAlchemy models
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
- duplicate pending-request checks
- employee-related business rules

### CRUD layer

Contains database read/write operations.

### Models and schemas

SQLAlchemy models represent database tables, while Pydantic schemas validate API input and output.

---

# 2. Authentication and Authorization

The application uses JWT authentication.

Login returns:

```json
{
  "access_token": "...",
  "refresh_token": "...",
  "token_type": "bearer"
}
```

Access tokens are used for protected employee and manager operations.

Refresh tokens are used to obtain a new access token without logging in again.

Roles currently include:

```text
employee
manager
```

Manager-only operations are protected by backend authorization.

---

# 3. Leave Management

## Applying for leave

A leave application creates a request with:

```text
status = Pending
```

The balance is validated, but it is not deducted yet.

The balance is deducted only after manager approval.

## Manager approval

```text
PUT /api/v1/leaves/{id}/approve
```

The backend:

1. verifies manager authorization
2. verifies that the leave is pending
3. re-validates the leave
4. checks the available balance
5. deducts the balance
6. changes the status to `Approved`

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

Cancellation can refund the balance when the request had already been approved and the balance had therefore been deducted.

---

# 4. Duplicate Pending Request Protection

A duplicate-request problem was encountered while testing the conversational manager workflow.

For example, the database could contain multiple records with exactly the same:

```text
employee
leave type
start date
end date
status = Pending
```

The leave service now checks for an existing identical pending request before creating another one.

This prevents repeated accidental submissions of the same leave request.

Existing duplicate records already present in the database are not automatically deleted. They are handled by the manager disambiguation flow described below.

---

# 5. Policy RAG System

The policy assistant is designed for questions about company leave rules rather than live database actions.

Current policy documents include:

- Leave Types and Eligibility
- Carry Forward Rules
- Holiday Calendar
- Approval Escalation

The current RAG evaluation uses:

```text
4 policy documents
9 retrievable chunks
```

## Retrieval pipeline

The newer `rag_lab` pipeline performs:

```text
User question
    ↓
Query decomposition
    ↓
Semantic retrieval
    ↓
Candidate merge + deduplication
    ↓
Cross-encoder reranking
    ↓
Top retrieved context
    ↓
Grounded LLM generation
```

### Embeddings

Current embedding model:

```text
sentence-transformers/all-MiniLM-L6-v2
```

Embedding dimension:

```text
384
```

### Vector retrieval

FAISS is used for semantic retrieval.

The system retrieves up to the top 10 candidates before reranking.

### Reranking

Current reranker:

```text
cross-encoder/ms-marco-MiniLM-L-6-v2
```

The retrieved candidates are reranked and the top 3 are used as grounded context.

### Generation

Current Groq model used by the RAG pipeline and conversational tool router:

```text
openai/gpt-oss-20b
```

The answer-generation step is grounded in the retrieved policy context.

---

# 6. RAG Evaluation

The RAG pipeline was evaluated using both a direct question set and a harder paraphrased/indirect question set.

## Original evaluation

```text
Semantic Top-1 Accuracy       100%
Reranked Top-1 Accuracy       100%
Semantic MRR                  1.000
Reranked MRR                  1.000
```

## HARD evaluation set

The harder evaluation contained 19 indirect and paraphrased questions.

```text
Semantic Top-1 Accuracy       84.2%  (16/19)
Reranked Top-1 Accuracy       100%
Semantic MRR                  0.908
Reranked MRR                  1.000
```

Reranking corrected the three semantic-retrieval failures.

Improvement:

```text
Top-1 accuracy: +15.8 percentage points
MRR:            +0.092
```

This demonstrates that semantic retrieval can surface relevant candidates while reranking improves the final selection of the most relevant policy chunk.

---

# 7. LLM Tool Calling

The conversational interface previously used keyword-based routing.

The routing system has now been upgraded to **LLM tool calling** so that users can speak naturally.

For example:

```text
I need Monday off
```

does not require the words:

```text
apply
request
leave
```

The LLM can select the appropriate tool based on the user's intent.

## Current high-level tools

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

The model is instructed to select one high-level operation.

Python executes that operation deterministically.

---

# 8. Natural-Language Leave Requests

The employee can now use conversational inputs such as:

```text
I need Monday off
```

or:

```text
I want to be away for three days starting next Tuesday
```

The system can collect missing information across multiple turns.

For example:

```text
User: I need Monday off

Assistant: What type of leave would you like to use?

User: Annual

Assistant: [submits the request]
```

The transaction date is normalized by deterministic Python logic rather than trusting the LLM to invent or reinterpret the final date.

Current date parsing supports examples such as:

```text
today
tomorrow
Monday
next Monday
this Monday
Friday
next Tuesday
2026-09-07
07/09/2026
7th September
three days
3 days
```

The exact date conversion is based on the application's current date and deterministic date parsing rules.

---

# 9. Multi-Turn Action State

Conversational actions use Streamlit session state.

Example:

```python
st.session_state.pending_action
```

This allows the application to remember that it is waiting for a missing piece of information.

This is especially important for manager approvals.

The application checks for an existing pending action before sending a new message through the LLM tool selector.

This prevents a clarification response from accidentally becoming a new leave application.

---

# 10. Manager Approval by Employee Name

Managers do not need to know the internal leave database ID.

A manager can say:

```text
approve Avinash leave
```

The system searches the pending leave requests and matches the employee name.

Example:

```text
Avinash Sinha — Sick leave, 2026-08-31 to 2026-08-31
Avinash Sinha — Annual leave, 2026-09-07 to 2026-09-07
Avinash Sinha — Annual leave, 2026-09-08 to 2026-09-08
```

If only one request matches, it can be approved directly.

If multiple requests match, the manager is asked for more information.

For genuinely identical requests, the system presents numbered choices:

```text
1. Annual leave — 2026-09-07 to 2026-09-07
2. Annual leave — 2026-09-07 to 2026-09-07
3. Annual leave — 2026-09-07 to 2026-09-07
```

The manager can then enter:

```text
2
```

The visible number is mapped internally to the database leave ID.

The internal ID is not required from the manager.

---

# 11. Pending Leave Requests

Managers can retrieve pending requests through:

```text
GET /api/v1/leaves/pending/
```

The response includes employee-facing information such as:

```json
{
  "id": 33,
  "employee_id": 5,
  "employee_name": "Udhaya Anbu",
  "leave_type": "Annual",
  "start_date": "2026-09-08",
  "end_date": "2026-09-10",
  "status": "Pending"
}
```

The internal ID is retained for backend operations while employee names, leave types, and dates are used in the conversational interface.

---

# 12. Separation of Concerns

The current assistant follows these rules:

### Policy question

```text
User
 ↓
LLM selects answer_policy_question
 ↓
RAG retrieval
 ↓
Reranking
 ↓
Grounded answer
```

### Live employee information

```text
User
 ↓
LLM selects get_leave_balance / get_leave_history
 ↓
FastAPI
 ↓
Authenticated employee data
```

### Leave action

```text
User
 ↓
LLM selects apply/cancel/approve/reject
 ↓
Deterministic Python
 ↓
FastAPI
 ↓
Business validation
 ↓
PostgreSQL transaction
```

The LLM never directly writes to the database.

---

# 13. FastAPI Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Health check |
| POST | `/api/v1/employees/` | Register employee |
| GET | `/api/v1/employees/` | List employees |
| GET | `/api/v1/employees/{id}` | Get employee |
| PUT | `/api/v1/employees/{id}` | Update employee |
| DELETE | `/api/v1/employees/{id}` | Delete employee |
| POST | `/api/v1/login` | Login |
| POST | `/api/v1/refresh` | Refresh JWT tokens |
| GET | `/api/v1/employees/me/` | Current employee profile |
| POST | `/api/v1/employees/{id}/leaves/` | Apply for leave |
| GET | `/api/v1/employees/{id}/leaves/` | Leave history |
| GET | `/api/v1/leaves/pending/` | Manager pending requests |
| PUT | `/api/v1/leaves/{id}/approve` | Manager approval |
| PUT | `/api/v1/leaves/{id}/reject` | Manager rejection |
| PUT | `/api/v1/leaves/{id}/cancel` | Cancel leave |
| POST | `/api/v1/knowledge/documents` | Ingest policy document |
| GET | `/api/v1/knowledge/documents` | List knowledge documents |
| DELETE | `/api/v1/knowledge/documents/{id}` | Delete knowledge document |
| POST | `/api/v1/assistant/ask` | Backend assistant endpoint |
| POST | `/api/v1/oauth/clients` | Register automation client |
| POST | `/api/v1/oauth/token` | OAuth2 client-credentials token |

Manager-only endpoints are protected by backend authorization.

---

# 14. OAuth2 for Automation

The backend supports OAuth2 Client Credentials for trusted automation clients such as n8n.

The flow is:

```text
Manager
  ↓
Register automation client
  ↓
client_id + client_secret
  ↓
OAuth token endpoint
  ↓
Service access token
  ↓
Automation endpoint
```

Service tokens are separate from employee JWT access/refresh tokens.

---

# 15. Environment Variables

The current tool-calling Streamlit application requires the Groq key from `.env`.

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

Do not commit `.env` or real API keys to source control.

---

# 16. Setup

## Prerequisites

- Python 3.11+
- PostgreSQL
- Groq API key for LLM tool calling and current RAG generation

## Create virtual environment

Windows:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

macOS/Linux:

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

## Configure `.env`

Copy `.env.example` to `.env` and add the required credentials.

## Run migrations

```bash
alembic upgrade head
```

## Start FastAPI

```bash
uvicorn app.main:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

Interactive API documentation:

```text
http://127.0.0.1:8000/docs
```

## Start Streamlit

In another terminal:

```bash
streamlit run streamlit_app.py
```

---

# 17. Seed Policy Knowledge

The backend sample policies can be loaded with:

```bash
python scripts/seed_knowledge_base.py
```

The newer `rag_lab` evaluation pipeline uses its policy corpus separately for retrieval/reranking experiments.

---

# 18. Testing

Run the complete Python test suite with:

```bash
pytest -q
```

The project has also been tested specifically for the RAG reranking pipeline.

The reranking test file currently verifies the expected retrieval/reranking behavior, and the broader test suite covers backend functionality.

Useful manual API checks include:

```text
GET  /openapi.json
GET  /api/v1/leaves/pending/
PUT  /api/v1/leaves/{id}/approve
PUT  /api/v1/leaves/{id}/reject
```

---

# 19. Example Conversational Flows

## Employee — leave balance

```text
User: How many annual leaves do I have?
```

The LLM selects:

```text
get_leave_balance
```

The application retrieves the authenticated employee's live balance through FastAPI.

## Employee — leave application

```text
User: I need Monday off
Assistant: What type of leave would you like to use?

User: Casual
```

The system resolves the date deterministically and submits the leave request through the API.

## Manager — pending requests

```text
User: Show pending leave requests
```

The LLM selects:

```text
get_pending_leave_requests
```

## Manager — approve by name

```text
User: approve Avinash leave
```

If several requests match, the assistant asks for a distinguishing detail.

For identical requests:

```text
1. Annual leave — 2026-09-07
2. Annual leave — 2026-09-07
3. Annual leave — 2026-09-07
```

Manager:

```text
2
```

The corresponding internal leave ID is then approved through FastAPI.

---

# 20. Key Engineering Decisions

### LLM for intent, deterministic code for execution

Natural-language understanding benefits from an LLM, but state-changing operations remain deterministic.

### Backend as source of truth

The backend decides whether a leave can be created, approved, rejected, or cancelled.

### RAG for policy, API for live data

Policy rules are retrieved from the knowledge base.

Employee balances and leave records come from the authenticated database.

### Reranking after retrieval

Semantic retrieval finds a candidate set, while a cross-encoder reranker improves the ordering of the final context.

### Deterministic date normalization

Dates are normalized from the original user message so that the LLM cannot silently invent a different transaction date.

### Multi-turn state for ambiguity

The system stores unresolved manager actions in session state rather than routing every follow-up message as a brand-new request.

### Hidden internal IDs

Managers work with employee names, leave types, dates, and numbered choices. Database IDs remain internal implementation details.

---

# 21. Known Data Cleanup Consideration

During testing, duplicate pending leave rows were created for some employees.

The application now prevents new identical pending requests, but existing duplicate rows may still need to be reviewed and cleaned from the database.

The conversational manager workflow handles these existing duplicates safely by requiring an explicit numbered selection.

---

# 22. Overall System

The project has evolved from a traditional layered FastAPI leave-management backend into a conversational leave-management system:

```text
Traditional API
      ↓
Policy RAG
      ↓
Semantic Retrieval
      ↓
Reranking
      ↓
Grounded Generation
      ↓
LLM Tool Calling
      ↓
Natural-Language Actions
      ↓
Deterministic Execution
      ↓
FastAPI Validation
      ↓
PostgreSQL
```

The result is a system where users can interact naturally while policy answers remain grounded and transactional operations remain controlled by deterministic backend business rules.
