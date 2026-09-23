"""
LangChain Orchestration and Integration Layer for Leave Management System with RAG.

Architecture:
- Acts strictly as an orchestration/integration layer.
- Does NOT replace the custom retrieval pipeline (FAISS, Pinecone, BM25, reranker).
- Wraps existing retrievers via LangChain's BaseRetriever interface.
- LangChain Document objects are created ONLY at the integration boundary,
  preserving all chunk metadata (title, category, leave_type, source, etc.).
- LLM generation is configured to use Qwen 3.8-27B (temperature=0.2, top_p=0.8, max_tokens=120)
  via ChatGroq, adhering to strict grounding prompts.
- Leave operation tools call FastAPI endpoints with JWT tokens;
  FastAPI remains authoritative for auth, permissions, leave balances, and PostgreSQL mutations.
"""
from __future__ import annotations

import json
import logging
import os
import re
from datetime import date
from typing import Any, List, Optional, Tuple

import requests
from dotenv import load_dotenv

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool

from rag_lab.agent import (
    DOCUMENTS_FOLDER,
    NO_CONTEXT_ANSWER,
    RELEVANCE_THRESHOLD_COSINE,
    RELEVANCE_THRESHOLD_CROSS_ENCODER,
    RERANK_TOP_N,
    RETRIEVE_PER_SUBQUESTION,
    _get_sub_questions,
    _is_relevant,
    answer_policy_question as answer_policy_question_custom,
    get_retriever as get_custom_retriever,
)
from rag_lab.generation import _fallback_answer
from rag_lab.reranking import rerank, using_cross_encoder

load_dotenv()

logger = logging.getLogger("langchain_integration")

# ---------------------------------------------------------------------------
# Qwen Generation Configuration (Step 8)
# ---------------------------------------------------------------------------
QWEN_MODEL_NAME = "qwen/qwen3.8-27b"
QWEN_TEMPERATURE = 0.2
QWEN_TOP_P = 0.8
QWEN_MAX_TOKENS = 120

SYSTEM_PROMPT = (
    "You are a leave-policy assistant. Answer ONLY using the provided "
    "context, in 1-3 short sentences — never repeat or quote the context "
    "verbatim, summarize it into a direct answer instead. If the context "
    "doesn't contain the answer, reply with exactly this sentence and "
    "nothing else: \"" + NO_CONTEXT_ANSWER + "\" "
    "Never guess or invent a policy detail that isn't in the context."
)

DEFAULT_API_BASE = os.getenv("API_BASE", "http://localhost:8000/api/v1")


# ---------------------------------------------------------------------------
# Step 4 & 6: Retriever Adapter (BaseRetriever wrapper preserving metadata)
# ---------------------------------------------------------------------------
class LangChainRetrieverAdapter(BaseRetriever):
    """
    Adapter wrapping custom retrievers (SemanticRetriever or PineconeRetriever)
    to conform to LangChain's BaseRetriever interface.

    Preserves:
    - Custom FAISS semantic retrieval
    - Custom Pinecone dense, sparse, and hybrid retrieval with alpha weighting
    - Metadata filtering
    - Cross-encoder reranking (ms-marco-MiniLM-L-6-v2)
    - Full chunk metadata in Document objects at the integration boundary
    """

    underlying_retriever: Any
    top_k: int = RETRIEVE_PER_SUBQUESTION
    rerank_top_n: int = RERANK_TOP_N
    use_reranker: bool = True
    alpha: Optional[float] = None
    filter_dict: Optional[dict[str, Any]] = None
    min_score_threshold: Optional[float] = None
    filter_relevant: bool = False

    model_config = {"arbitrary_types_allowed": True}

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None,
    ) -> List[Document]:
        """
        Executes underlying retrieval, applies reranking if configured,
        and returns LangChain Document objects.
        """
        retriever_kwargs: dict[str, Any] = {}
        if hasattr(self.underlying_retriever, "retrieve"):
            import inspect

            sig = inspect.signature(self.underlying_retriever.retrieve)
            if "alpha" in sig.parameters and self.alpha is not None:
                retriever_kwargs["alpha"] = self.alpha
            if "filter_dict" in sig.parameters and self.filter_dict is not None:
                retriever_kwargs["filter_dict"] = self.filter_dict

            candidates: list[tuple[dict[str, Any], float]] = (
                self.underlying_retriever.retrieve(
                    query, top_k=self.top_k, **retriever_kwargs
                )
            )
        else:
            candidates = []

        if not candidates:
            return []

        # Cross-encoder reranking
        if self.use_reranker:
            ranked_pairs = rerank(query, candidates, top_n=self.rerank_top_n)
        else:
            ranked_pairs = candidates[: self.rerank_top_n]

        # Convert to LangChain Document objects preserving full metadata
        documents: list[Document] = []
        for chunk, score in ranked_pairs:
            # Relevance check
            if self.min_score_threshold is not None:
                if score < self.min_score_threshold:
                    continue
            if self.filter_relevant and not _is_relevant(score):
                continue

            content = chunk.get("content") or chunk.get("text") or ""
            metadata = {
                "title": chunk.get("title", ""),
                "chunk_index": chunk.get("chunk_index", 0),
                "policy_category": chunk.get("policy_category", "general"),
                "leave_type": chunk.get("leave_type", "general"),
                "document_version": chunk.get("document_version", "v1.0"),
                "source": chunk.get("source", ""),
                "filename": chunk.get("filename", ""),
                "score": float(score),
            }
            documents.append(Document(page_content=content, metadata=metadata))

        return documents


def create_policy_retriever(
    retriever_type: str = "custom",
    top_k: int = RETRIEVE_PER_SUBQUESTION,
    rerank_top_n: int = RERANK_TOP_N,
    use_reranker: bool = True,
    alpha: Optional[float] = None,
    filter_dict: Optional[dict[str, Any]] = None,
    **kwargs: Any,
) -> LangChainRetrieverAdapter:
    """
    Factory creating a LangChain-compatible retriever wrapping the custom pipeline.
    """
    if retriever_type.lower() == "pinecone":
        from rag_lab.pinecone_store import PineconeRetriever, PineconeStore
        from rag_lab.sparse import BM25SparseEncoder

        store = PineconeStore()
        sparse_encoder = BM25SparseEncoder()
        underlying = PineconeRetriever(
            store=store,
            sparse_encoder=sparse_encoder,
            default_alpha=alpha if alpha is not None else 1.0,
        )
    else:
        underlying = get_custom_retriever()

    return LangChainRetrieverAdapter(
        underlying_retriever=underlying,
        top_k=top_k,
        rerank_top_n=rerank_top_n,
        use_reranker=use_reranker,
        alpha=alpha,
        filter_dict=filter_dict,
    )


# ---------------------------------------------------------------------------
# Step 8: Qwen Chat Model Connection
# ---------------------------------------------------------------------------
def get_langchain_chat_model(
    model_name: str = QWEN_MODEL_NAME,
    temperature: float = QWEN_TEMPERATURE,
    max_tokens: int = QWEN_MAX_TOKENS,
    top_p: float = QWEN_TOP_P,
):
    """
    Instantiate LangChain chat model wrapper with exact Qwen configuration.
    Falls back gracefully to an extractive runnable if no API key is available.
    """
    groq_api_key = os.getenv("GROQ_API_KEY")
    if groq_api_key:
        try:
            from langchain_groq import ChatGroq

            return ChatGroq(
                model=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                model_kwargs={"top_p": top_p},
                api_key=groq_api_key,
            )
        except Exception as exc:
            logger.warning("Could not initialize ChatGroq: %s. Using fallback.", exc)

    # Deterministic fallback runnable matching _fallback_answer
    def _run_fallback(messages) -> str:
        # Extract user content from messages
        question = ""
        context = ""
        for m in messages:
            content = getattr(m, "content", str(m))
            if "Question:" in content and "Context:" in content:
                parts = content.split("Context:\n", 1)
                q_part = parts[0].replace("Question:", "").strip()
                c_part = parts[1].strip() if len(parts) > 1 else ""
                question = q_part
                context = c_part
            elif getattr(m, "type", "") == "human":
                question = content

        return _fallback_answer(context, question)

    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    class FallbackRunnable(RunnableLambda):
        def __init__(self):
            super().__init__(func=_run_fallback)

    return FallbackRunnable()


# ---------------------------------------------------------------------------
# Step 7: LangChain Policy RAG Chain
# ---------------------------------------------------------------------------
def create_policy_rag_chain(
    llm: Any = None,
    prompt_template: Optional[ChatPromptTemplate] = None,
):
    """
    Create an LCEL chain for answering policy questions given context and question.
    """
    if llm is None:
        llm = get_langchain_chat_model()

    if prompt_template is None:
        prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT),
                ("human", "Question: {question}\n\nContext:\n{context}"),
            ]
        )

    chain = prompt_template | llm | StrOutputParser()
    return chain


def answer_policy_question_langchain(
    question: str,
    retriever: Optional[LangChainRetrieverAdapter] = None,
    chain: Any = None,
) -> Tuple[str, List[dict[str, Any]]]:
    """
    LangChain RAG pipeline answering policy questions.
    Decomposes multi-turn queries, retrieves via LangChainRetrieverAdapter,
    and generates grounded answers with Qwen.
    """
    question = (question or "").strip()
    if not question:
        return "I couldn't identify a question. Please try again.", []

    if retriever is None:
        retriever = create_policy_retriever()

    if chain is None:
        chain = create_policy_rag_chain()

    # Log orchestration path
    print(f"[LANGCHAIN] Policy query")
    underlying = getattr(retriever, "underlying_retriever", None)
    retriever_name = (
        "Pinecone"
        if underlying and "pinecone" in type(underlying).__name__.lower()
        else "FAISS Semantic"
    )
    print(f"[RETRIEVAL] {retriever_name}")
    print(f"[RERANKER] ms-marco-MiniLM-L-6-v2")
    print(f"[GENERATION] Qwen 3.8-27B")

    sub_questions = _get_sub_questions(question)
    answers: list[str] = []
    sources: list[dict[str, Any]] = []
    seen_sources: set[tuple[str, int]] = set()

    for index, sub_q in enumerate(sub_questions, start=1):
        docs: list[Document] = retriever.invoke(sub_q)

        relevant_docs = [
            doc for doc in docs
            if _is_relevant(doc.metadata.get("score", 0.0))
        ]

        if not relevant_docs:
            answers.append(f"{index}. {NO_CONTEXT_ANSWER}")
            continue

        context_parts = []
        for doc in relevant_docs:
            title = doc.metadata.get("title", "Policy")
            context_parts.append(f"[{title}]\n{doc.page_content}")

            source_key = (title, doc.metadata.get("chunk_index", 0))
            if source_key not in seen_sources:
                seen_sources.add(source_key)
                sources.append(doc.metadata)

        context = "\n\n".join(context_parts)

        # Generate answer using LangChain chain
        try:
            answer = chain.invoke({"question": sub_q, "context": context})
        except Exception as exc:
            logger.warning("LangChain generation failed: %s. Using fallback.", exc)
            answer = _fallback_answer(context, sub_q)

        if not answer or not str(answer).strip():
            answer = NO_CONTEXT_ANSWER

        answers.append(f"{index}. {str(answer).strip()}")

    final_answer = "\n\n".join(answers)
    return final_answer, sources


# ---------------------------------------------------------------------------
# Step 9: LangChain Tools with FastAPI Business Validation
# ---------------------------------------------------------------------------
def _call_fastapi(
    method: str,
    path: str,
    token: Optional[str] = None,
    json_data: Optional[dict[str, Any]] = None,
    api_base: str = DEFAULT_API_BASE,
) -> dict[str, Any]:
    """Helper ensuring all tool calls authenticate via FastAPI HTTP endpoints."""
    headers: dict[str, str] = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    url = f"{api_base}{path}"
    resp = requests.request(
        method=method,
        url=url,
        json=json_data,
        headers=headers,
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json() if resp.content else {}


def create_leave_tools(
    token: Optional[str] = None,
    api_base: str = DEFAULT_API_BASE,
) -> list[Any]:
    """
    Creates and returns the list of LangChain tools bound with session token and API base.
    Every tool enforces FastAPI authorization and validation rules.
    """

    @tool
    def answer_policy_question(question: str) -> str:
        """Answer company leave policy and holiday rules using grounded RAG."""
        ans, sources = answer_policy_question_langchain(question)
        if sources:
            titles = sorted({s.get("title") for s in sources if s.get("title")})
            if titles:
                ans += f"\n\n*Sources: {', '.join(titles)}*"
        return ans

    @tool
    def get_leave_balance() -> str:
        """Get the current authenticated user's annual, sick, and casual leave balances."""
        print("[LANGCHAIN] Leave operation")
        print("[TOOL] get_leave_balance")
        print("[BACKEND] FastAPI")
        print("[DATABASE] PostgreSQL")

        if not token:
            return "Please log in first (sidebar) — this needs your account's live data."
        try:
            profile = _call_fastapi("GET", "/employees/me/", token, api_base=api_base)
            return (
                f"Your current balances — "
                f"Annual: {profile['annual_leave_balance']} days, "
                f"Sick: {profile['sick_leave_balance']} days, "
                f"Casual: {profile['casual_leave_balance']} days."
            )
        except requests.HTTPError as exc:
            return f"Failed to fetch balances from backend: {exc}"

    @tool
    def get_leave_history() -> str:
        """Get the current authenticated user's historical and pending leave requests."""
        print("[LANGCHAIN] Leave operation")
        print("[TOOL] get_leave_history")
        print("[BACKEND] FastAPI")
        print("[DATABASE] PostgreSQL")

        if not token:
            return "Please log in first (sidebar) — this needs your account's live data."
        try:
            history = _call_fastapi("GET", "/leaves/me/", token, api_base=api_base)
            if not history:
                return "You have no leave requests on record yet."
            lines = [
                f"- {l['leave_type']} leave, {l['start_date']} to {l['end_date']} — {l['status']}"
                for l in history
            ]
            return "Your leave history:\n" + "\n".join(lines)
        except requests.HTTPError as exc:
            return f"Failed to fetch leave history: {exc}"

    @tool
    def get_pending_leave_requests() -> str:
        """View pending leave requests awaiting approval (manager/admin only)."""
        print("[LANGCHAIN] Leave operation")
        print("[TOOL] get_pending_leave_requests")
        print("[BACKEND] FastAPI")
        print("[DATABASE] PostgreSQL")

        if not token:
            return "Please log in first (sidebar) — this needs your account's live data."
        try:
            profile = _call_fastapi("GET", "/employees/me/", token, api_base=api_base)
            role = str(profile.get("role", "")).lower()
            if role not in {"manager", "admin"}:
                return "Only managers and admins can view pending leave requests."

            # Fetch pending requests
            pending = _call_fastapi("GET", "/leaves/pending/", token, api_base=api_base)
            if not pending:
                return "There are no pending leave requests."
            lines = ["Pending leave requests:"]
            for l in pending:
                emp_name = (
                    l.get("employee", {}).get("name", "Employee")
                    if isinstance(l.get("employee"), dict)
                    else "Employee"
                )
                lines.append(
                    f"- ID {l['id']}: {emp_name}, {l['leave_type']} leave, {l['start_date']} to {l['end_date']}"
                )
            return "\n".join(lines)
        except requests.HTTPError as exc:
            return f"Backend authorization or request failure: {exc}"

    @tool
    def apply_leave(leave_type: str, start_date: str, end_date: str) -> str:
        """Apply for leave by submitting a request to the FastAPI backend."""
        print("[LANGCHAIN] Leave operation")
        print("[TOOL] apply_leave")
        print("[BACKEND] FastAPI")
        print("[DATABASE] PostgreSQL")

        if not token:
            return "Please log in first (sidebar) — this needs your account's live data."
        try:
            payload = {
                "leave_type": leave_type.capitalize(),
                "start_date": start_date,
                "end_date": end_date,
            }
            res = _call_fastapi("POST", "/leaves/", token, json_data=payload, api_base=api_base)
            return (
                f"Applied for {res.get('leave_type')} leave from "
                f"{res.get('start_date')} to {res.get('end_date')} (Status: {res.get('status')})."
            )
        except requests.HTTPError as exc:
            detail = exc.response.text if exc.response is not None else str(exc)
            return f"Leave application rejected by backend: {detail}"

    @tool
    def cancel_leave(leave_id: int) -> str:
        """Cancel a pending leave request by ID."""
        print("[LANGCHAIN] Leave operation")
        print("[TOOL] cancel_leave")
        print("[BACKEND] FastAPI")
        print("[DATABASE] PostgreSQL")

        if not token:
            return "Please log in first (sidebar) — this needs your account's live data."
        try:
            res = _call_fastapi("PUT", f"/leaves/{leave_id}/cancel", token, api_base=api_base)
            return f"Leave request {leave_id} cancelled successfully."
        except requests.HTTPError as exc:
            return f"Failed to cancel leave request: {exc}"

    @tool
    def approve_leave(
        leave_id: int,
        employee_name: Optional[str] = None,
        start_date: Optional[str] = None,
        leave_type: Optional[str] = None,
    ) -> str:
        """Approve a pending leave request (manager/admin only)."""
        print("[LANGCHAIN] Leave operation")
        print("[TOOL] approve_leave")
        print("[BACKEND] FastAPI")
        print("[DATABASE] PostgreSQL")

        if not token:
            return "Please log in first (sidebar) — this needs your account's live data."
        try:
            res = _call_fastapi("PUT", f"/leaves/{leave_id}/approve", token, api_base=api_base)
            return f"Approved leave request {leave_id} for {res.get('leave_type')} leave."
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 403:
                return "Manager or admin authorization is required to approve leave requests."
            return f"Backend approval failed: {exc}"

    @tool
    def reject_leave(
        leave_id: int,
        employee_name: Optional[str] = None,
        start_date: Optional[str] = None,
        leave_type: Optional[str] = None,
    ) -> str:
        """Reject a pending leave request (manager/admin only)."""
        print("[LANGCHAIN] Leave operation")
        print("[TOOL] reject_leave")
        print("[BACKEND] FastAPI")
        print("[DATABASE] PostgreSQL")

        if not token:
            return "Please log in first (sidebar) — this needs your account's live data."
        try:
            res = _call_fastapi("PUT", f"/leaves/{leave_id}/reject", token, api_base=api_base)
            return f"Rejected leave request {leave_id}."
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 403:
                return "Manager or admin authorization is required to reject leave requests."
            return f"Backend rejection failed: {exc}"

    return [
        answer_policy_question,
        get_leave_balance,
        get_leave_history,
        get_pending_leave_requests,
        apply_leave,
        cancel_leave,
        approve_leave,
        reject_leave,
    ]


# ---------------------------------------------------------------------------
# Step 10: Tool Router & Agent Orchestration
# ---------------------------------------------------------------------------
TOOL_ROUTER_SYSTEM_PROMPT = """
You are the intent-and-tool router for a leave-management assistant.

Choose exactly ONE tool. Do not answer the user yourself.

Policy:
- answer_policy_question = general company policy/rules.
Employee live data:
- get_leave_balance = user's own current balances.
- get_leave_history = user's own history/status.
Manager live data:
- get_pending_leave_requests = a manager or admin asks to list/show/view pending requests they are allowed to review.
Actions:
- apply_leave = user is actually asking the system to request time off.
- cancel_leave = user wants to withdraw/remove/cancel their own request.
- approve_leave / reject_leave = manager or admin wants to act on a pending request.
"""


def choose_tool_langchain(
    message: str,
    tools: Optional[list[Any]] = None,
) -> Tuple[str, dict[str, Any]]:
    """
    Selects one tool using LangChain ChatGroq model bound with tools.
    Falls back cleanly to heuristic extraction if Groq API is offline.
    """
    groq_api_key = os.getenv("GROQ_API_KEY")
    if groq_api_key:
        try:
            from langchain_groq import ChatGroq

            llm = ChatGroq(
                model=QWEN_MODEL_NAME,
                temperature=QWEN_TEMPERATURE,
                max_tokens=QWEN_MAX_TOKENS,
                model_kwargs={"top_p": QWEN_TOP_P},
                api_key=groq_api_key,
            )
            tool_list = tools or create_leave_tools()
            llm_with_tools = llm.bind_tools(tool_list)

            prompt = (
                f"{TOOL_ROUTER_SYSTEM_PROMPT}\n"
                f"Today's date is {date.today().isoformat()}.\n\n"
                f"User: {message}"
            )
            response = llm_with_tools.invoke(prompt)

            tool_calls = getattr(response, "tool_calls", None) or []
            if tool_calls:
                call = tool_calls[0]
                return call["name"], call.get("args", {})
        except Exception as exc:
            logger.warning("LangChain LLM tool routing failed: %s. Using heuristic fallback.", exc)

    # Heuristic fallback matching deterministic router
    lowered = message.lower()
    if any(k in lowered for k in ["balance", "how many days do i have", "my leaves left"]):
        return "get_leave_balance", {}
    if any(k in lowered for k in ["history", "past leaves", "my requests", "previous leave"]):
        return "get_leave_history", {}
    if any(k in lowered for k in ["pending request", "pending leaves", "review leave"]):
        return "get_pending_leave_requests", {}
    if any(k in lowered for k in ["approve"]):
        return "approve_leave", {"employee_name": message}
    if any(k in lowered for k in ["reject"]):
        return "reject_leave", {"employee_name": message}
    if any(k in lowered for k in ["cancel"]):
        return "cancel_leave", {}
    if any(k in lowered for k in ["apply", "take leave", "take off", "request leave", "i need monday off"]):
        return "apply_leave", {}

    return "answer_policy_question", {"question": message}


def is_langchain_enabled() -> bool:
    """Check whether USE_LANGCHAIN is enabled in environment or settings."""
    val = os.getenv("USE_LANGCHAIN", "false").strip().lower()
    return val in {"1", "true", "yes", "on"}
