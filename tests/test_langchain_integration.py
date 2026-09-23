"""
Unit tests for LangChain Integration Layer.

Verifies:
1. LangChain integration imports successfully.
2. Existing custom retriever can be wrapped via LangChainRetrieverAdapter.
3. Metadata is preserved when converting chunks to Documents.
4. Policy RAG returns grounded context.
5. LangChain policy tool calls the existing RAG.
6. get_leave_balance calls the existing backend with authorization.
7. Authorization is still enforced by FastAPI.
8. apply_leave does not bypass backend validation.
9. LangChain mode and custom mode return equivalent retrieval results for the same query.
10. Pinecone integration still works through the LangChain adapter.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from rag_lab.chunking import chunk_documents
from rag_lab.ingestion import load_documents
from rag_lab.langchain_integration import (
    LangChainRetrieverAdapter,
    NO_CONTEXT_ANSWER,
    answer_policy_question_custom,
    answer_policy_question_langchain,
    choose_tool_langchain,
    create_leave_tools,
    create_policy_rag_chain,
    create_policy_retriever,
    get_langchain_chat_model,
)
from rag_lab.pinecone_store import PineconeRetriever, PineconeStore
from rag_lab.reranking import rerank
from rag_lab.retrieval import SemanticRetriever
from rag_lab.sparse import BM25SparseEncoder


# ---------------------------------------------------------------------------
# 1. Imports
# ---------------------------------------------------------------------------
def test_langchain_imports_successfully():
    import langchain
    import langchain_community
    import langchain_core
    import langchain_groq
    import rag_lab.langchain_integration as li

    assert li.LangChainRetrieverAdapter is not None
    assert li.create_policy_retriever is not None
    assert li.create_leave_tools is not None


# ---------------------------------------------------------------------------
# 2. Existing Retriever Wrapped
# ---------------------------------------------------------------------------
def test_existing_custom_retriever_can_be_wrapped():
    mock_retriever = MagicMock()
    sample_chunk = {
        "title": "Sample Policy",
        "content": "Employees get 20 days annual leave.",
        "chunk_index": 0,
    }
    mock_retriever.retrieve.return_value = [(sample_chunk, 0.95)]

    adapter = LangChainRetrieverAdapter(
        underlying_retriever=mock_retriever,
        use_reranker=False,
    )

    assert isinstance(adapter, BaseRetriever)
    docs = adapter.invoke("annual leave")

    assert len(docs) == 1
    assert isinstance(docs[0], Document)
    assert docs[0].page_content == "Employees get 20 days annual leave."
    mock_retriever.retrieve.assert_called_once()


# ---------------------------------------------------------------------------
# 3. Metadata Preservation
# ---------------------------------------------------------------------------
def test_metadata_is_preserved_when_converting_chunks_to_documents():
    mock_retriever = MagicMock()
    sample_chunk = {
        "title": "Holiday Calendar",
        "content": "August 15 is a company holiday.",
        "chunk_index": 3,
        "policy_category": "holiday",
        "leave_type": "general",
        "document_version": "v1.2",
        "source": "rag_lab/documents/holiday_calendar.md",
        "filename": "holiday_calendar.md",
    }
    mock_retriever.retrieve.return_value = [(sample_chunk, 0.88)]

    adapter = LangChainRetrieverAdapter(
        underlying_retriever=mock_retriever,
        use_reranker=False,
    )
    docs = adapter.invoke("August 15")

    assert len(docs) == 1
    meta = docs[0].metadata

    assert meta["title"] == "Holiday Calendar"
    assert meta["chunk_index"] == 3
    assert meta["policy_category"] == "holiday"
    assert meta["leave_type"] == "general"
    assert meta["document_version"] == "v1.2"
    assert meta["source"] == "rag_lab/documents/holiday_calendar.md"
    assert meta["filename"] == "holiday_calendar.md"
    assert meta["score"] == pytest.approx(0.88)


# ---------------------------------------------------------------------------
# 4. Grounded Policy RAG
# ---------------------------------------------------------------------------
def test_policy_rag_returns_grounded_context_and_refuses_hallucination():
    mock_adapter = MagicMock(spec=LangChainRetrieverAdapter)
    mock_adapter.invoke.return_value = [
        Document(
            page_content="Every employee is entitled to 5 Casual Leave days per calendar year.",
            metadata={"title": "Leave Types And Eligibility", "chunk_index": 0, "score": 1.5},
        )
    ]

    answer, sources = answer_policy_question_langchain(
        "How many casual leave days do I get?",
        retriever=mock_adapter,
    )

    assert "5" in answer or "casual" in answer.lower()
    assert len(sources) >= 1
    assert sources[0]["title"] == "Leave Types And Eligibility"

    # Test refusal when context is empty
    mock_adapter.invoke.return_value = []
    refusal_answer, refusal_sources = answer_policy_question_langchain(
        "What is the capital of France?",
        retriever=mock_adapter,
    )
    assert NO_CONTEXT_ANSWER in refusal_answer
    assert len(refusal_sources) == 0


# ---------------------------------------------------------------------------
# 5. LangChain Policy Tool Calls RAG
# ---------------------------------------------------------------------------
def test_langchain_policy_tool_calls_existing_rag():
    tools = create_leave_tools(token="mock_token")
    policy_tool = next(t for t in tools if t.name == "answer_policy_question")

    with patch(
        "rag_lab.langchain_integration.answer_policy_question_langchain",
        return_value=("You get 20 days annual leave.", [{"title": "Leave Policy"}]),
    ) as mock_rag:
        res = policy_tool.invoke({"question": "Annual leave policy?"})
        mock_rag.assert_called_once_with("Annual leave policy?")
        assert "20 days annual leave" in res
        assert "*Sources: Leave Policy*" in res


# ---------------------------------------------------------------------------
# 6. get_leave_balance Calls Backend
# ---------------------------------------------------------------------------
def test_get_leave_balance_calls_existing_backend():
    tools = create_leave_tools(token="valid_jwt_token", api_base="http://testserver/api/v1")
    balance_tool = next(t for t in tools if t.name == "get_leave_balance")

    mock_profile = {
        "annual_leave_balance": 18,
        "sick_leave_balance": 9,
        "casual_leave_balance": 4,
    }

    with patch("rag_lab.langchain_integration._call_fastapi", return_value=mock_profile) as mock_api:
        res = balance_tool.invoke({})
        mock_api.assert_called_once_with(
            "GET", "/employees/me/", "valid_jwt_token", api_base="http://testserver/api/v1"
        )
        assert "Annual: 18 days" in res
        assert "Sick: 9 days" in res
        assert "Casual: 4 days" in res

    # Unauthenticated attempt
    tools_no_auth = create_leave_tools(token=None)
    balance_tool_no_auth = next(t for t in tools_no_auth if t.name == "get_leave_balance")
    res_unauth = balance_tool_no_auth.invoke({})
    assert "Please log in first" in res_unauth


# ---------------------------------------------------------------------------
# 7. Authorization Still Enforced by FastAPI
# ---------------------------------------------------------------------------
def test_authorization_is_still_enforced_by_fastapi():
    import requests

    tools = create_leave_tools(token="employee_token")
    approve_tool = next(t for t in tools if t.name == "approve_leave")

    # Simulate FastAPI returning 403 Forbidden
    mock_resp = MagicMock()
    mock_resp.status_code = 403
    http_error = requests.HTTPError(response=mock_resp)

    with patch("rag_lab.langchain_integration._call_fastapi", side_effect=http_error):
        res = approve_tool.invoke({"leave_id": 42})
        assert "Manager or admin authorization is required" in res


# ---------------------------------------------------------------------------
# 8. apply_leave Does Not Bypass Backend Validation
# ---------------------------------------------------------------------------
def test_apply_leave_does_not_bypass_backend_validation():
    import requests

    tools = create_leave_tools(token="valid_token")
    apply_tool = next(t for t in tools if t.name == "apply_leave")

    # Backend rejects because of probation or insufficient balance
    mock_resp = MagicMock()
    mock_resp.status_code = 400
    mock_resp.text = '{"detail": "Insufficient leave balance"}'
    http_error = requests.HTTPError(response=mock_resp)

    with patch("rag_lab.langchain_integration._call_fastapi", side_effect=http_error):
        res = apply_tool.invoke({
            "leave_type": "Annual",
            "start_date": "2026-10-01",
            "end_date": "2026-10-10",
        })
        assert "Leave application rejected by backend" in res
        assert "Insufficient leave balance" in res


# ---------------------------------------------------------------------------
# 9. Retrieval Parity Between LangChain Mode and Custom Mode
# ---------------------------------------------------------------------------
def test_langchain_mode_and_custom_mode_return_equivalent_retrieval_results():
    docs = load_documents("rag_lab/documents")
    chunks = chunk_documents(docs, chunk_size=80, overlap=20)
    custom_retriever = SemanticRetriever(chunks)

    langchain_adapter = LangChainRetrieverAdapter(
        underlying_retriever=custom_retriever,
        top_k=5,
        rerank_top_n=3,
        use_reranker=True,
    )

    query = "How many casual leave days do I get?"

    # 1. Custom retrieval + rerank
    custom_retrieved = custom_retriever.retrieve(query, top_k=5)
    custom_reranked = rerank(query, custom_retrieved, top_n=3)
    custom_titles = [c["title"] for c, _ in custom_reranked]

    # 2. LangChain retrieval adapter
    lc_docs = langchain_adapter.invoke(query)
    lc_titles = [d.metadata["title"] for d in lc_docs]

    assert custom_titles == lc_titles
    assert len(lc_docs) == len(custom_reranked)


# ---------------------------------------------------------------------------
# 10. Pinecone Integration via Adapter
# ---------------------------------------------------------------------------
def test_pinecone_adapter_integration_works():
    docs = load_documents("rag_lab/documents")
    chunks = chunk_documents(docs, chunk_size=80, overlap=20)

    # Use MockPineconeIndex under the hood for fast reliable unit test
    store = PineconeStore(force_mock=True)
    sparse_enc = BM25SparseEncoder().fit([c["content"] for c in chunks])
    from rag_lab.embeddings import embed_texts

    embeddings = embed_texts([c["content"] for c in chunks])
    store.upsert_chunks(chunks, embeddings, sparse_enc)

    pinecone_retriever = PineconeRetriever(store, sparse_encoder=sparse_enc, default_alpha=0.75)

    adapter = LangChainRetrieverAdapter(
        underlying_retriever=pinecone_retriever,
        top_k=5,
        rerank_top_n=3,
        alpha=0.75,
        filter_dict={"policy_category": "eligibility"},
    )

    docs = adapter.invoke("sick leave requirements")
    assert len(docs) > 0
    for doc in docs:
        assert doc.metadata["policy_category"] == "eligibility"
        assert "score" in doc.metadata
