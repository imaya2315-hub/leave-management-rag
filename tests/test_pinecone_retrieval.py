"""
Unit tests for Pinecone vector database integration, hybrid retrieval,
metadata filtering, and evaluation metrics.

All tests run completely offline using in-memory mock structures without
requiring a real Pinecone API key or network access.
"""
from __future__ import annotations

import os
from unittest.mock import patch

import numpy as np
import pytest

from rag_lab.chunking import chunk_documents, derive_metadata
from rag_lab.pinecone_store import (
    MockPineconeIndex,
    PineconeConfig,
    PineconeRetriever,
    PineconeStore,
)
from rag_lab.sparse import BM25SparseEncoder, hybrid_scale


# ==============================================================================
# 1. Pinecone configuration loading
# ==============================================================================
def test_pinecone_config_loading_from_env():
    env_vars = {
        "PINECONE_API_KEY": "test-key-123",
        "PINECONE_INDEX_NAME": "custom-leave-index",
        "PINECONE_CLOUD": "gcp",
        "PINECONE_REGION": "us-central1",
        "PINECONE_METRIC": "dotproduct",
    }
    with patch.dict(os.environ, env_vars, clear=False):
        cfg = PineconeConfig()
        assert cfg.api_key == "test-key-123"
        assert cfg.index_name == "custom-leave-index"
        assert cfg.cloud == "gcp"
        assert cfg.region == "us-central1"
        assert cfg.metric == "dotproduct"
        assert cfg.is_configured is True


def test_pinecone_config_default_fallback():
    with patch.dict(os.environ, {"PINECONE_API_KEY": ""}, clear=False):
        cfg = PineconeConfig()
        assert cfg.api_key == ""
        assert cfg.index_name == "leave-policy-index"
        assert cfg.cloud == "aws"
        assert cfg.region == "us-east-1"
        assert cfg.metric == "dotproduct"
        assert cfg.dimension == 384
        assert cfg.is_configured is False


# ==============================================================================
# 2. Chunk metadata creation
# ==============================================================================
def test_chunk_metadata_creation_and_schema():
    sample_doc = {
        "title": "Leave Types And Eligibility",
        "filename": "leave_types_and_eligibility.txt",
        "source": "rag_lab/documents/leave_types_and_eligibility.txt",
        "content": (
            "Annual Leave: Every employee is entitled to 20 Annual Leave days per calendar year. "
            "Annual Leave should be planned in advance with 5 days notice."
        ),
    }

    chunks = chunk_documents([sample_doc], chunk_size=40, overlap=10)
    assert len(chunks) >= 1

    first_chunk = chunks[0]
    required_keys = [
        "source",
        "filename",
        "policy_category",
        "leave_type",
        "document_version",
        "chunk_index",
        "text",
        "title",
        "content",
    ]
    for key in required_keys:
        assert key in first_chunk, f"Missing required metadata key: {key}"

    assert first_chunk["policy_category"] == "eligibility"
    assert first_chunk["leave_type"] == "annual"
    assert first_chunk["filename"] == "leave_types_and_eligibility.txt"
    assert first_chunk["text"] == first_chunk["content"]


def test_metadata_derivation_categories():
    docs_to_test = [
        ({"title": "Carry Forward Rules", "filename": "carry_forward_rules.txt"}, "carry_forward"),
        ({"title": "Holiday Calendar", "filename": "holiday_calendar.txt"}, "holiday"),
        ({"title": "Approval Escalation", "filename": "approval_escalation.txt"}, "approval"),
        ({"title": "Leave Types and Eligibility", "filename": "leave_types_and_eligibility.txt"}, "eligibility"),
    ]
    for doc, expected_cat in docs_to_test:
        meta = derive_metadata(doc, "Policy details content", 0)
        assert meta["policy_category"] == expected_cat


# ==============================================================================
# 3. Pinecone upsert payload structure
# ==============================================================================
def test_pinecone_upsert_payload_structure():
    store = PineconeStore(force_mock=True)
    chunks = [
        {
            "title": "Carry Forward Rules",
            "filename": "carry_forward_rules.txt",
            "source": "rag_lab/documents/carry_forward_rules.txt",
            "content": "Up to 5 unused Annual Leave days may be carried forward.",
            "chunk_index": 0,
            "policy_category": "carry_forward",
            "leave_type": "annual",
            "document_version": "v1.0",
            "text": "Up to 5 unused Annual Leave days may be carried forward.",
        }
    ]
    dense_embeddings = [np.random.rand(384).astype(np.float32)]
    sparse_enc = BM25SparseEncoder().fit([c["content"] for c in chunks])

    upsert_count = store.upsert_chunks(chunks, dense_embeddings, sparse_enc)
    assert upsert_count == 1

    # Inspect stored payload in index
    vector_entry = store._index.vectors["carry_forward_rules.txt_0"]
    assert "id" in vector_entry
    assert "values" in vector_entry
    assert len(vector_entry["values"]) == 384
    assert "sparse_values" in vector_entry
    assert "indices" in vector_entry["sparse_values"]
    assert "values" in vector_entry["sparse_values"]
    assert "metadata" in vector_entry

    meta = vector_entry["metadata"]
    assert meta["policy_category"] == "carry_forward"
    assert meta["leave_type"] == "annual"
    assert meta["document_version"] == "v1.0"
    assert meta["source"] == "rag_lab/documents/carry_forward_rules.txt"
    assert meta["filename"] == "carry_forward_rules.txt"
    assert meta["chunk_index"] == 0
    assert meta["text"] == chunks[0]["content"]


# ==============================================================================
# 4. Metadata filters ($eq, $in, $and)
# ==============================================================================
def test_metadata_filters():
    mock_idx = MockPineconeIndex("test-idx")
    mock_idx.upsert([
        {
            "id": "v1",
            "values": [1.0, 0.0],
            "metadata": {"policy_category": "eligibility", "leave_type": "annual"},
        },
        {
            "id": "v2",
            "values": [0.0, 1.0],
            "metadata": {"policy_category": "carry_forward", "leave_type": "annual"},
        },
        {
            "id": "v3",
            "values": [1.0, 1.0],
            "metadata": {"policy_category": "carry_forward", "leave_type": "sick"},
        },
    ])

    # 1. Exact equality filter
    res_eq = mock_idx.query(vector=[1.0, 1.0], filter={"policy_category": "carry_forward"})
    assert len(res_eq["matches"]) == 2
    assert {m["id"] for m in res_eq["matches"]} == {"v2", "v3"}

    # 2. Operator $eq filter
    res_op = mock_idx.query(vector=[1.0, 1.0], filter={"leave_type": {"$eq": "sick"}})
    assert len(res_op["matches"]) == 1
    assert res_op["matches"][0]["id"] == "v3"

    # 3. $in filter
    res_in = mock_idx.query(vector=[1.0, 1.0], filter={"policy_category": {"$in": ["eligibility", "holiday"]}})
    assert len(res_in["matches"]) == 1
    assert res_in["matches"][0]["id"] == "v1"

    # 4. $and filter
    res_and = mock_idx.query(
        vector=[1.0, 1.0],
        filter={"$and": [{"policy_category": "carry_forward"}, {"leave_type": "annual"}]},
    )
    assert len(res_and["matches"]) == 1
    assert res_and["matches"][0]["id"] == "v2"


# ==============================================================================
# 5. Dense retrieval
# ==============================================================================
def test_dense_retrieval():
    mock_idx = MockPineconeIndex("test-idx", dimension=3)
    # Unit vectors
    mock_idx.upsert([
        {"id": "c1", "values": [1.0, 0.0, 0.0], "metadata": {"title": "Doc 1", "text": "Annual leave"}},
        {"id": "c2", "values": [0.0, 1.0, 0.0], "metadata": {"title": "Doc 2", "text": "Sick leave"}},
        {"id": "c3", "values": [0.0, 0.0, 1.0], "metadata": {"title": "Doc 3", "text": "Casual leave"}},
    ])

    # Query closest to c2
    res = mock_idx.query(vector=[0.1, 0.9, 0.0], top_k=2)
    assert len(res["matches"]) == 2
    assert res["matches"][0]["id"] == "c2"
    assert pytest.approx(res["matches"][0]["score"], 0.01) == 0.9


# ==============================================================================
# 6. Hybrid retrieval and alpha scaling
# ==============================================================================
def test_hybrid_scaling():
    dense = [1.0, 2.0, 3.0]
    sparse = {"indices": [10, 20], "values": [4.0, 8.0]}

    # alpha = 1.0 (dense only)
    sd_1, ss_1 = hybrid_scale(dense, sparse, alpha=1.0)
    assert sd_1 == [1.0, 2.0, 3.0]
    assert ss_1["values"] == [0.0, 0.0]

    # alpha = 0.0 (sparse only)
    sd_0, ss_0 = hybrid_scale(dense, sparse, alpha=0.0)
    assert sd_0 == [0.0, 0.0, 0.0]
    assert ss_0["values"] == [4.0, 8.0]

    # alpha = 0.5 (balanced)
    sd_half, ss_half = hybrid_scale(dense, sparse, alpha=0.5)
    assert sd_half == [0.5, 1.0, 1.5]
    assert ss_half["values"] == [2.0, 4.0]

    # Invalid alpha raises ValueError
    with pytest.raises(ValueError):
        hybrid_scale(dense, sparse, alpha=1.5)
    with pytest.raises(ValueError):
        hybrid_scale(dense, sparse, alpha=-0.1)


def test_hybrid_search_scoring():
    mock_idx = MockPineconeIndex("test-idx", dimension=2)
    mock_idx.upsert([
        {
            "id": "semantically_close",
            "values": [1.0, 0.0],
            "sparse_values": {"indices": [1], "values": [0.1]},
            "metadata": {"title": "Semantic"},
        },
        {
            "id": "keyword_exact",
            "values": [0.0, 1.0],
            "sparse_values": {"indices": [1], "values": [5.0]},
            "metadata": {"title": "Keyword"},
        },
    ])

    # When query has dense [1.0, 0.0] and sparse {indices: [1], values: [1.0]}
    q_dense = [1.0, 0.0]
    q_sparse = {"indices": [1], "values": [1.0]}

    # Alpha = 1.0 (Dense only): semantically_close should win
    sd, ss = hybrid_scale(q_dense, q_sparse, alpha=1.0)
    res_dense = mock_idx.query(vector=sd, sparse_vector=ss, top_k=1)
    assert res_dense["matches"][0]["id"] == "semantically_close"

    # Alpha = 0.0 (Sparse only): keyword_exact should win
    sd, ss = hybrid_scale(q_dense, q_sparse, alpha=0.0)
    res_sparse = mock_idx.query(vector=sd, sparse_vector=ss, top_k=1)
    assert res_sparse["matches"][0]["id"] == "keyword_exact"


# ==============================================================================
# 7. FAISS vs Pinecone result normalization
# ==============================================================================
def test_result_normalization():
    store = PineconeStore(force_mock=True)
    chunks = [
        {
            "title": "Leave Types And Eligibility",
            "filename": "leave_types_and_eligibility.txt",
            "content": "Annual leave entitlement is 20 days per year.",
            "chunk_index": 0,
            "policy_category": "eligibility",
            "leave_type": "annual",
            "document_version": "v1.0",
            "source": "rag_lab/documents/leave_types_and_eligibility.txt",
            "text": "Annual leave entitlement is 20 days per year.",
        }
    ]
    dense_embeddings = [np.ones(384, dtype=np.float32) * 0.1]
    store.upsert_chunks(chunks, dense_embeddings)

    retriever = PineconeRetriever(store)
    results = retriever.retrieve("annual leave", top_k=1)

    assert isinstance(results, list)
    assert len(results) == 1
    chunk_dict, score = results[0]

    # Must match the (chunk_dict, float) shape expected by cross-encoder rerank()
    assert isinstance(chunk_dict, dict)
    assert isinstance(score, float)
    assert "title" in chunk_dict
    assert "content" in chunk_dict
    assert "chunk_index" in chunk_dict
    assert "policy_category" in chunk_dict
    assert "leave_type" in chunk_dict
    assert chunk_dict["title"] == "Leave Types And Eligibility"


# ==============================================================================
# 8. Evaluation metrics calculation
# ==============================================================================
def test_metric_calculations():
    # Helper functions commonly used in evaluation
    def calc_top1(ranked_titles: list[str], gold: str) -> int:
        return 1 if ranked_titles and ranked_titles[0] == gold else 0

    def calc_mrr(ranked_titles: list[str], gold: str) -> float:
        for r, title in enumerate(ranked_titles, start=1):
            if title == gold:
                return 1.0 / r
        return 0.0

    def calc_rel_top3(ranked_titles: list[str], gold: str) -> int:
        return 1 if gold in ranked_titles[:3] else 0

    # Test case 1: rank 1 hit
    r1 = ["Carry Forward Rules", "Holiday Calendar", "Leave Types And Eligibility"]
    assert calc_top1(r1, "Carry Forward Rules") == 1
    assert calc_mrr(r1, "Carry Forward Rules") == 1.0
    assert calc_rel_top3(r1, "Carry Forward Rules") == 1

    # Test case 2: rank 2 hit
    r2 = ["Holiday Calendar", "Carry Forward Rules", "Approval Escalation"]
    assert calc_top1(r2, "Carry Forward Rules") == 0
    assert pytest.approx(calc_mrr(r2, "Carry Forward Rules")) == 0.5
    assert calc_rel_top3(r2, "Carry Forward Rules") == 1

    # Test case 3: rank 3 hit
    r3 = ["Holiday Calendar", "Approval Escalation", "Carry Forward Rules"]
    assert calc_top1(r3, "Carry Forward Rules") == 0
    assert pytest.approx(calc_mrr(r3, "Carry Forward Rules"), 0.01) == 1.0 / 3
    assert calc_rel_top3(r3, "Carry Forward Rules") == 1

    # Test case 4: rank 4 (not in top 3)
    r4 = ["Holiday Calendar", "Approval Escalation", "Other", "Carry Forward Rules"]
    assert calc_top1(r4, "Carry Forward Rules") == 0
    assert calc_mrr(r4, "Carry Forward Rules") == 0.25
    assert calc_rel_top3(r4, "Carry Forward Rules") == 0
