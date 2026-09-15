"""
Stage 4B: Pinecone Vector Database Integration.

Provides:
- PineconeStore: Production Pinecone client wrapper supporting:
  - Index verification and serverless index creation
  - Upsert with dense vector, sparse vector, and rich metadata
  - Dense, sparse, and hybrid query with alpha weighting
  - Metadata filtering ($eq, $in, $and, etc.)
  - Vector update and deletion
- MockPineconeIndex: High-fidelity in-memory Pinecone simulation
  used automatically when PINECONE_API_KEY is not set or in unit tests.
- PineconeRetriever: Drop-in retriever interface matching SemanticRetriever,
  returning [(chunk_dict, score), ...] directly compatible with rerank()
  and agent.py.

IMPORTANT:
- Multi-tenancy / namespaces are explicitly NOT used (all operations use default namespace).
- Existing FAISS implementation in vector_store.py / retrieval.py remains untouched.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from dotenv import load_dotenv

from rag_lab.embeddings import embed_texts, embedding_dim
from rag_lab.sparse import BM25SparseEncoder, hybrid_scale

load_dotenv()


@dataclass
class PineconeConfig:
    api_key: str = field(default_factory=lambda: os.getenv("PINECONE_API_KEY", ""))
    index_name: str = field(default_factory=lambda: os.getenv("PINECONE_INDEX_NAME", "leave-policy-index"))
    cloud: str = field(default_factory=lambda: os.getenv("PINECONE_CLOUD", "aws"))
    region: str = field(default_factory=lambda: os.getenv("PINECONE_REGION", "us-east-1"))
    metric: str = field(default_factory=lambda: os.getenv("PINECONE_METRIC", "dotproduct"))
    dimension: int = 384

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key and self.api_key.strip())


class MockPineconeIndex:
    """
    In-memory simulation of a Pinecone index supporting dotproduct similarity,
    hybrid sparse+dense scoring, and metadata filtering.
    Used for unit testing and offline development without requiring Pinecone cloud.
    """

    def __init__(self, name: str, dimension: int = 384, metric: str = "dotproduct"):
        self.name = name
        self.dimension = dimension
        self.metric = metric
        self.vectors: dict[str, dict[str, Any]] = {}

    def upsert(self, vectors: list[dict[str, Any]], namespace: str | None = None) -> dict[str, int]:
        if namespace:
            raise ValueError("Namespaces are explicitly disabled in this Leave Management System")
        upserted_count = 0
        for vec in vectors:
            vec_id = str(vec["id"])
            self.vectors[vec_id] = {
                "id": vec_id,
                "values": [float(v) for v in vec.get("values", [])],
                "sparse_values": vec.get("sparse_values") or {"indices": [], "values": []},
                "metadata": dict(vec.get("metadata", {})),
            }
            upserted_count += 1
        return {"upserted_count": upserted_count}

    def delete(self, ids: list[str] | None = None, delete_all: bool = False, namespace: str | None = None) -> dict:
        if namespace:
            raise ValueError("Namespaces are explicitly disabled in this Leave Management System")
        if delete_all:
            self.vectors.clear()
        elif ids:
            for vid in ids:
                self.vectors.pop(vid, None)
        return {}

    def update(
        self,
        id: str,
        values: list[float] | None = None,
        set_metadata: dict[str, Any] | None = None,
        sparse_values: dict[str, list[Any]] | None = None,
        namespace: str | None = None,
    ) -> dict:
        if namespace:
            raise ValueError("Namespaces are explicitly disabled in this Leave Management System")
        if id not in self.vectors:
            raise KeyError(f"Vector {id} not found")
        if values is not None:
            self.vectors[id]["values"] = [float(v) for v in values]
        if sparse_values is not None:
            self.vectors[id]["sparse_values"] = sparse_values
        if set_metadata is not None:
            self.vectors[id]["metadata"].update(set_metadata)
        return {}

    @staticmethod
    def _matches_filter(metadata: dict[str, Any], filter_dict: dict[str, Any]) -> bool:
        if not filter_dict:
            return True

        for k, v in filter_dict.items():
            if k == "$and":
                if not all(MockPineconeIndex._matches_filter(metadata, sub) for sub in v):
                    return False
            elif k == "$or":
                if not any(MockPineconeIndex._matches_filter(metadata, sub) for sub in v):
                    return False
            else:
                meta_val = metadata.get(k)
                if isinstance(v, dict):
                    for op, target in v.items():
                        if op == "$eq" and meta_val != target:
                            return False
                        elif op == "$ne" and meta_val == target:
                            return False
                        elif op == "$in" and meta_val not in target:
                            return False
                        elif op == "$nin" and meta_val in target:
                            return False
                else:
                    if meta_val != v:
                        return False
        return True

    def query(
        self,
        vector: list[float] | None = None,
        sparse_vector: dict[str, list[Any]] | None = None,
        top_k: int = 5,
        filter: dict[str, Any] | None = None,
        include_metadata: bool = True,
        namespace: str | None = None,
    ) -> dict[str, Any]:
        if namespace:
            raise ValueError("Namespaces are explicitly disabled in this Leave Management System")

        candidates = []
        q_dense = np.array(vector, dtype="float32") if vector is not None else None

        q_sparse_map: dict[int, float] = {}
        if sparse_vector:
            for idx, val in zip(sparse_vector.get("indices", []), sparse_vector.get("values", [])):
                q_sparse_map[int(idx)] = float(val)

        for vec_id, data in self.vectors.items():
            meta = data.get("metadata", {})
            if filter and not self._matches_filter(meta, filter):
                continue

            score = 0.0

            # Dense dotproduct
            if q_dense is not None and data.get("values"):
                d_dense = np.array(data["values"], dtype="float32")
                if len(d_dense) == len(q_dense):
                    score += float(np.dot(q_dense, d_dense))

            # Sparse dotproduct
            if q_sparse_map and data.get("sparse_values"):
                s_indices = data["sparse_values"].get("indices", [])
                s_values = data["sparse_values"].get("values", [])
                for s_idx, s_val in zip(s_indices, s_values):
                    if int(s_idx) in q_sparse_map:
                        score += q_sparse_map[int(s_idx)] * float(s_val)

            match_entry = {
                "id": vec_id,
                "score": float(score),
            }
            if include_metadata:
                match_entry["metadata"] = meta
            candidates.append(match_entry)

        # Sort by score descending
        candidates.sort(key=lambda x: x["score"], reverse=True)
        return {
            "matches": candidates[:top_k],
            "namespace": "",
        }


class PineconeStore:
    """
    Manages Pinecone index lifecycle, upserts, queries, and metadata filtering.
    """

    def __init__(self, config: PineconeConfig | None = None, force_mock: bool = False):
        self.config = config or PineconeConfig()
        self.force_mock = force_mock
        self._pc = None
        self._index = None

        if self.force_mock or not self.config.is_configured:
            # Use mock in-memory index
            self._index = MockPineconeIndex(
                name=self.config.index_name,
                dimension=self.config.dimension,
                metric=self.config.metric,
            )
            self.is_live = False
        else:
            self._init_live_pinecone()

    def _init_live_pinecone(self) -> None:
        try:
            from pinecone import Pinecone, ServerlessSpec

            self._pc = Pinecone(api_key=self.config.api_key)
            existing_indexes = [idx.name for idx in self._pc.list_indexes()]

            if self.config.index_name not in existing_indexes:
                self._pc.create_index(
                    name=self.config.index_name,
                    dimension=self.config.dimension,
                    metric=self.config.metric,
                    spec=ServerlessSpec(
                        cloud=self.config.cloud,
                        region=self.config.region,
                    ),
                )
                # Wait briefly for index initialization
                time.sleep(2)

            self._index = self._pc.Index(self.config.index_name)
            self.is_live = True
        except Exception as exc:
            # Fallback to mock index with a clear diagnostic flag
            self._index = MockPineconeIndex(
                name=self.config.index_name,
                dimension=self.config.dimension,
                metric=self.config.metric,
            )
            self.is_live = False
            self.last_error = str(exc)

    def upsert_chunks(
        self,
        chunks: list[dict[str, Any]],
        dense_embeddings: list[list[float]] | np.ndarray,
        sparse_encoder: BM25SparseEncoder | None = None,
    ) -> int:
        """
        Upserts chunk vectors with full metadata schema into Pinecone.
        Required metadata schema per chunk:
          - source
          - filename
          - policy_category
          - leave_type
          - document_version
          - chunk_index
          - text
          - title
        """
        if len(chunks) != len(dense_embeddings):
            raise ValueError("Mismatch between number of chunks and dense embeddings")

        vectors = []
        for idx, (chunk, dense_emb) in enumerate(zip(chunks, dense_embeddings)):
            dense_list = dense_emb.tolist() if isinstance(dense_emb, np.ndarray) else list(dense_emb)

            sparse_vals = {"indices": [], "values": []}
            if sparse_encoder is not None:
                sparse_vals = sparse_encoder.encode_documents(chunk["content"])

            # Ensure all required metadata fields exist
            metadata = {
                "source": chunk.get("source", chunk.get("filename", chunk.get("title", ""))),
                "filename": chunk.get("filename", ""),
                "policy_category": chunk.get("policy_category", "general"),
                "leave_type": chunk.get("leave_type", "general"),
                "document_version": chunk.get("document_version", "v1.0"),
                "chunk_index": int(chunk.get("chunk_index", idx)),
                "text": chunk.get("text", chunk.get("content", "")),
                "title": chunk.get("title", ""),
            }

            vec_id = f"{metadata['filename']}_{metadata['chunk_index']}"
            vectors.append({
                "id": vec_id,
                "values": dense_list,
                "sparse_values": sparse_vals,
                "metadata": metadata,
            })

        # Batch upsert
        batch_size = 100
        total_upserted = 0
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i : i + batch_size]
            resp = self._index.upsert(vectors=batch)
            total_upserted += len(batch)

        return total_upserted

    def query(
        self,
        dense_vector: list[float] | None = None,
        sparse_vector: dict[str, list[Any]] | None = None,
        top_k: int = 5,
        filter_dict: dict[str, Any] | None = None,
        alpha: float = 1.0,
    ) -> list[dict[str, Any]]:
        """
        Executes dense, sparse, or hybrid query with alpha weighting and metadata filtering.
        """
        scaled_dense, scaled_sparse = hybrid_scale(dense_vector, sparse_vector, alpha)

        query_args: dict[str, Any] = {
            "top_k": top_k,
            "include_metadata": True,
        }
        if scaled_dense is not None:
            query_args["vector"] = scaled_dense
        if scaled_sparse is not None and scaled_sparse.get("indices"):
            query_args["sparse_vector"] = scaled_sparse
        if filter_dict:
            query_args["filter"] = filter_dict

        resp = self._index.query(**query_args)
        matches = resp.get("matches", []) if isinstance(resp, dict) else resp.matches
        return [m if isinstance(m, dict) else m.to_dict() for m in matches]


class PineconeRetriever:
    """
    Retriever class conforming to the [(chunk_dict, score), ...] interface.
    Drop-in compatible with cross-encoder rerank() and agent.py.
    """

    def __init__(
        self,
        store: PineconeStore,
        sparse_encoder: BM25SparseEncoder | None = None,
        default_alpha: float = 1.0,
    ):
        self.store = store
        self.sparse_encoder = sparse_encoder
        self.default_alpha = default_alpha

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filter_dict: dict[str, Any] | None = None,
        alpha: float | None = None,
    ) -> list[tuple[dict[str, Any], float]]:
        alpha_val = self.default_alpha if alpha is None else alpha

        # 1. Compute dense query embedding
        dense_emb = embed_texts([query])[0].tolist()

        # 2. Compute sparse query vector if encoder present
        sparse_vec = None
        if self.sparse_encoder is not None:
            sparse_vec = self.sparse_encoder.encode_queries(query)

        # 3. Query Pinecone
        matches = self.store.query(
            dense_vector=dense_emb,
            sparse_vector=sparse_vec,
            top_k=top_k,
            filter_dict=filter_dict,
            alpha=alpha_val,
        )

        # 4. Normalize to [(chunk_dict, score), ...]
        results = []
        for m in matches:
            meta = m.get("metadata", {})
            chunk_dict = {
                "title": meta.get("title", ""),
                "content": meta.get("text", ""),
                "chunk_index": meta.get("chunk_index", 0),
                "policy_category": meta.get("policy_category", "general"),
                "leave_type": meta.get("leave_type", "general"),
                "document_version": meta.get("document_version", "v1.0"),
                "source": meta.get("source", ""),
                "filename": meta.get("filename", ""),
                "text": meta.get("text", ""),
            }
            score = float(m.get("score", 0.0))
            results.append((chunk_dict, score))

        return results
