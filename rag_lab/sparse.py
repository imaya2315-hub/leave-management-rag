"""
Sparse and Hybrid Vector Encoding for Pinecone.

Provides:
- BM25SparseEncoder: fits Okapi BM25 on corpus chunks and generates
  Pinecone-compatible sparse vectors: {"indices": list[int], "values": list[float]}.
- hybrid_scale: scales dense and sparse vectors by the convex parameter alpha:
    Final Score = (alpha * Dense) + ((1 - alpha) * Sparse)
    alpha = 1.0 -> dense only
    alpha = 0.0 -> sparse only
    0 < alpha < 1 -> hybrid combination
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

# Default Okapi BM25 hyperparameters
DEFAULT_K1 = 1.2
DEFAULT_B = 0.75

_TOKEN_RE = re.compile(r"\b[a-zA-Z0-9_]+\b")


def tokenize(text: str) -> list[str]:
    """Tokenize text into lowercase alphanumeric words."""
    return _TOKEN_RE.findall(text.lower())


class BM25SparseEncoder:
    """
    Fits BM25 statistics on a corpus and produces sparse vectors formatted for Pinecone:
    {"indices": [int, ...], "values": [float, ...]}.
    """

    def __init__(self, k1: float = DEFAULT_K1, b: float = DEFAULT_B):
        self.k1 = k1
        self.b = b
        self.vocab: dict[str, int] = {}
        self.idf: dict[int, float] = {}
        self.avgdl: float = 0.0
        self.doc_count: int = 0

    def fit(self, corpus: list[str]) -> "BM25SparseEncoder":
        """Calculate vocabulary, document frequencies, average length, and IDF."""
        self.doc_count = len(corpus)
        if self.doc_count == 0:
            return self

        doc_lengths = []
        df: Counter[str] = Counter()

        tokenized_corpus = []
        for doc in corpus:
            tokens = tokenize(doc)
            tokenized_corpus.append(tokens)
            doc_lengths.append(len(tokens))
            df.update(set(tokens))

        self.avgdl = sum(doc_lengths) / self.doc_count if self.doc_count > 0 else 0.0

        # Build vocabulary sorted for deterministic index assignment
        sorted_terms = sorted(df.keys())
        self.vocab = {term: idx for idx, term in enumerate(sorted_terms)}

        # Compute Robertson-Spärck Jones IDF with smoothing
        self.idf = {}
        for term, doc_freq in df.items():
            idx = self.vocab[term]
            # Standard smoothed BM25 IDF: ln(1 + (N - df + 0.5) / (df + 0.5))
            self.idf[idx] = math.log(1.0 + (self.doc_count - doc_freq + 0.5) / (doc_freq + 0.5))

        return self

    def encode_documents(self, text: str) -> dict[str, list[Any]]:
        """
        Encode a document into Pinecone sparse vector:
        {"indices": [int, ...], "values": [float, ...]}.
        """
        tokens = tokenize(text)
        doc_len = len(tokens)
        if not tokens or self.doc_count == 0:
            return {"indices": [], "values": []}

        tf = Counter(tokens)
        indices: list[int] = []
        values: list[float] = []

        for term, count in sorted(tf.items(), key=lambda item: self.vocab.get(item[0], -1)):
            idx = self.vocab.get(term)
            if idx is None:
                continue

            # Okapi BM25 term frequency saturation
            idf_val = self.idf.get(idx, 0.0)
            if idf_val <= 0:
                continue

            numerator = count * (self.k1 + 1.0)
            denominator = count + self.k1 * (1.0 - self.b + self.b * (doc_len / (self.avgdl or 1.0)))
            bm25_tf = numerator / (denominator or 1.0)
            weight = float(idf_val * bm25_tf)

            if weight > 0.0:
                indices.append(idx)
                values.append(round(weight, 6))

        return {"indices": indices, "values": values}

    def encode_queries(self, query: str) -> dict[str, list[Any]]:
        """
        Encode a query into Pinecone sparse vector.
        When Pinecone performs dotproduct(query_sparse, doc_sparse), the result
        equals the sum of doc BM25 weights for all matching query tokens.
        """
        tokens = tokenize(query)
        if not tokens or not self.vocab:
            return {"indices": [], "values": []}

        tf = Counter(tokens)
        indices: list[int] = []
        values: list[float] = []

        for term, count in sorted(tf.items(), key=lambda item: self.vocab.get(item[0], -1)):
            idx = self.vocab.get(term)
            if idx is None:
                continue

            # Query term weight (typically 1.0 or normalized query frequency)
            indices.append(idx)
            values.append(float(count))

        return {"indices": indices, "values": values}


def hybrid_scale(
    dense_vector: list[float] | None,
    sparse_vector: dict[str, list[Any]] | None,
    alpha: float,
) -> tuple[list[float] | None, dict[str, list[Any]] | None]:
    """
    Scales dense and sparse vectors by alpha according to Pinecone's hybrid search:
      scaled_dense = [v * alpha for v in dense_vector]
      scaled_sparse = {
          "indices": sparse["indices"],
          "values": [v * (1 - alpha) for v in sparse["values"]]
      }

    alpha = 1.0 -> Pure dense (semantic)
    alpha = 0.0 -> Pure sparse (lexical)
    0 < alpha < 1 -> Hybrid weighting
    """
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"Alpha must be between 0.0 and 1.0, got {alpha}")

    # Scale dense
    scaled_dense: list[float] | None = None
    if dense_vector is not None:
        if alpha == 0.0:
            # All dense values set to 0.0 for pure sparse
            scaled_dense = [0.0 for _ in dense_vector]
        else:
            scaled_dense = [float(v) * alpha for v in dense_vector]

    # Scale sparse
    scaled_sparse: dict[str, list[Any]] | None = None
    if sparse_vector is not None:
        raw_indices = sparse_vector.get("indices", [])
        raw_values = sparse_vector.get("values", [])
        sparse_weight = 1.0 - alpha

        if sparse_weight == 0.0:
            scaled_sparse = {"indices": raw_indices, "values": [0.0 for _ in raw_values]}
        else:
            scaled_sparse = {
                "indices": list(raw_indices),
                "values": [float(v) * sparse_weight for v in raw_values],
            }

    return scaled_dense, scaled_sparse
