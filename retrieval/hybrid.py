from typing import Any, Dict, List

from retrieval.embedder import embed_one
from retrieval.vectorstore import query as dense_query
from retrieval.bm25_retriever import bm25_retrieve
from retrieval.reranker import rerank
from config import TOP_K

_RRF_K = 60


def _rrf_fuse(
    dense_hits: List[Dict[str, Any]],
    sparse_hits: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Reciprocal Rank Fusion: RRF(d) = Σ 1 / (k + rank)"""
    scores: Dict[str, float] = {}
    by_id: Dict[str, Dict[str, Any]] = {}

    for rank, hit in enumerate(dense_hits):
        key = hit["id"]
        scores[key] = scores.get(key, 0.0) + 1.0 / (_RRF_K + rank + 1)
        by_id[key] = hit

    for rank, hit in enumerate(sparse_hits):
        key = hit["id"]
        scores[key] = scores.get(key, 0.0) + 1.0 / (_RRF_K + rank + 1)
        by_id[key] = hit

    ranked_ids = sorted(scores, key=lambda k: scores[k], reverse=True)
    return [by_id[k] for k in ranked_ids]


def hybrid_retrieve(query: str, top_k: int = TOP_K) -> List[Dict[str, Any]]:
    """
    Dense retrieval + BM25 → RRF fusion → cross-encoder reranker.
    Falls back to dense-only if the collection is empty or BM25 yields nothing.
    """
    embedding = embed_one(query)
    dense_hits = dense_query(embedding, top_k=top_k * 3)
    sparse_hits = bm25_retrieve(query, top_k=top_k * 3)

    if not dense_hits and not sparse_hits:
        return []

    fused = _rrf_fuse(dense_hits, sparse_hits)
    return rerank(query, fused[: top_k * 2], top_k=top_k)
