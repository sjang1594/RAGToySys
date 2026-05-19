from typing import Any, Dict, List, Optional

from sentence_transformers import CrossEncoder

from config import RERANKER_MODEL

_model: Optional[CrossEncoder] = None

def _get_model() -> CrossEncoder:
    global _model
    if _model is None:
        _model = CrossEncoder(RERANKER_MODEL)
    return _model

def rerank(query: str, candidates: List[Dict[str, Any]], top_k: int = 4) -> List[Dict[str, Any]]:
    """Score (query, document) pairs with a cross-encoder and return top_k."""
    if not candidates:
        return []
    model = _get_model()
    scores = model.predict([(query, c["document"]) for c in candidates])
    ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
    return [
        {**candidate, "rerank_score": float(score)}
        for score, candidate in ranked[:top_k]
    ]
