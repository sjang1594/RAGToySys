import re
from typing import Any, Dict, List

from rank_bm25 import BM25Okapi

from retrieval.vectorstore import get_all

def _tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower())

def bm25_retrieve(query: str, top_k: int = 10) -> List[Dict[str, Any]]:
    """BM25 sparse retrieval over all documents in ChromaDB."""
    all_docs = get_all()
    if not all_docs:
        return []

    corpus = [d["document"] for d in all_docs]
    tokenized = [_tokenize(doc) for doc in corpus]
    bm25 = BM25Okapi(tokenized)

    scores = bm25.get_scores(_tokenize(query))
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    return [
        {**all_docs[i], "bm25_score": float(scores[i])}
        for i in ranked
        if scores[i] > 0
    ]
