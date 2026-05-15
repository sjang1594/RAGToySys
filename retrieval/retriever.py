from typing import List, Dict, Any

from retrieval.embedder import embed_one
from retrieval.vectorstore import query
from config import TOP_K


def retrieve(question: str, top_k: int = TOP_K) -> List[Dict[str, Any]]:
    """
    Query string → top-k relevant chunks.
    Each result: {document, metadata, distance}
    """
    embedding = embed_one(question)
    return query(embedding, top_k=top_k)

def format_results(results: List[Dict[str, Any]]) -> str:
    """Pretty-print retrieved chunks for debugging."""
    lines = []
    for i, r in enumerate(results):
        meta = r["metadata"]
        source = f"{meta['filename']} (page {meta['page']}, chunk {meta['chunk_index']})"
        lines.append(f"[{i+1}] {source} | distance={r['distance']:.4f}")
        lines.append(r["document"])
        lines.append("")
    return "\n".join(lines)

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m retrieval.retriever \"your question\"")
        sys.exit(1)

    question = " ".join(sys.argv[1:])
    results = retrieve(question)
    print(format_results(results))
