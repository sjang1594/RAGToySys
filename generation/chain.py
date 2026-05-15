from typing import List, Dict, Any

import anthropic

from retrieval.retriever import retrieve
from config import ANTHROPIC_API_KEY, ANTHROPIC_BASE_URL, LLM_MODEL


_client = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(
            api_key=ANTHROPIC_API_KEY,
            base_url=ANTHROPIC_BASE_URL,
        )
    return _client


def _build_prompt(question: str, chunks: List[Dict[str, Any]]) -> str:
    context_parts = []
    for i, chunk in enumerate(chunks):
        meta = chunk["metadata"]
        source = f"{meta['filename']} (page {meta['page']})"
        context_parts.append(f"[Source {i+1}: {source}]\n{chunk['document']}")

    context = "\n\n".join(context_parts)
    return f"{context}\n\nQuestion: {question}"


def ask(question: str, top_k: int = 4) -> Dict[str, Any]:
    """
    Full RAG query: retrieve chunks → build prompt → generate answer.
    Returns {answer, sources}
    """
    chunks = retrieve(question, top_k=top_k)

    if not chunks:
        return {"answer": "No relevant documents found.", "sources": []}

    prompt = _build_prompt(question, chunks)

    response = _get_client().messages.create(
        model=LLM_MODEL,
        max_tokens=1024,
        system=(
            "You are a helpful assistant. Answer the question using only the provided context. "
            "If the context does not contain enough information, say so. "
            "At the end of your answer, list the sources you used."
        ),
        messages=[{"role": "user", "content": prompt}],
    )

    sources = [
        f"{c['metadata']['filename']} (page {c['metadata']['page']})"
        for c in chunks
    ]

    return {
        "answer": response.content[0].text,
        "sources": sources,
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m generation.chain \"your question\"")
        sys.exit(1)

    question = " ".join(sys.argv[1:])
    result = ask(question)
    print(result["answer"])
    print("\nSources:", result["sources"])
