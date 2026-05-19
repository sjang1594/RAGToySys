from typing import List, Dict, Any

from providers import get_llm_provider
from providers.base import LLMProvider
from retrieval.retriever import retrieve

_provider: LLMProvider | None = None

def _get_provider() -> LLMProvider:
    global _provider
    if _provider is None:
        _provider = get_llm_provider()
    return _provider

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

    response = _get_provider().complete(
        messages=[{"role": "user", "content": prompt}],
        system=(
            "You are a helpful assistant. Answer the question using only the provided context. "
            "If the context does not contain enough information, say so. "
            "At the end of your answer, list the sources you used."
        ),
        max_tokens=1024,
    )

    sources = [
        f"{c['metadata']['filename']} (page {c['metadata']['page']})"
        for c in chunks
    ]

    text = next((b.text for b in response.content if b.type == "text"), "")
    return {"answer": text, "sources": sources}

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m generation.chain \"your question\"")
        sys.exit(1)

    question = " ".join(sys.argv[1:])
    result = ask(question)
    print(result["answer"])
    print("\nSources:", result["sources"])