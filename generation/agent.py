import json
import math
from typing import Any, Dict, List

from providers import get_llm_provider
from providers.base import LLMProvider

MAX_ITERATIONS = 10

TOOLS = [
    {
        "name": "search_knowledge_base",
        "description": (
            "Search the knowledge base for relevant information. "
            "Use this to answer questions about ingested documents."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query"},
                "top_k": {
                    "type": "integer",
                    "description": "Number of results to return (default: 4)",
                    "default": 4,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "ingest_document",
        "description": "Ingest a document (image or PDF) into the knowledge base.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file to ingest"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "list_documents",
        "description": "List all documents currently in the knowledge base.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "calculator",
        "description": (
            "Evaluate a mathematical expression. "
            "Supports arithmetic and standard math functions (sqrt, log, sin, cos, etc.)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Math expression to evaluate (e.g. '2 + 2 * 3', 'sqrt(144)')",
                },
            },
            "required": ["expression"],
        },
    },
    {
        "name": "get_reading_list",
        "description": (
            "Given an arXiv paper ID, return a ranked reading list of related papers. "
            "Uses B→D→C strategy: B=direct references, D=foundational lineage, C=semantically similar. "
            "Use this when the user asks what to read before/alongside a paper."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "arxiv_id": {
                    "type": "string",
                    "description": "arXiv ID (e.g. '2310.06825') or full URL",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of papers to return (default: 10)",
                    "default": 10,
                },
            },
            "required": ["arxiv_id"],
        },
    },
    {
        "name": "ingest_arxiv",
        "description": "Download and ingest an arXiv paper into the knowledge base by ID or URL.",
        "input_schema": {
            "type": "object",
            "properties": {
                "arxiv_id": {
                    "type": "string",
                    "description": "arXiv ID (e.g. '2310.06825') or full URL",
                },
            },
            "required": ["arxiv_id"],
        },
    },
    {
        "name": "analyze_paper_architecture",
        "description": (
            "Analyze a paper's architecture: extracts structured JSON (backbone, components, loss, results) "
            "and describes key figures/diagrams. Also returns the foundational lineage (D tier). "
            "Use this when the user asks to understand or explain a paper's architecture or design."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "arxiv_id": {
                    "type": "string",
                    "description": "arXiv ID (e.g. '2310.06825') or full URL",
                },
                "include_figures": {
                    "type": "boolean",
                    "description": "Whether to extract figure descriptions via Vision (default: false). Only effective when VISION_PROVIDER=claude.",
                    "default": False,
                },
            },
            "required": ["arxiv_id"],
        },
    },
]

_MATH_NS = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
_MATH_NS.update({"abs": abs, "round": round})


def _tool_search(query: str, top_k: int = 4) -> str:
    from retrieval.hybrid import hybrid_retrieve
    results = hybrid_retrieve(query, top_k=int(top_k))
    if not results:
        return "No relevant documents found."
    parts = []
    for i, r in enumerate(results):
        meta = r["metadata"]
        source = f"{meta['filename']} (page {meta['page']})"
        parts.append(f"[{i + 1}] {source}\n{r['document']}")
    return "\n\n".join(parts)


def _tool_ingest(path: str) -> str:
    from pathlib import Path
    from ingestion.pipeline import ingest
    p = Path(path)
    if not p.exists():
        return f"Error: file not found: {path}"
    count = ingest(p)
    return f"Ingested '{p.name}' — {count} chunks stored."


def _tool_list() -> str:
    from retrieval.vectorstore import list_sources, count
    sources = list_sources()
    total = count()
    if not sources:
        return "Knowledge base is empty."
    lines = [f"{total} chunks across {len(sources)} document(s):"]
    for s in sources:
        lines.append(f"  - {s}")
    return "\n".join(lines)


def _tool_calculator(expression: str) -> str:
    try:
        result = eval(expression, {"__builtins__": {}}, _MATH_NS)  # noqa: S307
        return str(result)
    except Exception as e:
        return f"Error: {e}"


def _tool_reading_list(arxiv_id: str, top_k: int = 10) -> str:
    from paper.ranker import build_reading_list
    papers = build_reading_list(arxiv_id, top_k=int(top_k))
    if not papers:
        return f"No related papers found for {arxiv_id}."
    lines = [f"Reading list for {arxiv_id} (B→D→C ranking):\n"]
    for p in papers:
        lines.append(str(p))
    return "\n".join(lines)


def _tool_ingest_arxiv(arxiv_id: str) -> str:
    from paper.ingestor import ingest_arxiv
    count = ingest_arxiv(arxiv_id)
    return f"Ingested arXiv:{arxiv_id} — {count} chunks stored."


def _tool_analyze_architecture(arxiv_id: str, include_figures: bool = False) -> str:
    import json
    from paper.arxiv import download_pdf, parse_id
    from config import PAPERS_DIR
    from extraction.architecture import extract_architecture

    bare_id = parse_id(arxiv_id)
    pdf_path = download_pdf(bare_id, PAPERS_DIR)

    arch = extract_architecture(pdf_path)
    lines = ["=== Architecture JSON ===", json.dumps(arch, indent=2, ensure_ascii=False)]

    if include_figures:
        from extraction.figures import extract_figures
        print(f"[agent] Extracting figures from {pdf_path.name}...")
        figs = extract_figures(pdf_path, max_pages=8)
        if figs:
            lines.append("\n=== Key Figures ===")
            for f in figs:
                lines.append(str(f))
        else:
            lines.append("\n=== Key Figures ===\nNo significant figures found.")

    # D-tier lineage (top 3 direct references for context)
    try:
        from paper.ranker import build_reading_list
        papers = build_reading_list(bare_id, top_k=3)
        b_papers = [p for p in papers if p.reason == "B_direct"]
        if b_papers:
            lines.append("\n=== Foundational References (D lineage) ===")
            for p in b_papers:
                lines.append(f"  {p.title} ({p.year})")
    except Exception:
        pass

    return "\n".join(lines)


def _dispatch(name: str, inputs: Dict[str, Any]) -> str:
    if name == "search_knowledge_base":
        return _tool_search(**inputs)
    if name == "ingest_document":
        return _tool_ingest(**inputs)
    if name == "ingest_arxiv":
        return _tool_ingest_arxiv(**inputs)
    if name == "list_documents":
        return _tool_list()
    if name == "calculator":
        return _tool_calculator(**inputs)
    if name == "get_reading_list":
        return _tool_reading_list(**inputs)
    if name == "analyze_paper_architecture":
        return _tool_analyze_architecture(**inputs)
    return f"Unknown tool: {name}"


_provider: LLMProvider | None = None


def _get_provider() -> LLMProvider:
    global _provider
    if _provider is None:
        _provider = get_llm_provider()
    return _provider


def run(task: str, verbose: bool = True) -> str:
    """
    ReAct agent loop: runs until Claude stops calling tools or MAX_ITERATIONS reached.
    Returns the final answer string.
    """
    provider = _get_provider()
    messages: List[Dict[str, Any]] = [{"role": "user", "content": task}]

    for _ in range(MAX_ITERATIONS):
        response = provider.complete(
            messages=messages,
            system=(
                "You are a research assistant with access to a knowledge base of ML/CV/CG papers. "
                "IMPORTANT: You MUST call tools to retrieve information — do NOT answer from memory alone. "
                "ALWAYS call search_knowledge_base first before answering any question about papers or techniques. "
                "Only provide a final answer after you have retrieved context from the knowledge base. "
                "Do not describe what you would do — just do it by calling the tool directly."
            ),
            tools=TOOLS,
            max_tokens=2048,
        )

        messages.append(provider.response_to_message(response))

        if response.stop_reason == "end_turn":
            for block in response.content:
                if hasattr(block, "text"):
                    return block.text
            return ""

        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            if verbose:
                print(f"\n[tool] {block.name}({json.dumps(block.input, ensure_ascii=False)})")
            result = _dispatch(block.name, block.input)
            if verbose:
                preview = result[:300] + "..." if len(result) > 300 else result
                print(f"[result] {preview}")
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": result,
            })

        messages.append({"role": "user", "content": tool_results})

    return "Max iterations reached without a final answer."


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print('Usage: python -m generation.agent "your task"')
        sys.exit(1)
    print(run(" ".join(sys.argv[1:])))
