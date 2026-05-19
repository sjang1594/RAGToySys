import json
from pathlib import Path
from typing import Any

from providers import get_llm_provider

_SYSTEM = (
    "You are an expert at reading ML/CV/CG research papers and extracting structured information. "
    "Return only valid JSON, no markdown fences."
)

_PROMPT_TMPL = """Extract the architecture and key information from the following paper text.
Return a JSON object with these fields (use null if not mentioned):
{{
  "title": "paper title",
  "problem": "one sentence — what problem does this paper solve?",
  "approach": "one sentence — what is the core approach?",
  "backbone": "backbone architecture if applicable (e.g. ViT-L/16, ResNet-50)",
  "key_components": ["list of novel components or modules"],
  "loss_functions": ["list of loss functions used"],
  "datasets": ["training/evaluation datasets"],
  "key_results": "main quantitative result (e.g. 89.2 top-1 on ImageNet)",
  "novelty": "what makes this paper different from prior work"
}}

--- PAPER TEXT (first {n_chars} chars) ---
{text}
"""

def _extract_text(pdf_path: Path, max_chars: int) -> str:
    """Extract plain text from a PDF using pymupdf (no OCR needed for arXiv papers)."""
    import fitz  # pymupdf
    doc = fitz.open(str(pdf_path))
    parts = []
    total = 0
    for page in doc:
        text = page.get_text()
        parts.append(text)
        total += len(text)
        if total >= max_chars:
            break
    doc.close()
    return "\n\n".join(parts)[:max_chars]

def extract_architecture(pdf_path: Path, max_chars: int = 8000) -> dict[str, Any]:
    """Extract structured architecture JSON from a paper PDF using LLM."""
    truncated = _extract_text(pdf_path, max_chars)

    prompt = _PROMPT_TMPL.format(text=truncated, n_chars=max_chars)
    provider = get_llm_provider()
    response = provider.complete(
        messages=[{"role": "user", "content": prompt}],
        system=_SYSTEM,
        max_tokens=1024,
    )

    raw = next((b.text for b in response.content if b.type == "text"), "")
    raw = raw.strip()

    # Extract the JSON object regardless of surrounding text or markdown fences
    start = raw.find("{")
    end = raw.rfind("}") + 1
    if start != -1 and end > start:
        raw = raw[start:end]

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"raw": raw, "error": "JSON parse failed"}