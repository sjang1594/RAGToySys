"""LLM-as-Judge for architecture extraction quality."""
import json

from providers import get_llm_provider

_SYSTEM = (
    "You are evaluating architecture information extracted from an ML research paper. "
    "Be strict but fair. Return only valid JSON, no markdown fences."
)

_PROMPT = """Paper: {title} (arXiv:{arxiv_id})

Expected key information:
{expected}

Extracted information:
{extracted}

Rate the extraction quality on a scale of 1-5:
5 = Accurate and complete
4 = Mostly correct, minor omissions
3 = Partially correct, notable gaps
2 = Significant errors or missing critical info
1 = Major hallucinations or wrong paper

Return JSON:
{{"score": N, "issues": ["list any problems"], "reasoning": "one sentence"}}"""


def judge_architecture(
    arxiv_id: str,
    title: str,
    expected: dict,
    extracted: dict,
) -> dict:
    prompt = _PROMPT.format(
        title=title,
        arxiv_id=arxiv_id,
        expected=json.dumps(expected, ensure_ascii=False),
        extracted=json.dumps(extracted, indent=2, ensure_ascii=False),
    )
    provider = get_llm_provider()
    response = provider.complete(
        messages=[{"role": "user", "content": prompt}],
        system=_SYSTEM,
        max_tokens=512,
    )
    raw = next((b.text for b in response.content if b.type == "text"), "")
    raw = raw.strip()
    start, end = raw.find("{"), raw.rfind("}") + 1
    if start != -1 and end > start:
        raw = raw[start:end]
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"score": 0, "issues": ["judge response parse failed"], "reasoning": raw[:200]}
