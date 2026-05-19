"""Metric calculations for search and architecture evaluation."""
import json
from pathlib import Path
from typing import Any


GOLDSET_DIR = Path(__file__).parent / "goldset"


# ── Search metrics ─────────────────────────────────────────────────────────────

def _load_search_goldset() -> list[dict]:
    with open(GOLDSET_DIR / "search.json") as f:
        return json.load(f)


def run_search_eval(top_k: int = 4) -> dict[str, Any]:
    from retrieval.hybrid import hybrid_retrieve

    goldset = _load_search_goldset()
    hits = 0
    results = []

    for item in goldset:
        retrieved = hybrid_retrieve(item["query"], top_k=top_k)
        retrieved_sources = {r["metadata"]["filename"] for r in retrieved}
        expected = set(item["expected_sources"])
        hit = bool(expected & retrieved_sources)
        hits += hit
        results.append({
            "id": item["id"],
            "query": item["query"],
            "expected": item["expected_sources"],
            "retrieved": sorted(retrieved_sources),
            "hit": hit,
        })

    total = len(goldset)
    return {
        "recall_at_k": hits / total if total else 0.0,
        "hit_rate": hits / total if total else 0.0,
        "k": top_k,
        "total": total,
        "hits": hits,
        "details": results,
    }


# ── Architecture metrics ────────────────────────────────────────────────────────

def _load_architecture_goldset() -> list[dict]:
    with open(GOLDSET_DIR / "architecture.json") as f:
        return json.load(f)


def _field_completeness(extracted: dict) -> float:
    scored_fields = ["title", "problem", "approach", "backbone",
                     "key_components", "loss_functions", "datasets", "key_results"]
    non_null = sum(
        1 for f in scored_fields
        if extracted.get(f) not in (None, [], "")
    )
    return non_null / len(scored_fields)


def _component_coverage(extracted: dict, must_include: list[str]) -> float:
    if not must_include:
        return 1.0
    components_text = json.dumps(extracted.get("key_components", [])).lower()
    found = sum(1 for kw in must_include if kw.lower() in components_text)
    return found / len(must_include)


def run_architecture_eval() -> dict[str, Any]:
    from config import PAPERS_DIR
    from paper.arxiv import download_pdf, parse_id
    from extraction.architecture import extract_architecture
    from evaluation.judge import judge_architecture

    goldset = _load_architecture_goldset()
    results = []
    judge_scores = []
    completeness_scores = []
    coverage_scores = []

    for item in goldset:
        print(f"  [arch] evaluating {item['arxiv_id']} ({item['title']})...")
        try:
            bare_id = parse_id(item["arxiv_id"])
            pdf_path = download_pdf(bare_id, PAPERS_DIR)
            extracted = extract_architecture(pdf_path)

            completeness = _field_completeness(extracted)
            coverage = _component_coverage(extracted, item["expected"].get("key_components_must_include", []))

            print(f"  [judge] running LLM judge for {item['arxiv_id']}...")
            verdict = judge_architecture(
                arxiv_id=item["arxiv_id"],
                title=item["title"],
                expected=item["expected"],
                extracted=extracted,
            )

            completeness_scores.append(completeness)
            coverage_scores.append(coverage)
            judge_scores.append(verdict.get("score", 0))

            results.append({
                "id": item["id"],
                "arxiv_id": item["arxiv_id"],
                "title": item["title"],
                "field_completeness": round(completeness, 3),
                "component_coverage": round(coverage, 3),
                "judge_score": verdict.get("score"),
                "judge_issues": verdict.get("issues", []),
                "judge_reasoning": verdict.get("reasoning", ""),
                "extracted": extracted,
            })
        except Exception as e:
            results.append({
                "id": item["id"],
                "arxiv_id": item["arxiv_id"],
                "title": item["title"],
                "error": str(e),
            })

    def avg(lst): return round(sum(lst) / len(lst), 3) if lst else 0.0

    return {
        "mean_field_completeness": avg(completeness_scores),
        "mean_component_coverage": avg(coverage_scores),
        "mean_judge_score": avg(judge_scores),
        "total": len(goldset),
        "details": results,
    }
