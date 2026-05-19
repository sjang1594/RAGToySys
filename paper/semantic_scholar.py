import time
from typing import Any

import requests

from config import SEMANTIC_SCHOLAR_API_KEY

_GRAPH = "https://api.semanticscholar.org/graph/v1"
_RECO = "https://api.semanticscholar.org/recommendations/v1"
_HEADERS = {"User-Agent": "RAGToySys/1.0"}

def _get(url: str, params: dict = None) -> Any:
    if SEMANTIC_SCHOLAR_API_KEY:
        _HEADERS["x-api-key"] = SEMANTIC_SCHOLAR_API_KEY
    for attempt in range(4):
        resp = requests.get(url, params=params, headers=_HEADERS, timeout=30)
        if resp.status_code == 429:
            wait = int(resp.headers.get("Retry-After", 2 ** attempt * 10))
            print(f"[s2] Rate limited, waiting {wait}s (attempt {attempt + 1}/4)")
            time.sleep(wait)
            continue
        if resp.status_code in (401, 403, 404):
            return None
        resp.raise_for_status()
        return resp.json()
    raise RuntimeError(f"Semantic Scholar rate limit exceeded: {url}")

def get_paper(arxiv_id: str) -> dict | None:
    """Fetch paper details by arXiv ID."""
    return _get(
        f"{_GRAPH}/paper/ARXIV:{arxiv_id}",
        params={"fields": "paperId,title,authors,year,externalIds,abstract"},
    )

def get_references(ss_id: str, limit: int = 50) -> list[dict]:
    """Papers this paper directly cites (B tier)."""
    data = _get(
        f"{_GRAPH}/paper/{ss_id}/references",
        params={"fields": "paperId,title,authors,year,externalIds", "limit": limit},
    )
    if not data:
        return []
    return [r["citedPaper"] for r in data.get("data", []) if r.get("citedPaper")]

def get_related(ss_id: str, limit: int = 20) -> list[dict]:
    """Semantically related papers (C tier)."""
    data = _get(
        f"{_RECO}/papers/forpaper/{ss_id}",
        params={"fields": "paperId,title,authors,year,externalIds", "limit": limit},
    )
    if not data:
        return []
    return data.get("recommendedPapers", [])

def arxiv_id_of(paper: dict) -> str | None:
    return (paper.get("externalIds") or {}).get("ArXiv")
