import re
import xml.etree.ElementTree as ET
from pathlib import Path

import time

import requests

_NS = {"atom": "http://www.w3.org/2005/Atom"}
_API = "https://export.arxiv.org/api/query"
_PDF = "https://arxiv.org/pdf/{arxiv_id}.pdf"
_HEADERS = {"User-Agent": "RAGToySys/1.0 (research tool; mailto:user@example.com)"}

def parse_id(id_or_url: str) -> str:
    """Extract bare arXiv ID from a URL or raw ID string.

    Accepts: '2301.00001', 'arxiv:2301.00001', 'https://arxiv.org/abs/2301.00001v2'
    """
    id_or_url = id_or_url.strip()
    match = re.search(r"(\d{4}\.\d{4,5}(?:v\d+)?)", id_or_url)
    if not match:
        raise ValueError(f"Cannot parse arXiv ID from: {id_or_url!r}")
    return match.group(1)

def fetch_metadata(arxiv_id: str) -> dict:
    """Fetch paper metadata from arXiv Atom API. Returns flat dict for ChromaDB metadata."""
    bare_id = re.sub(r"v\d+$", "", arxiv_id)  # strip version for API lookup
    for attempt in range(3):
        resp = requests.get(_API, params={"id_list": bare_id}, headers=_HEADERS, timeout=30)
        if resp.status_code == 429:
            time.sleep(2 ** attempt * 3)
            continue
        resp.raise_for_status()
        break
    else:
        raise RuntimeError(f"arXiv API rate-limited after 3 attempts for {bare_id}")

    root = ET.fromstring(resp.text)
    entry = root.find("atom:entry", _NS)
    if entry is None:
        raise ValueError(f"arXiv ID not found: {arxiv_id}")

    title = _text(entry, "atom:title").replace("\n", " ").strip()
    abstract = _text(entry, "atom:summary").replace("\n", " ").strip()
    published = _text(entry, "atom:published")  # e.g. "2023-01-01T00:00:00Z"
    year = published[:4] if published else ""

    authors = [
        _text(a, "atom:name")
        for a in entry.findall("atom:author", _NS)
    ]

    return {
        "arxiv_id": bare_id,
        "title": title,
        "authors": ", ".join(authors),
        "year": year,
        "abstract": abstract[:500],  # ChromaDB metadata values must be short
    }

def download_pdf(arxiv_id: str, dest_dir: Path) -> Path:
    """Download arXiv PDF to dest_dir. Returns local path."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    bare_id = re.sub(r"v\d+$", "", arxiv_id)
    dest = dest_dir / f"{bare_id}.pdf"

    if dest.exists():
        print(f"[arxiv] PDF already cached: {dest}")
        return dest

    url = _PDF.format(arxiv_id=bare_id)
    print(f"[arxiv] Downloading {url}")
    resp = requests.get(url, headers=_HEADERS, timeout=120, stream=True)
    resp.raise_for_status()

    with dest.open("wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)

    print(f"[arxiv] Saved to {dest}")
    return dest

def _text(element, tag: str) -> str:
    node = element.find(tag, _NS)
    return (node.text or "").strip() if node is not None else ""
