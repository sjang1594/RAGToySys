from config import PAPERS_DIR
from ingestion.pipeline import ingest
from paper.arxiv import download_pdf, fetch_metadata, parse_id

def ingest_arxiv(id_or_url: str) -> int:
    """Fetch arXiv paper, download PDF, and ingest with paper metadata."""
    arxiv_id = parse_id(id_or_url)

    print(f"[paper] Fetching metadata for {arxiv_id}")
    meta = fetch_metadata(arxiv_id)
    print(f"[paper] {meta['title']} ({meta['year']})")

    pdf_path = download_pdf(arxiv_id, PAPERS_DIR)
    return ingest(pdf_path, extra_metadata=meta)
