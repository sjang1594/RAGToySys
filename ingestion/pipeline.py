from datetime import datetime, timezone
from pathlib import Path
from typing import List, Union

from ingestion.ocr.extractor import extract
from ingestion.cleaner import clean
from ingestion.chunker import chunk
from retrieval.embedder import embed
from retrieval.vectorstore import add
from config import CHUNK_SIZE, CHUNK_OVERLAP


def ingest(path: Union[str, Path]) -> int:
    """
    Full ingestion pipeline: image/PDF → ChromaDB.
    Returns number of chunks stored.
    """
    path = Path(path)
    filename = path.name
    timestamp = datetime.now(timezone.utc).isoformat()

    # OCR — list of page texts
    pages: List[str] = extract(path)

    ids, embeddings, documents, metadatas = [], [], [], []

    for page_num, raw_text in enumerate(pages):
        cleaned = clean(raw_text)
        if not cleaned.strip():
            continue

        chunks = chunk(cleaned, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

        for chunk_idx, chunk_text in enumerate(chunks):
            if not chunk_text.strip():
                continue

            chunk_id = f"{filename}::p{page_num}::c{chunk_idx}"
            metadata = {
                "filename": filename,
                "page": page_num,
                "chunk_index": chunk_idx,
                "timestamp": timestamp,
            }
            ids.append(chunk_id)
            documents.append(chunk_text)
            metadatas.append(metadata)

    if not ids:
        print(f"[ingest] No text extracted from {filename}")
        return 0

    embeddings = embed(documents)
    add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)

    print(f"[ingest] {filename} → {len(ids)} chunks stored")
    return len(ids)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m ingestion.pipeline --input <file>")
        sys.exit(1)

    target = sys.argv[-1]
    ingest(target)
