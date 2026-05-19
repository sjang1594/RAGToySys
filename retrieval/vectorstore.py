from typing import List, Dict, Any, Optional
import chromadb
from chromadb import Collection

from config import STORAGE_DIR, CHROMA_COLLECTION

_client: Optional[chromadb.PersistentClient] = None
_collection: Optional[Collection] = None

def _get_collection() -> Collection:
    global _client, _collection
    if _collection is None:
        STORAGE_DIR.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(path=str(STORAGE_DIR))
        _collection = _client.get_or_create_collection(
            name=CHROMA_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection

def add(
    ids: List[str],
    embeddings: List[List[float]],
    documents: List[str],
    metadatas: List[Dict[str, Any]],
) -> None:
    _get_collection().add(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )

def query(
    embedding: List[float],
    top_k: int = 4,
) -> List[Dict[str, Any]]:
    """Return top-k results as list of {id, document, metadata, distance}."""
    result = _get_collection().query(
        query_embeddings=[embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    hits = []
    for id_, doc, meta, dist in zip(
        result["ids"][0],
        result["documents"][0],
        result["metadatas"][0],
        result["distances"][0],
    ):
        hits.append({"id": id_, "document": doc, "metadata": meta, "distance": dist})
    return hits

def get_all() -> List[Dict[str, Any]]:
    """Return all stored chunks as list of {id, document, metadata}."""
    col = _get_collection()
    if col.count() == 0:
        return []
    data = col.get(include=["documents", "metadatas"])
    return [
        {"id": id_, "document": doc, "metadata": meta}
        for id_, doc, meta in zip(data["ids"], data["documents"], data["metadatas"])
    ]

def list_sources() -> List[str]:
    """Return unique filenames stored in the collection."""
    col = _get_collection()
    if col.count() == 0:
        return []
    all_meta = col.get(include=["metadatas"])["metadatas"]
    return sorted({m["filename"] for m in all_meta})

def delete_source(filename: str) -> int:
    """Delete all chunks belonging to a filename. Returns number of chunks deleted."""
    col = _get_collection()
    result = col.get(where={"filename": filename}, include=["metadatas"])
    ids = result["ids"]
    if ids:
        col.delete(ids=ids)
    return len(ids)

def count() -> int:
    return _get_collection().count()
