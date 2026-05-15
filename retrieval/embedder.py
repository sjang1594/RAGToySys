from typing import List, Optional
from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL

_model: Optional[SentenceTransformer] = None

def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model

def embed(texts: List[str]) -> List[List[float]]:
    """Encode a list of strings into embedding vectors."""
    model = _get_model()
    vectors = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return vectors.tolist()

def embed_one(text: str) -> List[float]:
    return embed([text])[0]
