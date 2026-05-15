from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

# Paths
ROOT_DIR = Path(__file__).parent
STORAGE_DIR = ROOT_DIR / "storage" / "chroma_db"
DOCS_DIR = ROOT_DIR / "docs"

# Claude
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_AUTH_TOKEN")
ANTHROPIC_BASE_URL = os.getenv("ANTHROPIC_BASE_URL")
LLM_MODEL = "claude-haiku-4-5-20251001"

# Embedding
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ChromaDB
CHROMA_COLLECTION = "rag_documents"

# Chunking
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Retrieval
TOP_K = 4