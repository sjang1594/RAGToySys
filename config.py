from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

# Paths
ROOT_DIR = Path(__file__).parent
STORAGE_DIR = ROOT_DIR / "storage" / "chroma_db"
PAPERS_DIR = ROOT_DIR / "storage" / "papers"
DOCS_DIR = ROOT_DIR / "docs"

# Claude
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_AUTH_TOKEN")
ANTHROPIC_BASE_URL = os.getenv("ANTHROPIC_BASE_URL")
LLM_MODEL = "claude-haiku-4-5-20251001"

# Provider selection: "claude" | "ollama"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "claude")
VISION_PROVIDER = os.getenv("VISION_PROVIDER", "claude")

# Ollama
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")
OLLAMA_VISION_MODEL = os.getenv("OLLAMA_VISION_MODEL", "llava")

# Embedding
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ChromaDB
CHROMA_COLLECTION = "rag_documents"

# Chunking
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Retrieval
TOP_K = 4

# Semantic Scholar (optional — higher rate limits with key)
SEMANTIC_SCHOLAR_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")