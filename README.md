# RAGSystem — Personal Knowledge Base with OCR

## Decisions Log

| Component       | Choice                              |
|----------------|-------------------------------------|
| OCR Engine      | Tesseract + OpenCV preprocessing    |
| Framework       | None (raw Python)                   |
| LLM             | Claude API (claude-haiku-4-5)       |
| Vector DB       | ChromaDB (local, persistent)        |
| Embedding       | sentence-transformers               |
| Chunking        | Custom recursive splitter           |
| Chunk size      | 500 chars, overlap 50               |
| Input types     | Mixed (scanned PDFs, photos, screenshots, handwriting) |
| Metadata        | filename, page number, timestamp    |

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   INGESTION PIPELINE                │
│                                                     │
│  [Image/PDF] → [Preprocessor] → [Tesseract OCR]    │
│                    ↓                                │
│              [Text Cleaner]                         │
│                    ↓                                │
│              [Chunker] ──── chunk_size=500          │
│                    ↓        chunk_overlap=50        │
│         [Metadata Tagger]                           │
│         (filename, page, ts)                        │
│                    ↓                                │
│         [ChromaDB Embedder & Store]                 │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│                   QUERY PIPELINE                    │
│                                                     │
│  [User Query] → [Embedder]                          │
│                    ↓                                │
│             [ChromaDB Search]  ← top-k chunks       │
│                    ↓                                │
│          [Prompt Builder]                           │
│          (query + retrieved chunks)                 │
│                    ↓                                │
│          [Claude API (Haiku)]                       │
│                    ↓                                │
│              [Answer + Sources]                     │
└─────────────────────────────────────────────────────┘
```

---

## Project Structure

```
RAGSystem/
├── ingestion/
│   ├── ocr/
│   │   ├── preprocessor.py    # OpenCV: deskew, binarize, denoise
│   │   └── extractor.py       # Tesseract wrapper
│   ├── cleaner.py             # Post-OCR text normalization
│   ├── chunker.py             # Custom recursive splitter (no LangChain)
│   └── pipeline.py            # Orchestrates full ingestion
├── retrieval/
│   ├── embedder.py            # sentence-transformers wrapper
│   ├── vectorstore.py         # ChromaDB init + CRUD
│   └── retriever.py           # Raw ChromaDB query → top-k chunks
├── generation/
│   └── chain.py               # Raw Anthropic SDK + prompt builder
├── storage/
│   └── chroma_db/             # Persisted vector store (gitignored)
├── app/
│   └── cli.py                 # CLI: ingest / query commands
├── docs/                      # Sample test documents
├── config.py                  # API keys, paths, constants
├── requirements.txt
└── PLAN.md                    # This file
```

---

### OCR Pipeline (this can be switched to PaddleOCR if Tesseract accuracy is too low, especially for handwriting)
**Goal:** Given any image, output clean text.

Steps:
1. `preprocessor.py` — OpenCV pipeline
   - Grayscale conversion
   - Deskew (Hough line detection → rotate)
   - Binarization (Otsu thresholding)
   - Denoise (fastNlMeansDenoising)
2. `extractor.py` — Tesseract wrapper
   - Single image → raw OCR text
   - PDF → per-page images → OCR each page
3. `cleaner.py` — Post-OCR normalization
   - Strip garbage characters
   - Normalize whitespace
   - Fix common Tesseract artifacts

Verify: Feed a scanned image, get readable clean text.

### Test 1 - data/clear.png

### Test 2 - data/sample.jpg

### Test 3 - data/handwritten.jpg
    
---

### Ingestion Pipeline
**Goal:** Image → ChromaDB entries with metadata.

Steps:
1. `chunker.py` — Custom recursive splitter (separator priority: \n\n → \n → space → char)
2. `pipeline.py` — Orchestrate: preprocess → OCR → clean → chunk → embed → store
3. ChromaDB collection setup in `vectorstore.py`
4. Metadata schema: `{filename, page, timestamp, chunk_index}`

Verify: Ingest a document, inspect ChromaDB collection.

---

### Retrieval
**Goal:** Query → top-k relevant chunks.

Steps:
1. `embedder.py` — sentence-transformers (all-MiniLM-L6-v2)
2. `retriever.py` — raw ChromaDB `.query()`, top-k=4
3. Return chunks + metadata (source citation)

Verify: Query returns chunks with correct source attribution.

---

### Generation
**Goal:** Query + chunks → answer via Claude.

Steps:
1. `chain.py` — manual prompt builder: system + chunks + query
2. Raw `anthropic.Anthropic().messages.create()` call
3. Claude Haiku as LLM backend
4. Return answer + source citations

Verify: End-to-end query produces grounded answer.

---

### CLI App
**Goal:** Usable interface for ingest and query.

Commands:
```bash
python app/cli.py ingest --input docs/scan.jpg
python app/cli.py ingest --input docs/report.pdf
python app/cli.py query "What is the beat frequency formula?"
python app/cli.py list   # show all ingested documents
```

Verify: Full workflow usable from terminal.

---

## Skill Stack

```
pytesseract           # Tesseract Python binding
opencv-python         # Image preprocessing
Pillow                # Image I/O
pdf2image             # PDF → image pages
chromadb              # Vector store
sentence-transformers # Embedding model
anthropic             # Claude API client (raw SDK)
python-dotenv         # API key management
```
---