# RagToySys — Architecture Document

## Overview

A personal knowledge base that ingests image-based documents (photos, scans, screenshots) via OCR and answers natural language queries using retrieved context. Built without frameworks — every layer is explicit raw Python.

---

## Design Decisions

| Component       | Choice                        | Reason                                              |
|----------------|-------------------------------|-----------------------------------------------------|
| OCR Engine      | Tesseract + OpenCV            | Local, free, forces real preprocessing engineering  |
| Framework       | None (raw Python)             | Learn every layer without abstraction hiding it     |
| LLM             | Claude Haiku (claude-haiku-4-5-20251001) | Cheapest per token, fast, sufficient quality |
| Vector DB       | ChromaDB                      | Local, persistent, zero infra, pip install          |
| Embedding       | all-MiniLM-L6-v2              | Small, fast, good quality for English text          |
| Chunking        | Custom recursive splitter     | Understand overlap mechanics firsthand              |
| Chunk size      | 500 chars, overlap 50         | Balanced context vs retrieval precision             |
| Input types     | Mixed                         | Photos, scanned PDFs, screenshots, handwriting      |
| Metadata        | filename, page, timestamp     | Source citation in answers                          |

---

## System Architecture

### Ingestion Pipeline

```
[Input: image / PDF]
        │
        ▼
[OpenCV Preprocessor]
  - Grayscale
  - Deskew (Hough lines → rotate)
  - Binarize (Otsu threshold)
  - Denoise (fastNlMeansDenoising)
        │
        ▼
[Tesseract OCR]
  - Image  → raw text
  - PDF    → per-page images → OCR each → concat
        │
        ▼
[Text Cleaner]
  - Strip garbage characters
  - Normalize whitespace
  - Fix common Tesseract artifacts
        │
        ▼
[Recursive Chunker]
  - Separators: \n\n → \n → space → char
  - chunk_size=500, chunk_overlap=50
        │
        ▼
[Metadata Tagger]
  - filename, page_number, chunk_index, timestamp
        │
        ▼
[sentence-transformers Embedder]
  - all-MiniLM-L6-v2
  - Each chunk → 384-dim float vector
        │
        ▼
[ChromaDB]
  - Persist vectors + metadata to disk
```

### Query Pipeline

```
[User Query: string]
        │
        ▼
[sentence-transformers Embedder]
  - Query → 384-dim float vector
        │
        ▼
[ChromaDB .query()]
  - Cosine similarity search
  - Returns top-k=4 chunks + metadata
        │
        ▼
[Prompt Builder]
  - system: "Answer using only the provided context."
  - context: chunk_1 ... chunk_4 (with source labels)
  - user: original query
        │
        ▼
[Anthropic SDK]
  - anthropic.Anthropic().messages.create()
  - model: claude-haiku-4-5-20251001
        │
        ▼
[Answer + Source Citations]
  - answer text
  - sources: [filename, page] per cited chunk
```
---

## Data Flow Summary

```
Image/PDF
  → OCR text (Tesseract)
  → Clean text (custom)
  → Chunks[] (custom splitter)
  → Vectors[] (sentence-transformers)
  → ChromaDB (persist)

Query
  → Vector (sentence-transformers)
  → Top-4 chunks (ChromaDB cosine)
  → Prompt (f-string)
  → Answer (Claude Haiku)
  → Output (text + sources)
```