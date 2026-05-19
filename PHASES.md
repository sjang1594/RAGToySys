# RAGToySys — CV/CG Paper Research Agent: Implementation Phases

## Overview

RAGToySys를 확장해 CV/CG 최신 논문 리서치 에이전트를 구축한다.  
두 가지 입력 모드를 지원한다:

- **Mode 1** — 자연어 검색: "X 기술 논문 찾아줘"
- **Mode 2** — 논문 분석: 최신 논문 입력 → 연관 논문 + 아키텍처 + 읽기 순서 반환

---

## Phase 0 — Core RAGToySys ✅

기존 RAGToySys 인프라 + Provider 추상화 완료.

| 컴포넌트 | 상태 |
|---------|------|
| OCR ingestion (Tesseract + OpenCV) | ✅ |
| Hybrid retrieval (Dense + BM25 + RRF + reranker) | ✅ |
| ReAct agent (tool calling loop) | ✅ |
| Provider 추상화 (Claude / Ollama / HuggingFace) | ✅ |
| CLI (`ingest`, `query`, `list`, `remove`, `agent`) | ✅ |

---

## Phase 1 — Paper Input Layer

**Goal:** 논문을 arXiv 링크 또는 PDF로 받아 paper-aware 메타데이터와 함께 ChromaDB에 저장한다.

### 새 파일

| 파일 | 역할 |
|------|------|
| `paper/arxiv.py` | arXiv ID 파싱, PDF 다운로드, 메타데이터 추출 |
| `paper/ingestor.py` | 기존 ingestion pipeline + paper 메타데이터 스키마 확장 |

### 메타데이터 스키마 확장

```python
# 기존
{filename, page, timestamp, chunk_index}

# Phase 1 이후
{filename, page, timestamp, chunk_index,
 title, authors, year, arxiv_id, abstract}
```

### CLI 확장

```bash
rag ingest --arxiv 2301.00001          # arXiv ID
rag ingest --input ./paper.pdf         # 기존 PDF (그대로)
```

### Success Criteria

```
rag ingest --arxiv 2301.00001
→ ChromaDB에 title, authors, year, arxiv_id 포함된 chunks 저장 확인
```

---

## Phase 2 — Semantic Scholar Integration + Ranking

**Goal:** 논문 입력 → B→D→C 순서로 읽기 목록 자동 생성.

### Ranking 전략

| 단계 | 기준 | 설명 |
|------|------|------|
| **B** | 직접 인용 | 입력 논문이 직접 cite한 논문 (저자가 읽어야 한다고 판단한 것) |
| **D** | 시간 계보 | B 논문들의 선행 연구, 연도순 정렬 (이 방법이 어떻게 발전해왔는가) |
| **C** | 의미 유사도 | 내용이 비슷한 병렬 연구 (Semantic Scholar 임베딩 기반) |

### 새 파일

| 파일 | 역할 |
|------|------|
| `paper/semantic_scholar.py` | Semantic Scholar API 클라이언트 (references, citations, related) |
| `paper/ranker.py` | B→D→C 3단계 ranking 구현 |

### 새 Agent Tool

```python
get_reading_list(arxiv_id: str, top_k: int = 10) -> list[Paper]
```

### Success Criteria

```
rag agent "2401.00001 논문 읽기 전에 뭘 읽어야 해?"
→ B→D→C 순서로 ranked된 논문 목록 반환
```

---

## Phase 3 — Architecture Extraction

**Goal:** 논문에서 구조화된 아키텍처 JSON + Figure 분석을 추출한다.

### 추출 대상

| 유형 | 출력 | 구현 |
|------|------|------|
| **B** 구조화 JSON | `{backbone, head, loss, key_contributions}` | LLM extraction |
| **C** Figure 분석 | 아키텍처 다이어그램 설명 | Vision provider (Claude / LLaVA) |
| **D** 기술 계보 | "ResNet → ViT → 이 논문" 흐름 | Phase 2 ranker 연동 |

### 새 파일

| 파일 | 역할 |
|------|------|
| `extraction/architecture.py` | LLM으로 아키텍처 JSON 추출 |
| `extraction/figures.py` | PDF → 페이지 이미지 → Vision provider 분석 |

### 새 Agent Tool

```python
analyze_paper_architecture(arxiv_id_or_path: str) -> ArchitectureReport
```

### Success Criteria

```
rag agent "Attention is All You Need 아키텍처 뽑아줘"
→ JSON 구조 + Figure 설명 + 기술 계보 반환
```

### Vision Model 평가 결과 (2026-05-19)

로컬 Vision 모델의 figure extraction 적용 가능성을 검증했다. 테스트 대상 논문: Mistral 7B (`2310.06825.pdf`).

| 모델 | 크기 | 결과 | 원인 |
|------|------|------|------|
| `moondream:latest` | 1.7GB | ❌ 심각한 hallucination | 1.7B 파라미터로 기술 문서 이해 불가. 아키텍처 다이어그램을 "컴퓨터 마우스 사용법", "물 펌프 그래프"로 오인식. |
| `llava:latest` | 4.7GB | ❌ OOM crash | Ollama model runner가 RAM 부족으로 비정상 종료. |
| `llama3.2-vision` | 8GB+ | ❌ 로드 불가 | RAM 부족으로 로드 자체 실패. |
| `claude-vision` (cloud) | — | ✅ 동작 | API 비용 발생. |

**결론:** 현재 하드웨어에서 로컬 Vision 모델로 ML 논문 아키텍처 다이어그램 추출은 불가능하다.

**후속 결정 (미확정):**
- Option A — `include_figures` 기본값을 `False`로 변경, 유저가 명시적으로 요청할 때만 Claude Vision 호출
- Option B — Figure extraction 자체를 로컬 전용 모드에서 비활성화, `VISION_PROVIDER=none` 추가

---

## Phase 4 — Fine-tuning Pipeline

**Goal:** CV/CG 도메인 데이터로 로컬 모델을 fine-tuning해 `OllamaProvider`와 연동한다.

### 접근 방식

- **데이터 생성**: RAG 쿼리 결과로 Q&A 쌍 자동 생성 (LLM-as-annotator)
- **Fine-tuning**: HuggingFace + LoRA (PEFT) — GPU 필요
- **배포**: fine-tuned 모델 → Ollama Modelfile로 export → `LLM_PROVIDER=ollama`로 전환

### 새 파일

| 파일 | 역할 |
|------|------|
| `training/data_prep.py` | RAG 결과 기반 Q&A 쌍 생성 |
| `training/finetune.py` | HuggingFace + LoRA fine-tuning 루프 |
| `training/export.py` | fine-tuned 모델 → Ollama 포맷 export |

### Success Criteria

```bash
# fine-tuning 실행
python -m training.finetune --data ./training/data --output ./models/cv-llm

# fine-tuned 모델로 agent 실행
LLM_PROVIDER=ollama OLLAMA_MODEL=cv-llm rag agent "NeRF 설명해줘"
```

---

## 의존성 순서

```
Phase 0 ✅
    │
    ▼
Phase 1 ── Paper Input Layer
    │
    ▼
Phase 2 ── Semantic Scholar + Ranking
    │
    ▼
Phase 3 ── Architecture Extraction
    │
Phase 4 ── Fine-tuning (Phase 1 이후 독립적으로 진행 가능)
```

---

## 최종 디렉토리 구조 (목표)

```
RAGToySys/
├── app/
│   └── cli.py
├── config.py
├── extraction/
│   ├── architecture.py
│   └── figures.py
├── generation/
│   ├── agent.py
│   └── chain.py
├── ingestion/
│   ├── ocr/
│   ├── chunker.py
│   ├── cleaner.py
│   └── pipeline.py
├── paper/
│   ├── arxiv.py
│   ├── ingestor.py
│   ├── ranker.py
│   └── semantic_scholar.py
├── providers/
│   ├── base.py
│   ├── claude.py
│   ├── ollama.py
│   └── __init__.py
├── retrieval/
│   ├── bm25_retriever.py
│   ├── embedder.py
│   ├── hybrid.py
│   ├── reranker.py
│   ├── retriever.py
│   └── vectorstore.py
└── training/
    ├── data_prep.py
    ├── finetune.py
    └── export.py
```
