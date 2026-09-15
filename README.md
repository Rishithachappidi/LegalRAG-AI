# LegalRAG-AI

### Intelligent Legal Document Analysis using Sentence-BERT, FAISS, DCBD and LLaMA

LegalRAG-AI is a Retrieval-Augmented Generation (RAG) system for legal documents. Upload one or more legal PDFs (contracts, agreements, policies), ask a question in plain English, and the system retrieves the most relevant clauses using semantic search and generates a structured, citation-grounded legal analysis using an LLM — with a built-in faithfulness/hallucination score for every answer.

The system combines:

- PDF text extraction (PyMuPDF)
- Rule-based clause-level document chunking
- Sentence-BERT semantic embeddings (`all-MiniLM-L6-v2`)
- FAISS vector similarity search (cosine similarity via inner product)
- Relevance validation (rejects out-of-scope questions)
- **DCBD** — a novel cumulative-similarity–based adaptive clause selection algorithm (replaces fixed top-k)
- LLaMA-based legal report generation (via Groq API, with a local Transformers fallback)
- Faithfulness / hallucination evaluation using embedding cosine similarity
- Interactive Streamlit dashboard

The core objective is to reduce irrelevant retrieval and unsupported ("hallucinated") LLM answers by grounding every generated answer strictly in clauses retrieved from the user's own documents.

---

##  Table of Contents

1. [Problem Statement](#-problem-statement)
2. [Key Features](#-key-features)
3. [System Architecture](#-system-architecture)
4. [Complete Workflow](#-complete-workflow)
5. [Detailed Methodology](#-detailed-methodology)
6. [Technology Stack](#-technology-stack)
7. [Project Structure](#-project-structure)
8. [Requirements](#-requirements)
9. [Installation](#-installation)
10. [Environment Configuration](#-environment-configuration)
11. [How to Run](#-how-to-run)
12. [How to Use](#-how-to-use)
13. [Evaluation Metrics](#-evaluation-metrics)
14. [Why DCBD](#-why-dcbd)
15. [Advantages](#-advantages)
16. [Limitations](#-limitations)
17. [Future Enhancements](#-future-enhancements)
18. [Troubleshooting](#-troubleshooting)
19. [Disclaimer](#-disclaimer)
20. [Author](#-author)

---

##  Problem Statement

Legal documents such as agreements, contracts, policies, and legal-service documents often contain a large number of clauses. Finding the correct information manually is time-consuming, and traditional keyword search fails when the user's wording differs from the document's wording.

For example, a user may ask:

> "Can either party end the agreement before the contract expires?"

while the document says:

> "Either party may terminate this agreement prior to the expiration date..."

A keyword search will often miss this connection. LegalRAG-AI uses semantic embeddings to understand the *meaning* of both the question and the clauses, so it can retrieve the right clause regardless of exact phrasing — and then constrains the LLM to answer using only those retrieved clauses.

---

##  Key Features

- **Multi-document upload** — analyze and query several PDFs in a single session.
- **Automatic clause segmentation** — splits documents on legal heading patterns (numbered clauses, `ARTICLE`, `SECTION`, `CLAUSE`).
- **Semantic retrieval** — Sentence-BERT embeddings + FAISS `IndexFlatIP` for cosine-similarity search.
- **Relevance validation** — rejects questions unrelated to the uploaded documents before any LLM call is made.
- **DCBD adaptive clause selection** — picks *as many clauses as the question actually needs* instead of a fixed top-k.
- **Grounded generation** — the LLM prompt explicitly forbids inventing facts, dates, penalties, or obligations not present in the retrieved clauses, and requires clause-ID citations.
- **Faithfulness scoring** — every answer is scored for how semantically consistent it is with the retrieved context, with an estimated hallucination risk.
- **Graceful degradation** — if no LLM backend is available, a rule-based fallback report is generated directly from the selected clauses instead of failing.
- **Downloadable report** — export the question, answer, and evaluation metrics as a `.txt` file.
- **Caching** — embeddings, FAISS indices, and extracted text are cached to disk so re-analyzing the same document is instant.

---

##  System Architecture

```text
Legal PDF
     ↓
PDF Text Extraction (PyMuPDF)
     ↓
Clause-Level Chunking (regex heading detection)
     ↓
Sentence-BERT Embeddings (all-MiniLM-L6-v2)
     ↓
FAISS Vector Index (IndexFlatIP, per document)
     ↓
User Question
     ↓
Semantic Retrieval (top-k per document, merged & sorted)
     ↓
Relevance Validation (top-score / avg-score thresholds)
     ↓
DCBD Cumulative-Similarity Selection
     ↓
Selected Legal Clauses
     ↓
LLaMA (Groq API or local Transformers)
     ↓
Structured Legal Analysis Report
     ↓
Faithfulness / Hallucination Evaluation
```

---

## Complete Workflow

**Phase 1 — Knowledge base creation** (runs once per uploaded PDF, on clicking **Analyze Documents**):

1. Extract raw text from the PDF with PyMuPDF.
2. Split the text into clauses using heading-pattern regex.
3. Encode each clause into a 384-dimensional embedding with Sentence-BERT.
4. Build (or load, if already cached) a FAISS `IndexFlatIP` index for the document, plus a metadata JSON mapping vector positions back to clause text.

**Phase 2 — Question answering** (runs on clicking **Search**):

1. Embed the user's question with the same Sentence-BERT model.
2. Search each uploaded document's FAISS index for its top-5 most similar clauses, merge all results, and take the overall top 10.
3. Validate relevance — if the best match is too weak, stop and tell the user the question isn't related to the uploaded documents.
4. Run DCBD to select the smallest set of clauses that together cover enough of the cumulative similarity mass.
5. Build a constrained prompt and send it to the LLM (Groq/LLaMA, or local fallback).
6. Score the generated answer's faithfulness against the selected clauses.
7. Display the report, metrics, and offer a downloadable `.txt` export.

---

##  Detailed Methodology

### 1. PDF Extraction (`preprocessing/pdf_extractor.py`)
Uses PyMuPDF (`fitz`) to read every page of the uploaded PDF and concatenate the extracted text. The raw text is cached to `extracted_text/<name>.txt`.

### 2. Clause Chunking (`preprocessing/clause_chunker.py`)
Splits the document text with a regex that looks ahead for legal heading patterns — numbered clauses like `1.` or `2.1`, or the keywords `ARTICLE`, `SECTION`, `CLAUSE` — and discards fragments shorter than 40 characters. Each surviving clause is saved with a `clause_id` and `document` name to `chunks/<name>_chunks.json`.

### 3. Sentence-BERT Embeddings (`models/sentence_bert.py`)
Loads `all-MiniLM-L6-v2` from `sentence-transformers` (cached via `st.cache_resource` so it loads once per session) and encodes each clause into a normalized 384-dim vector. Embeddings are cached as `.npy` files under `embeddings/`.

### 4. FAISS Vector Store (`vectorstore/faiss_manager.py`)
Builds a `faiss.IndexFlatIP` (inner product, which is equivalent to cosine similarity since embeddings are normalized) per document, and persists it to `faiss_index/<name>.index` alongside a metadata JSON. Existing indices are reused rather than rebuilt.

### 5. Semantic Retrieval (`retrieval/retrieval.py`)
Embeds the query with the same Sentence-BERT model, searches each document's FAISS index for its top-`k` matches, and returns clause text, document name, clause ID, and similarity score. Index/metadata are cached in memory per session.

### 6. Relevance Validation (`retrieval/relevance_validator.py`)
Rejects the query before generation if either:
- the top retrieved similarity score is below `0.30`, or
- the average similarity across retrieved results is below `0.20`.

This prevents the LLM from ever being asked to answer a question that isn't actually covered by the uploaded documents.

### 7. DCBD — Dynamic Clause Boundary Detection (`retrieval/dcbd.py`)
The project's core retrieval-selection contribution. Instead of always feeding the LLM a fixed number of clauses:

1. Sort retrieved clauses by similarity score, descending.
2. Compute the total similarity mass across all retrieved clauses.
3. Walk down the sorted list, accumulating clauses one at a time, until the cumulative similarity reaches a **coverage threshold (75%)** — subject to a minimum of **3** and a maximum of **6** clauses.

This means a narrow, well-matched question pulls in only a few highly relevant clauses, while a broader question that spreads similarity across many clauses pulls in more — reducing both noisy over-retrieval and under-retrieval.

### 8. LLaMA Legal Analysis (`llm/llama_manager.py`)
Builds a strict prompt instructing the model to answer **only** from the provided clauses (no invented facts, dates, penalties, or obligations), to cite clause IDs (e.g. `[Clause 3]`), and to structure the answer as: Executive Summary → Relevant Legal Provisions → Detailed Legal Analysis → Synthesis → Limitations → Conclusion.

Two backends are supported, selected automatically:
- **Groq API** (`llama-3.1-8b-instant`) — used if `GROQ_API_KEY` is set. This is the primary, recommended path.
- **Local Transformers fallback** — used if Groq is unavailable and `transformers` + `torch` are installed (defaults to `distilgpt2`, configurable via `LOCAL_LLM_MODEL`).

If neither is available, the app falls back to a non-LLM report built directly from the DCBD-selected clause texts (`build_fallback_report` in `app.py`).

### 9. Faithfulness & Hallucination Evaluation (`evaluation/faithfulness.py`)
Embeds the full generated answer and the full selected-clause context with Sentence-BERT, and computes their cosine similarity:

- **Faithfulness score** = `similarity × 100` (0–100%)
- **Estimated hallucination risk** = `100 − faithfulness score`
- **Answer-context similarity** = raw cosine similarity (0–1)
- **Qualitative label**: Excellent (≥85%) / Good (≥70%) / Moderate (≥50%) / Low grounding — high hallucination risk (<50%)

---

##  Technology Stack

| Layer | Technology |
|---|---|
| Frontend / UI | Streamlit |
| PDF parsing | PyMuPDF (`fitz`) |
| Embeddings | Sentence-Transformers (`all-MiniLM-L6-v2`) |
| Vector search | FAISS (`faiss-cpu`, `IndexFlatIP`) |
| LLM (primary) | Groq API (`llama-3.1-8b-instant`) |
| LLM (fallback) | HuggingFace Transformers (local, e.g. `distilgpt2`) + PyTorch |
| Numerical computing | NumPy |
| Config | `python-dotenv` |
| Language | Python |

---

##  Project Structure

```text
LegalRAG-AI/
├── app.py                          # Streamlit app — orchestrates the full pipeline
├── requirements.txt
├── .gitignore
│
├── preprocessing/
│   ├── pdf_extractor.py            # PyMuPDF text extraction
│   └── clause_chunker.py           # Regex-based clause segmentation
│
├── models/
│   └── sentence_bert.py            # Sentence-BERT loading + embedding generation
│
├── vectorstore/
│   └── faiss_manager.py            # FAISS index build/load
│
├── retrieval/
│   ├── retrieval.py                # Query embedding + FAISS search
│   ├── relevance_validator.py      # Top-score / avg-score relevance gate
│   └── dcbd.py                     # Dynamic Clause Boundary Detection
│
├── llm/
│   └── llama_manager.py            # Groq / local-Transformers backend + prompt builder
│
├── evaluation/
│   └── faithfulness.py             # Embedding-similarity faithfulness scoring
│
├── extracted_text/                 # Cached raw text per uploaded PDF (generated)
├── chunks/                         # Cached clause JSON per document (generated)
├── embeddings/                     # Cached .npy embeddings per document (generated)
└── faiss_index/                    # Cached FAISS indices + metadata (generated)
```

> The `extracted_text/`, `chunks/`, `embeddings/`, and `faiss_index/` folders are populated automatically the first time you analyze a document, and are reused on subsequent runs so you don't have to re-embed unchanged files.

---

##  Requirements

- Python 3.9+ recommended
- pip

From `requirements.txt`:

```text
streamlit
PyMuPDF
numpy
faiss-cpu
sentence-transformers
groq
python-dotenv
```

> **Note:** for the local-LLM fallback path (used only if you don't set `GROQ_API_KEY`), you also need `transformers` and `torch`, which are **not** in `requirements.txt` by default:
> ```bash
> pip install transformers torch
> ```

---

##  Installation

```bash
# 1. Clone the repository
git clone https://github.com/yakkalakarthikeya/LegalRAG-AI.git
cd LegalRAG-AI

# 2. (Recommended) create a virtual environment
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

##  Environment Configuration

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here

# Optional — only used if GROQ_API_KEY is not set and transformers/torch are installed
LOCAL_LLM_MODEL=distilgpt2
```

Get a free Groq API key at [console.groq.com](https://console.groq.com/). If you don't set `GROQ_API_KEY` and don't install `transformers`/`torch`, the app still runs — it will fall back to a clause-based summary report instead of an LLM-generated one, with a warning shown in the UI.

---

##  How to Run

```bash
streamlit run app.py
```

This opens the app in your browser (by default at `http://localhost:8501`). On first launch, Streamlit will download the `all-MiniLM-L6-v2` Sentence-BERT model automatically.

---

##  How to Use

1. **Upload documents** — drag and drop one or more legal PDFs into the uploader.
2. **Click " Analyze Documents"** — the app extracts text, chunks clauses, generates embeddings, and builds a FAISS index for each document (shown with progress/success messages and a preview of the first three clauses).
3. **Ask a question** — type a legal question about the uploaded documents in the "Ask a Legal Question" box.
4. **Click "Search"** — the app:
   - retrieves and displays the top matching clauses with similarity scores,
   - runs DCBD to narrow down to the most relevant subset,
   - generates a structured legal analysis report,
   - displays faithfulness, hallucination risk, and answer-context similarity metrics.
5. **Download the report** — use the " Download Legal Report (.txt)" button to export the question, answer, and evaluation metrics.

If your question isn't related to the uploaded documents, the relevance validator will stop the pipeline before any LLM call and ask you to rephrase or upload the right document.

---

##  Evaluation Metrics

| Metric | Meaning |
|---|---|
| **Faithfulness Score** | How semantically consistent the generated answer is with the retrieved clauses (0–100%) |
| **Estimated Hallucination Risk** | `100% − Faithfulness Score` |
| **Answer-Context Similarity** | Raw cosine similarity between answer and context embeddings (0–1) |
| **Retrieval scores** (per clause) | Cosine similarity between the question and each retrieved clause, shown alongside every clause |

---

##  Why DCBD

Fixed top-k retrieval has two failure modes: too few clauses for broad questions, or too many irrelevant clauses for narrow ones, which dilutes the LLM's context and increases hallucination risk. DCBD instead selects clauses based on how much of the *cumulative similarity mass* they cover (75% coverage, bounded between 3 and 6 clauses), so the amount of context adapts to how concentrated or spread out the relevant information actually is in the document.

---

##  Advantages

- Retrieval is meaning-based, not keyword-based — differently worded questions can still match the right clause.
- The LLM is explicitly constrained to the retrieved context, reducing fabricated legal claims.
- Adaptive clause selection (DCBD) instead of a one-size-fits-all top-k.
- Works even without an LLM API key, via the local-model or clause-based fallback paths.
- Per-document caching avoids redundant re-processing of the same PDF.

##  Limitations

- Clause segmentation relies on regex heading patterns and may not perfectly split documents with unconventional formatting.
- The local-model fallback (`distilgpt2` by default) produces much lower-quality analysis than the Groq-hosted LLaMA path.
- Faithfulness scoring is an embedding-similarity proxy, not a verified fact-checking mechanism.
- Not a substitute for professional legal advice (see [Disclaimer](#-disclaimer)).

##  Future Enhancements

- Support for additional document formats (DOCX, scanned/OCR PDFs).
- Persistent multi-user knowledge bases instead of session-scoped uploads.
- More granular citation linking (highlighting exact source text in-app).
- Configurable LLM backends beyond Groq/local Transformers.

---

##  Troubleshooting

| Issue | Fix |
|---|---|
| `LLM initialization failed` warning in the UI | Set `GROQ_API_KEY` in `.env`, or install `transformers` + `torch` for the local fallback |
| "The question is not related to the uploaded legal documents" | Rephrase the question to more closely match the document's content, or confirm the right PDF is uploaded |
| Slow first run | The Sentence-BERT model download and initial embedding generation happen once; subsequent runs reuse cached embeddings/indices |
| Import errors on startup | Re-run `pip install -r requirements.txt` inside your active virtual environment |

---

##  Disclaimer

LegalRAG-AI is a research/engineering project for retrieval-augmented legal document analysis. It is **not** a substitute for professional legal advice. Always consult a qualified lawyer for actual legal decisions.

---


