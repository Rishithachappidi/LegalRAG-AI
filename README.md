# ⚖️ LegalRAG-AI

## Intelligent Legal Document Analysis using Sentence-BERT, FAISS, DCBD and LLaMA

LegalRAG-AI is an intelligent legal document analysis system that uses Retrieval-Augmented Generation (RAG) to retrieve relevant legal clauses from uploaded PDF documents and generate context-grounded legal analysis using a Large Language Model.

The system combines:

- PDF text extraction
- Clause-level document chunking
- Sentence-BERT semantic embeddings
- FAISS vector similarity search
- Relevance validation
- Cumulative Similarity-based DCBD clause selection
- LLaMA-based legal report generation
- Faithfulness and hallucination evaluation
- Streamlit interactive dashboard

The main objective is to reduce irrelevant retrieval and unsupported LLM-generated answers by grounding the generation process in clauses retrieved from the user's legal documents.

---

# 📌 Table of Contents

1. [Project Overview](#-project-overview)
2. [Problem Statement](#-problem-statement)
3. [Motivation](#-motivation)
4. [Objectives](#-objectives)
5. [Key Features](#-key-features)
6. [System Architecture](#-system-architecture)
7. [Complete Workflow](#-complete-workflow)
8. [Detailed Methodology](#-detailed-methodology)
9. [PDF Extraction](#1-pdf-extraction)
10. [Clause Chunking](#2-clause-chunking)
11. [Sentence-BERT Embeddings](#3-sentence-bert-embeddings)
12. [FAISS Vector Store](#4-faiss-vector-store)
13. [Semantic Retrieval](#5-semantic-retrieval)
14. [Relevance Validation](#6-relevance-validation)
15. [DCBD Clause Selection](#7-dcbd-clause-selection)
16. [LLaMA Legal Analysis](#8-llama-legal-analysis)
17. [Faithfulness and Hallucination Evaluation](#9-faithfulness-and-hallucination-evaluation)
18. [Technology Stack](#-technology-stack)
19. [Project Structure](#-project-structure)
20. [Requirements](#-requirements)
21. [Installation](#-installation)
22. [Environment Configuration](#-environment-configuration)
23. [How to Run](#-how-to-run)
24. [How to Use](#-how-to-use)
25. [Example Workflow](#-example-workflow)
26. [Evaluation Metrics](#-evaluation-metrics)
27. [Why Sentence-BERT](#-why-sentence-bert)
28. [Why FAISS](#-why-faiss)
29. [Why Semantic Similarity](#-why-semantic-similarity)
30. [Why DCBD](#-why-dcbd)
31. [Hallucination Handling](#-hallucination-handling)
32. [Advantages](#-advantages)
33. [Limitations](#-limitations)
34. [Future Enhancements](#-future-enhancements)
35. [Security](#-security)
36. [Troubleshooting](#-troubleshooting)
37. [Research Contribution](#-research-contribution)
38. [Disclaimer](#-disclaimer)
39. [Author](#-author)

---

# 📖 Project Overview

Legal documents such as agreements, contracts, policies, and legal-service documents often contain a large number of clauses.

Finding the correct information manually can be time-consuming.

Traditional keyword-based search also has limitations because the wording of the user's question may be different from the wording used in the document.

For example, a user may ask:

> "Can either party end the agreement before the contract expires?"

The document might contain:

> "Either party may terminate this agreement prior to the expiration date..."

A keyword search may not always recognize the semantic relationship between these sentences.

LegalRAG-AI uses semantic embeddings to understand the meaning of both the question and the legal clauses.

The overall architecture is:

```text
Legal PDF
     ↓
PDF Text Extraction
     ↓
Clause-Level Chunking
     ↓
Sentence-BERT Embeddings
     ↓
FAISS Vector Index
     ↓
User Question
     ↓
Semantic Retrieval
     ↓
Relevance Validation
     ↓
DCBD Cumulative Similarity Selection
     ↓
Selected Legal Clauses
     ↓
LLaMA
     ↓
Legal Analysis Report
     ↓
Faithfulness / Hallucination Evaluation
