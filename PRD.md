# 📄 Product Requirements Document (PRD)

**Project Name:** Contexto  
**Tagline:** Deep document understanding, simplified.  
**Version:** 1.0.0  
**Status:** In Development  

---

## 1. Executive Summary
**Contexto** is an intelligent, high-performance web application designed to eliminate manual searching through complex documents. By letting users upload PDF files and engage in context-aware conversations, **Contexto** transforms static documents into dynamic, interactive knowledge hubs. Built with FastAPI, LangChain, ChromaDB, and open-source models via Hugging Face (or Google Gemini/OpenAI), it provides a fast, accurate, and cost-effective implementation of Retrieval-Augmented Generation (RAG).

---

## 2. Goals & Key Objectives
* **Grounded Insights:** Deliver rapid Q&A strictly backed by uploaded document context to eliminate LLM hallucinations.
* **Source Traceability:** Provide page-level and text-excerpt source citations for every answer generated.
* **Modular Architecture:** Design a clean FastAPI backend capable of swapping underlying embedding models, vector stores, or LLM providers with zero friction.
* **Cost Efficiency:** Prioritize free, open-source Hugging Face models (`bge-small-en-v1.5` for embeddings, `Mistral`/`Llama-3.2` for generation) to enable cost-free local development.

---

## 3. Technology Stack Architecture

| Layer | Component / Tool | Role |
| :--- | :--- | :--- |
| **Backend Framework** | **FastAPI** | Async RESTful API routes, Pydantic data validation, OpenAPI/Swagger auto-docs |
| **Orchestration Engine**| **LangChain** | Document loaders, text splitters, RAG chains, prompt templates |
| **Vector Database** | **ChromaDB** | Local persistent vector storage with per-session collection isolated namespaces |
| **Embedding Model** | **Hugging Face (`BAAI/bge-small-en-v1.5`)** | Converts document text chunks into 384-dimensional dense vector representations |
| **LLM Provider** | **Hugging Face Inference API / Gemini** | High-speed, context-grounded response generation |
| **PDF Parser** | **PyPDF / Unstructured** | Extracts raw text and page metadata from uploaded PDF documents |

---

## 4. System Workflow Architecture

```
[User PDF Upload] 
       │
       ▼
[FastAPI Ingestion Endpoint] ──> [Text Splitter] ──> [HF Embedding Engine] ──> [ChromaDB Vector Store]
                                                                                      │
[User Question] ───────────────> [Vector Query] ──> [Top-K Context Chunks] ─────────────┤
                                                                                      ▼
[User Output] <───────────────── [LLM Output] <─── [Prompt + Context Injection] ──────┘
```

---

## 5. Functional Requirements

### 5.1 Document Ingestion & Vector Indexing
* **PDF Upload Endpoint:** Accepts single/multi-page `.pdf` files via multi-part form upload.
* **Recursive Character Splitting:** Splits documents using `RecursiveCharacterTextSplitter` (`chunk_size = 1000`, `chunk_overlap = 200`) to preserve structural boundaries across pages.
* **Vector Indexing:** Converts text chunks into dense vector embeddings via Hugging Face and stores them inside persistent ChromaDB collections indexed by a unique `session_id`.

### 5.2 Retrieval-Augmented Generation (RAG) Pipeline
* **Similarity Search:** Retrieves the top-$k$ most relevant text blocks ($k=4$) using cosine similarity distance.
* **Strict Grounding Guardrails:** Enforces zero-hallucination policies via system prompting:
  > *"You are Contexto, an AI document assistant. Answer the user's question strictly using the provided context blocks. If the answer is not present in the context, explicitly state that the document does not contain enough information."*
* **Citation Delivery:** Returns structured answers along with reference payloads containing page numbers and excerpt previews.

### 5.3 Session & Memory Lifecycle Management
* **Multi-Turn Chat History:** Maintains short-term conversational context per document session using LangChain buffer memory.
* **Session Isolation:** Guarantees isolated vector store collections per user session to avoid cross-document data leaks.

---

## 6. API Endpoint Specification

| Route | Method | Payload | Response | Description |
| :--- | :--- | :--- | :--- | :--- |
| `/api/v1/context/ingest` | `POST` | `file: UploadFile` | `{ session_id: str, total_chunks: int }` | Ingests PDF, generates embeddings, and indexes chunks in ChromaDB. |
| `/api/v1/context/query` | `POST` | `{ session_id: str, query: str }` | `{ answer: str, sources: List[Source] }` | Performs vector search and returns grounded response with sources. |
| `/api/v1/context/clear` | `DELETE` | `{ session_id: str }` | `{ status: "success" }` | Flushes session vector collection and memory cache. |

---

## 7. Performance & Quality Benchmarks

* **Ingestion Throughput:** Ingests and indexes standard 20-page PDFs in $\le 5$ seconds.
* **Query Latency:** Returns RAG search and generation responses in $\le 3$ seconds.
* **File Constraints:** Enforces max file size limit of 20MB per upload during MVP testing.