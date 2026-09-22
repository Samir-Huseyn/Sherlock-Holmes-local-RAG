# Sherlock Holmes RAG System 🕵️

A local Retrieval-Augmented Generation (RAG) system that answers questions based on the complete Sherlock Holmes canon by Arthur Conan Doyle. 100% offline, running with Ollama + Qdrant.

## Architecture

*1. Ingestion (ingest.py):*
- Loads 12 Sherlock Holmes books from books/
- Chunks text into 800-character pieces (100 char overlap filter)
- Converts each chunk to a 384-dim vector using all-MiniLM-L6-v2
- Stores 4203 vectors in Qdrant

*2. Retrieval + Generation (sherlock_rag.py):*
- User question -> embedding
- Cosine similarity search in Qdrant (top 5 chunks)
- Context + Question -> Llama3 (via Ollama) -> Answer

This solves the context window problem: Instead of feeding 1M+ tokens, we only feed the 5 most relevant paragraphs.

## Tech Stack
- *LLM:* Llama3 8B (Ollama)
- *Embeddings:* SentenceTransformers all-MiniLM-L6-v2
- *Vector DB:* Qdrant (local)
- *Language:* Python

## Installation & Run

1. Install Ollama and pull models:
```bash
ollama pull llama3