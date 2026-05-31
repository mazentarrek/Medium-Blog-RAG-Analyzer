# LangChain RAG Demo

A simple Retrieval-Augmented Generation (RAG) application built with LangChain, OpenAI Embeddings, Pinecone, and Groq.

## Features

* Document ingestion and chunking
* OpenAI embedding generation
* Pinecone vector storage
* Semantic document retrieval
* Answer generation using Llama 3.3 70B via Groq

## Workflow

```text
Documents
   ↓
Chunking
   ↓
Embeddings
   ↓
Pinecone
   ↓
Retriever
   ↓
LLM
   ↓
Answer
```

## Setup

Create a `.env` file:

```env
OPEN_API_KEY=your_openai_api_key
GROQ_API_KEY=your_groq_api_key
INDEX_name=your_pinecone_index_name
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run

Index documents:

```bash
python ingestion.py
```

Query the knowledge base:

```bash
python retrieval.py
```

## Purpose

This project demonstrates the core concepts of RAG, including indexing, retrieval, vector databases, and context-aware answer generation using LLMs.
