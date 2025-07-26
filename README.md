# Bangla RAG Question Answering System

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline for answering questions from a Bangla textbook (PDF) using **locally-hosted Ollama with Mistral** and **LaBSE embeddings**. The goal is to retrieve relevant information from textbook content and generate coherent answers to natural language queries in Bangla.

---

## Setup Guide

Prerequisites :

- Python 3.8+
- Ollama installed: [https://ollama.com](https://ollama.com)
- Mistral model downloaded (`ollama run mistral`)

### ✅ Installation

```bash
# Clone the repository
https://github.com/Md-Saharan-Evan/Bangla_QA_Chatbot
cd Bangla_QA_Chatbot

# Create a virtual environment
python3 -m venv AI
source AI/bin/activate

# Install dependencies
pip install -r requirements.txt

# Prepare data
python prepare_data.py

# Run Ollama server (in new terminal)
ollama run mistral

# Start FastAPI server
uvicorn app.api:app --reload --port 8000

# Open frontend
open frontend/index.html
```

---

## 📦 Used Tools and Libraries

- `langchain` – vector search, embeddings
- `pdfplumber` – PDF text extraction
- `sentence-transformers` – LaBSE embedding model
- `chromadb` – persistent vector DB
- `FastAPI` – backend API
- `Ollama` – local LLM host for Mistral
- `requests` – API call to Ollama

---

## 📤 Sample Queries & Outputs

### Input (Bangla):

> "বিয়ের সময় কল্যাণীর প্রকৃত বয়স কত ছিল?"

### Output:

> "কল্যাণীর বিয়ের সময় বয়স ছিল ১৫ বছর।"

## 📡 API Documentation

### POST `/ask`

**Body:** `{ "question": "<your_question>" }`
**Response:** `{ "answer": "<generated_answer>" }`

### GET `/debug`

**Query param:** `?query=...`
**Response:** `{ "chunks": [<relevant_context>] }`

---

## 📊 Evaluation Matrix (Optional if Applicable)

- ❌ BLEU/ROUGE scores not computed (no reference dataset)
- ✅ Manual relevance inspection confirms most results are meaningful for direct factual questions

---

## 🧠 Mandatory Questions & Answers

### 1. **Text Extraction Method & Challenges**

We used `pdfplumber` for Bangla PDF extraction due to its reliable line-by-line parsing and unicode support. Yes, formatting was a challenge:

- Like, many pages contained MCQs and non-text elements
- We removed MCQs using regex filters to retain only prose content

### 2. **Chunking Strategy**

We used **character-based chunking** (`RecursiveCharacterTextSplitter`) with:

- `chunk_size=300`
- `chunk_overlap=50`

This balances chunk size with meaningfulness. Sentence-based chunking was avoided due to irregular punctuation in Bangla PDF text.

### 3. **Embedding Model & Rationale**

Used: `sentence-transformers/LaBSE`

- Supports over 100 languages including Bangla
- Captures semantic meaning across multilingual input
- Pretrained and ready for use with HuggingFace/Transformers

### 4. **Similarity Method and Storage**

- We use `ChromaDB` with cosine similarity.
- Embedding vectors are indexed and retrieved based on cosine closeness between the query vector and stored chunk vectors.

This is fast and suitable for small to medium scale semantic search.

### 5. **Ensuring Meaningful Comparison**

- Query and chunks are embedded using the same LaBSE model
- Only top-k relevant chunks are passed to the Mistral model
- The prompt clearly separates context and query to guide generation

If the query is vague, irrelevant chunks might be retrieved. In such cases, the model may give generic or incorrect answers.

### 6. **Result Relevance & Future Improvements**

- Most results were relevant for factual Bangla questions
- To improve:

  - Increase `top_k` to 5 or more
  - Use hybrid chunking (paragraph + char-based)
  - Add Bangla QA dataset for fine-tuning
  - Try other embedding models like `indic-sbert` or `bengaliBERT`

---

---

## 📫 Contact

Maintainer: \[Your Name] – \[[mdsaharanevan20001@gmail.com](mailto:mdsaharanevan20001@email.com)] – GitHub: \[@ymd-saharan-evan]
