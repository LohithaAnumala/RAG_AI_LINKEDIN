# 🏥 Healthcare AI Chatbot (RAG + FastAPI + Groq)

Built an end-to-end **Retrieval-Augmented Generation (RAG)** based chatbot that answers healthcare-related queries using real-time context retrieval.

---

## 🔹 Key Features

* ⚡ FastAPI backend with a simple UI
* 🌐 Dynamic data retrieval using Wikipedia API
* 🧠 Semantic chunking for improved context understanding
* 🔍 Hybrid retrieval (BM25 + Vector Search - in progress)
* 🤖 LLM integration via Groq (LLaMA 3.1 models)
* 📊 Evaluation metrics: Faithfulness & Relevance (hallucination-aware)

---

## 🔹 How It Works

1. User submits a query
2. Query is processed and sent to Wikipedia
3. Relevant content is retrieved and chunked
4. Hybrid retrieval (BM25 + embeddings) identifies top chunks
5. Reranking improves context relevance
6. Context is passed to LLM (Groq)
7. Final grounded answer is generated with evaluation scores

---

## 🔹 Challenges Solved

* ❗ Retrieval ambiguity (e.g., generic queries like "cancer")
* ❗ Context mismatch due to naive chunking
* ❗ Hallucination control using strict prompting
* ❗ Latency vs accuracy trade-offs
* ❗ End-to-end API + frontend integration

---

## 🔹 Improvements in Progress

* 🚀 Fully optimized hybrid retrieval (BM25 + semantic search)
* 🧹 Better query preprocessing & intent detection
* 🧩 Advanced chunking strategies (sliding window + semantic)
* 📈 Enhanced evaluation (RAGAS / model-based scoring)

---

## 🔹 Tech Stack

**Python | FastAPI | FAISS | Sentence Transformers | Groq | Wikipedia API**

---

💡 Focus: Building a **robust, real-world RAG system** that minimizes hallucinations and improves answer reliability — not just a basic chatbot.
