from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

import wikipedia
import time
import logging

from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi

from groq import Groq

# ------------------ INIT ------------------

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)

# ------------------ INPUT ------------------

class Query(BaseModel):
    question: str

# ------------------ MODELS ------------------

embedder = SentenceTransformer("all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

groq_client = Groq(api_key="")

# ------------------ WIKI (FIXED) ------------------

def fetch_wikipedia(query):
    try:
        # 🔥 exact page first
        try:
            page = wikipedia.page(query, auto_suggest=False)
            print("\n✅ EXACT PAGE:", page.title)
            return page.content[:2000]
        except:
            pass

        results = wikipedia.search(query)

        # 🎯 try exact title match
        for title in results:
            if query.lower() == title.lower():
                page = wikipedia.page(title)
                print("\n🎯 MATCHED PAGE:", title)
                return page.content[:2000]

        # fallback
        for title in results[:5]:
            try:
                page = wikipedia.page(title)
                print("\n⚠️ FALLBACK PAGE:", title)
                return page.content[:2000]
            except:
                continue

        return None
    except:
        return None

# ------------------ CHUNKING ------------------

def semantic_chunk(text, chunk_size=300, overlap=50):
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)

    return chunks

# ------------------ QUERY TYPE ------------------

def is_definition_query(query):
    q = query.lower()
    return any(x in q for x in ["what is", "define", "meaning"])

# ------------------ RETRIEVAL ------------------

def hybrid_retrieval(chunks, query):

    print("\n🔍 QUERY:", query)
    print("📊 TOTAL CHUNKS:", len(chunks))

    tokenized = [c.lower().split() for c in chunks]
    bm25 = BM25Okapi(tokenized)
    bm25_scores = bm25.get_scores(query.lower().split())

    chunk_embeds = embedder.encode(chunks)
    query_embed = embedder.encode([query])[0]

    import numpy as np
    scores = []

    for i, chunk in enumerate(chunks):
        sim = np.dot(chunk_embeds[i], query_embed) / (
            np.linalg.norm(chunk_embeds[i]) * np.linalg.norm(query_embed)
        )

        final_score = 0.5 * bm25_scores[i] + 0.5 * sim
        scores.append((chunk, final_score))

    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    top_chunks = [s[0] for s in scores[:5]]

    print("\n📦 TOP CHUNKS:")
    for i, c in enumerate(top_chunks):
        print(f"\nChunk {i+1}:\n{c[:200]}...")

    return top_chunks

# ------------------ RERANK ------------------

def rerank(query, chunks):

    pairs = [(query, c) for c in chunks]
    scores = reranker.predict(pairs)

    ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)

    top = [r[0] for r in ranked[:3]]

    print("\n🏆 FINAL CHUNKS:")
    for i, c in enumerate(top):
        print(f"\nFinal {i+1}:\n{c[:200]}...")

    return top

# ------------------ CONTEXT ------------------

def build_context(chunks):
    return "\n".join(chunks)

# ------------------ LLM ------------------

def generate_answer(context, question):

    prompt = f"""
You are a strict medical assistant.

RULES:
- Answer ONLY from the context
- If definition is asked, give exact definition
- Do NOT guess
- If not found, say "I don't know"

Context:
{context}

Question:
{question}

Answer:
"""

    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1
    )

    return response.choices[0].message.content.strip()

# ------------------ EVALUATION ------------------

def evaluate_faithfulness(context, answer):
    score = reranker.predict([(answer, context)])[0]
    return "High" if score > 0.5 else "Low"

def evaluate_relevance(question, answer):
    score = reranker.predict([(question, answer)])[0]
    return "High" if score > 0.5 else "Low"

# ------------------ PIPELINE ------------------

def rag_pipeline(query):

    text = fetch_wikipedia(query)

    if not text:
        return {
            "answer": "No data found",
            "faithfulness": "Low",
            "relevance": "Low"
        }

    chunks = semantic_chunk(text)

    # 🔥 definition priority
    if is_definition_query(query):
        print("\n🎯 DEFINITION MODE")
        chunks = chunks[:2] + chunks

    retrieved = hybrid_retrieval(chunks, query)
    reranked = rerank(query, retrieved)

    context = build_context(reranked)

    print("\n📄 FINAL CONTEXT:\n", context[:500])

    answer = generate_answer(context, query)

    f = evaluate_faithfulness(context, answer)
    r = evaluate_relevance(query, answer)

    return {
        "answer": answer,
        "faithfulness": f,
        "relevance": r
    }

# ------------------ API ------------------

@app.post("/ask")
def ask(q: Query):

    start = time.time()

    result = rag_pipeline(q.question)

    latency = round(time.time() - start, 2)
    result["latency"] = f"{latency}s"

    return result

# ------------------ FRONTEND ------------------

@app.get("/")
def home():
    return FileResponse("index.html")
