"""MP2 · Mini-RAG — Ask Your Documents

RAG pipeline over 5 Sherlock Holmes story summaries.

    corpus/*.txt  →  chunks  →  embeddings  →  Qdrant
                                                  ↓
                              question  →  retrieve  →  answer + citations

Run sequence:
    pip install -r requirements.txt
    python mp2_rag.py ingest    # builds the collection (run once)
    python mp2_rag.py ask       # interactive Q&A loop
    python mp2_rag.py validate  # runs against data/predefined_questions.jsonl
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

load_dotenv()

# Windows terminals default to cp1252 which can't print → or ✓
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

# ── Configuration ────────────────────────────────────────────────────────

CORPUS_DIR        = Path(__file__).parent / "corpus"
DATA_DIR          = Path(__file__).parent / "data"
COLLECTION_NAME   = "mp2_sherlock"
EMBEDDING_MODEL   = "text-embedding-3-small"
EMBEDDING_DIM     = 1536
CHAT_MODEL        = "gpt-4o-mini"
TARGET_CHUNK_SIZE = 500
CHUNK_OVERLAP     = 80

openai_client = OpenAI(
    base_url=os.environ.get("VOCAREUM_OPENAI_BASE_URL"),
)
qdrant = QdrantClient(
    url=os.environ["QDRANT_URL"],
    api_key=os.environ.get("QDRANT_API_KEY") or None,
)


# ── Step 1: Load the corpus ──────────────────────────────────────────────

def load_corpus(corpus_dir: Path) -> list[dict[str, Any]]:
    docs = []
    for path in sorted(corpus_dir.glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        title = next(line.strip() for line in text.splitlines() if line.strip())
        docs.append({"source": path.name, "title": title, "text": text})
    return docs


# ── Step 2: Chunk each document ──────────────────────────────────────────

def chunk_document(doc: dict[str, Any]) -> list[dict[str, Any]]:
    """Paragraph-aware chunking with section-header detection."""
    paragraphs = [p.strip() for p in re.split(r"\n\n+", doc["text"]) if p.strip()]

    chunks: list[dict[str, Any]] = []
    current_section = "Introduction"
    buffer = ""

    for para in paragraphs:
        # Short lines without terminal punctuation are section headers
        if len(para) < 60 and para[-1] not in ".?!:":
            if buffer.strip():
                chunks.append(_make_chunk(doc, current_section, buffer))
                buffer = ""
            current_section = para
            continue

        if buffer and len(buffer) + len(para) + 2 > TARGET_CHUNK_SIZE:
            chunks.append(_make_chunk(doc, current_section, buffer))
            buffer = buffer[-CHUNK_OVERLAP:].strip() + "\n\n" + para
        else:
            buffer = (buffer + "\n\n" + para).strip() if buffer else para

    if buffer.strip():
        chunks.append(_make_chunk(doc, current_section, buffer))

    return chunks


def _make_chunk(doc: dict[str, Any], section: str, text: str) -> dict[str, Any]:
    return {
        "source": doc["source"],
        "title": doc["title"],
        "section": section,
        "text": text.strip(),
    }


# ── Step 3: Embed text ───────────────────────────────────────────────────

def embed_texts(texts: list[str]) -> list[list[float]]:
    response = openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
    )
    return [item.embedding for item in response.data]


# ── Step 4: Set up the Qdrant collection ────────────────────────────────

def setup_collection() -> None:
    if qdrant.collection_exists(COLLECTION_NAME):
        qdrant.delete_collection(COLLECTION_NAME)
    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
    )


# ── Step 5: Ingest chunks into Qdrant ───────────────────────────────────

def ingest_chunks(chunks: list[dict[str, Any]]) -> None:
    texts = [chunk["text"] for chunk in chunks]
    vectors = embed_texts(texts)
    points = [
        PointStruct(id=str(uuid.uuid4()), vector=vector, payload=chunk)
        for chunk, vector in zip(chunks, vectors)
    ]
    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)


# ── Step 6: Retrieve ─────────────────────────────────────────────────────

def retrieve(query: str, k: int = 3) -> list[dict[str, Any]]:
    query_vector = embed_texts([query])[0]
    response = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=k,
    )
    chunks = []
    for result in response.points:
        chunk = dict(result.payload)
        chunk["score"] = result.score
        chunks.append(chunk)
    return chunks


# ── Step 7: Generate the answer ──────────────────────────────────────────

SYSTEM_PROMPT = """You are a helpful assistant answering questions about a small
collection of Sherlock Holmes stories. You will be given the user's question and
several relevant excerpts. Use ONLY the provided excerpts to answer. If the
excerpts don't contain the answer, say so plainly. Cite the source (story title
+ section) in your answer."""


def answer(question: str, k: int = 3) -> dict[str, Any]:
    start = time.time()

    chunks = retrieve(question, k=k)

    context_parts = []
    for chunk in chunks:
        header = f"[Source: {chunk['title']} — {chunk['section']}]"
        context_parts.append(f"{header}\n{chunk['text']}")

    context = "\n\n---\n\n".join(context_parts)
    user_message = f"Context:\n{context}\n\nQuestion: {question}"

    response = openai_client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ],
        temperature=0.0,
    )

    latency_ms = int((time.time() - start) * 1000)

    return {
        "question":  question,
        "answer":    response.choices[0].message.content,
        "citations": [
            {"source": c["source"], "title": c["title"], "section": c["section"]}
            for c in chunks
        ],
        "latency_ms": latency_ms,
    }


# ── Validation harness (provided — do not modify) ────────────────────────

def validate_against(jsonl_path: Path) -> None:
    questions = [json.loads(line) for line in jsonl_path.read_text().splitlines() if line.strip()]
    print(f"\n  Validating {len(questions)} questions from {jsonl_path.name}…\n")

    hits = 0
    for q in questions:
        result = answer(q["question"], k=3)
        cited_sources = {cit["source"] for cit in result["citations"]}
        source_hit = q["expected_source"] in cited_sources

        ans_lower = result["answer"].lower()
        facts_hit = sum(1 for fact in q.get("expected_facts", []) if fact.lower() in ans_lower)
        facts_total = len(q.get("expected_facts", []))

        verdict = "✓" if source_hit else "✗"
        print(f"  {verdict} {q['id']}")
        print(f"      Q: {q['question']}")
        print(f"      Cited: {', '.join(cited_sources)}")
        print(f"      Expected: {q['expected_source']}")
        print(f"      Facts matched: {facts_hit}/{facts_total}")
        print(f"      Latency: {result.get('latency_ms', '?')}ms")
        print()
        if source_hit:
            hits += 1

    print(f"  Source-match: {hits}/{len(questions)}")


# ── CLI (provided — do not modify) ───────────────────────────────────────

def cmd_ingest() -> None:
    print("→ Loading corpus…")
    docs = load_corpus(CORPUS_DIR)
    print(f"  {len(docs)} documents loaded")

    print("→ Chunking…")
    all_chunks: list[dict[str, Any]] = []
    for doc in docs:
        chunks = chunk_document(doc)
        all_chunks.extend(chunks)
        print(f"  {doc['source']}: {len(chunks)} chunks")

    print(f"→ Total chunks: {len(all_chunks)}")
    print("→ Setting up Qdrant collection…")
    setup_collection()

    print("→ Ingesting…")
    ingest_chunks(all_chunks)
    print("\n✓ Done. Try: python mp2_rag.py ask")


def cmd_ask() -> None:
    print("Mini-RAG over the Sherlock Holmes corpus.")
    print("Type your question. Empty line or Ctrl-C to exit.\n")
    while True:
        try:
            q = input("? ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not q:
            return
        result = answer(q, k=3)
        print(f"\n{result['answer']}\n")
        print("  Sources:")
        for c in result["citations"]:
            print(f"    - {c['title']} — {c['section']}")
        print(f"  Latency: {result.get('latency_ms', '?')}ms\n")


def cmd_validate() -> None:
    validate_against(DATA_DIR / "predefined_questions.jsonl")
    learner_path = DATA_DIR / "learner_questions.jsonl"
    if learner_path.exists():
        first = json.loads(learner_path.read_text().splitlines()[0])
        if not first["question"].startswith("Replace this"):
            validate_against(learner_path)


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)
    cmd = sys.argv[1]
    if cmd == "ingest":
        cmd_ingest()
    elif cmd == "ask":
        cmd_ask()
    elif cmd == "validate":
        cmd_validate()
    else:
        print(f"Unknown command: {cmd}\n")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
