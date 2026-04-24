"""
memory.py — Long-term memory for the Code Reviewer Agent.

Upgraded from Day 2 (flat JSON file) to use a vector database (ChromaDB).

The key difference:

  JSON approach (Day 2):
    - Store: append a review dict to a list
    - Retrieve: load the entire list, dump last 5 into the prompt
    - Problem: 200 past reviews → massive context, slow, expensive

  Vector DB approach (Day 5):
    - Store: embed the code + issues into a vector, save to ChromaDB
    - Retrieve: embed the *current* code, find the top 3 most similar past reviews
    - Result: agent gets *relevant* context, not just recent context

How embeddings work:
  An embedding is a list of ~380 numbers (a vector) that captures the
  "meaning" of a piece of text. Two similar texts (e.g., two files with
  a division-by-zero bug) will have vectors that point in the same direction.
  ChromaDB stores these vectors and can quickly find the closest ones —
  that's semantic search.

  We use ChromaDB's built-in sentence-transformers embedding function
  ('all-MiniLM-L6-v2'), which runs locally with no extra API calls.
"""

import json
import os
import shutil
from datetime import datetime

import chromadb
from chromadb import EmbeddingFunction, Documents, Embeddings
from google import genai
from dotenv import load_dotenv

load_dotenv()

# ChromaDB data lives in the user's home directory, not the project folder.
# Reason: ChromaDB uses SQLite under the hood, and SQLite requires OS-level file
# locking that doesn't work on network mounts or FUSE filesystems. Storing in
# home dir (~/.code_reviewer_db/) is the standard practice for local app data.
_HOME_DATA_DIR = os.path.join(os.path.expanduser("~"), ".code_reviewer_db")
DB_DIR     = os.path.join(_HOME_DATA_DIR, "chroma")
PREFS_FILE = os.path.join(_HOME_DATA_DIR, "preferences.json")

os.makedirs(DB_DIR, exist_ok=True)


def _embed_gemini(texts: list[str]) -> list[list[float]]:
    """
    Embeds texts using the Gemini text-embedding-004 model (768 dimensions).
    This is the production path — high quality, semantic understanding of code.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    client  = genai.Client(api_key=api_key)
    result  = client.models.embed_content(
        model="text-embedding-004",
        contents=texts,
    )
    return [emb.values for emb in result.embeddings]


def _embed_local(texts: list[str]) -> list[list[float]]:
    """
    Fallback: local bag-of-words embedding using random hash projections.

    How it works:
      1. Tokenise the text into code tokens (keywords, names, operators).
      2. Hash each token to one of 256 buckets using Python's hash().
      3. Count occurrences per bucket, then L2-normalise.

    This won't capture deep semantics (e.g. it won't know "division" ≈
    "ZeroDivisionError"), but it does capture code-level similarity:
    two files that both use open() / read() will score higher than a file
    about division. Good enough to demonstrate the concept.

    Produces 256-dimensional float vectors.
    """
    import re
    import math

    DIM = 256

    def tokenise(text: str) -> list[str]:
        # Split on whitespace, punctuation, operators — keep alphanumerics
        return re.findall(r"[A-Za-z_]\w*|\d+", text.lower())

    def vectorise(text: str) -> list[float]:
        tokens = tokenise(text)
        vec = [0.0] * DIM
        for tok in tokens:
            bucket = hash(tok) % DIM
            vec[bucket] += 1.0
        # L2 normalise so cosine similarity works correctly
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]

    return [vectorise(t) for t in texts]


def _embed(texts: list[str]) -> list[list[float]]:
    """
    Embed texts — tries Gemini API first, falls back to local embedder.

    In production (on a machine with unrestricted Gemini API access), the
    Gemini path is used. In sandboxed or offline environments, the local
    bag-of-words embedder activates automatically.
    """
    try:
        return _embed_gemini(texts)
    except Exception:
        return _embed_local(texts)


def _get_collection():
    """
    Returns the ChromaDB collection, creating it if it doesn't exist.

    We do NOT pass an embedding_function here — we use the "bring your own
    embeddings" pattern: we call Gemini ourselves in save_memory/load_memory
    and pass raw float vectors directly. This avoids embedding function
    version conflicts if the collection was previously created with a
    different embedder.
    """
    client = chromadb.PersistentClient(path=DB_DIR)
    return client.get_or_create_collection(
        name="code_reviews_v2",           # v2 = Gemini embeddings (vs default ONNX)
        metadata={"hnsw:space": "cosine"},
    )


def _load_prefs() -> dict:
    """Load user preferences from the sidecar JSON file."""
    if not os.path.exists(PREFS_FILE):
        return {}
    with open(PREFS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_prefs(prefs: dict) -> None:
    with open(PREFS_FILE, "w", encoding="utf-8") as f:
        json.dump(prefs, f, indent=2)


# ── Tools the agent can call ──────────────────────────────────────────────────

def load_memory(query: str = "") -> str:
    """
    Retrieves relevant past reviews from the vector database.

    If a query is provided (pass the code currently being reviewed),
    this performs a semantic search — it finds the past reviews whose
    code and issues are most similar to the current code. This means
    the agent gets relevant context, not just the most recent reviews.

    If no query is provided, returns the 3 most recently added reviews.

    Call this at the start of every review. Always pass the current
    code as the query argument for best results.

    Args:
        query: The code currently being reviewed (used for semantic search).

    Returns:
        Formatted string of relevant past reviews and user preferences.
    """
    collection = _get_collection()
    prefs = _load_prefs()

    lines = []

    if prefs:
        lines.append("=== User Preferences ===")
        for key, value in prefs.items():
            lines.append(f"  {key}: {value}")

    count = collection.count()
    if count == 0:
        lines.append("No review history yet — this appears to be the first session.")
        return "\n".join(lines) if lines else "No memory yet — this appears to be the first review session."

    lines.append(f"\n=== Relevant Past Reviews (semantic search over {count} stored) ===")

    if query.strip():
        # Semantic search: embed current code → find closest past reviews
        query_vec = _embed([query[:2000]])   # returns list of one vector
        results = collection.query(
            query_embeddings=query_vec,
            n_results=min(3, count),
            include=["documents", "metadatas", "distances"],
        )
        docs      = results["documents"][0]
        metas     = results["metadatas"][0]
        distances = results["distances"][0]

        for doc, meta, dist in zip(docs, metas, distances):
            similarity = round((1 - dist) * 100, 1)
            lines.append(f"\n  File: {meta['file']} | Date: {meta['date']} | Similarity: {similarity}%")
            if meta.get("issues"):
                lines.append(f"  Issues: {meta['issues']}")

    else:
        # Fallback: no query provided → return most recently added reviews
        all_results = collection.get(include=["metadatas"])
        recent_metas = all_results["metadatas"][-3:]
        for meta in recent_metas:
            lines.append(f"\n  File: {meta['file']} | Date: {meta['date']}")
            if meta.get("issues"):
                lines.append(f"  Issues: {meta['issues']}")

    return "\n".join(lines)


def save_memory(
    file_reviewed: str,
    issues_found: list[str],
    preference_notes: str = "",
    code_snippet: str = "",
) -> str:
    """
    Saves the result of a review to the vector database so future sessions
    can find it via semantic search.

    The document that gets embedded is the code + issues combined —
    so semantic search will match on both code patterns AND issue types.
    For example, two different files that both have division-by-zero bugs
    will have similar embeddings and will surface each other as context.

    Call this at the end of every review.

    Args:
        file_reviewed:    The name or path of the file that was reviewed.
        issues_found:     A list of the key issues found during the review.
        preference_notes: Observations about the user's coding style (optional).
        code_snippet:     The code that was reviewed — used to create the embedding.

    Returns:
        A confirmation message.
    """
    collection = _get_collection()

    issues_str = ", ".join(issues_found)

    # The embedded document combines code + issues so both dimensions are searchable
    document = (
        f"File: {file_reviewed}\n"
        f"Issues: {issues_str}\n"
        f"Code:\n{code_snippet[:1500]}"
    )

    # Generate the embedding via Gemini and store it directly
    embedding = _embed([document])[0]  # returns [[...]] → take first vector

    # Unique ID: filename + timestamp (handles reviewing the same file multiple times)
    review_id = f"{file_reviewed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    collection.add(
        documents=[document],
        embeddings=[embedding],          # raw float vector from Gemini
        ids=[review_id],
        metadatas=[{
            "file": file_reviewed,
            "date": datetime.now().strftime("%Y-%m-%d"),
            "issues": issues_str[:500],  # ChromaDB metadata has size limits
        }],
    )

    # User preferences live in a sidecar JSON file — no need to embed them
    if preference_notes:
        prefs = _load_prefs()
        prefs["notes"] = preference_notes
        prefs["last_updated"] = datetime.now().strftime("%Y-%m-%d")
        _save_prefs(prefs)

    return (
        f"Memory saved to vector DB: review of '{file_reviewed}' "
        f"with {len(issues_found)} issues recorded."
    )


def clear_memory() -> None:
    """
    Wipes all stored reviews and preferences.
    Used by the eval harness to isolate tasks from each other.
    """
    collection = _get_collection()
    all_ids = collection.get()["ids"]
    if all_ids:
        collection.delete(ids=all_ids)

    if os.path.exists(PREFS_FILE):
        os.remove(PREFS_FILE)
