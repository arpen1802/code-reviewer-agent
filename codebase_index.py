"""
codebase_index.py — RAG over the target codebase.

This is different from memory.py (which remembers *past reviews*).
This module indexes the *code being reviewed* — the repository itself —
so agents can retrieve related code while reviewing:

  - "Who calls this function?"
  - "How does this codebase usually handle errors?"
  - "Is there an existing helper that does the same thing?"

How it works:
  1. index_codebase(root) walks the repo, parses every .py file with `ast`,
     and splits it into semantic chunks (one per top-level function / class,
     plus a module-level chunk for imports & constants).
  2. Each chunk is embedded (same embedder as memory.py) and stored in a
     dedicated ChromaDB collection with path + line metadata.
  3. search_codebase(query) embeds the query and returns the most similar
     chunks, formatted as `path:start-end` + code excerpt.

The agents call search_codebase as a tool during review, which makes the
review context-aware: findings can reference actual callers, conventions,
and duplicated logic elsewhere in the repository.
"""

import ast
import os

import chromadb

from memory import DB_DIR, _embed
from observability import observed

COLLECTION_NAME = "codebase_index"

# Directories that are never part of the reviewed source
SKIP_DIRS = {
    ".git", "__pycache__", ".venv", "venv", "ai-agent", "env",
    "node_modules", "chroma_db", ".mypy_cache", ".pytest_cache",
    "logs", "results",
}

MAX_CHUNK_CHARS = 3000   # keep embeddings + retrieved context bounded
MAX_RESULT_CHARS = 1200  # per-chunk excerpt returned to the agent


def _get_collection():
    client = chromadb.PersistentClient(path=DB_DIR)
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def _iter_python_files(root: str):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for name in filenames:
            if name.endswith(".py"):
                yield os.path.join(dirpath, name)


def _chunk_file(filepath: str, rel_path: str) -> list[dict]:
    """
    Splits one Python file into semantic chunks using the AST:
      - one chunk per top-level function / async function / class
      - one "module" chunk with everything else (imports, constants, script code)

    Returns a list of {id, document, metadata} dicts ready for ChromaDB.
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            source = f.read()
        tree = ast.parse(source)
    except (SyntaxError, UnicodeDecodeError, OSError):
        return []  # skip unparseable files rather than failing the whole index

    lines = source.splitlines()
    chunks = []
    covered_lines: set[int] = set()

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            start, end = node.lineno, node.end_lineno or node.lineno
            covered_lines.update(range(start, end + 1))
            segment = "\n".join(lines[start - 1:end])[:MAX_CHUNK_CHARS]
            kind = "class" if isinstance(node, ast.ClassDef) else "function"
            chunks.append({
                "id": f"{rel_path}::{node.name}",
                "document": f"# {rel_path}:{start}-{end} ({kind} {node.name})\n{segment}",
                "metadata": {
                    "path": rel_path, "symbol": node.name, "kind": kind,
                    "start_line": start, "end_line": end,
                },
            })

    # Module-level chunk: imports, constants, top-level statements
    module_lines = [
        line for i, line in enumerate(lines, start=1)
        if i not in covered_lines and line.strip()
    ]
    if module_lines:
        segment = "\n".join(module_lines)[:MAX_CHUNK_CHARS]
        chunks.append({
            "id": f"{rel_path}::__module__",
            "document": f"# {rel_path} (module-level code)\n{segment}",
            "metadata": {
                "path": rel_path, "symbol": "__module__", "kind": "module",
                "start_line": 1, "end_line": len(lines),
            },
        })

    return chunks


@observed
def index_codebase(root: str = ".") -> str:
    """
    Indexes every Python file under `root` into the codebase collection.
    Rebuilds from scratch each run — simple and always consistent
    (a full reindex of a mid-size repo takes seconds).

    Returns a human-readable summary string.
    """
    root = os.path.abspath(root)
    collection = _get_collection()

    # Drop previous index so deleted/renamed files don't leave stale chunks
    existing = collection.get()["ids"]
    if existing:
        collection.delete(ids=existing)

    all_chunks = []
    file_count = 0
    for filepath in _iter_python_files(root):
        rel_path = os.path.relpath(filepath, root)
        chunks = _chunk_file(filepath, rel_path)
        if chunks:
            file_count += 1
            all_chunks.extend(chunks)

    if not all_chunks:
        return f"No Python files found to index under {root}."

    # Embed in batches to keep request sizes reasonable
    BATCH = 32
    for i in range(0, len(all_chunks), BATCH):
        batch = all_chunks[i:i + BATCH]
        embeddings = _embed([c["document"] for c in batch])
        collection.add(
            ids=[c["id"] for c in batch],
            documents=[c["document"] for c in batch],
            embeddings=embeddings,
            metadatas=[c["metadata"] for c in batch],
        )

    return (
        f"Indexed {len(all_chunks)} chunks from {file_count} Python files "
        f"under {root}."
    )


@observed
def search_codebase(query: str, n_results: int = 5) -> str:
    """
    Retrieves the code chunks most relevant to the query from the indexed
    codebase. Use this while reviewing to find related code: callers of a
    changed function, existing helpers, error-handling conventions, or
    duplicated logic.

    Args:
        query: Natural language or code — e.g. a function name, a diff hunk,
               or "how are exceptions handled in this codebase".
        n_results: How many chunks to return (default 5).

    Returns:
        Formatted string of matching chunks with path:line locations,
        or a notice if the codebase has not been indexed yet.
    """
    collection = _get_collection()
    count = collection.count()
    if count == 0:
        return (
            "Codebase index is empty. Run `python main.py --index <path>` "
            "first to enable codebase-aware review."
        )

    query_vec = _embed([query[:2000]])
    results = collection.query(
        query_embeddings=query_vec,
        n_results=min(n_results, count),
        include=["documents", "metadatas", "distances"],
    )

    out = [f"=== Codebase context (top {len(results['documents'][0])} of {count} chunks) ==="]
    for doc, meta, dist in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        similarity = round((1 - dist) * 100, 1)
        out.append(
            f"\n--- {meta['path']}:{meta['start_line']}-{meta['end_line']} "
            f"({meta['kind']} {meta['symbol']}, similarity {similarity}%) ---"
        )
        out.append(doc[:MAX_RESULT_CHARS])

    return "\n".join(out)
