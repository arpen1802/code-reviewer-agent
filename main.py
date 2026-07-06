"""
main.py — CLI entry point for the Code Reviewer Agent.

Usage:
    # Index a codebase for RAG-powered, repository-aware review:
    python main.py --index /path/to/repo

    # Review the uncommitted changes in a repo (diff review mode):
    python main.py --diff /path/to/repo [base_ref]

    # Review a file:
    python main.py sample_code.py

    # Review a code screenshot (multimodality):
    python main.py --image screenshot.png

    # Paste code directly (interactive mode):
    python main.py
"""

import sys
from agents.orchestrator import run_orchestrator
from codebase_index import index_codebase
from tools import get_git_diff
from guardrails import is_input_clean
from observability import flush as flush_langfuse, is_enabled as langfuse_enabled


def main():
    print("=" * 60)
    print("         AI Code Reviewer — Multi-Agent")
    print("=" * 60)

    image_path = None
    user_input = None

    # ── Parse arguments ───────────────────────────────────────────────────────
    if len(sys.argv) > 1:
        if sys.argv[1] == "--index":
            # Index mode: build the codebase RAG index, then exit
            target = sys.argv[2] if len(sys.argv) > 2 else "."
            print(f"\nIndexing codebase: {target}\n")
            print(index_codebase(target))
            return

        elif sys.argv[1] == "--diff":
            # Diff review mode: review uncommitted changes with codebase context
            repo = sys.argv[2] if len(sys.argv) > 2 else "."
            base_ref = sys.argv[3] if len(sys.argv) > 3 else "HEAD"
            print(f"\nReviewing diff of '{repo}' against '{base_ref}'\n")
            diff = get_git_diff(repo_path=repo, base_ref=base_ref)
            if diff.startswith("Error") or diff.startswith("(no changes"):
                print(diff)
                return
            user_input = (
                f"Please review the following git diff (changes against {base_ref}). "
                f"Focus on what changed; use the codebase context to check whether "
                f"changes break callers or violate existing conventions.\n\n"
                f"```diff\n{diff}\n```"
            )

        elif sys.argv[1] == "--image" and len(sys.argv) > 2:
            # Multimodal mode: review a screenshot
            image_path = sys.argv[2]
            print(f"\nReviewing image: {image_path}")
            user_input = "Please review the Python code in the provided screenshot."

        else:
            # File path mode — read the file upfront so the orchestrator has
            # the actual code for memory embedding (semantic search needs the code,
            # not just the filename)
            filepath = sys.argv[1]
            print(f"\nReviewing file: {filepath}\n")
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    code = f.read()
                user_input = f"Please review this code from file '{filepath}':\n\n```python\n{code}\n```"
            except FileNotFoundError:
                print(f"Error: File '{filepath}' not found.")
                return

    else:
        # Interactive paste mode
        print("\nPaste your Python code below.")
        print("When done, type 'END' on a new line and press Enter.\n")
        lines = []
        while True:
            line = input()
            if line.strip() == "END":
                break
            lines.append(line)
        code = "\n".join(lines)
        user_input = f"Please review this code:\n\n```python\n{code}\n```"

    # ── Input guardrail ───────────────────────────────────────────────────────
    clean, reason = is_input_clean(user_input)
    if not clean:
        print(f"\n⚠️  {reason}")
        return

    # ── Run the multi-agent orchestrator ──────────────────────────────────────
    print("\nOrchestrating multi-agent review...\n")
    try:
        review = run_orchestrator(user_input, image_path=image_path)
        print("\n" + review)
    finally:
        # Short-lived CLI process: flush any pending Langfuse traces before exit
        # so they show up in the UI promptly (no-op if Langfuse isn't configured).
        if langfuse_enabled():
            flush_langfuse()


if __name__ == "__main__":
    main()
