"""
main.py — CLI entry point for the Code Reviewer Agent.

Usage:
    # Review a file:
    python main.py sample_code.py

    # Or paste code directly (interactive mode):
    python main.py
"""

import sys
from agent import run_agent
from guardrails import is_input_clean


def main():
    print("=" * 60)
    print("         AI Code Reviewer — Day 1")
    print("=" * 60)

    # If a file path was passed as an argument, use that
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
        print(f"\nReviewing file: {filepath}\n")
        user_input = f"Please review the code in this file: {filepath}"

    # Otherwise, ask the user to paste code
    else:
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

    print("\nAgent is thinking...\n")
    review = run_agent(user_input)

    print("\n" + "=" * 60)
    print("REVIEW")
    print("=" * 60)
    print(review)


if __name__ == "__main__":
    main()
