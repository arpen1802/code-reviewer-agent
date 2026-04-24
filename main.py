"""
main.py — CLI entry point for the Code Reviewer Agent.

Usage:
    # Review a file:
    python main.py sample_code.py

    # Review a code screenshot (multimodality):
    python main.py --image screenshot.png

    # Paste code directly (interactive mode):
    python main.py
"""

import sys
from agents.orchestrator import run_orchestrator
from guardrails import is_input_clean


def main():
    print("=" * 60)
    print("         AI Code Reviewer — Multi-Agent")
    print("=" * 60)

    image_path = None
    user_input = None

    # ── Parse arguments ───────────────────────────────────────────────────────
    if len(sys.argv) > 1:
        if sys.argv[1] == "--image" and len(sys.argv) > 2:
            # Multimodal mode: review a screenshot
            image_path = sys.argv[2]
            print(f"\nReviewing image: {image_path}")
            user_input = "Please review the Python code in the provided screenshot."

        else:
            # File path mode
            filepath = sys.argv[1]
            print(f"\nReviewing file: {filepath}\n")
            user_input = f"Please review the code in this file: {filepath}"

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
    review = run_orchestrator(user_input, image_path=image_path)

    print("\n" + review)


if __name__ == "__main__":
    main()
