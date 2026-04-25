"""
agents/orchestrator.py — The multi-agent orchestrator.

This is the "manager" agent. It doesn't do code review itself.
Instead, it:
  1. Delegates to three specialist sub-agents in parallel
  2. Collects their reports
  3. Merges everything into one final structured review

The sub-agents are treated like tools — the orchestrator calls them
by name, gets their output as a string, and builds on it.

This file also adds multimodality: the orchestrator can accept an
image (e.g. a screenshot of code) in addition to text, using Gemini's
vision capability to extract code from the image before delegating.
"""

import os
import concurrent.futures
from google import genai
from google.genai import types
from dotenv import load_dotenv
from memory import load_memory, save_memory
from agents.reviewer_agent import run_reviewer_agent
from agents.security_agent import run_security_agent
from agents.test_writer_agent import run_test_writer_agent

load_dotenv()


# ── Multimodality: extract code from an image ─────────────────────────────────

def extract_code_from_image(image_path: str) -> str:
    """
    Uses Gemini's vision capability to extract Python code from an image.
    This is the multimodality feature: the agent can accept a screenshot
    instead of (or alongside) text.

    Args:
        image_path: Path to a PNG or JPEG image file.

    Returns:
        Extracted Python code as a string, or an error message.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)

    try:
        with open(image_path, "rb") as f:
            image_bytes = f.read()
    except FileNotFoundError:
        return f"Error: Image file '{image_path}' not found."

    # Determine mime type from extension
    ext = os.path.splitext(image_path)[1].lower()
    mime_map = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}
    mime_type = mime_map.get(ext, "image/png")

    # Build a multimodal message: image + text instruction
    image_part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
    text_part = types.Part.from_text(
        "Extract all Python code visible in this image. "
        "Return only the code, no explanation."
    )

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[image_part, text_part],
    )
    return response.text


# ── Orchestrator ──────────────────────────────────────────────────────────────

def run_orchestrator(user_input: str, image_path: str | None = None) -> str:
    """
    Orchestrates a full multi-agent code review.

    Args:
        user_input: Code snippet or file path.
        image_path: Optional path to a code screenshot (multimodality).

    Returns:
        A merged final review from all three specialist agents.
    """

    # ── Step 0: Multimodality — extract code from image if provided ───────────
    if image_path:
        print(f"\n  [orchestrator] Extracting code from image: {image_path}")
        extracted_code = extract_code_from_image(image_path)
        print(f"  [orchestrator] Extracted {len(extracted_code)} chars of code")
        # Prepend extracted code to the user's input
        user_input = f"Extracted from screenshot:\n```python\n{extracted_code}\n```\n\n{user_input}"

    # ── Step 1: Load memory ───────────────────────────────────────────────────
    # Pass the current code as the query so ChromaDB does semantic search:
    # "find past reviews whose code is most similar to what we're reviewing now"
    print("\n  [orchestrator] Loading memory...")
    memory_context = load_memory(query=user_input[:2000])
    # Show what was retrieved so the user can see semantic search in action
    non_empty = [l for l in memory_context.splitlines() if l.strip()]
    first_line = non_empty[0] if non_empty else "none"
    print(f"  [orchestrator] Memory: {first_line}")
    full_input = f"Memory context:\n{memory_context}\n\n---\n\n{user_input}"

    # ── Step 2: Run all three agents in parallel ──────────────────────────────
    # This is the key multi-agent pattern: parallel execution.
    # Each agent runs independently and returns its report.
    print("\n  [orchestrator] Dispatching to specialist agents in parallel...\n")

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        future_review = executor.submit(run_reviewer_agent, full_input)
        future_security = executor.submit(run_security_agent, full_input)
        future_tests = executor.submit(run_test_writer_agent, full_input)

        print("  Waiting for all agents to finish...")
        review_report = future_review.result()
        security_report = future_security.result()
        test_report = future_tests.result()

    print("\n  [orchestrator] All agents done. Merging reports...")

    # ── Step 3: Merge reports ─────────────────────────────────────────────────
    # The orchestrator assembles the final output — it's the only one who sees
    # all three reports.
    merged = _merge_reports(review_report, security_report, test_report)

    # ── Step 4: Save memory ───────────────────────────────────────────────────
    # Pass code_snippet so the embedding captures the actual code —
    # this is what makes future semantic searches find similar bugs/patterns.
    filename = _extract_filename(user_input)
    issues_summary = [
        f"Quality: {_first_line(review_report)}",
        f"Security: {_first_line(security_report)}",
    ]
    save_memory(
        file_reviewed=filename,
        issues_found=issues_summary,
        code_snippet=user_input[:2000],  # embed the actual code, not just metadata
    )
    print("  [orchestrator] Memory saved.")

    return merged


def _merge_reports(review: str, security: str, tests: str) -> str:
    """Combines the three specialist reports into one readable final review."""
    review   = (review   or "(Reviewer agent returned no output.)").strip()
    security = (security or "(Security agent returned no output.)").strip()
    tests    = (tests    or "(Test writer agent returned no output.)").strip()
    return f"""
{'=' * 60}
 MULTI-AGENT CODE REVIEW
{'=' * 60}

━━━ CODE QUALITY (Reviewer Agent) ━━━━━━━━━━━━━━━━━━━━━━━━━━

{review}

━━━ SECURITY ANALYSIS (Security Agent) ━━━━━━━━━━━━━━━━━━━━━

{security}

━━━ SUGGESTED TESTS (Test Writer Agent) ━━━━━━━━━━━━━━━━━━━━

{tests}

{'=' * 60}
""".strip()


def _extract_filename(text: str) -> str:
    """Best-effort extraction of a filename from user input for memory logging."""
    import re
    match = re.search(r"[\w/.-]+\.py", text)
    return match.group(0) if match else "unknown"


def _first_line(text: str) -> str:
    """Returns the first non-empty line of a report."""
    for line in text.splitlines():
        if line.strip():
            return line.strip()[:80]
    return "(no summary)"
