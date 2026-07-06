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

Observability (Lecture 5, post-Langfuse migration):
  - One trace per orchestrator invocation (@observed).
  - Memory load / save and the vision extraction are nested spans.
  - Each parallel sub-agent runs in its own thread and creates its own
    @observed trace. We pass the orchestrator's session_id into each
    so all four traces group in Langfuse's Sessions view as one user
    request. (Trying to nest spans across thread boundaries would mean
    fighting OpenTelemetry context propagation; the session_id model
    matches the lecture's "Session → Trace → Step" hierarchy and is
    queryable without that complexity.)
"""

import os
import uuid
import concurrent.futures
from google import genai
from google.genai import types
from dotenv import load_dotenv

from memory import load_memory, save_memory
from codebase_index import search_codebase
from agents.reviewer_agent import run_reviewer_agent
from agents.security_agent import run_security_agent
from agents.test_writer_agent import run_test_writer_agent
from observability import generation, gemini_usage, observed, set_session, span

load_dotenv()


# ── Multimodality: extract code from an image ─────────────────────────────────

def _extract_code_from_image(image_path: str) -> str:
    """
    Uses Gemini's vision capability to extract Python code from an image.
    Wrapped in a generation observation so the vision call shows up in the
    trace alongside the rest of the orchestrator's work.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)

    try:
        with open(image_path, "rb") as f:
            image_bytes = f.read()
    except FileNotFoundError:
        return f"Error: Image file '{image_path}' not found."

    ext = os.path.splitext(image_path)[1].lower()
    mime_map = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}
    mime_type = mime_map.get(ext, "image/png")

    image_part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
    text_part = types.Part.from_text(
        "Extract all Python code visible in this image. "
        "Return only the code, no explanation."
    )

    with generation(
        "gemini:orchestrator:vision",
        model="gemini-2.5-flash",
        input={"image_path": image_path, "mime_type": mime_type},
    ) as g:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[image_part, text_part],
        )
        g.update(output=response.text or "", usage_details=gemini_usage(response))
    return response.text


# Keep the old public name as a thin alias for callers / docs that referenced it.
extract_code_from_image = _extract_code_from_image


# ── Orchestrator ──────────────────────────────────────────────────────────────

@observed
def run_orchestrator(user_input: str, image_path: str | None = None) -> str:
    """
    Orchestrates a full multi-agent code review.

    Args:
        user_input: Code snippet or file path.
        image_path: Optional path to a code screenshot (multimodality).

    Returns:
        A merged final review from all three specialist agents.
    """
    # One shared session_id across the orchestrator trace and the 3 sub-agent
    # traces. View them grouped under "Sessions" in the Langfuse UI.
    session_id = str(uuid.uuid4())
    set_session(session_id, tags=["orchestrator"])

    # ── Step 0: Multimodality — extract code from image if provided ───────────
    if image_path:
        print(f"\n  [orchestrator] Extracting code from image: {image_path}")
        with span("orchestrator:vision_extract", input={"image_path": image_path}):
            extracted_code = _extract_code_from_image(image_path)
        print(f"  [orchestrator] Extracted {len(extracted_code)} chars of code")
        user_input = f"Extracted from screenshot:\n```python\n{extracted_code}\n```\n\n{user_input}"

    # ── Step 1: Load memory ───────────────────────────────────────────────────
    # Pass the current code as the query so ChromaDB does semantic search:
    # "find past reviews whose code is most similar to what we're reviewing now"
    print("\n  [orchestrator] Loading memory...")
    memory_context = load_memory(query=user_input[:2000])
    non_empty = [l for l in memory_context.splitlines() if l.strip()]
    first_line = non_empty[0] if non_empty else "none"
    print(f"  [orchestrator] Memory: {first_line}")

    # ── Step 1.5: Retrieve codebase context (RAG over the repo under review) ──
    # One retrieval pass here, shared by all three sub-agents — cheaper than
    # each agent retrieving separately, and keeps their prompts consistent.
    print("  [orchestrator] Retrieving codebase context...")
    with span("orchestrator:codebase_rag", input={"query_chars": min(len(user_input), 2000)}):
        codebase_context = search_codebase(user_input[:2000])
    cb_first = next((l for l in codebase_context.splitlines() if l.strip()), "none")
    print(f"  [orchestrator] Codebase: {cb_first}")

    full_input = (
        f"Memory context:\n{memory_context}\n\n"
        f"Codebase context (related code retrieved from the repository — use it to "
        f"ground findings in actual callers, helpers, and conventions):\n"
        f"{codebase_context}\n\n---\n\n{user_input}"
    )

    # ── Step 2: Run all three agents in parallel ──────────────────────────────
    # Each agent runs in its own thread and creates its own @observed trace.
    # session_id ties them together in Langfuse's Sessions view.
    print("\n  [orchestrator] Dispatching to specialist agents in parallel...\n")

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        future_review   = executor.submit(run_reviewer_agent,    full_input, session_id=session_id)
        future_security = executor.submit(run_security_agent,    full_input, session_id=session_id)
        future_tests    = executor.submit(run_test_writer_agent, full_input, session_id=session_id)

        print("  Waiting for all agents to finish...")
        review_report   = future_review.result()
        security_report = future_security.result()
        test_report     = future_tests.result()

    print("\n  [orchestrator] All agents done. Merging reports...")

    # ── Step 3: Merge reports ─────────────────────────────────────────────────
    merged = _merge_reports(review_report, security_report, test_report)

    # ── Step 4: Save memory ───────────────────────────────────────────────────
    filename = _extract_filename(user_input)
    issues_summary = [
        f"Quality: {_first_line(review_report)}",
        f"Security: {_first_line(security_report)}",
    ]

    save_memory(
        file_reviewed=filename,
        issues_found=issues_summary,
        code_snippet=user_input[:2000],
    )
    print("  [orchestrator] Memory saved.")

    return merged


# ── Helpers ────────────────────────────────────────────────────────────────────

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
