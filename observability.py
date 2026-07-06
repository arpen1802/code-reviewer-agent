"""
observability.py — Langfuse tracing helpers.

Single source of truth for "should this be traced, and how." Every other
file in the project goes through `observed`, `span`, `generation`, and
`set_session` — they have no `import langfuse` of their own. That means:

  - One config check per process (`is_enabled()`).
  - Missing credentials → every helper becomes a no-op silently. The
    project still runs without a Langfuse account.
  - SDK swap-out is one file.

Replaces the homegrown logger.py (JSONL writer + log_viewer.py CLI).
Langfuse gives us the trace UI, accurate token counts via Gemini's
`usage_metadata`, and a Sessions view that groups the orchestrator and
the three specialist agents under a shared session_id.

Usage:

    from observability import observed, span, generation, set_session, flush

    @observed                                        # one trace per call
    def run_agent(user_input, session_id=None):
        if session_id:
            set_session(session_id, tags=["reviewer"])
        with generation("gemini:reviewer", model="gemini-2.5-flash") as g:
            g.update(input={"prompt": user_input})
            resp = client.models.generate_content(...)
            g.update(output=resp.text, usage_details={
                "input": resp.usage_metadata.prompt_token_count,
                "output": resp.usage_metadata.candidates_token_count,
            })
        with span("tool:run_python_code", input={"code": code}) as s:
            result = run_python_code(code)
            s.update(output={"result": result})
        return resp.text
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Iterator

from dotenv import load_dotenv

# Ensure .env is loaded before we check for credentials. Safe to call
# multiple times; later load_dotenv() calls in callers are no-ops.
load_dotenv()

_enabled: bool = bool(
    os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY")
)


def is_enabled() -> bool:
    """True iff Langfuse credentials are present in the environment."""
    return _enabled


# ── No-op fallback ────────────────────────────────────────────────────────────


class _NoopObs:
    """Returned from span/generation context managers when Langfuse is
    disabled. .update() swallows kwargs so caller code doesn't branch."""

    def update(self, **kwargs: Any) -> None:
        pass


# ── @observed decorator ───────────────────────────────────────────────────────


def observed(fn):
    """Decorator: turn this function call into one Langfuse trace.

    Auto-captures function args and return value as the trace input/output.
    No-op when Langfuse isn't configured. Equivalent to `@observe()` from
    the Langfuse SDK, but indirected so we have one place to disable it.
    """
    if not _enabled:
        return fn
    try:
        from langfuse import observe

        return observe()(fn)
    except Exception:
        # Import failure or version mismatch: degrade to no-op rather than
        # breaking the agent for an observability reason.
        return fn


# ── Trace metadata (session_id / user_id / tags) ──────────────────────────────


def set_session(session_id: str, *, user_id: str | None = None, tags: list[str] | None = None) -> None:
    """Tag the current trace with a session_id (and optionally user_id /
    tags). Call once near the top of an @observed function so the trace
    appears in Langfuse's Sessions view alongside any sibling agent traces
    that share the same session_id.
    """
    if not _enabled:
        return
    try:
        from langfuse import get_client

        get_client().update_current_trace(
            session_id=session_id,
            user_id=user_id,
            tags=tags or [],
        )
    except Exception:
        pass


# ── Nested observations: spans and generations ────────────────────────────────


@contextmanager
def span(name: str, *, input: Any = None) -> Iterator[Any]:
    """Open a nested span inside the current trace. Yields an object whose
    .update(output=..., metadata=...) you can call as the span produces
    data. No-op when Langfuse isn't configured.
    """
    if not _enabled:
        yield _NoopObs()
        return
    try:
        from langfuse import get_client

        with get_client().start_as_current_observation(as_type="span", name=name) as s:
            if input is not None:
                s.update(input=input)
            yield s
    except Exception:
        yield _NoopObs()


@contextmanager
def generation(name: str, *, model: str, input: Any = None) -> Iterator[Any]:
    """Open a generation observation (the LLM call itself) inside the
    current trace. Use .update(output=..., usage_details={"input": N,
    "output": M}) to record token counts. No-op when Langfuse isn't
    configured.
    """
    if not _enabled:
        yield _NoopObs()
        return
    try:
        from langfuse import get_client

        with get_client().start_as_current_observation(
            as_type="generation", name=name, model=model
        ) as g:
            if input is not None:
                g.update(input=input)
            yield g
    except Exception:
        yield _NoopObs()


# ── Flush ─────────────────────────────────────────────────────────────────────


def flush() -> None:
    """Flush pending traces. Call at the end of a short-lived process
    (CLI script, eval script). The orchestrator flushes after each review
    so long-running CLI sessions see traces appear promptly.
    """
    if not _enabled:
        return
    try:
        from langfuse import get_client

        get_client().flush()
    except Exception:
        pass


# ── Convenience: extract Gemini usage_metadata into the dict Langfuse wants ───


def gemini_usage(response: Any) -> dict[str, int] | None:
    """Pull token counts off a google-genai response into the dict shape
    Langfuse's `usage_details` field expects. Returns None when the
    response has no usage_metadata (e.g. a stub).
    """
    meta = getattr(response, "usage_metadata", None)
    if meta is None:
        return None
    return {
        "input": getattr(meta, "prompt_token_count", 0) or 0,
        "output": getattr(meta, "candidates_token_count", 0) or 0,
        "total": getattr(meta, "total_token_count", 0) or 0,
    }
