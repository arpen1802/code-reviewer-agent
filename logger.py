"""
logger.py — Structured observability for the multi-agent code reviewer.

Every agent action emits one JSON line to a .jsonl log file.
This implements the "Layer 0: Infrastructure" from the Agent Stack:
  - Structured logs (machine-readable, easy to grep/query)
  - Trace hierarchy: Session → Trace → Step
  - Key metrics: latency_ms, token counts, error
  - No stdout noise — all logging goes to the file only

Log file location: ~/.code_reviewer_db/runs.jsonl
Each line is a valid JSON object — easy to tail, grep, or load into pandas.

Usage:
    from logger import get_logger
    log = get_logger()                 # one per process
    log.session_start("my-session")   # marks the beginning
    log.agent_start("reviewer")
    log.tool_call("reviewer", "read_file", {"path": "foo.py"}, latency_ms=120)
    log.agent_done("reviewer", latency_ms=3400, output_chars=800)
    log.session_end(total_latency_ms=5000)

ROADMAP NOTE
------------
This is a homegrown observability layer written as part of the
production-engineering exercise. It's slated to be replaced by Langfuse
(via the LangChain `CallbackHandler`) — see the "Roadmap" section in
README.md. The shape of the data this module emits matches what Langfuse
captures automatically, plus Langfuse uses real token counts from
`response.usage_metadata` instead of the 4-chars-per-token approximation
in log_viewer.py.
"""

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any


# ── Log file path ──────────────────────────────────────────────────────────────
# Stored in the same home-dir location as the ChromaDB to keep everything together.
LOG_DIR = Path.home() / ".code_reviewer_db"
LOG_FILE = LOG_DIR / "runs.jsonl"


def _ensure_log_dir() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)


# ── Logger ─────────────────────────────────────────────────────────────────────

class AgentLogger:
    """
    Emits structured JSON log lines.

    One instance should be created per process (use get_logger()).
    The session_id is fixed for the lifetime of the process.
    The trace_id is reset each time session_start() is called.

    Every log line has these guaranteed fields:
        ts           — ISO-8601 timestamp
        session_id   — UUID for the whole process run
        trace_id     — UUID for a single code-review request
        event        — event type (see EVENT_* constants below)
    """

    # ── Event types (mirrors the Trace Hierarchy from the slides) ──────────────
    EVENT_SESSION_START  = "session_start"
    EVENT_SESSION_END    = "session_end"
    EVENT_AGENT_START    = "agent_start"
    EVENT_AGENT_DONE     = "agent_done"
    EVENT_TOOL_CALL      = "tool_call"
    EVENT_MEMORY_LOAD    = "memory_load"
    EVENT_MEMORY_SAVE    = "memory_save"
    EVENT_ERROR          = "error"

    def __init__(self) -> None:
        _ensure_log_dir()
        self.session_id: str = str(uuid.uuid4())
        self.trace_id: str = str(uuid.uuid4())
        self._file = open(LOG_FILE, "a", encoding="utf-8")

    # ── Public API ─────────────────────────────────────────────────────────────

    def session_start(self, input_summary: str = "") -> None:
        """Call once at the start of each review request."""
        self.trace_id = str(uuid.uuid4())   # new trace for each request
        self._emit(
            event=self.EVENT_SESSION_START,
            input_chars=len(input_summary),
            input_preview=input_summary[:120],
        )

    def session_end(self, total_latency_ms: float = 0) -> None:
        """Call once when the full review is complete."""
        self._emit(
            event=self.EVENT_SESSION_END,
            total_latency_ms=round(total_latency_ms, 1),
        )

    def agent_start(self, agent: str) -> None:
        """Call when an agent thread begins."""
        self._emit(event=self.EVENT_AGENT_START, agent=agent)

    def agent_done(
        self,
        agent: str,
        latency_ms: float,
        output_chars: int = 0,
        error: str | None = None,
    ) -> None:
        """Call when an agent thread finishes (success or error)."""
        self._emit(
            event=self.EVENT_AGENT_DONE,
            agent=agent,
            latency_ms=round(latency_ms, 1),
            output_chars=output_chars,
            error=error,
        )

    def tool_call(
        self,
        agent: str,
        tool_name: str,
        tool_args: dict[str, Any],
        latency_ms: float,
        result_len: int = 0,
        error: str | None = None,
    ) -> None:
        """Call after each tool execution inside an agent."""
        # Redact large args — we log the shape, not the full content
        safe_args = {k: (str(v)[:80] if isinstance(v, str) else v) for k, v in tool_args.items()}
        self._emit(
            event=self.EVENT_TOOL_CALL,
            agent=agent,
            tool=tool_name,
            args=safe_args,
            latency_ms=round(latency_ms, 1),
            result_len=result_len,
            error=error,
        )

    def memory_load(self, query_chars: int, results_found: int, latency_ms: float) -> None:
        self._emit(
            event=self.EVENT_MEMORY_LOAD,
            query_chars=query_chars,
            results_found=results_found,
            latency_ms=round(latency_ms, 1),
        )

    def memory_save(self, file_reviewed: str, latency_ms: float) -> None:
        self._emit(
            event=self.EVENT_MEMORY_SAVE,
            file_reviewed=file_reviewed,
            latency_ms=round(latency_ms, 1),
        )

    def error(self, agent: str, message: str) -> None:
        self._emit(event=self.EVENT_ERROR, agent=agent, error=message)

    # ── Internal ───────────────────────────────────────────────────────────────

    def _emit(self, **fields: Any) -> None:
        """Write one JSON line to the log file."""
        record = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "session_id": self.session_id,
            "trace_id": self.trace_id,
            **fields,
        }
        # Filter out None values to keep logs clean
        record = {k: v for k, v in record.items() if v is not None}
        self._file.write(json.dumps(record) + "\n")
        self._file.flush()   # ensure each line is written even if process crashes

    def __del__(self) -> None:
        try:
            self._file.close()
        except Exception:
            pass


# ── Singleton ──────────────────────────────────────────────────────────────────
# One logger per process. Import it anywhere with:  from logger import get_logger
_logger_instance: AgentLogger | None = None

def get_logger() -> AgentLogger:
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = AgentLogger()
    return _logger_instance
