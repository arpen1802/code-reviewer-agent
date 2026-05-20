"""
log_viewer.py — Pretty-print the structured JSONL log file.

Usage:
    python log_viewer.py              # show all traces
    python log_viewer.py --last 1     # show only the most recent trace
    python log_viewer.py --tail       # live tail (like tail -f)

Each trace shows:
  - Which file was reviewed
  - Per-agent latency breakdown
  - Tool calls with timing
  - Total session latency
  - Estimated cost (based on Gemini 2.5 Flash pricing)

This is the "Alerts > Dashboards" principle from the slides:
you should be able to understand a run in < 30 seconds by reading this output.
"""

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

LOG_FILE = Path.home() / ".code_reviewer_db" / "runs.jsonl"

# Gemini 2.5 Flash pricing (USD per 1M tokens, as of 2025)
# We don't have exact token counts from the Gemini SDK without extra work,
# so we estimate from output_chars: ~4 chars per token is a reasonable approximation.
COST_PER_1M_INPUT_TOKENS  = 0.15
COST_PER_1M_OUTPUT_TOKENS = 0.60
CHARS_PER_TOKEN = 4


def load_traces(last_n: int | None = None) -> list[list[dict]]:
    """
    Load the JSONL log and group lines by trace_id.
    Returns a list of traces, each trace is a list of log lines.
    """
    if not LOG_FILE.exists():
        return []

    by_trace: dict[str, list[dict]] = {}
    trace_order: list[str] = []

    with open(LOG_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            tid = record.get("trace_id", "unknown")
            if tid not in by_trace:
                by_trace[tid] = []
                trace_order.append(tid)
            by_trace[tid].append(record)

    traces = [by_trace[tid] for tid in trace_order]
    if last_n is not None:
        traces = traces[-last_n:]
    return traces


def format_ms(ms: float) -> str:
    if ms >= 1000:
        return f"{ms / 1000:.1f}s"
    return f"{ms:.0f}ms"


def estimate_cost(output_chars: int) -> float:
    """Rough cost estimate from output character count."""
    tokens = output_chars / CHARS_PER_TOKEN
    return (tokens / 1_000_000) * COST_PER_1M_OUTPUT_TOKENS


def print_trace(trace: list[dict]) -> None:
    """Print a human-readable summary of one trace."""
    if not trace:
        return

    # ── Header ────────────────────────────────────────────────────────────────
    start_event = next((r for r in trace if r.get("event") == "session_start"), None)
    end_event   = next((r for r in trace if r.get("event") == "session_end"),   None)
    ts          = trace[0].get("ts", "?")
    trace_id    = trace[0].get("trace_id", "?")[:8]   # short form

    total_ms = end_event.get("total_latency_ms", 0) if end_event else 0
    preview  = (start_event.get("input_preview", "") if start_event else "")[:60]

    print(f"\n{'─' * 60}")
    print(f"  Trace {trace_id}  |  {ts}")
    if preview:
        print(f"  Input: {preview}...")
    print(f"  Total latency: {format_ms(total_ms)}")

    # ── Memory events ─────────────────────────────────────────────────────────
    mem_load = next((r for r in trace if r.get("event") == "memory_load"), None)
    mem_save = next((r for r in trace if r.get("event") == "memory_save"), None)
    if mem_load:
        print(f"\n  Memory load: {format_ms(mem_load.get('latency_ms', 0))}  "
              f"({mem_load.get('results_found', 0)} past reviews retrieved)")
    if mem_save:
        print(f"  Memory save: {format_ms(mem_save.get('latency_ms', 0))}  "
              f"(file: {mem_save.get('file_reviewed', '?')})")

    # ── Per-agent breakdown ───────────────────────────────────────────────────
    agent_done_events = [r for r in trace if r.get("event") == "agent_done"]
    tool_events       = [r for r in trace if r.get("event") == "tool_call"]

    if agent_done_events:
        print(f"\n  {'Agent':<14}  {'Latency':>8}  {'Output':>8}  {'Est. Cost':>10}  {'Tools'}")
        print(f"  {'─' * 56}")
        total_output_chars = 0
        for ev in sorted(agent_done_events, key=lambda e: e.get("latency_ms", 0), reverse=True):
            agent      = ev.get("agent", "?")
            latency    = ev.get("latency_ms", 0)
            out_chars  = ev.get("output_chars", 0)
            err        = ev.get("error", "")
            cost       = estimate_cost(out_chars)
            total_output_chars += out_chars

            # Count tool calls for this agent
            agent_tools = [t for t in tool_events if t.get("agent") == agent]
            tool_summary = ", ".join(
                f"{t.get('tool', '?')}({format_ms(t.get('latency_ms', 0))})"
                for t in agent_tools
            ) or "none"

            status = "✗ ERROR" if err else "✓"
            print(f"  {status} {agent:<12}  {format_ms(latency):>8}  {out_chars:>6}ch  "
                  f"${cost:.4f}       {tool_summary}")

        total_cost = estimate_cost(total_output_chars)
        print(f"  {'─' * 56}")
        print(f"  {'TOTAL':<14}  {format_ms(total_ms):>8}  {total_output_chars:>6}ch  ${total_cost:.4f}")

    # ── Errors ────────────────────────────────────────────────────────────────
    errors = [r for r in trace if r.get("error")]
    if errors:
        print(f"\n  ⚠  Errors ({len(errors)}):")
        for e in errors:
            print(f"     [{e.get('agent', '?')}] {e.get('error', '')[:100]}")


def tail_mode() -> None:
    """Live tail: print new log lines as they arrive."""
    print(f"Tailing {LOG_FILE} (Ctrl-C to stop)...\n")
    if not LOG_FILE.exists():
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        LOG_FILE.touch()
    with open(LOG_FILE, encoding="utf-8") as f:
        f.seek(0, 2)  # jump to end
        while True:
            line = f.readline()
            if line:
                try:
                    record = json.loads(line.strip())
                    event  = record.get("event", "")
                    agent  = record.get("agent", "")
                    ts     = record.get("ts", "")
                    ms     = record.get("latency_ms") or record.get("total_latency_ms")
                    ms_str = f"  {format_ms(ms)}" if ms else ""
                    print(f"{ts}  {event:<18}  {agent:<12}{ms_str}")
                except Exception:
                    print(line.strip())
            else:
                time.sleep(0.2)


def main() -> None:
    args = sys.argv[1:]

    if "--tail" in args:
        tail_mode()
        return

    last_n = None
    if "--last" in args:
        idx = args.index("--last")
        try:
            last_n = int(args[idx + 1])
        except (IndexError, ValueError):
            last_n = 1

    traces = load_traces(last_n=last_n)

    if not traces:
        print(f"No log entries found at {LOG_FILE}")
        print("Run python main.py <file.py> first to generate logs.")
        return

    print(f"{'=' * 60}")
    print(f"  CODE REVIEWER LOG  —  {len(traces)} trace(s)")
    print(f"  Log file: {LOG_FILE}")
    print(f"{'=' * 60}")

    for trace in traces:
        print_trace(trace)

    print(f"\n{'─' * 60}\n")


if __name__ == "__main__":
    main()
