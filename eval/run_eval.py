"""
eval/run_eval.py — Entry point for the evaluation suite.

Usage:
    cd code-reviewer-agent
    python eval/run_eval.py

    # Skip the LLM judge (no extra API cost beyond the agent itself):
    python eval/run_eval.py --no-llm-judge

    # Run a single task by ID:
    python eval/run_eval.py --task task_001

    # Save a structured result JSON to eval/results/<ts>-<sha>.json:
    python eval/run_eval.py --save

    # Compare this run against the most recent saved baseline:
    python eval/run_eval.py --compare

This ties together all four pipeline components from Lecture 4:
  1. Task Suite     → tasks.json
  2. Infrastructure → harness.py (runner + trajectory capture)
  3. Criteria       → defined inside tasks.json (expected_findings)
  4. Grading        → graders.py (code_grader + llm_judge)
"""

import sys
import os
import json
import argparse
import subprocess
from collections import defaultdict
from datetime import datetime, timezone

# Make sure the parent directory is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.harness import run_task
from eval.graders import code_grader, llm_judge
from eval.version import PROMPT_VERSION


RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


# ----- Helpers ---------------------------------------------------------------


def _tag_value(tags: list[str], namespace: str) -> str | None:
    """Return the value of e.g. 'difficulty:easy' → 'easy', or None if absent."""
    prefix = f"{namespace}:"
    for t in tags:
        if t.startswith(prefix):
            return t.split(":", 1)[1]
    return None


def _git_sha() -> str:
    """Short git SHA of HEAD, plus '-dirty' if the working tree has uncommitted changes."""
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL
        ).decode().strip()
        return f"{sha}-dirty" if dirty else sha
    except Exception:
        return "unknown"


def _pct(passed: int, total: int) -> float:
    return round(100 * passed / total, 1) if total else 0.0


def load_tasks(task_id: str | None = None) -> list[dict]:
    tasks_path = os.path.join(os.path.dirname(__file__), "tasks.json")
    with open(tasks_path, "r") as f:
        tasks = json.load(f)
    if task_id:
        tasks = [t for t in tasks if t["id"] == task_id]
        if not tasks:
            print(f"Task '{task_id}' not found.")
            sys.exit(1)
    return tasks


# ----- Aggregation -----------------------------------------------------------


def aggregate(task_results: list[dict], use_llm_judge: bool) -> dict:
    """Compute totals and per-category / per-difficulty breakdowns."""

    def fresh() -> dict:
        return {"code": {"passed": 0, "total": 0}, "llm": {"passed": 0, "total": 0}}

    totals = fresh()
    by_cat: dict[str, dict] = defaultdict(fresh)
    by_diff: dict[str, dict] = defaultdict(fresh)

    for r in task_results:
        cat = r.get("category") or "uncategorized"
        diff = r.get("difficulty") or "unknown"

        for bucket in (totals, by_cat[cat], by_diff[diff]):
            bucket["code"]["total"] += 1
            if r["code_passed"]:
                bucket["code"]["passed"] += 1
            if r["llm_passed"] is not None:
                bucket["llm"]["total"] += 1
                if r["llm_passed"]:
                    bucket["llm"]["passed"] += 1

    def finalize(b: dict) -> dict:
        out = {
            "code": {
                "passed": b["code"]["passed"],
                "total": b["code"]["total"],
                "pct": _pct(b["code"]["passed"], b["code"]["total"]),
            },
        }
        if use_llm_judge and b["llm"]["total"]:
            out["llm"] = {
                "passed": b["llm"]["passed"],
                "total": b["llm"]["total"],
                "pct": _pct(b["llm"]["passed"], b["llm"]["total"]),
            }
        else:
            out["llm"] = None
        return out

    return {
        "totals": finalize(totals),
        "by_category": {k: finalize(v) for k, v in sorted(by_cat.items())},
        "by_difficulty": {k: finalize(v) for k, v in sorted(by_diff.items())},
    }


# ----- I/O on results --------------------------------------------------------


def save_result(result: dict) -> str:
    """Persist a result JSON. Filename starts with timestamp so sort-by-name == sort-by-time."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    ts = result["timestamp"].replace(":", "").replace("-", "").split(".")[0]
    path = os.path.join(RESULTS_DIR, f"{ts}-{result['git_sha']}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    return path


def latest_baseline() -> dict | None:
    """Most recent saved result, if any."""
    if not os.path.isdir(RESULTS_DIR):
        return None
    candidates = sorted(
        os.path.join(RESULTS_DIR, f)
        for f in os.listdir(RESULTS_DIR)
        if f.endswith(".json")
    )
    if not candidates:
        return None
    with open(candidates[-1]) as f:
        return json.load(f)


# ----- Pretty printers -------------------------------------------------------


def print_table(current: dict, use_llm_judge: bool) -> None:
    """Per-task pass/fail table. Format unchanged from the original runner."""
    print("\n" + "=" * 65)
    print("  RESULTS SUMMARY")
    print("=" * 65)
    print(f"\n  Task ID      Code Grader   LLM Judge")
    print(f"  {'-' * 50}")
    for r in current["tasks"]:
        code_status = "✓ pass" if r["code_passed"] else "✗ fail"
        if r["llm_passed"] is None:
            llm_status = "—  skip"
        else:
            llm_status = "✓ pass" if r["llm_passed"] else "✗ fail"
        print(f"  {r['task_id']:<12} {code_status:<13} {llm_status}")

    totals = current["totals"]
    print(
        f"\n  Code grader: {totals['code']['passed']}/{totals['code']['total']} "
        f"passed ({totals['code']['pct']}%)"
    )
    if use_llm_judge and totals["llm"]:
        print(
            f"  LLM judge:   {totals['llm']['passed']}/{totals['llm']['total']} "
            f"passed ({totals['llm']['pct']}%)"
        )


def print_breakdowns(current: dict) -> None:
    print("\n  By category:")
    for cat, stats in current["by_category"].items():
        code = stats["code"]
        line = f"    {cat:<14} code {code['passed']}/{code['total']} ({code['pct']}%)"
        if stats["llm"]:
            l = stats["llm"]
            line += f"  ·  llm {l['passed']}/{l['total']} ({l['pct']}%)"
        print(line)

    print("\n  By difficulty:")
    for diff, stats in current["by_difficulty"].items():
        code = stats["code"]
        line = f"    {diff:<14} code {code['passed']}/{code['total']} ({code['pct']}%)"
        if stats["llm"]:
            l = stats["llm"]
            line += f"  ·  llm {l['passed']}/{l['total']} ({l['pct']}%)"
        print(line)


def print_diff(current: dict, baseline: dict) -> None:
    print("\n" + "=" * 65)
    print(f"  DIFF vs baseline {baseline['git_sha']} (prompt {baseline['prompt_version']})")
    print("=" * 65)

    c_total = current["totals"]["code"]
    b_total = baseline["totals"]["code"]
    delta = round(c_total["pct"] - b_total["pct"], 1)
    arrow = "↑" if delta > 0 else ("↓" if delta < 0 else "·")
    print(
        f"\n  Code grader: {b_total['pct']}% → {c_total['pct']}%  "
        f"{arrow} {delta:+}pp  ({b_total['passed']}/{b_total['total']} → "
        f"{c_total['passed']}/{c_total['total']})"
    )

    if current["totals"]["llm"] and baseline["totals"]["llm"]:
        c_l = current["totals"]["llm"]
        b_l = baseline["totals"]["llm"]
        delta_l = round(c_l["pct"] - b_l["pct"], 1)
        arrow_l = "↑" if delta_l > 0 else ("↓" if delta_l < 0 else "·")
        print(
            f"  LLM judge:   {b_l['pct']}% → {c_l['pct']}%  "
            f"{arrow_l} {delta_l:+}pp  ({b_l['passed']}/{b_l['total']} → "
            f"{c_l['passed']}/{c_l['total']})"
        )

    # Per-task regressions / improvements (code grader is the strict signal)
    baseline_by_id = {t["task_id"]: t for t in baseline["tasks"]}
    regressions, improvements = [], []
    for t in current["tasks"]:
        prev = baseline_by_id.get(t["task_id"])
        if prev is None:
            continue
        if prev["code_passed"] and not t["code_passed"]:
            regressions.append(t["task_id"])
        elif not prev["code_passed"] and t["code_passed"]:
            improvements.append(t["task_id"])

    if regressions:
        print(f"\n  ✗ Regressions ({len(regressions)}):  {', '.join(regressions)}")
    if improvements:
        print(f"  ✓ Improvements ({len(improvements)}): {', '.join(improvements)}")
    if not regressions and not improvements:
        print("\n  No per-task changes vs baseline.")


# ----- Main runner -----------------------------------------------------------


def run_eval(
    use_llm_judge: bool = True,
    task_id: str | None = None,
    verbose: bool = False,
    save: bool = False,
    compare: bool = False,
) -> dict:
    tasks = load_tasks(task_id)

    print("=" * 65)
    print(f"  EVAL SUITE — {len(tasks)} task(s) — prompt {PROMPT_VERSION}")
    print("=" * 65)

    # Capture the baseline *before* we save, so --save and --compare can run together
    # without comparing the current run against itself.
    baseline = latest_baseline() if compare else None

    task_results: list[dict] = []

    for task in tasks:
        print(f"\n▶ [{task['id']}] {task['description']}")
        print(f"  Tags: {', '.join(task['tags'])}")

        # ── Run the agent on this task ────────────────────────────────────────
        print("  Running agent...", end="", flush=True)
        result = run_task(task)
        print(f" done ({result.duration_seconds}s)")

        category = _tag_value(task["tags"], "category")
        difficulty = _tag_value(task["tags"], "difficulty")

        if result.error:
            print(f"  ✗ Agent crashed: {result.error}")
            task_results.append({
                "task_id": task["id"],
                "description": task["description"],
                "tags": task["tags"],
                "category": category,
                "difficulty": difficulty,
                "code_passed": False,
                "code_reason": f"Agent crashed: {result.error}",
                "llm_passed": None,
                "llm_score": None,
                "llm_reason": None,
                "duration_seconds": result.duration_seconds,
                "trajectory_length": 0,
                "error": result.error,
            })
            continue

        # ── Show trajectory ───────────────────────────────────────────────────
        print(f"  Trajectory ({len(result.trajectory)} tool calls):")
        for tc in result.trajectory:
            args_display = str(tc.args)[:60]
            print(f"    → {tc.tool_name}({args_display})")

        if verbose:
            print(f"\n  --- Agent output ---\n{result.agent_output[:800]}\n  ---")

        # ── Apply code grader ─────────────────────────────────────────────────
        cg = code_grader(task, result.agent_output, result.trajectory)
        status = "✓" if cg.passed else "✗"
        print(f"  {status} Code grader: {cg.reason}")

        # ── Apply LLM judge ───────────────────────────────────────────────────
        lj = None
        if use_llm_judge:
            print("  Running LLM judge...", end="", flush=True)
            lj = llm_judge(task, result.agent_output)
            status = "✓" if lj.passed else "✗"
            print(f"\r  {status} LLM judge:   {lj.reason}")

        task_results.append({
            "task_id": task["id"],
            "description": task["description"],
            "tags": task["tags"],
            "category": category,
            "difficulty": difficulty,
            "code_passed": cg.passed,
            "code_reason": cg.reason,
            "llm_passed": lj.passed if lj else None,
            "llm_score": lj.score if lj else None,
            "llm_reason": lj.reason if lj else None,
            "duration_seconds": result.duration_seconds,
            "trajectory_length": len(result.trajectory),
            "error": None,
        })

    # ── Assemble the result document ──────────────────────────────────────────
    aggregated = aggregate(task_results, use_llm_judge)
    current = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "prompt_version": PROMPT_VERSION,
        "use_llm_judge": use_llm_judge,
        "task_count": len(task_results),
        "totals": aggregated["totals"],
        "by_category": aggregated["by_category"],
        "by_difficulty": aggregated["by_difficulty"],
        "tasks": task_results,
    }

    # ── Print summary / breakdowns ────────────────────────────────────────────
    print_table(current, use_llm_judge)
    print_breakdowns(current)

    # ── Compare against baseline if requested (uses pre-save snapshot) ────────
    if compare:
        if baseline is None:
            print("\n  (no prior result to compare against — run with --save first to start a history)")
        else:
            print_diff(current, baseline)

    # ── Persist if requested ──────────────────────────────────────────────────
    if save:
        path = save_result(current)
        print(f"\n  Saved → {os.path.relpath(path)}")

    print()
    return current


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the code reviewer eval suite.")
    parser.add_argument("--no-llm-judge", action="store_true", help="Skip the LLM judge grader")
    parser.add_argument("--task", type=str, help="Run a single task by ID")
    parser.add_argument("--verbose", action="store_true", help="Print agent output for each task")
    parser.add_argument(
        "--save",
        action="store_true",
        help="Persist results to eval/results/<ts>-<sha>.json",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Diff against the most recent saved baseline in eval/results/",
    )
    args = parser.parse_args()

    run_eval(
        use_llm_judge=not args.no_llm_judge,
        task_id=args.task,
        verbose=args.verbose,
        save=args.save,
        compare=args.compare,
    )
