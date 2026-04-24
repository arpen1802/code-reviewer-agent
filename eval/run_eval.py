"""
eval/run_eval.py — Entry point for the evaluation suite.

Usage:
    cd code-reviewer-agent
    python eval/run_eval.py

    # Run only code grader (no LLM API calls, fast):
    python eval/run_eval.py --no-llm-judge

    # Run a single task by ID:
    python eval/run_eval.py --task task_001

This ties together all four pipeline components from Lecture 4:
  1. Task Suite  → tasks.json
  2. Infrastructure → harness.py (runner + trajectory capture)
  3. Criteria    → defined inside tasks.json (expected_findings)
  4. Grading     → graders.py (code_grader + llm_judge)
"""

import sys
import os
import json
import argparse

# Make sure the parent directory is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.harness import run_task
from eval.graders import code_grader, llm_judge


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


def run_eval(use_llm_judge: bool = True, task_id: str | None = None, verbose: bool = False):
    tasks = load_tasks(task_id)

    print("=" * 65)
    print(f"  EVAL SUITE — {len(tasks)} task(s)")
    print("=" * 65)

    results_summary = []

    for task in tasks:
        print(f"\n▶ [{task['id']}] {task['description']}")
        print(f"  Tags: {', '.join(task['tags'])}")

        # ── Run the agent on this task ────────────────────────────────────────
        print("  Running agent...", end="", flush=True)
        result = run_task(task)
        print(f" done ({result.duration_seconds}s)")

        if result.error:
            print(f"  ✗ Agent crashed: {result.error}")
            results_summary.append({
                "task_id": task["id"],
                "code_passed": False,
                "llm_passed": None,
                "error": result.error
            })
            continue

        # ── Show trajectory ───────────────────────────────────────────────────
        print(f"  Trajectory ({len(result.trajectory)} tool calls):")
        for tc in result.trajectory:
            args_display = str(tc.args)[:60]
            print(f"    → {tc.tool_name}({args_display})")

        # ── Optionally show agent output ──────────────────────────────────────
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

        results_summary.append({
            "task_id": task["id"],
            "code_passed": cg.passed,
            "llm_passed": lj.passed if lj else None,
            "duration": result.duration_seconds,
        })

    # ── Final report ──────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  RESULTS SUMMARY")
    print("=" * 65)

    code_passes = sum(1 for r in results_summary if r["code_passed"])
    total = len(results_summary)

    print(f"\n  Task ID      Code Grader   LLM Judge")
    print(f"  {'-'*50}")
    for r in results_summary:
        code_status = "✓ pass" if r["code_passed"] else "✗ fail"
        llm_status = (
            "✓ pass" if r["llm_passed"]
            else ("✗ fail" if r["llm_passed"] is False else "—  skip")
        )
        print(f"  {r['task_id']:<12} {code_status:<13} {llm_status}")

    print(f"\n  Code grader: {code_passes}/{total} passed ({100*code_passes//total}%)")

    if use_llm_judge:
        llm_passes = sum(1 for r in results_summary if r.get("llm_passed"))
        llm_total = sum(1 for r in results_summary if r.get("llm_passed") is not None)
        if llm_total:
            print(f"  LLM judge:   {llm_passes}/{llm_total} passed ({100*llm_passes//llm_total}%)")

    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the code reviewer eval suite.")
    parser.add_argument("--no-llm-judge", action="store_true", help="Skip the LLM judge grader")
    parser.add_argument("--task", type=str, help="Run a single task by ID")
    parser.add_argument("--verbose", action="store_true", help="Print agent output for each task")
    args = parser.parse_args()

    run_eval(
        use_llm_judge=not args.no_llm_judge,
        task_id=args.task,
        verbose=args.verbose,
    )
