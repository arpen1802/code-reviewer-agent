# Evals

This is the rigor layer of the project. The agent's quality is **measured**, not just claimed: every prompt change, model swap, and refactor runs through a hand-curated task suite, and the score lands in a file you can `git blame`.

If you're reading this to understand the project, skim sections in this order: [Overview](#overview) → [Running locally](#running-locally) → [Score history](#score-history).

---

## Overview

| Piece | What it is | Where |
|---|---|---|
| **Task suite** | Hand-written code samples with `expected_findings` keywords | [`eval/tasks.json`](../eval/tasks.json) |
| **Code grader** | Deterministic keyword matcher (zero API cost) | [`eval/graders.py`](../eval/graders.py) `code_grader()` |
| **LLM judge** | Gemini Flash scoring the review on a 1–5 rubric, pass ≥ 3 | [`eval/graders.py`](../eval/graders.py) `llm_judge()` |
| **Harness** | Runs the agent on a task, captures trajectory + output | [`eval/harness.py`](../eval/harness.py) |
| **Runner** | CLI; aggregates results, prints summary, saves baselines, diffs | [`eval/run_eval.py`](../eval/run_eval.py) |
| **Versioning** | `PROMPT_VERSION` constant baked into each result | [`eval/version.py`](../eval/version.py) |
| **History** | Saved result JSONs, one per run | `eval/results/` |
| **CI** | Runs the suite on PRs to `main` and on every push to `main` | [`.github/workflows/evals.yml`](../.github/workflows/evals.yml) |

### Task structure

```json
{
  "id": "task_013",
  "description": "SQL injection via f-string",
  "tags": ["difficulty:medium", "category:security"],
  "input": "def get_user(conn, username):\n    ...",
  "expected_findings": ["sql injection", "injection", "parameterized", "?"],
  "should_not_find": []
}
```

- **`tags`** use a `namespace:value` scheme. Two namespaces today: `difficulty` (`easy`/`medium`/`hard`) and `category` (`bug`/`security`/`performance`/`correctness`/`concurrency`/`no_bugs`).
- **`expected_findings`** is a **synonym list** — the agent passes if it mentions any one of them. "Division by zero" and "ZeroDivisionError" both count.
- **`should_not_find`** is the hallucination guard, used on clean-code (`category:no_bugs`) tasks. If the agent claims a clean function has a bug, it fails.

### Two graders, on purpose

| Grader | Strength | Weakness |
|---|---|---|
| `code_grader` | Cheap, fast, deterministic, easy to debug — perfect for CI | Can't tell a good review from a thorough-but-bad one if the right keyword appears |
| `llm_judge` | Catches review *quality* (clarity, actionability, calibration) | Costs Gemini calls; introduces model bias |

CI runs only the deterministic grader (cost discipline). Local runs default to both.

---

## Running locally

From the repo root (`code-reviewer-agent/`):

```bash
# Full suite, both graders (default)
python eval/run_eval.py

# Skip the LLM judge — keyword grader only, no extra API cost
python eval/run_eval.py --no-llm-judge

# Single task (useful when iterating on a regression)
python eval/run_eval.py --task task_023

# Save the result as a baseline
python eval/run_eval.py --save

# Diff this run against the most recent saved baseline
python eval/run_eval.py --compare

# All four flags compose — typical "I changed a prompt, did I break anything?" run:
python eval/run_eval.py --no-llm-judge --save --compare
```

> The agent itself calls Gemini for *every* task, so even `--no-llm-judge` is ~30 API calls. Budget accordingly.

### Reading the output

The runner prints, in order:

1. **Trajectory per task** — every tool call the agent made (`load_memory`, `run_python_code`, `save_memory`). Useful for catching cases where the agent skipped reading the code before reviewing it.
2. **Pass/fail table** by task.
3. **Breakdowns** by category and difficulty — where the weak spots are.
4. **Diff vs baseline** (if `--compare`) — overall percentage point change plus per-task regressions/improvements.

A regression list like:

```
✗ Regressions (2):  task_015, task_022
```

…means those two tasks **used to pass and now fail**. That's the first thing to look at after a prompt change.

---

## Adding a new task

1. Pick the next free ID (`task_NNN`).
2. Choose `category:` and `difficulty:` tags from the existing namespaces. If you need a new category, add it consistently in `tasks.json` and update the table above.
3. Write the smallest possible `input` that exercises the issue. Realistic Python only — the agent runs it via `run_python_code`, so it must at least parse.
4. List **synonyms** in `expected_findings`. Think about what a *good* reviewer would say, and include the common phrasings.
5. For clean-code probes (`category:no_bugs`), leave `expected_findings: []` and put **likely hallucinations** in `should_not_find`.
6. Run just that task to sanity-check: `python eval/run_eval.py --task task_NNN`.
7. If it passes and the synonym list feels right, commit the change.

**Avoid:** trick questions, tasks whose `expected_findings` are too broad (e.g. `["bug"]` matches almost any review), tasks where the bug is so subtle that *humans* would disagree about whether it's a bug.

---

## CI integration

[`.github/workflows/evals.yml`](../.github/workflows/evals.yml) runs the suite on:

- Every PR to `main` (paths-filtered so doc-only PRs don't trigger).
- Every push to `main` — and commits the baseline JSON back into `eval/results/` automatically.
- Manual `workflow_dispatch`.

The workflow uses only the deterministic grader (`--no-llm-judge --save --compare`) and posts a markdown summary to the **GitHub Step Summary** (visible under the workflow run) with totals, per-category breakdown, and a list of failed tasks. The result JSON is also uploaded as a 90-day artifact.

**Setup checklist:**

1. Add `GEMINI_API_KEY` to repo secrets (`Settings → Secrets and variables → Actions`).
2. That's it. The first run on `main` populates `eval/results/` with the initial baseline.

> **Note:** GitHub's documented behavior is that secrets are not exposed to PRs from forks. For a personal portfolio repo where only the owner pushes, that's not a problem.

---

## Saved result schema

Each run that uses `--save` writes a JSON to `eval/results/<YYYYMMDDTHHMMSS>-<git-sha>.json`. Schema (abbreviated):

```jsonc
{
  "timestamp": "2026-05-14T18:30:00+00:00",
  "git_sha": "abc1234",
  "prompt_version": "v1",
  "use_llm_judge": false,
  "task_count": 30,
  "totals": {
    "code": {"passed": 23, "total": 30, "pct": 76.7},
    "llm":  null
  },
  "by_category":   { "bug": {"code": {...}, "llm": null}, ... },
  "by_difficulty": { "easy": {"code": {...}, "llm": null}, ... },
  "tasks": [
    {
      "task_id": "task_001",
      "description": "Division by zero on empty list",
      "tags": ["difficulty:easy", "category:bug"],
      "category": "bug",
      "difficulty": "easy",
      "code_passed": true,
      "code_reason": "Found: ['empty', 'ZeroDivisionError']",
      "llm_passed":  null,
      "llm_score":   null,
      "llm_reason":  null,
      "duration_seconds": 7.5,
      "trajectory_length": 3,
      "error": null
    }
  ]
}
```

Filenames sort lexicographically by timestamp, so `ls -1t eval/results/` is also sorted by time.

---

## Score history

Each entry below corresponds to a prompt version. Update when bumping `PROMPT_VERSION`.

| Prompt | What changed | Code grader | LLM judge | Notes |
|---|---|---|---|---|
| `v1` | Initial multi-agent + ChromaDB memory | _TBD_ (first CI run on main) | _TBD_ | Baseline — 30-task suite |

> Newer rows go on top once we add them. Keep the table short (latest 6–8 versions); older rows can move into a collapsed section.

---

## Roadmap

Things not yet wired but worth doing:

- **Cost telemetry** in the result JSON (sum input + output tokens per task) so we can chart cost vs score over time.
- **Trajectory grading** — assert that the agent always called `run_python_code` before claiming a bug exists. Belongs in `graders.py` as a third grader.
- **Per-category pass-rate floors** as CI gates — fail the build if, say, `security` drops below 70%.
- **Smaller, faster smoke set** (one task per category) for `--quick` runs during prompt iteration.
