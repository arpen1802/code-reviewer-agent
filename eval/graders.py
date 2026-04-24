"""
eval/graders.py — Grading logic for the Code Reviewer Agent eval suite.

Two graders, as described in Lecture 4:

  1. code_grader   — fast, deterministic keyword check (no API call)
  2. llm_judge     — LLM scores review quality on a 1-5 scale

Each grader takes the task definition and the agent's output,
and returns a GradeResult with a pass/fail and a short explanation.
"""

import os
import re
from dataclasses import dataclass
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()


@dataclass
class GradeResult:
    """Result from one grader on one task."""
    grader: str          # which grader produced this
    passed: bool         # pass or fail
    score: float         # 0.0 to 1.0
    reason: str          # human-readable explanation


# ── Grader 1: Code-based keyword checker ─────────────────────────────────────

def code_grader(task: dict, agent_output: str, trajectory: list = None) -> GradeResult:
    """
    Checks the agent's output for expected keywords (and absence of
    hallucinated terms for clean-code tasks).

    Checks two sources (lecture principle: evaluate the full chain):
      1. agent_output — the final text the agent returned to the user
      2. trajectory   — the save_memory call args (what the agent recorded)

    Pass conditions:
      - ANY expected keyword appears in either source
      - NONE of the should_not_find keywords appear in either source
    """
    # Build a combined search corpus: final text + save_memory issues
    corpus = agent_output or ""
    if trajectory:
        for tc in trajectory:
            if tc.tool_name == "save_memory":
                corpus += " " + str(tc.args)

    output_lower = corpus.lower()

    expected = task.get("expected_findings", [])
    should_not = task.get("should_not_find", [])

    # Check expected findings — ANY keyword match counts (synonym list logic).
    # The agent may say "ZeroDivisionError" or "division by zero" — both are correct.
    if expected:
        matched = [kw for kw in expected if kw.lower() in output_lower]
        if not matched:
            return GradeResult(
                grader="code_grader",
                passed=False,
                score=0.0,
                reason=f"None of the expected findings were mentioned. Expected any of: {expected}"
            )

    # Check hallucinations (for clean-code tasks)
    hallucinated = [kw for kw in should_not if kw.lower() in output_lower]
    if hallucinated:
        return GradeResult(
            grader="code_grader",
            passed=False,
            score=0.0,
            reason=f"Hallucinated problems in clean code: {hallucinated}"
        )

    matched_display = [kw for kw in expected if kw.lower() in output_lower] if expected else []
    return GradeResult(
        grader="code_grader",
        passed=True,
        score=1.0,
        reason=f"Found: {matched_display}" if matched_display else "No issues expected and none hallucinated."
    )


# ── Grader 2: LLM-as-Judge ────────────────────────────────────────────────────

JUDGE_PROMPT = """You are an expert evaluator of AI code review tools.

You will be given:
1. A Python code snippet
2. An AI-generated code review of that snippet

Score the review on a scale of 1 to 5:
  5 — Excellent: finds all real issues, explains why they matter, gives actionable fixes
  4 — Good: finds the main issues, mostly clear and useful
  3 — Acceptable: finds some issues but misses important ones or is vague
  2 — Poor: largely misses the point or gives generic feedback
  1 — Bad: wrong, misleading, or hallucinates problems that don't exist

Reply with ONLY this format (no other text):
SCORE: <number>
REASON: <one sentence>
"""

def llm_judge(task: dict, agent_output: str) -> GradeResult:
    """
    Uses a separate LLM call to score the quality of the agent's review.

    The judge sees: the original code + the agent's review.
    It does NOT see the expected_findings — it grades independently.

    Pass threshold: score >= 3 out of 5.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)

    prompt = f"""Code being reviewed:
```python
{task['input']}
```

AI-generated review:
{agent_output}
"""

    config = types.GenerateContentConfig(
        system_instruction=JUDGE_PROMPT,
    )

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=config,
        )
        text = response.text.strip()

        # Parse the structured response
        score_match = re.search(r"SCORE:\s*(\d)", text)
        reason_match = re.search(r"REASON:\s*(.+)", text)

        if not score_match:
            return GradeResult(
                grader="llm_judge",
                passed=False,
                score=0.0,
                reason=f"Could not parse judge response: {text}"
            )

        score = int(score_match.group(1))
        reason = reason_match.group(1).strip() if reason_match else "No reason given."
        normalized = score / 5.0

        return GradeResult(
            grader="llm_judge",
            passed=score >= 3,
            score=normalized,
            reason=f"Judge score {score}/5 — {reason}"
        )

    except Exception as e:
        return GradeResult(
            grader="llm_judge",
            passed=False,
            score=0.0,
            reason=f"Judge error: {str(e)}"
        )
