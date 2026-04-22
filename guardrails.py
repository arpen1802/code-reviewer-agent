"""
guardrails.py — Safety checks for the Code Reviewer Agent.

Day 2 addition. Guardrails are enforced in Python, NOT in the LLM prompt.
Why? Because prompts can be bypassed (jailbreaking, injection). Python code cannot.

Two layers of protection:
  1. is_code_safe(code)     — action limiter: blocks dangerous code before execution
  2. is_input_clean(text)   — content filter: blocks prompt injection attempts in user input

The rule of thumb from the lecture:
  - Use the LLM for judgment calls (is this code well-written?)
  - Use Python for hard rules (never run os.system under any circumstances)
"""

import re

# ── Dangerous patterns ────────────────────────────────────────────────────────
# These are patterns we NEVER want executed, regardless of what the LLM decides.
# Organized by category so it's easy to extend later.

DANGEROUS_PATTERNS = {
    "system_commands": [
        r"os\.system\s*\(",
        r"subprocess\.call\s*\(",
        r"subprocess\.Popen\s*\(",
        r"subprocess\.run\s*\(",
    ],
    "file_destruction": [
        r"shutil\.rmtree\s*\(",
        r"os\.remove\s*\(",
        r"os\.unlink\s*\(",
        r"rmdir\s*\(",
    ],
    "network_access": [
        r"urllib\.request",
        r"requests\.get\s*\(",
        r"requests\.post\s*\(",
        r"socket\.connect\s*\(",
    ],
    "code_execution": [
        r"\beval\s*\(",
        r"\bexec\s*\(",
        r"__import__\s*\(",
        r"importlib\.import_module\s*\(",
    ],
}

# ── Prompt injection indicators ───────────────────────────────────────────────
# Phrases commonly used in injection attacks embedded inside user-submitted code.

INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"disregard\s+(your\s+)?system\s+prompt",
    r"you\s+are\s+now\s+a",
    r"pretend\s+you\s+(are|have\s+no)",
    r"new\s+instructions?:",
    r"system\s+override",
]


# ── Guardrail functions ───────────────────────────────────────────────────────

def is_code_safe(code: str) -> tuple[bool, str]:
    """
    Checks whether a code snippet is safe to execute.

    Returns:
        (True, "")               if the code is safe
        (False, reason: str)     if the code contains dangerous patterns
    """
    for category, patterns in DANGEROUS_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, code):
                return False, (
                    f"Blocked: code contains a '{category}' pattern "
                    f"({pattern.strip()}). Execution refused for safety."
                )
    return True, ""


def is_input_clean(text: str) -> tuple[bool, str]:
    """
    Checks whether user input contains prompt injection attempts.

    Returns:
        (True, "")               if the input looks clean
        (False, reason: str)     if an injection pattern was detected
    """
    lowered = text.lower()
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, lowered):
            return False, (
                f"Blocked: input contains a suspected prompt injection pattern. "
                f"Please submit only valid Python code or file paths."
            )
    return True, ""
