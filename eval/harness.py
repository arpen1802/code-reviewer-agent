"""
eval/harness.py — The eval runner (infrastructure layer).

This is pipeline component 2 of 4 from Lecture 4.
Its job: take a task, run the agent, capture everything that happened,
return a structured result.

Key design decisions:
  - We test agent.py (the single agent), not the orchestrator.
    This isolates what we're measuring.
  - We capture the trajectory by wrapping TOOL_REGISTRY so every
    tool call is logged automatically.
  - Each task run is independent — no shared state between tasks.
"""

import sys
import os
import json
import time
from dataclasses import dataclass, field
from typing import Any

# Make sure the parent directory is on the path so we can import agent.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import run_agent
import tools as tools_module

MEMORY_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "memory.json")


def reset_memory():
    """Delete memory.json so each eval task starts with a clean slate."""
    if os.path.exists(MEMORY_FILE):
        os.remove(MEMORY_FILE)


@dataclass
class ToolCall:
    """One recorded tool call from the agent's trajectory."""
    tool_name: str
    args: dict
    result: str


@dataclass
class TaskResult:
    """Everything captured from running one task."""
    task_id: str
    task_description: str
    agent_output: str           # the final text the agent returned
    trajectory: list[ToolCall]  # ordered list of all tool calls made
    duration_seconds: float
    error: str | None = None    # if the agent crashed


def run_task(task: dict) -> TaskResult:
    """
    Runs the agent on a single task and captures the full trajectory.

    We intercept tool calls by temporarily wrapping each function in
    TOOL_REGISTRY with a logging wrapper. The agent never knows it's
    being monitored — same code path as production.
    """
    # Reset memory so this task is independent of all previous tasks
    reset_memory()

    trajectory: list[ToolCall] = []

    # ── Trajectory capture: wrap every tool in TOOL_REGISTRY ─────────────────
    # This is how we log tool calls without changing agent.py at all.
    # Each wrapper records the call, then calls the real function.
    original_registry = dict(tools_module.TOOL_REGISTRY)

    def make_wrapper(name, fn):
        def wrapper(**kwargs):
            result = fn(**kwargs)
            trajectory.append(ToolCall(
                tool_name=name,
                args=kwargs,
                result=str(result)[:300]  # truncate long results for storage
            ))
            return result
        return wrapper

    # Patch the registry with wrapped versions
    for name, fn in original_registry.items():
        tools_module.TOOL_REGISTRY[name] = make_wrapper(name, fn)

    # ── Run the agent ─────────────────────────────────────────────────────────
    start = time.time()
    error = None
    agent_output = ""

    try:
        user_input = f"Please review this code:\n\n```python\n{task['input']}\n```"
        agent_output = run_agent(user_input) or ""
    except Exception as e:
        error = str(e)
        agent_output = ""
    finally:
        # Always restore the original registry, even if agent crashed
        tools_module.TOOL_REGISTRY.update(original_registry)

    duration = time.time() - start

    return TaskResult(
        task_id=task["id"],
        task_description=task["description"],
        agent_output=agent_output,
        trajectory=trajectory,
        duration_seconds=round(duration, 2),
        error=error,
    )
