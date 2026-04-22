"""
memory.py — Long-term memory for the Code Reviewer Agent.

This is what Day 2 adds to our agent. Instead of starting fresh every run,
the agent can now:
  - Load past context at the start of a review (what files were reviewed before,
    what the user's coding preferences are)
  - Save new learnings at the end of a review (issues found, style notes)

Storage: a simple local JSON file (memory.json).
In production (Day 5) this would be a real database, but for learning
purposes a JSON file makes the data fully visible and easy to inspect.

Memory structure:
{
  "preferences": {
    "style": "...",      # e.g. PEP8, Google style
    "notes": "..."       # free-form notes about the user's coding habits
  },
  "history": [
    {
      "file": "...",
      "date": "...",
      "issues_found": [...]
    },
    ...
  ]
}
"""

import json
import os
from datetime import datetime

MEMORY_FILE = "memory.json"


def _load_raw() -> dict:
    """Internal helper: loads the raw memory dict from disk."""
    if not os.path.exists(MEMORY_FILE):
        return {"preferences": {}, "history": []}
    with open(MEMORY_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_raw(data: dict) -> None:
    """Internal helper: writes the raw memory dict to disk."""
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


# ── Tools the agent can call ──────────────────────────────────────────────────

def load_memory() -> str:
    """
    Loads the agent's long-term memory from disk and returns it as a
    formatted string. Call this at the start of every review to recall
    past context: previously reviewed files, known issues, and user preferences.
    """
    data = _load_raw()

    if not data["preferences"] and not data["history"]:
        return "No memory yet — this appears to be the first review session."

    lines = []

    if data["preferences"]:
        lines.append("=== User Preferences ===")
        for key, value in data["preferences"].items():
            lines.append(f"  {key}: {value}")

    if data["history"]:
        lines.append("\n=== Review History ===")
        for entry in data["history"][-5:]:  # show last 5 reviews only
            lines.append(f"  File: {entry['file']} | Date: {entry['date']}")
            if entry.get("issues_found"):
                lines.append(f"  Issues: {', '.join(entry['issues_found'])}")

    return "\n".join(lines)


def save_memory(file_reviewed: str, issues_found: list[str], preference_notes: str = "") -> str:
    """
    Saves the result of a review to long-term memory so future sessions
    can learn from it. Call this at the end of every review.

    Args:
        file_reviewed: The name or path of the file that was reviewed.
        issues_found: A list of the key issues found during the review.
        preference_notes: Any observations about the user's coding style
                          or preferences worth remembering (optional).

    Returns:
        A confirmation message.
    """
    data = _load_raw()

    # Save the review to history
    data["history"].append({
        "file": file_reviewed,
        "date": datetime.now().strftime("%Y-%m-%d"),
        "issues_found": issues_found
    })

    # Update preferences if new notes were provided
    if preference_notes:
        data["preferences"]["notes"] = preference_notes
        data["preferences"]["last_updated"] = datetime.now().strftime("%Y-%m-%d")

    _save_raw(data)
    return f"Memory saved: review of '{file_reviewed}' with {len(issues_found)} issues recorded."
