"""
tools.py — The actual functions our agent can call.

Remember: the LLM never runs these directly.
It just says "I want to call this tool with these args."
The agent loop in agent.py is what actually executes them.
"""

import subprocess
import sys


def run_python_code(code: str) -> str:
    """
    Executes a Python code snippet in a subprocess and returns
    the combined stdout + stderr output.

    We use a subprocess (not exec/eval) so that:
    - Crashes don't kill our main program
    - We can enforce a timeout
    - Output is cleanly captured as a string
    """
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=10  # safety limit: kill if code runs too long
        )
        output = result.stdout
        if result.stderr:
            output += f"\n[stderr]:\n{result.stderr}"
        return output.strip() if output.strip() else "(no output)"

    except subprocess.TimeoutExpired:
        return "Error: Code execution timed out after 10 seconds."
    except Exception as e:
        return f"Error running code: {str(e)}"


def read_file(filepath: str) -> str:
    """
    Reads a file from disk and returns its full contents as a string.
    Used when the user passes a file path instead of pasting code directly.
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return f"Error: File '{filepath}' not found."
    except Exception as e:
        return f"Error reading file: {str(e)}"


# Registry: maps tool names (strings) to actual Python functions.
# The agent loop uses this to know which function to call when
# the LLM returns a tool call with a given name.
TOOL_REGISTRY = {
    "run_python_code": run_python_code,
    "read_file": read_file,
}
