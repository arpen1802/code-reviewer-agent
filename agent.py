"""
agent.py — The core agent loop.

This is the heart of the project. It wires together:
  - The Gemini LLM (the "brain" that decides what to do)
  - The tools in tools.py (the "hands" that actually do things)
  - The loop that keeps going until the LLM is done

Flow for each review:
  1. User gives us code (or a file path)
  2. We send it to Gemini with a system prompt + tool definitions
  3. Gemini replies with either:
       a. A tool call  → we execute it, send the result back, loop again
       b. Plain text   → that's the final review, we're done
"""

import os
from google import genai
from google.genai import types
from dotenv import load_dotenv
from tools import TOOL_REGISTRY, run_python_code, read_file

load_dotenv()

# ── System prompt ────────────────────────────────────────────────────────────
# This tells the LLM what role it plays and how to behave.
# Later (Day 2) we will move this into a more structured config.

SYSTEM_PROMPT = """You are an expert Python code reviewer.

When given code to review, you should:
1. First use the run_python_code tool to actually execute the code and observe its real output.
2. If the user gives a file path instead of code, use read_file to get the code first.
3. After running it, provide a structured review covering:
   - What the code does
   - Bugs or errors found (with line numbers if possible)
   - Code quality issues (naming, structure, readability)
   - Specific, actionable suggestions for improvement

Be concise but thorough. Use examples in your suggestions where helpful.
"""


# ── Agent loop ────────────────────────────────────────────────────────────────

def run_agent(user_input: str) -> str:
    """
    Runs the full agent loop for a single review request.

    Args:
        user_input: Either a code snippet (string) or a file path.

    Returns:
        The final review as a string.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found. Copy .env.example to .env and add your key.")

    client = genai.Client(api_key=api_key)

    # We pass the actual Python functions as tools.
    # The new SDK reads their names and docstrings automatically.
    # We disable automatic_function_calling so we control the loop ourselves —
    # this makes the agent loop visible and educational.
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        tools=[run_python_code, read_file],
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )

    chat = client.chats.create(model="gemini-2.5-flash", config=config)

    # Send the user's first message
    response = chat.send_message(user_input)

    # ── The loop ──────────────────────────────────────────────────────────────
    # This is the core of what makes it an "agent" rather than a one-shot call.
    # We keep going as long as the LLM wants to call tools.

    max_iterations = 10  # safety cap to prevent infinite loops

    for _ in range(max_iterations):

        # If no function calls, the LLM is done → return the final review
        if not response.function_calls:
            return response.text

        # Otherwise, execute every tool call the LLM requested
        tool_results = []
        for fc in response.function_calls:
            tool_name = fc.name
            tool_args = dict(fc.args)

            print(f"\n  → Agent calling: {tool_name}({tool_args})")
            result = TOOL_REGISTRY[tool_name](**tool_args)
            print(f"  ← Result: {result[:120]}{'...' if len(result) > 120 else ''}")

            tool_results.append(
                types.Part.from_function_response(
                    name=tool_name,
                    response={"result": result}
                )
            )

        # Send all results back to the LLM and get its next response
        response = chat.send_message(tool_results)

    return "Error: Agent exceeded maximum iterations without finishing."
