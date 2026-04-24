"""
agent.py — The core agent loop.

This is the heart of the project. It wires together:
  - The Gemini LLM (the "brain" that decides what to do)
  - The tools in tools.py (the "hands" that actually do things)
  - The memory tools in memory.py (long-term knowledge across sessions)
  - The loop that keeps going until the LLM is done

Flow for each review:
  1. User gives us code (or a file path)
  2. We send it to Gemini with a system prompt + tool definitions
  3. Gemini replies with either:
       a. A tool call  → we execute it, send the result back, loop again
       b. Plain text   → that's the final review, we're done

Day 2 additions:
  - load_memory and save_memory tools
  - Updated system prompt that instructs the agent to use memory
"""

import os
from google import genai
from google.genai import types
from dotenv import load_dotenv
from tools import TOOL_REGISTRY, run_python_code, read_file
from memory import load_memory, save_memory

load_dotenv()

# Register memory tools so the agent loop can call them by name
TOOL_REGISTRY["load_memory"] = load_memory
TOOL_REGISTRY["save_memory"] = save_memory

# ── System prompt ────────────────────────────────────────────────────────────
# Updated for Day 2: the agent now has memory.
# Notice how we simply describe the expected behavior in plain English —
# the LLM figures out when and how to call each tool.

SYSTEM_PROMPT = """You are an expert Python code reviewer with memory across sessions.

At the start of every review:
1. Call load_memory to recall past reviews and user preferences.
2. If the user gives a file path, call read_file to get the code.
3. Call run_python_code to execute the code and observe its real output.
4. Provide a structured review covering:
   - What the code does
   - Bugs or errors found (with line numbers if possible)
   - Code quality issues (naming, structure, readability)
   - Specific, actionable suggestions for improvement
   - If you've seen this file before, note what has changed or improved.

At the end of every review:
5. Call save_memory with the filename, a list of the key issues found,
   and any observations about the user's coding style worth remembering.

Be concise but thorough. Personalize your feedback using past context when available.
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

    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        tools=[run_python_code, read_file, load_memory, save_memory],
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )

    chat = client.chats.create(model="gemini-2.5-flash", config=config)

    # Send the user's first message
    response = chat.send_message(user_input)

    # ── The loop ──────────────────────────────────────────────────────────────
    max_iterations = 15  # bumped up slightly — memory calls add extra turns

    for _ in range(max_iterations):

        # If no function calls, the LLM is done → return the final review
        if not response.function_calls:
            if response.text:
                return response.text
            # Gemini finished tool calls but wrote no review — nudge it
            followup = chat.send_message(
                "Now write your complete structured review for the user based on everything you found."
            )
            return followup.text or "(Agent completed but returned no text.)"

        # Execute every tool call the LLM requested
        tool_results = []
        for fc in response.function_calls:
            tool_name = fc.name
            tool_args = dict(fc.args)

            print(f"\n  → Agent calling: {tool_name}({tool_args})")
            result = TOOL_REGISTRY[tool_name](**tool_args)

            # Memory results can be long — truncate display only, not the actual result
            display = str(result)
            print(f"  ← Result: {display[:120]}{'...' if len(display) > 120 else ''}")

            tool_results.append(
                types.Part.from_function_response(
                    name=tool_name,
                    response={"result": result}
                )
            )

        # Send all results back to the LLM and get its next response
        response = chat.send_message(tool_results)

    return "Error: Agent exceeded maximum iterations without finishing."
