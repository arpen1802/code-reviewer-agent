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

from tools import TOOL_REGISTRY, run_python_code, read_file, get_git_diff
from memory import load_memory, save_memory
from codebase_index import search_codebase
from observability import generation, gemini_usage, observed, set_session

load_dotenv()

# Register memory + codebase RAG tools so the agent loop can call them by name
TOOL_REGISTRY["load_memory"] = load_memory
TOOL_REGISTRY["save_memory"] = save_memory
TOOL_REGISTRY["search_codebase"] = search_codebase

# ── System prompt ────────────────────────────────────────────────────────────
# Updated for Day 2: the agent now has memory.
# Notice how we simply describe the expected behavior in plain English —
# the LLM figures out when and how to call each tool.

SYSTEM_PROMPT = """You are an expert Python code reviewer with memory across sessions.

At the start of every review:
1. Call load_memory(query=<the code being reviewed>) — pass the actual code as the
   query so the vector database can find the most similar past reviews.
   This uses semantic search, so you'll get relevant context, not just recent history.
2. If the user gives a file path, call read_file to get the code.
   If the user asks to review changes/a diff, call get_git_diff instead.
3. Call search_codebase with the code (or each significant diff hunk) to
   retrieve related code from the repository: callers of changed functions,
   existing helpers, and the codebase's conventions. Ground your findings in
   this context — e.g. flag when a change breaks a caller or duplicates an
   existing helper, citing the file:line locations you retrieved.
4. Call run_python_code to execute the code and observe its real output.
5. Provide a structured review covering:
   - What the code does
   - Bugs or errors found (with line numbers if possible)
   - Code quality issues (naming, structure, readability)
   - Specific, actionable suggestions for improvement
   - If you've seen similar code or issues before (from memory), mention it.
   - Repository-specific findings from search_codebase (broken callers,
     duplicated logic, convention violations), with file:line references.

At the end of every review:
6. Call save_memory with:
   - file_reviewed: the filename or description
   - issues_found: list of key issues found
   - preference_notes: any observations about the user's coding style (optional)
   - code_snippet: the code that was reviewed (so it can be embedded for future search)

Be concise but thorough. Personalize your feedback using past context when available.
"""


# ── Agent loop ────────────────────────────────────────────────────────────────

MODEL = "gemini-2.5-flash"


@observed
def run_agent(user_input: str, session_id: str | None = None) -> str:
    """
    Runs the full agent loop for a single review request.

    Args:
        user_input: Either a code snippet (string) or a file path.
        session_id: Optional session id. Used by the eval harness to tie all
                    per-task reviews under a recognizable session in Langfuse.

    Returns:
        The final review as a string.
    """
    if session_id:
        set_session(session_id, tags=["single_agent", MODEL])

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found. Copy .env.example to .env and add your key.")

    client = genai.Client(api_key=api_key)

    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        tools=[run_python_code, read_file, get_git_diff, load_memory, save_memory, search_codebase],
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )

    chat = client.chats.create(model=MODEL, config=config)

    # First Gemini call — captured as a generation observation
    with generation("gemini:agent:initial", model=MODEL, input={"user": user_input}) as g:
        response = chat.send_message(user_input)
        g.update(output=response.text or "(tool calls)", usage_details=gemini_usage(response))

    # ── The loop ──────────────────────────────────────────────────────────────
    max_iterations = 15  # bumped up slightly — memory calls add extra turns

    for turn in range(max_iterations):

        # If no function calls, the LLM is done → return the final review
        if not response.function_calls:
            if response.text:
                return response.text
            with generation(
                "gemini:agent:final-nudge", model=MODEL,
                input={"nudge": "write the final structured review"},
            ) as g:
                followup = chat.send_message(
                    "Now write your complete structured review for the user based on everything you found."
                )
                g.update(output=followup.text or "", usage_details=gemini_usage(followup))
            return followup.text or "(Agent completed but returned no text.)"

        # Execute every tool call the LLM requested (each tool is @observed)
        tool_results = []
        for fc in response.function_calls:
            tool_name = fc.name
            tool_args = dict(fc.args)

            print(f"\n  → Agent calling: {tool_name}({tool_args})")
            result = TOOL_REGISTRY[tool_name](**tool_args)

            display = str(result)
            print(f"  ← Result: {display[:120]}{'...' if len(display) > 120 else ''}")

            tool_results.append(
                types.Part.from_function_response(
                    name=tool_name,
                    response={"result": result}
                )
            )

        # Send all results back to the LLM and get its next response
        with generation(
            f"gemini:agent:turn-{turn + 1}", model=MODEL,
            input={"tool_results_count": len(tool_results)},
        ) as g:
            response = chat.send_message(tool_results)
            g.update(output=response.text or "(more tool calls)", usage_details=gemini_usage(response))

    return "Error: Agent exceeded maximum iterations without finishing."
