"""
agents/test_writer_agent.py — Test generation specialist agent.

This sub-agent reads code and writes pytest test cases for it.
It also runs the generated tests to verify they work.

It's the only agent that *generates* code, not just analyzes it.

Observability: each invocation is one Langfuse trace (via @observed).
Each Gemini call is captured as a `generation` observation with real
token counts. Tool calls are traced at the tool function in tools.py.
"""

import os
from google import genai
from google.genai import types
from dotenv import load_dotenv

from tools import run_python_code, read_file, TOOL_REGISTRY
from observability import generation, gemini_usage, observed, set_session

load_dotenv()

SYSTEM_PROMPT = """You are a Python testing specialist. Your only job is to write pytest test cases for the code you're given.

Your process:
1. If given a file path, use read_file to get the code.
2. Understand what each function is supposed to do.
3. Write pytest tests covering:
   - Normal (happy path) cases
   - Edge cases (empty input, None, zero, very large values)
   - Error cases (inputs that should raise exceptions)
4. Use run_python_code to verify your tests actually run (even if they fail on buggy code).
5. Output the full test file content as your final response.

Rules:
- Do NOT comment on code quality — that is handled by another agent.
- Do NOT identify security issues — that is handled by another agent.
- Use pytest conventions: test functions named test_*, clear assert statements.
- If the code has bugs, still write tests — tests document expected behavior even when code is broken.
"""

MODEL = "gemini-2.5-flash"


@observed
def run_test_writer_agent(user_input: str, session_id: str | None = None) -> str:
    """
    Runs the test writing agent.

    Args:
        user_input: Code snippet or file path to generate tests for.
        session_id: Optional shared session id (see reviewer_agent.py).

    Returns:
        A pytest test file as a string.
    """
    if session_id:
        set_session(session_id, tags=["test_writer", MODEL])

    api_key = os.getenv("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)

    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        tools=[run_python_code, read_file],
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )

    chat = client.chats.create(model=MODEL, config=config)

    with generation("gemini:test_writer:initial", model=MODEL, input={"user": user_input}) as g:
        response = chat.send_message(user_input)
        g.update(output=response.text or "(tool calls)", usage_details=gemini_usage(response))

    for turn in range(10):
        if not response.function_calls:
            if response.text:
                return response.text
            with generation(
                "gemini:test_writer:final-nudge", model=MODEL,
                input={"nudge": "write the complete pytest test file"},
            ) as g:
                followup = chat.send_message(
                    "Now write the complete pytest test file based on everything you found."
                )
                g.update(output=followup.text or "", usage_details=gemini_usage(followup))
            return followup.text or "(Test writer agent returned no output.)"

        tool_results = []
        for fc in response.function_calls:
            tool_name = fc.name
            tool_args = dict(fc.args)
            print(f"    [test_writer] → {tool_name}({tool_args})")
            # Tool function itself is @observed → automatic span
            result = TOOL_REGISTRY[tool_name](**tool_args)
            tool_results.append(
                types.Part.from_function_response(
                    name=tool_name,
                    response={"result": result},
                )
            )

        with generation(
            f"gemini:test_writer:turn-{turn + 1}", model=MODEL,
            input={"tool_results_count": len(tool_results)},
        ) as g:
            response = chat.send_message(tool_results)
            g.update(output=response.text or "(more tool calls)", usage_details=gemini_usage(response))

    return "Error: Test writer agent exceeded max iterations."
