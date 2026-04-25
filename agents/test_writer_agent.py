"""
agents/test_writer_agent.py — Test generation specialist agent.

This sub-agent reads code and writes pytest test cases for it.
It also runs the generated tests to verify they work.

It's the only agent that *generates* code, not just analyzes it.
"""

import os
from google import genai
from google.genai import types
from dotenv import load_dotenv
from tools import run_python_code, read_file, TOOL_REGISTRY

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


def run_test_writer_agent(user_input: str) -> str:
    """
    Runs the test writing agent.

    Args:
        user_input: Code snippet or file path to generate tests for.

    Returns:
        A pytest test file as a string.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)

    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        tools=[run_python_code, read_file],
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )

    chat = client.chats.create(model="gemini-2.5-flash", config=config)
    response = chat.send_message(user_input)

    for _ in range(10):
        if not response.function_calls:
            if response.text:
                return response.text
            followup = chat.send_message(
                "Now write the complete pytest test file based on everything you found."
            )
            return followup.text or "(Test writer agent returned no output.)"

        tool_results = []
        for fc in response.function_calls:
            tool_name = fc.name
            tool_args = dict(fc.args)
            print(f"    [test_writer] → {tool_name}({tool_args})")
            result = TOOL_REGISTRY[tool_name](**tool_args)
            tool_results.append(
                types.Part.from_function_response(
                    name=tool_name,
                    response={"result": result}
                )
            )
        response = chat.send_message(tool_results)

    return "Error: Test writer agent exceeded max iterations."
