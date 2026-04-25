"""
agents/security_agent.py — Security specialist agent.

This sub-agent focuses exclusively on security vulnerabilities.
It doesn't care about code style or test coverage — just threats.

It does NOT execute code (running unknown code for security review
would be counterproductive and dangerous). It reads and analyzes only.
"""

import os
from google import genai
from google.genai import types
from dotenv import load_dotenv
from tools import read_file, TOOL_REGISTRY

load_dotenv()

SYSTEM_PROMPT = """You are a Python security specialist. Your only job is to identify security vulnerabilities in code.

Look for:
1. Injection vulnerabilities (SQL, shell, eval/exec)
2. Hardcoded secrets (API keys, passwords, tokens)
3. Unsafe deserialization (pickle, yaml.load)
4. Path traversal vulnerabilities
5. Insecure network calls (no SSL verification, HTTP instead of HTTPS)
6. Race conditions or insecure temp file usage
7. Overly broad exception handling that hides errors

Rules:
- Do NOT comment on code quality or style — that is handled by another agent.
- Do NOT write test cases — that is handled by another agent.
- Do NOT execute code. Read and analyze only.
- If given a file path, use read_file to get the code.
- Rate each finding as HIGH / MEDIUM / LOW severity.
- If no security issues are found, say so clearly.
"""


def run_security_agent(user_input: str) -> str:
    """
    Runs the security review agent.

    Args:
        user_input: Code snippet or file path to review.

    Returns:
        A security-focused review with severity ratings.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)

    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        tools=[read_file],
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )

    chat = client.chats.create(model="gemini-2.5-flash", config=config)
    response = chat.send_message(user_input)

    for _ in range(10):
        if not response.function_calls:
            if response.text:
                return response.text
            followup = chat.send_message(
                "Now write your complete security analysis with severity ratings based on everything you found."
            )
            return followup.text or "(Security agent returned no output.)"

        tool_results = []
        for fc in response.function_calls:
            tool_name = fc.name
            tool_args = dict(fc.args)
            print(f"    [security] → {tool_name}({tool_args})")
            result = TOOL_REGISTRY[tool_name](**tool_args)
            tool_results.append(
                types.Part.from_function_response(
                    name=tool_name,
                    response={"result": result}
                )
            )
        response = chat.send_message(tool_results)

    return "Error: Security agent exceeded max iterations."
