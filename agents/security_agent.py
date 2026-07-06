"""
agents/security_agent.py — Security specialist agent.

This sub-agent focuses exclusively on security vulnerabilities.
It doesn't care about code style or test coverage — just threats.

It does NOT execute code (running unknown code for security review
would be counterproductive and dangerous). It reads and analyzes only.

Observability: each invocation is one Langfuse trace (via @observed).
The Gemini call is captured as a `generation` observation with real
token counts from response.usage_metadata. Tool calls (read_file) are
traced at the tool function itself in tools.py.
"""

import os
from google import genai
from google.genai import types
from dotenv import load_dotenv

from tools import read_file, TOOL_REGISTRY
from observability import generation, gemini_usage, observed, set_session

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

MODEL = "gemini-2.5-flash"


@observed
def run_security_agent(user_input: str, session_id: str | None = None) -> str:
    """
    Runs the security review agent.

    Args:
        user_input: Code snippet or file path to review.
        session_id: Optional shared session id so the orchestrator and all
                    specialist agents group together in the Langfuse Sessions
                    view. Passed by the orchestrator; harmless if omitted.

    Returns:
        A security-focused review with severity ratings.
    """
    if session_id:
        set_session(session_id, tags=["security", MODEL])

    api_key = os.getenv("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)

    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        tools=[read_file],
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )

    chat = client.chats.create(model=MODEL, config=config)

    # First Gemini call — capture as a generation observation
    with generation("gemini:security:initial", model=MODEL, input={"user": user_input}) as g:
        response = chat.send_message(user_input)
        g.update(output=response.text or "(tool calls)", usage_details=gemini_usage(response))

    for turn in range(10):
        if not response.function_calls:
            if response.text:
                return response.text
            # Gemini finished tool calls but wrote no review — nudge it
            with generation(
                "gemini:security:final-nudge", model=MODEL,
                input={"nudge": "write final security analysis"},
            ) as g:
                followup = chat.send_message(
                    "Now write your complete security analysis with severity ratings based on everything you found."
                )
                g.update(output=followup.text or "", usage_details=gemini_usage(followup))
            return followup.text or "(Security agent returned no output.)"

        tool_results = []
        for fc in response.function_calls:
            tool_name = fc.name
            tool_args = dict(fc.args)
            print(f"    [security] → {tool_name}({tool_args})")
            # Tool function itself is @observed → automatic span
            result = TOOL_REGISTRY[tool_name](**tool_args)
            tool_results.append(
                types.Part.from_function_response(
                    name=tool_name,
                    response={"result": result},
                )
            )

        with generation(
            f"gemini:security:turn-{turn + 1}", model=MODEL,
            input={"tool_results_count": len(tool_results)},
        ) as g:
            response = chat.send_message(tool_results)
            g.update(output=response.text or "(more tool calls)", usage_details=gemini_usage(response))

    return "Error: Security agent exceeded max iterations."
