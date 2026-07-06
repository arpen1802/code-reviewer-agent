"""
agents/reviewer_agent.py — Code quality specialist agent.

This is a sub-agent. It has one job: review code quality.
It doesn't think about security or tests — that's other agents' business.

Notice that this agent has the same structure as the original agent.py —
same loop, same tool registry pattern. The only thing that changes is
the system prompt, which narrows the agent's focus to quality alone.

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

SYSTEM_PROMPT = """You are a code quality specialist. Your only job is to review Python code for:

1. Bugs and logic errors (with line numbers)
2. Code readability and naming conventions
3. Structure, complexity, and maintainability
4. Performance issues

Rules:
- Do NOT comment on security — that is handled by another agent.
- Do NOT write test cases — that is handled by another agent.
- If given a file path, use read_file to get the code first.
- Use run_python_code to execute the code and see its real output before reviewing.
- Be specific and actionable. Include line numbers where possible.
- Output a clean, structured review.
"""

MODEL = "gemini-2.5-flash"


@observed
def run_reviewer_agent(user_input: str, session_id: str | None = None) -> str:
    """
    Runs the code quality review agent.

    Args:
        user_input: Code snippet or file path to review.
        session_id: Optional shared session id so the orchestrator and all
                    specialist agents group together in the Langfuse Sessions
                    view. Passed by the orchestrator; harmless if omitted.

    Returns:
        A structured quality review as a string.
    """
    if session_id:
        set_session(session_id, tags=["reviewer", MODEL])

    api_key = os.getenv("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)

    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        tools=[run_python_code, read_file],
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )

    chat = client.chats.create(model=MODEL, config=config)

    with generation("gemini:reviewer:initial", model=MODEL, input={"user": user_input}) as g:
        response = chat.send_message(user_input)
        g.update(output=response.text or "(tool calls)", usage_details=gemini_usage(response))

    for turn in range(10):
        if not response.function_calls:
            if response.text:
                return response.text
            with generation(
                "gemini:reviewer:final-nudge", model=MODEL,
                input={"nudge": "write the final quality review"},
            ) as g:
                followup = chat.send_message(
                    "Now write your complete structured code quality review based on everything you found."
                )
                g.update(output=followup.text or "", usage_details=gemini_usage(followup))
            return followup.text or "(Reviewer agent returned no output.)"

        tool_results = []
        for fc in response.function_calls:
            tool_name = fc.name
            tool_args = dict(fc.args)
            print(f"    [reviewer] → {tool_name}({tool_args})")
            # Tool function itself is @observed → automatic span
            result = TOOL_REGISTRY[tool_name](**tool_args)
            tool_results.append(
                types.Part.from_function_response(
                    name=tool_name,
                    response={"result": result},
                )
            )

        with generation(
            f"gemini:reviewer:turn-{turn + 1}", model=MODEL,
            input={"tool_results_count": len(tool_results)},
        ) as g:
            response = chat.send_message(tool_results)
            g.update(output=response.text or "(more tool calls)", usage_details=gemini_usage(response))

    return "Error: Reviewer agent exceeded max iterations."
