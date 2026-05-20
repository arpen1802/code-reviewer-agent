"""Eval-related version constants.

Bump PROMPT_VERSION whenever you change the agent's system prompt, switch
the underlying model, or restructure tool calling — anything that could
move the eval scores. The value is written into every saved eval result
so historical pass rates stay attributable to a specific prompt/model.

Convention:
    v1       initial multi-agent + ChromaDB memory
    v1.1     minor wording tweak, same scaffolding
    v2       structural change (e.g. new specialist agent, new prompt
             format) — bump the major version

This is intentionally a hand-edited constant rather than a hash, because
the version is a human story about *why* the agent changed, not a fingerprint.
"""

PROMPT_VERSION = "v1"
