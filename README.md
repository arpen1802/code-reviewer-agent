# AI Code Reviewer Agent

A multi-agent Python code reviewer that executes the code it reviews, retrieves context from the target codebase (RAG), reviews git diffs with repository awareness, remembers past sessions, and enforces safety guardrails — built progressively from a basic agent loop into a production-ready system with evals and observability.

---

## Project Structure

```
code-reviewer-agent/
├── .github/workflows/
│   └── evals.yml             # CI: runs the eval suite on PRs to main
├── agents/
│   ├── orchestrator.py       # Multi-agent coordinator + multimodality + session_id
│   ├── reviewer_agent.py     # Code quality specialist
│   ├── security_agent.py     # Security vulnerability specialist
│   └── test_writer_agent.py  # Test generation specialist
├── docs/
│   └── evals.md              # Eval suite docs: how to run, how to add tasks
├── eval/
│   ├── tasks.json            # Task suite: 30 test cases with expected findings
│   ├── graders.py            # Code-based grader + LLM-as-judge grader
│   ├── harness.py            # Runner: executes tasks, captures trajectories
│   ├── version.py            # PROMPT_VERSION baked into every saved result
│   ├── run_eval.py           # Entry point: --save baselines, --compare diffs
│   └── results/              # Saved baselines: <ts>-<sha>.json
├── agent.py          # Core single-agent loop (also used by eval harness)
├── tools.py          # Tool implementations (run_python_code, read_file, get_git_diff)
├── codebase_index.py # RAG over the target repo (AST chunking + ChromaDB retrieval)
├── memory.py         # ChromaDB long-term memory (load/save across sessions)
├── observability.py  # Langfuse @observed / span / generation helpers
├── guardrails.py     # Safety checks (content filter + action limiter)
├── main.py           # CLI entry point (flushes Langfuse on exit)
├── sample_code.py    # Buggy test file for trying the agent
├── requirements.txt  # Python dependencies
├── .env.example      # GEMINI_API_KEY + optional LANGFUSE_* keys
└── .gitignore        # Ignores chroma_db/, logs/, memory.json
```

---

## Setup

```bash
# 1. Clone the repo
git clone <your-repo-url>
cd code-reviewer-agent

# 2. Create and activate virtual environment
python -m venv ai-agent
source ai-agent/bin/activate  # Mac/Linux
ai-agent\Scripts\activate     # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your Gemini API key
cp .env.example .env
# Open .env and paste your key from https://aistudio.google.com
```

---

## Usage

```bash
# Index a codebase for repository-aware (RAG) review — run once per repo
python main.py --index /path/to/repo

# Review the uncommitted changes in a repo (diff review mode)
python main.py --diff /path/to/repo          # against HEAD
python main.py --diff /path/to/repo main     # feature branch against main

# Review a Python file
python main.py sample_code.py

# Review a screenshot of code (multimodality)
python main.py --image screenshot.png

# Paste code interactively
python main.py
```

---

## How It Works

### Multi-Agent Architecture

```
User Input (code, file path, git diff, or screenshot)
        ↓
[Guardrail] Input checked for injection attempts
        ↓
[RAG] Memory (similar past reviews) + Codebase index (related code)
        ↓
Orchestrator Agent
  ├── [parallel] Reviewer Agent    → code quality, bugs, readability
  ├── [parallel] Security Agent    → vulnerabilities, severity ratings
  └── [parallel] Test Writer Agent → pytest test cases
        ↓
Orchestrator merges all three reports → Final Review
        ↓
Memory saved for next session
```

The three specialist agents run **in parallel** using Python threads, so the full review takes roughly the same time as a single agent call.

### Multimodality

If you pass `--image screenshot.png`, the orchestrator first sends the image to Gemini's vision model to extract the Python code, then routes the extracted code through the normal review pipeline. You can review code from screenshots, not just text files.

---

## Features

### ✅ Core Agent + Tools
- Basic agent loop with `run_python_code` and `read_file` tools
- LLM executes code before reviewing so feedback is grounded in real output
- Manual tool-calling loop (no automatic function calling) so the flow is transparent

### ✅ Memory
- **Long-term memory** (`memory.py`): agent persists review history and user preferences across sessions in `memory.json`
- `load_memory` called at the start of each review to recall past context
- `save_memory` called at the end to record issues found

### ✅ Guardrails
- Two-layer safety system via `guardrails.py`
  - Layer 1 (content filter): blocks prompt injection in user input before any API call
  - Layer 2 (action limiter): blocks dangerous code patterns before subprocess execution

### ✅ Multi-Agent Systems + Multimodality
- Orchestrator pattern: one coordinator delegates to three specialist sub-agents
- Parallel execution: all three agents run simultaneously via `ThreadPoolExecutor`
- Specialist agents: Reviewer (quality), Security (vulnerabilities), Test Writer (pytest generation)
- Multimodality: accepts code screenshots via Gemini's vision API (`--image` flag)

### ✅ Agent Evaluation
- 4-component eval pipeline: Task Suite → Infrastructure → Criteria → Grading
- Task suite (`eval/tasks.json`): 5 test cases covering bugs, security, performance, clean code, and edge cases
- Two graders: code-based keyword checker (fast, free) + LLM-as-judge (quality scoring 1–5)
- Trajectory capture: harness records every tool call made during each review
- Memory isolation: eval resets `memory.json` between tasks to prevent contamination
- CLI flags: `--no-llm-judge` for fast runs, `--task` for single task debug, `--verbose` for full output

### ✅ Vector Database Memory

Upgraded `memory.py` from a flat JSON file to a **ChromaDB vector database** with semantic search.

Instead of "give me the last 5 reviews," the agent now asks "give me the reviews most similar to *this specific code*."
This is the RAG (Retrieval-Augmented Generation) pattern.

**How it works:**

1. `save_memory` embeds the reviewed code + issues into a 768-dim float vector (via Gemini `text-embedding-004`) and stores it in ChromaDB.
2. `load_memory(query=<current code>)` embeds the current code, then finds the top-3 most similar past reviews using cosine similarity.
3. Only relevant context enters the prompt — no token bloat as history grows.

**Local fallback:** In environments without Gemini API access, a bag-of-words local embedder activates automatically. In production (on your own machine), the Gemini path is used.

```
JSON file memory (Day 2):          Vector DB memory (Day 5):
load last 5 reviews                embed current code
      ↓                                    ↓
all 5 go into prompt               semantic search → top 3 matches
                                          ↓
                             only relevant reviews go into prompt
```

### ✅ Codebase RAG: repository-aware review

Reviews are grounded in the *actual repository*, not just the snippet under review. This is a second, separate RAG pipeline (`codebase_index.py`) alongside review memory:

1. `--index <repo>` walks the repo and splits every `.py` file into semantic chunks using the **AST** — one chunk per top-level function/class, plus a module-level chunk for imports and constants.
2. Chunks are embedded and stored in a dedicated ChromaDB collection with `path`, `symbol`, and line-range metadata.
3. During review, agents retrieve the most relevant chunks via the `search_codebase` tool (the orchestrator also does one shared retrieval pass for all three specialists).

The result: findings like *"this change breaks `run_eval.py:41`, which still calls the old signature"* or *"this duplicates the existing helper in `guardrails.py`"* — with real file:line references.

```
Review request (file or diff)
        ↓
embed → search codebase index → top-k related chunks
        ↓
callers / helpers / conventions injected into agent context
        ↓
repository-specific findings with file:line citations
```

### ✅ Diff review mode

`--diff <repo> [base_ref]` reviews *what changed* instead of whole files — the way real code review works:

- Pulls `git diff --unified=5` against `base_ref` (default `HEAD`, pass `main` to review a feature branch)
- Combines each hunk with retrieved codebase context, so agents can check whether a change breaks callers or violates existing conventions
- Available to the agent as a `get_git_diff` tool as well, so it can fetch diffs itself when asked

### ✅ Production Engineering: Langfuse tracing

Tracing lives in [`observability.py`](./observability.py) — a thin wrapper over the [Langfuse](https://langfuse.com) Python SDK that silently no-ops when `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` are unset, so the project still runs without an account.

**What's captured** when Langfuse is configured:

- **One trace per agent invocation** — `run_orchestrator`, `run_reviewer_agent`, `run_security_agent`, `run_test_writer_agent`, `run_agent`, and the four tool functions are all `@observed`. Input/output are auto-captured from function args and return values.
- **Real Gemini token counts** — every `chat.send_message(...)` is wrapped in a `generation` observation that reads `response.usage_metadata.prompt_token_count` / `candidates_token_count`. Langfuse computes per-trace cost from these via its model pricing table.
- **Sessions view** — the orchestrator generates one `session_id` per user request and propagates it to each specialist agent. All four traces (orchestrator + reviewer + security + test_writer) appear together in Langfuse's Sessions view, so you can see a complete multi-agent run at a glance.
- **Parallel-agent fan-out** — each specialist runs in its own thread and creates its own trace. They share the same session_id (queryable as one unit) without us having to fight OpenTelemetry context propagation across threads.
- **Spans for vision and memory** — the orchestrator's image-extraction call (multimodality) and ChromaDB memory load/save are captured as nested observations inside the orchestrator trace.

This is the same observability stack used in [scalable-take-home](https://github.com/arpen1802/scalable-take-home) ([`src/app/observability.py`](https://github.com/arpen1802/scalable-take-home/blob/main/src/app/observability.py)), so all my agent projects share one tool and one mental model.

**Setup:** sign up for Langfuse Cloud (free tier), grab the public+secret keys from `Settings → API Keys`, and put them in `.env` (see `.env.example`).

---

## Roadmap

These are the next concrete things this repo wants but doesn't have yet.

- **Cost telemetry in eval results.** Sum input + output tokens per task and persist them alongside pass rates so we can chart cost vs score over time across prompt versions.
- **Trajectory-based grader.** Assert that the agent always called `run_python_code` before claiming a bug exists. Belongs in `eval/graders.py` as a third grader.
- **Per-category pass-rate floors as CI gates.** Fail the build if, say, `security` drops below 70% — see `.github/workflows/evals.yml`.
- **`--quick` smoke set** (one task per category) for fast prompt-iteration runs without paying the full 30-task tax.
- **Langfuse dataset wiring.** Push the eval task suite into a Langfuse dataset so eval runs land in the Langfuse UI alongside production traces — same UI for both QA and prod debugging.

---

## Tech Stack

- **LLM**: Gemini 2.5 Flash via [Google GenAI SDK](https://github.com/google-gemini/generative-ai-python)
- **Embeddings**: Gemini `text-embedding-004` (768-dim) with local bag-of-words fallback
- **Vector DB**: ChromaDB (persistent, cosine similarity, stored in `~/.code_reviewer_db/`)
- **Language**: Python 3.10+
- **Concurrency**: `concurrent.futures.ThreadPoolExecutor` for parallel agents
