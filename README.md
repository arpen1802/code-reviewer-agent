# AI Code Reviewer Agent

A Python agent that reviews code by actually executing it, remembering past sessions, and enforcing safety guardrails — built progressively from a basic agent loop into a production-ready system.

---

## Project Structure

```
code-reviewer-agent/
├── agents/
│   ├── orchestrator.py       # Multi-agent coordinator + multimodality
│   ├── reviewer_agent.py     # Code quality specialist
│   ├── security_agent.py     # Security vulnerability specialist
│   └── test_writer_agent.py  # Test generation specialist
├── eval/
│   ├── tasks.json            # Task suite: 5 test cases with expected findings
│   ├── graders.py            # Code-based grader + LLM-as-judge grader
│   ├── harness.py            # Runner: executes tasks, captures trajectories
│   └── run_eval.py           # Entry point: runs suite, prints results
├── agent.py          # Core single agent loop (also used by eval)
├── tools.py          # Tool implementations (run_python_code, read_file)
├── memory.py         # Long-term memory (load/save across sessions)
├── guardrails.py     # Safety checks (content filter + action limiter)
├── main.py           # CLI entry point
├── sample_code.py    # Buggy test file for trying the agent
├── memory.json       # Auto-generated: persisted review history (gitignored)
├── requirements.txt  # Python dependencies
├── .env.example      # API key template
└── .gitignore
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
User Input (code, file path, or screenshot)
        ↓
[Guardrail] Input checked for injection attempts
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

### 🔜 Production Engineering

---

## Tech Stack

- **LLM**: Gemini 2.5 Flash via [Google GenAI SDK](https://github.com/google-gemini/generative-ai-python)
- **Embeddings**: Gemini `text-embedding-004` (768-dim) with local bag-of-words fallback
- **Vector DB**: ChromaDB (persistent, cosine similarity, stored in `~/.code_reviewer_db/`)
- **Language**: Python 3.10+
- **Concurrency**: `concurrent.futures.ThreadPoolExecutor` for parallel agents
