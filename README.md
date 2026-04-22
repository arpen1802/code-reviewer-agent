# AI Code Reviewer Agent

A Python agent that reviews code by actually executing it, remembering past sessions, and enforcing safety guardrails — built progressively from a basic agent loop into a production-ready system.

---

## Project Structure

```
code-reviewer-agent/
├── agent.py          # Core agent loop (LLM + tool orchestration)
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
# Review a file
python main.py sample_code.py

# Paste code interactively
python main.py
```

---

## How It Works

The agent follows a loop:

```
User submits code / file path
        ↓
[Guardrail] Input checked for injection attempts
        ↓
LLM loads memory → reads file → runs code → writes review → saves memory
        ↓
Each tool call: LLM decides → runtime executes → result sent back to LLM
        ↓
Loop ends when LLM produces plain text (the final review)
```

The LLM is the **brain** — it decides which tools to call and in what order.  
The Python runtime is the **hands** — it actually executes everything.

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

### 🔜 AI Agent Workflows, Multi-Agent Systems, Multimodality
### 🔜 Agent Evaluation
### 🔜 Production Engineering

---

## Tech Stack

- **LLM**: Gemini 2.5 Flash via [Google GenAI SDK](https://github.com/google-gemini/generative-ai-python)
- **Language**: Python 3.10+
- **Memory**: JSON file (to be upgraded to a database in Day 5)
