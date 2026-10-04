# 🏛️ Table-Ronde

> **Dynamic multi-agent orchestrator for technical debate, codebase auditing, and automated implementation plans.**

`table-ronde` is an advanced AI roundtable orchestrator powered by **LangChain**, **Textual**, and **Rich**. It simulates collaborative technical discussions between specialized AI agents (by default: the Architect, the Skeptic, and the Enthusiast) to analyze architectural questions, audit existing codebases, and synthesize actionable, structured implementation plans.

Now featuring **Smart Router v2.0** with **Laya** neural routing, full-screen **Textual TUI**, **multi-tier memory fabric**, and **universal cross-platform support** (Linux, macOS with Apple Silicon MPS, and Windows).

---

## ⚡ Key Features

- **🖥️ Full-Screen Interactive TUI (`Textual`)**: Modern terminal interface with real-time streaming, reactive agent panels, debate timeline, interactive human-in-the-loop modals, and a built-in plan viewer. (Use `--no-tui` for the classic Rich CLI).
- **🧠 Smart Router v2.0 (Hybrid Laya + Heuristics + LLM)**:
  - **Tier 1 (Laya)**: Ultra-fast (~33ms) neural routing based on [ConvAI Laya](https://huggingface.co/convaiinnovations/laya), running locally on NVIDIA CUDA or Apple Silicon (MPS).
  - **Tier 2 (Heuristics)**: Deterministic, zero-latency (0ms), offline keyword and complexity analyzer.
  - **Tier 3 (Structured LLM)**: Deep reasoning fallback using structured outputs (Gemini Flash, GPT-4o-mini, Claude, Ollama).
  - **Continuous Learning Loop**: Automatically tracks routing feedback and outcomes in `feedback_store.jsonl` and includes a built-in `finetune-router` CLI.
- **🌍 Universal Multi-Platform Compatibility**:
  - **macOS**: Native GPU acceleration via Apple Silicon (`mps`) and resilient IPv4/IPv6 networking.
  - **Windows**: Full PowerShell compatibility, automated UTF-8 console reconfiguration, POSIX-normalized `.gitignore` matching, and file-lock retries.
  - **Linux**: Zero-configuration support across Intel, AMD, and ARM architectures.
- **💾 Multi-Tier Resilient Memory Fabric**:
  - **Tier 1 (Working Memory)**: Redis-backed shared memory state and LangGraph checkpointing for session persistence and recovery.
  - **Tier 2 (Semantic Knowledge)**: Long-term PostgreSQL + `pgvector` store for architectural patterns and past debates.
  - **Graceful Fallbacks**: Transparently operates with in-memory state when external databases are offline.
- **🎭 Fully Configurable Personas & Hybrid Models**:
  - Define custom experts, temperatures, prompts, and distinct models per role via `config.yml` (e.g., GPT-4o for Architect, Gemini 3.8 Flash for Skeptic, Claude for Security).
- **🔄 Multi-Round Debates & Human-in-the-Loop**:
  - Multi-round consensus checking. Intervene between rounds to provide feedback or type `/round` to trigger an additional debate cycle based on your guidance.
- **📁 Smart Codebase Scanner**:
  - Priority-based file scanner respecting `.gitignore` rules, handling UTF-8 BOM, and automatically ignoring binaries (`.dll`, `.pyd`, `.so`, `.exe`).

---

## 🎭 Default Agents

| Emoji | Role | Persona | Mission | Temp |
| :---: | :--- | :--- | :--- | :---: |
| 😈 | **Agent 1** | **The Skeptic** | Devil's Advocate. Identifies security flaws, technical debt, performance bottlenecks, and edge cases. | `0.6` |
| 🚀 | **Agent 2** | **The Enthusiast** | Visionary & Builder. Counter-argues, proposes modern solutions, frameworks, and fast paths to value. | `0.8` |
| 🏛️ | **Agent 3** | **The Architect** | Moderator & Lead. Synthesizes perspectives, settles debates, and drafts the **Final Implementation Plan**. | `0.3` |

---

## 🛠️ Installation

The project uses [`uv`](https://github.com/astral-sh/uv) (Python ≥ 3.12).

### 1. Standard Lightweight Installation (Default)
By default, heavy dependencies like PyTorch and Laya are optional, keeping the install fast and lightweight:

```bash
git clone https://github.com/art0-v1r0s/table-ronde.git
cd table-ronde
uv sync
```

### 2. Optional: With Laya Neural Router (GPU / Apple Silicon)
To enable local Laya neural routing on CUDA or Apple Silicon:

```bash
uv sync --extra laya
```

---

## 🔑 Environment Variables & API Keys

Set the environment variables for your chosen providers (Linux/macOS: `export VAR="val"`, Windows PowerShell: `$env:VAR="val"`):

### LLM Providers
```bash
# Google Gemini (supports both GEMINI_API_KEY and GOOGLE_API_KEY)
export GEMINI_API_KEY="your_gemini_api_key"

# OpenAI / GitHub Models / Copilot
export OPENAI_API_KEY="your_openai_key"
export GITHUB_TOKEN="your_github_token"
export COPILOT_API_KEY="your_copilot_key"

# Anthropic Claude
export ANTHROPIC_API_KEY="your_anthropic_api_key"

# Local Ollama (no key needed; custom host supported)
export OLLAMA_HOST="http://localhost:11434"
```

### Storage & Endpoints (Optional)
```bash
# Custom directory for data and feedback store (defaults to ~/.table_ronde)
export TABLE_RONDE_DATA_DIR="/path/to/data"

# Redis URL for Tier 1 memory & LangGraph checkpointing (defaults to redis://localhost:6379)
export REDIS_URL="redis://localhost:6379"

# PostgreSQL URL for Tier 2 vector memory (defaults to postgresql://...:5432/table_ronde_memory)
export DATABASE_URL="postgresql://postgres:postgres@localhost:5432/table_ronde_memory"
```

---

## 🚀 Usage

### 1. Interactive TUI Mode (Default)
Run without arguments to launch the guided interactive wizard in the full-screen terminal UI:
```bash
uv run table-ronde
```

### 2. Command Line Direct Launch
Analyze a concept or system design directly:
```bash
uv run table-ronde "Design a real-time event-driven microservices architecture"
```

### 3. Codebase Audit & Refactoring
Point Table-Ronde to any local code repository:
```bash
uv run table-ronde --path /path/to/project "Audit security, scalability, and suggest improvements"
```

### 4. Smart Routing & Hardware Acceleration
Leverage Smart Router v2.0 with automatic hardware selection:
```bash
uv run table-ronde --smart-routing --router-strategy auto --router-device auto "Design an OAuth2 authentication service"
```
*Options for `--router-device`: `auto` (picks CUDA if available, then Apple Silicon MPS, then CPU), `cuda`, `mps`, `cpu`.*

### 5. Custom Personas (`config.yml`)
Define custom experts and assigned models in YAML:
```yaml
orchestrator:
  rounds: 2
  default_provider: gemini
  default_model: gemini-3.8-flash

architect:
  role: architect
  title: "Principal Architect"
  emoji: "🏛️"
  temperature: 0.2
  prompt: "Synthesize debates into an actionable production plan."

personas:
  - role: security
    title: "Security Auditor"
    emoji: "🛡️"
    temperature: 0.2
    prompt: "Identify OWASP Top 10 risks and zero-trust vulnerabilities."
  - role: devops
    title: "SRE Lead"
    emoji: "⚙️"
    temperature: 0.4
    prompt: "Focus on Kubernetes, CI/CD, observability, and failover."
```
Run with:
```bash
uv run table-ronde --config my_config.yml "Deploy a multi-region payment processing platform"
```

### 6. Fine-Tuning Laya on Accumulated Feedback
Fine-tune the neural router on real debate outcomes gathered in your environment:
```bash
uv run table-ronde finetune-router --epochs 3 --batch-size 8
```

### 7. Headless / Non-TUI Mode
For CI/CD pipelines, scripting, or minimal environments:
```bash
uv run table-ronde --no-tui "Refactor authentication flow" -o plan.md
```

---

## 🛠️ CLI Reference

### `table-ronde [prompt] [OPTIONS]`

| Option | Shortcut | Description | Default |
| :--- | :--- | :--- | :--- |
| `--path` | `-p` | Path to an existing project directory to scan | `None` |
| `--output` | `-o` | Output file path for the final implementation plan | `plan_v2.md` |
| `--config` | `-c` | Path to custom YAML configuration file | `None` |
| `--rounds` | `-r` | Number of debate rounds | `1` |
| `--provider` | `-pr` | LLM Provider (`gemini`, `openai`, `copilot`, `claude`, `ollama`) | `gemini` |
| `--model` | `-m` | Model identifier (e.g. `gemini-3.8-flash`, `gpt-4o`) | Provider default |
| `--interactive` | `-i` | Pause after each round for human guidance or `/round` | `False` |
| `--export-transcript` | `-t` | Export file path for the complete debate transcript | `None` |
| `--smart-routing` | | Enable Smart Router (System One task analysis & consensus) | `False` |
| `--router-strategy` | | Strategy: `auto` (Laya ➔ Heuristic ➔ LLM), `laya`, `heuristic`, `llm` | `auto` |
| `--router-device` | | Device for Laya inference: `auto`, `cuda`, `mps`, `cpu` | `auto` |
| `--no-tui` | | Disable full-screen Textual TUI and use classic streaming CLI | `False` |
| `--save-session` | | Path to save the debate session state (JSON) | `None` |
| `--resume` | | Path to resume a debate session from JSON | `None` |

---

## 🧪 Testing & Quality

Run the test suite across all modules (94+ unit and integration tests):
```bash
uv run pytest
```

Run code formatting and static analysis:
```bash
uv run ruff check .
```

---

## 📄 License

This project is licensed under the MIT License.
