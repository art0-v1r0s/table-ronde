# 🏛️ Table-Ronde

> **Dynamic multi-agent orchestrator for technical debate and generating implementation plans v2.0 / v3.0.**

`table-ronde` is a Command Line Interface (CLI) tool powered by **LangChain** and **Rich**. It simulates a roundtable discussion between several AI agents with complementary roles and personalities (by default: the Architect, the Skeptic, and the Enthusiast) to analyze a topic, an architectural concept, or an existing codebase and produce a highly structured action plan.

Since version 2.0, **Table-Ronde is completely dynamic**: you can configure your own experts, use different LLMs simultaneously, and orchestrate debates across multiple rounds!

---

## ⚡ Features (v2.0 / v3.0)

- **Interactive Setup Menu (NEW)**: Launching `table-ronde` with zero arguments triggers a guided terminal menu (`questionary`) to configure the topic, provider, rounds, and paths interactively.
- **Immersive Visual Terminal UI (NEW)**: ASCII art banner (`pyfiglet`), colored streaming panels per agent role, round progress bars, tool call indicators, and execution duration summaries.
- **Dynamic Configuration (`config.yml`)**: Define your own personas (e.g., Security Expert, Junior Developer), their emojis, their prompts, and individual temperatures.
- **Hybrid Models**: Ability to assign a different LLM provider/model to each agent (e.g., `gpt-4o` for the Architect, `gemini-3.6-flash` for the Skeptic).
- **Multiple Rounds**: Launch in-depth debates across multiple loops (`--rounds N`).
- **Advanced Interactive Mode**: Intervene before the final synthesis to give your directives, or type `/round` to force the agents into a new debate round based on your remarks (`--interactive`).
- **Codebase Analysis**: Smart project scanner with built-in `.gitignore` filtering via `pathspec`.
- **Highly Optimized Network**: Network implementation (forcing IPv4) for instant connection times, avoiding IPv6 blackholes on the Google API.
- **Real-time Streaming**: Fluid chunk-by-chunk display of each agent's thoughts using `rich.Live`.
- **Deliverable Export**: Export the final plan in Markdown format and optionally export the full transcript of the exchanges.

---

## 🎭 Default Agents

If you don't use a custom configuration, the system launches these 3 default agents:

| Emoji | Role | Name | Description & Role in the debate | Temperature |
| :---: | :--- | :--- | :--- | :---: |
| 😈 | **Agent 1** | **The Skeptic** | The Devil's Advocate. Actively looks for flaws, technical debt, security risks, and unnecessary complexity. | `0.6` |
| 🚀 | **Agent 2** | **The Enthusiast** | The Visionary. Responds to criticisms, proposes modern solutions, seeks the fastest path to deliver value. | `0.8` |
| 🏛️ | **Agent 3** | **The Architect** | The Moderator. Guides the exchanges, pragmatically settles debates, and generates the **Final Implementation Plan** in Markdown. | `0.3` |

---

## 🛠️ Installation

The project uses [`uv`](https://github.com/astral-sh/uv) for dependency and virtual environment management (Python ≥ 3.12).

```bash
# Clone the repository
git clone https://github.com/art0-v1r0s/table-ronde.git
cd table-ronde

# Sync the environment and install dependencies
uv sync
```

---

## 🔑 API Keys Configuration

Set the environment variable corresponding to the provider you want to use:

### For Google Gemini (default)
```bash
export GEMINI_API_KEY="your_gemini_api_key"
```

### For GitHub Copilot / GitHub Models / OpenAI
```bash
export GITHUB_TOKEN="your_github_token"
# or
export COPILOT_API_KEY="your_copilot_key"
# or
export OPENAI_API_KEY="your_openai_key"
```

---

## 🚀 Usage

### 0. Guided Interactive Menu (Zero Arguments)

Launch without any arguments to start the interactive wizard:
```bash
uv run table-ronde
```
*Guides you through choosing the topic, LLM provider (Gemini, OpenAI, Copilot), debate rounds, directory scanning, and custom configs with interactive terminal prompts.*

### 1. Simple Analysis (Default Prompts and Agents)

```bash
uv run table-ronde "Design a real-time microservices architecture for an auction platform"
```

### 2. Audit and Refactoring of an Existing Project

```bash
uv run table-ronde --path /path/to/my-project "Optimize the security and scalability of the project"
```

### 3. Use Your Own Persona Configuration (NEW)

Create a `my_config.yml` file to define your custom experts:
```yaml
orchestrator:
  rounds: 2
  default_provider: gemini
  default_model: gemini-3.6-flash

architect:
  role: architect
  title: "The Architect"
  emoji: "🏛️"
  temperature: 0.2
  prompt: "You are the Architect. Moderate the experts and propose a highly detailed plan."
  # model: gpt-4o (you can override the default model here)

personas:
  - role: security
    title: "Security Expert"
    emoji: "🛡️"
    temperature: 0.3
    prompt: "You systematically look for OWASP vulnerabilities in the proposed idea."
  - role: dev
    title: "Lead Developer"
    emoji: "💻"
    temperature: 0.6
    prompt: "You talk software architecture, design patterns, and scalability."
```
Then run the tool:
```bash
uv run table-ronde --config my_config.yml "Implement an OAuth2 SSO"
```

### 4. Interactive Mode (Human-in-the-loop) and Multi-Rounds

```bash
uv run table-ronde --interactive --rounds 2 "Modernize our data pipeline architecture"
```
*The debate will pause at the end of each round to allow you to insert your directives for the Architect. If you reply with the magic command `/round`, the agents will do a full new cycle of debate based on your remark!*

---

## 🛠️ Full CLI Options

```bash
uv run table-ronde [OPTIONS] [PROMPT]
```

| Option | Shortcut | Description | Default Value |
| :--- | :--- | :--- | :--- |
| `--config` | `-c` | YAML configuration file for custom personas | `None` |
| `--rounds` | `-r` | Number of debate rounds (overrides YAML config) | `1` |
| `--interactive` | `-i` | Pause before resolution to inject your note or `/round` | `False` |
| `--path` | `-p` | Path to an existing project directory to scan | `None` |
| `--output` | `-o` | Output file for the final implementation plan | `plan_v2.md` |
| `--provider` | `-pr` | LLM Provider (`gemini`, `copilot`, `github`, `openai`) | `gemini` |
| `--model` | `-m` | Specific model to use (e.g., `gemini-3.6-flash`, `gpt-4o`) | Auto based on provider |
| `--export-transcript` | `-t` | File path to export the entire debate transcript | `None` |

---

## 🧪 Testing & Quality

Run the unit test suite with `pytest`:

```bash
uv run pytest
```

Check code quality with `ruff`:

```bash
uv run ruff check .
```

---

## 📄 License

This project is licensed under the MIT License.
