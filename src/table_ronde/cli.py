import logging
import os
import socket
import time
import warnings
from collections.abc import Callable, Generator
from pathlib import Path
from typing import Any

import typer
import yaml
from langchain_core.messages import BaseMessageChunk
from rich.prompt import Prompt
from rich.rule import Rule

# Network optimization: Force IPv4 to prevent 80s IPv6 timeout (blackholing)
old_getaddrinfo = socket.getaddrinfo


def new_getaddrinfo(*args, **kwargs):
    responses = old_getaddrinfo(*args, **kwargs)
    return [response for response in responses if response[0] == socket.AF_INET]


socket.getaddrinfo = new_getaddrinfo

# Suppress unnecessary warnings
logging.getLogger("google_genai.models").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*fixed sampling defaults.*")

from table_ronde import ui
from table_ronde.agents import TableRondeAgents
from table_ronde.menu import run_interactive_menu
from table_ronde.orchestrator import Orchestrator

app = typer.Typer(
    name="table-ronde",
    help="Dynamic multi-agent orchestrator to debate and design implementation plans",
)


def make_stream_callback(agents: TableRondeAgents) -> Callable:
    def callback(role: str, gen: Generator[BaseMessageChunk, None, None]) -> str:
        title = role
        emoji = "🤖"
        agent_index = 0

        if role == agents.architect_cfg.get("role"):
            title = agents.architect_cfg.get("title", "Architect")
            emoji = agents.architect_cfg.get("emoji", "🏛️")
        else:
            for idx, p in enumerate(agents.personas):
                if p["role"] == role:
                    title = p.get("title", role)
                    emoji = p.get("emoji", "🤖")
                    agent_index = idx
                    break

        panel = ui.AgentStreamPanel(
            title=title, emoji=emoji, role=role, agent_index=agent_index
        )
        full_text = ""

        try:
            panel.start()
            for chunk in gen:
                if hasattr(chunk, "content") and chunk.content:
                    if isinstance(chunk.content, str):
                        full_text += chunk.content
                    elif isinstance(chunk.content, list):
                        for part in chunk.content:
                            if isinstance(part, str):
                                full_text += part
                            elif isinstance(part, dict) and "text" in part:
                                full_text += part["text"]
                    panel.update(full_text)
        finally:
            panel.finish()

        return full_text

    return callback


def make_phase_callback() -> Callable[[str, dict[str, Any]], None]:
    def callback(phase: str, data: dict[str, Any]) -> None:
        if phase == "opening":
            architect_name = data.get("architect", "Architect")
            ui.print_phase_header(
                "⚡ PHASE 1 — OPENING", f"{architect_name} sets the stage"
            )
        elif phase == "round":
            ui.print_round_header(data["current"], data["total"])
        elif phase == "resolution":
            architect_name = data.get("architect", "Architect")
            ui.print_phase_header(
                "🏁 PHASE 3 — RESOLUTION",
                f"{architect_name} synthesizes the final plan",
            )
        elif phase == "tool_call":
            ui.print_tool_call(data["name"], data.get("args", ""))

    return callback


def make_human_input_callback(interactive: bool) -> Callable[[], str | None] | None:
    if not interactive:
        return None

    def ask() -> str | None:
        ui.console.print()
        ui.console.print(Rule(" 💬 Human-in-the-Loop 💬 ", style="bold yellow"))
        note = Prompt.ask(
            "[bold yellow]Your note for the Architect (or '/round' for another debate round)[/bold yellow]\n"
            "[dim](Press Enter to proceed without notes)[/dim]",
            default="",
            console=ui.console,
        )
        ui.console.print()
        return note.strip() or None

    return ask


@app.command()
def main(
    prompt: str | None = typer.Argument(
        None, help="Project description or topic to analyze"
    ),
    path: Path | None = typer.Option(
        None, "--path", "-p", help="Path to an existing project directory to analyze"
    ),
    output: Path = typer.Option(
        Path("plan_v2.md"), "--output", "-o", help="Output file for the final plan"
    ),
    config_file: Path | None = typer.Option(
        None, "--config", "-c", help="YAML configuration file for personas and models"
    ),
    rounds: int | None = typer.Option(
        None, "--rounds", "-r", help="Number of debate rounds (overrides config)"
    ),
    provider: str = typer.Option(
        "gemini",
        "--provider",
        "-pr",
        help="LLM Provider ('gemini', 'openai', 'copilot' / 'github', 'claude', or 'ollama')",
    ),
    model: str | None = typer.Option(
        None,
        "--model",
        "-m",
        help="Default model to use (e.g., 'gemini-3.6-flash' or 'gpt-4o')",
    ),
    export_transcript: Path | None = typer.Option(
        None,
        "--export-transcript",
        "-t",
        help="File to export the full debate transcript",
    ),
    interactive: bool = typer.Option(
        False,
        "--interactive",
        "-i",
        help="Pauses the debate before resolution to gather your note or start a new round",
    ),
    save_session: Path | None = typer.Option(
        None, "--save-session", help="Path to save the debate session state (JSON)"
    ),
    resume: Path | None = typer.Option(
        None, "--resume", help="Path to a saved session JSON to resume from"
    ),
    no_tui: bool = typer.Option(
        False,
        "--no-tui",
        help="Disable the full-screen Textual TUI and use legacy Rich CLI mode instead",
    ),
    jev: bool = typer.Option(
        False,
        "--jev/--no-jev",
        help="Enable JevEngine consensus and note routing",
    ),
):
    # ── TUI mode (default) ──
    if not no_tui and not resume:
        from table_ronde.tui import run_tui

        if prompt or path:
            # CLI args provided → skip setup screen, launch debate directly
            tui_config = {
                "prompt": prompt or "Analysis and improvement of the provided project.",
                "provider": provider,
                "model": model,
                "rounds": rounds or 1,
                "interactive": interactive,
                "path": str(path) if path else None,
                "config_file": str(config_file) if config_file else None,
                "output": str(output),
            }
            run_tui(config=tui_config)
        else:
            # No args → show setup screen
            run_tui()
        return

    # ── Legacy Rich CLI mode (--no-tui or --resume) ──
    if not prompt and not path and not resume and not config_file:
        try:
            menu_opts = run_interactive_menu()
        except (KeyboardInterrupt, SystemExit):
            raise typer.Exit(code=0)

        prompt = menu_opts["prompt"]
        provider = menu_opts["provider"]
        rounds = menu_opts["rounds"]
        interactive = menu_opts["interactive"]
        if menu_opts.get("path"):
            path = Path(menu_opts["path"])
        if menu_opts.get("config_file"):
            config_file = Path(menu_opts["config_file"])
        if menu_opts.get("output"):
            output = Path(menu_opts["output"])

    if not prompt and not path and not resume:
        ui.console.print(
            "[bold red]Error: You must provide a topic, a path, or a session to resume.[/bold red]"
        )
        raise typer.Exit(code=1)

    prov_clean = provider.lower()
    if prov_clean in ("copilot", "github"):
        if not (os.getenv("GITHUB_TOKEN") or os.getenv("COPILOT_API_KEY")):
            ui.console.print(
                "[bold yellow]Warning: GITHUB_TOKEN or COPILOT_API_KEY is not set in the environment.[/bold yellow]"
            )
    elif prov_clean == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            ui.console.print(
                "[bold yellow]Warning: OPENAI_API_KEY is not set in the environment.[/bold yellow]"
            )
    elif prov_clean == "gemini":
        if not os.getenv("GEMINI_API_KEY"):
            ui.console.print(
                "[bold yellow]Warning: GEMINI_API_KEY is not set in the environment.[/bold yellow]"
            )
    elif prov_clean == "claude":
        if not os.getenv("ANTHROPIC_API_KEY"):
            ui.console.print(
                "[bold yellow]Warning: ANTHROPIC_API_KEY is not set in the environment.[/bold yellow]"
            )
    elif prov_clean == "ollama":
        pass  # No API key needed for local Ollama

    config = None
    if config_file:
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
                ui.console.print(
                    f"[dim]⚙️  Configuration loaded from: {config_file}[/dim]"
                )
        except Exception as e:
            ui.console.print(
                f"[bold red]Error loading configuration {config_file}: {e}[/bold red]"
            )
            raise typer.Exit(code=1)

    if config is None:
        config = {}
    
    if "orchestrator" not in config:
        config["orchestrator"] = {}

    if rounds is not None:
        config["orchestrator"]["rounds"] = rounds

    config["orchestrator"]["jev_enabled"] = jev

    user_prompt = prompt or "Analysis and improvement of the provided project."

    # ── Display visual banner, participants, and configuration ──
    ui.print_banner(provider)

    try:
        agents = TableRondeAgents(config=config, provider=provider, model_name=model)
        orchestrator = Orchestrator(
            agents,
            on_message_callback=make_stream_callback(agents),
            human_input_callback=make_human_input_callback(interactive),
            on_phase_callback=make_phase_callback(),
        )

        ui.print_agents_table(agents.personas, agents.architect_cfg)

        active_rounds = 1
        if isinstance(agents.config, dict):
            active_rounds = agents.config.get("orchestrator", {}).get("rounds", 1)

        ui.print_config_summary(
            rounds=active_rounds,
            interactive=interactive,
            project_path=str(path.resolve()) if path else None,
            output_path=str(output.resolve()),
        )

        if resume:
            ui.console.print(
                f"[dim]🔄 Resuming session from: {resume.resolve()}[/dim]\n"
            )
            orchestrator.load_session(str(resume))

        start_time = time.monotonic()

        result = orchestrator.run_simulation(
            user_prompt,
            project_path=str(path) if path else None,
            is_resume=bool(resume),
            save_path=str(save_session) if save_session else None,
        )

        elapsed = time.monotonic() - start_time

        final_plan = result["final_plan"]
        output.write_text(final_plan, encoding="utf-8")

        if export_transcript:
            full_transcript = result.get("full_transcript", "")
            export_transcript.write_text(full_transcript, encoding="utf-8")

        ui.print_final_summary(
            output_path=str(output.resolve()),
            transcript_path=str(export_transcript.resolve())
            if export_transcript
            else None,
            session_path=str(save_session.resolve()) if save_session else None,
            duration_secs=elapsed,
        )

    except Exception as e:
        ui.console.print(
            f"\n[bold red]An error occurred during the simulation: {e}[/bold red]"
        )
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
