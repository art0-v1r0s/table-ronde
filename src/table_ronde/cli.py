import logging
import os
import socket
import warnings
from collections.abc import Callable, Generator
from pathlib import Path

import typer
import yaml
from langchain_core.messages import BaseMessageChunk
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.text import Text

# Network optimization: Force IPv4 to prevent 80s IPv6 timeout (blackholing)
old_getaddrinfo = socket.getaddrinfo
def new_getaddrinfo(*args, **kwargs):
    responses = old_getaddrinfo(*args, **kwargs)
    return [response for response in responses if response[0] == socket.AF_INET]
socket.getaddrinfo = new_getaddrinfo

# Suppress unnecessary warnings
logging.getLogger("google_genai.models").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*fixed sampling defaults.*")

from table_ronde.agents import TableRondeAgents
from table_ronde.orchestrator import Orchestrator, StreamCallback

app = typer.Typer(
    name="table-ronde",
    help="Multi-agent orchestrator to debate and design implementation plans v2.0 / v3.0",
)
console = Console()

def get_role_style(role: str, agents_cfg: TableRondeAgents) -> tuple[str, str]:
    if role == agents_cfg.architect_cfg.get("role"):
        title = agents_cfg.architect_cfg.get("title", f"Agent ({role})")
        emoji = agents_cfg.architect_cfg.get("emoji", "🏛️")
        return (f"{emoji} {title}", "bold blue")
    
    colors = ["bold red", "bold green", "bold magenta", "bold yellow", "bold cyan"]
    for idx, p in enumerate(agents_cfg.personas):
        if p["role"] == role:
            title = p.get("title", f"Agent ({role})")
            emoji = p.get("emoji", "🤖")
            color = colors[idx % len(colors)]
            return (f"{emoji} {title}", color)
            
    return (f"Agent ({role})", "white")


def make_stream_callback(agents: TableRondeAgents) -> Callable:
    def callback(role: str, gen) -> str:
        # We need to map role -> title, emoji
        title = role
        emoji = "🤖"
        if role == agents.architect_cfg["role"]:
            title = agents.architect_cfg.get("title", "Architect")
            emoji = agents.architect_cfg.get("emoji", "🏛️")
        else:
            for p in agents.personas:
                if p["role"] == role:
                    title = p.get("title", role)
                    emoji = p.get("emoji", "🤖")
                    break

        full_text = ""
        live = None

        try:
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

                    if full_text.strip() and live is None:
                        panel = Panel(
                            Markdown(full_text),
                            title=f"{emoji} {title}",
                            border_style="blue",
                            padding=(1, 2),
                        )
                        live = Live(panel, refresh_per_second=10)
                        live.start()

                    if live:
                        panel = Panel(
                            Markdown(full_text),
                            title=f"{emoji} {title}",
                            border_style="blue",
                            padding=(1, 2),
                        )
                        live.update(panel)
        finally:
            if live:
                live.stop()

        return full_text

    return callback


def make_human_input_callback(interactive: bool) -> Callable[[], str | None] | None:
    if not interactive:
        return None

    def ask() -> str | None:
        console.print()
        note = Prompt.ask(
            "[bold yellow]💬 Your note for the Architect (or '/round' for another debate round)[/bold yellow]\n(Press Enter to skip)",
            default="",
            console=console,
        )
        console.print()
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
        "gemini", "--provider", "-pr", help="LLM Provider ('gemini' or 'copilot' / 'github')"
    ),
    model: str | None = typer.Option(
        None, "--model", "-m", help="Default model to use (e.g., 'gemini-3.6-flash' or 'gpt-4o')"
    ),
    export_transcript: Path | None = typer.Option(
        None, "--export-transcript", "-t", help="File to export the full debate transcript"
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
):
    if not prompt and not path and not resume:
        console.print(
            "[bold red]Error: You must provide a topic, a path, or a session to resume.[/bold red]"
        )
        raise typer.Exit(code=1)

    prov_clean = provider.lower()
    if prov_clean in ("copilot", "github"):
        if not (os.getenv("GITHUB_TOKEN") or os.getenv("COPILOT_API_KEY")):
            console.print(
                "[bold yellow]Warning: GITHUB_TOKEN or COPILOT_API_KEY is not set in the environment.[/bold yellow]"
            )
    elif prov_clean == "gemini":
        if not os.getenv("GEMINI_API_KEY"):
            console.print(
                "[bold yellow]Warning: GEMINI_API_KEY is not set in the environment.[/bold yellow]"
            )
            
    config = None
    if config_file:
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
                console.print(f"[dim]⚙️  Configuration loaded from: {config_file}[/dim]")
        except Exception as e:
            console.print(f"[bold red]Error loading configuration {config_file}: {e}[/bold red]")
            raise typer.Exit(code=1)
            
    if rounds is not None and config:
        config.setdefault("orchestrator", {})["rounds"] = rounds
    elif rounds is not None:
        config = {"orchestrator": {"rounds": rounds}}

    user_prompt = prompt or "Analysis and improvement of the provided project."

    console.print("[bold cyan]====================================================[/bold cyan]")
    console.print(f"[bold cyan]  TABLE-RONDE : MULTI-AGENT DEBATE ({provider.upper()}) [/bold cyan]")
    console.print("[bold cyan]====================================================[/bold cyan]\n")

    if path:
        console.print(f"[dim]📁 Analyzing project at path: {path.resolve()}[/dim]\n")
    if resume:
        console.print(f"[dim]🔄 Resuming session from: {resume.resolve()}[/dim]\n")

    try:
        agents = TableRondeAgents(config=config, provider=provider, model_name=model)
        orchestrator = Orchestrator(
            agents,
            on_message_callback=make_stream_callback(agents),
            human_input_callback=make_human_input_callback(interactive),
        )

        if resume:
            orchestrator.load_session(str(resume))

        result = orchestrator.run_simulation(
            user_prompt, 
            project_path=str(path) if path else None,
            is_resume=bool(resume),
            save_path=str(save_session) if save_session else None
        )

        final_plan = result["final_plan"]
        output.write_text(final_plan, encoding="utf-8")

        summary_msg = (
            f"[bold green]✨ Implementation Plan successfully generated![/bold green]\n"
            f"The file was saved to: [bold white]{output.resolve()}[/bold white]"
        )

        if export_transcript:
            full_transcript = result.get("full_transcript", "")
            export_transcript.write_text(full_transcript, encoding="utf-8")
            summary_msg += f"\n[dim]📝 Transcript exported to: {export_transcript.resolve()}[/dim]"

        console.print(
            Panel(
                summary_msg,
                title="🎉 Finished",
                border_style="green",
            )
        )

    except Exception as e:
        console.print(f"\n[bold red]An error occurred during the simulation: {e}[/bold red]")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
