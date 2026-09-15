import os
import logging
import socket
import warnings
import yaml
from collections.abc import Callable, Generator
from pathlib import Path

import typer
from langchain_core.messages import BaseMessageChunk
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.text import Text

# Optimisation réseau : Forcer IPv4 pour éviter le timeout IPv6 de 80s (blackholing)
old_getaddrinfo = socket.getaddrinfo
def new_getaddrinfo(*args, **kwargs):
    responses = old_getaddrinfo(*args, **kwargs)
    return [response for response in responses if response[0] == socket.AF_INET]
socket.getaddrinfo = new_getaddrinfo

# Supprimer les avertissements inutiles
logging.getLogger("google_genai.models").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*fixed sampling defaults.*")

from table_ronde.agents import TableRondeAgents
from table_ronde.orchestrator import Orchestrator, StreamCallback

app = typer.Typer(
    name="table-ronde",
    help="Orchestrateur multi-agents pour débattre et concevoir des plans d'implémentation v2.0 / v3.0",
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


def make_stream_callback(agents: TableRondeAgents) -> StreamCallback:
    """Crée le callback de streaming : affiche en temps réel avec rich.Live."""
    def stream_callback(role: str, gen: Generator[BaseMessageChunk, None, None]) -> str:
        title, border_style = get_role_style(role, agents)
        content = ""
        try:
            with Live(
                Panel(Text("…"), title=title, border_style=border_style, padding=(1, 2)),
                refresh_per_second=15,
                console=console,
            ) as live:
                for chunk in gen:
                    if hasattr(chunk, "content") and chunk.content:
                        if isinstance(chunk.content, str):
                            content += chunk.content
                        elif isinstance(chunk.content, list):
                            for part in chunk.content:
                                if isinstance(part, str):
                                    content += part
                                elif isinstance(part, dict) and "text" in part:
                                    content += part["text"]
                        
                        live.update(Panel(Text(content), title=title, border_style=border_style, padding=(1, 2)))
        except Exception as e:
            console.print(f"[yellow]⚠️ Stream interrompu ({e}), réponse partielle conservée.[/yellow]")

        console.print(Panel(Markdown(content), title=title, border_style=border_style, padding=(1, 2)))
        console.print()
        return content

    return stream_callback


def make_human_input_callback(interactive: bool) -> Callable[[], str | None] | None:
    if not interactive:
        return None

    def ask() -> str | None:
        console.print()
        note = Prompt.ask(
            "[bold yellow]💬 Votre note pour l'Architecte (ou '/tour' pour refaire un tour de débat)[/bold yellow]\n(Entrée pour ignorer)",
            default="",
            console=console,
        )
        console.print()
        return note.strip() or None

    return ask


@app.command()
def main(
    prompt: str | None = typer.Argument(
        None, help="Description du projet ou sujet à analyser"
    ),
    path: Path | None = typer.Option(
        None, "--path", "-p", help="Chemin vers le répertoire du projet existant à analyser"
    ),
    output: Path = typer.Option(
        Path("plan_v2.md"), "--output", "-o", help="Fichier de sortie pour le plan final"
    ),
    config_file: Path | None = typer.Option(
        None, "--config", "-c", help="Fichier YAML de configuration des personas et modèles"
    ),
    rounds: int | None = typer.Option(
        None, "--rounds", "-r", help="Nombre de tours de débat (surcharge la config)"
    ),
    provider: str = typer.Option(
        "gemini", "--provider", "-pr", help="Fournisseur LLM ('gemini' ou 'copilot' / 'github')"
    ),
    model: str | None = typer.Option(
        None, "--model", "-m", help="Modèle à utiliser par défaut (ex: 'gemini-3.6-flash' ou 'gpt-4o')"
    ),
    export_transcript: Path | None = typer.Option(
        None, "--export-transcript", "-t", help="Fichier pour exporter le transcript complet du débat"
    ),
    interactive: bool = typer.Option(
        False,
        "--interactive",
        "-i",
        help="Met le débat en pause avant la résolution pour recueillir votre note ou relancer un tour",
    ),
):
    if not prompt and not path:
        console.print(
            "[bold red]Erreur : Vous devez fournir au moins un sujet (prompt) ou un chemin vers un projet (--path).[/bold red]"
        )
        raise typer.Exit(code=1)

    prov_clean = provider.lower()
    if prov_clean in ("copilot", "github"):
        if not (os.getenv("GITHUB_TOKEN") or os.getenv("COPILOT_API_KEY")):
            console.print(
                "[bold yellow]Attention : GITHUB_TOKEN ou COPILOT_API_KEY n'est pas définie dans l'environnement.[/bold yellow]"
            )
    elif prov_clean == "gemini":
        if not os.getenv("GEMINI_API_KEY"):
            console.print(
                "[bold yellow]Attention : GEMINI_API_KEY n'est pas définie dans l'environnement.[/bold yellow]"
            )
            
    config = None
    if config_file:
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
                console.print(f"[dim]⚙️  Configuration chargée depuis : {config_file}[/dim]")
        except Exception as e:
            console.print(f"[bold red]Erreur lors du chargement de la configuration {config_file}: {e}[/bold red]")
            raise typer.Exit(code=1)
            
    if rounds is not None and config:
        config.setdefault("orchestrator", {})["rounds"] = rounds
    elif rounds is not None:
        config = {"orchestrator": {"rounds": rounds}}

    user_prompt = prompt or "Analyse et amélioration du projet fourni."

    console.print("[bold cyan]====================================================[/bold cyan]")
    console.print(f"[bold cyan]  TABLE-RONDE : DEBAT MULTI-AGENTS ({provider.upper()}) [/bold cyan]")
    console.print("[bold cyan]====================================================[/bold cyan]\n")

    if path:
        console.print(f"[dim]📁 Analyse du projet à l'emplacement : {path.resolve()}[/dim]\n")

    try:
        agents = TableRondeAgents(config=config, provider=provider, model_name=model)
        orchestrator = Orchestrator(
            agents,
            on_message_callback=make_stream_callback(agents),
            human_input_callback=make_human_input_callback(interactive),
        )

        result = orchestrator.run_simulation(user_prompt, project_path=str(path) if path else None)

        final_plan = result["final_plan"]
        output.write_text(final_plan, encoding="utf-8")

        summary_msg = (
            f"[bold green]✨ Plan d'Implémentation généré avec succès ![/bold green]\n"
            f"Le fichier a été enregistré dans : [bold white]{output.resolve()}[/bold white]"
        )

        if export_transcript:
            full_transcript = result.get("full_transcript", "")
            export_transcript.write_text(full_transcript, encoding="utf-8")
            summary_msg += f"\n[dim]📝 Transcript exporté dans : {export_transcript.resolve()}[/dim]"

        console.print(
            Panel(
                summary_msg,
                title="🎉 Terminé",
                border_style="green",
            )
        )

    except Exception as e:
        console.print(f"\n[bold red]Une erreur s'est produite lors de la simulation : {e}[/bold red]")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
