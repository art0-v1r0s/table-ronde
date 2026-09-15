import os
from pathlib import Path

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from table_ronde.agents import TableRondeAgents
from table_ronde.orchestrator import Orchestrator

app = typer.Typer(
    name="table-ronde",
    help="Orchestrateur multi-agents pour débattre et concevoir des plans d'implémentation v2.0 / v3.0",
)
console = Console()


def render_agent_message(role: str, content: str):
    if role == "architect":
        title = "🏛️ Agent 3 : L'Architecte"
        border_style = "bold blue"
    elif role == "skeptic":
        title = "😈 Agent 1 : Le Sceptique"
        border_style = "bold red"
    elif role == "enthusiast":
        title = "🚀 Agent 2 : L'Enthousiaste"
        border_style = "bold green"
    else:
        title = f"Agent ({role})"
        border_style = "white"

    md = Markdown(content)
    panel = Panel(md, title=title, border_style=border_style, padding=(1, 2))
    console.print(panel)
    console.print()


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
    provider: str = typer.Option(
        "gemini", "--provider", "-pr", help="Fournisseur LLM ('gemini' ou 'copilot' / 'github')"
    ),
    model: str | None = typer.Option(
        None, "--model", "-m", help="Modèle à utiliser (ex: 'gemini-2.5-flash' ou 'gpt-4o')"
    ),
    export_transcript: Path | None = typer.Option(
        None, "--export-transcript", "-t", help="Fichier pour exporter le transcript complet du débat"
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

    user_prompt = prompt or "Analyse et amélioration du projet fourni."

    console.print("[bold cyan]====================================================[/bold cyan]")
    console.print(f"[bold cyan]  TABLE-RONDE : DEBAT MULTI-AGENTS ({provider.upper()}) [/bold cyan]")
    console.print("[bold cyan]====================================================[/bold cyan]\n")

    if path:
        console.print(f"[dim]📁 Analyse du projet à l'emplacement : {path.resolve()}[/dim]\n")

    try:
        agents = TableRondeAgents(provider=provider, model_name=model)
        orchestrator = Orchestrator(agents, on_message_callback=render_agent_message)

        with console.status("[bold yellow]La Table-Ronde débute ses échanges...[/bold yellow]"):
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

