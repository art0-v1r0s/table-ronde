"""Interactive setup menu for table-ronde (zero-argument mode)."""

from pathlib import Path
from typing import Any

import questionary
from rich.panel import Panel

from table_ronde import ui


def run_interactive_menu() -> dict[str, Any]:
    """Guide the user through an interactive setup when no CLI arguments are given.

    Returns a dict with resolved options matching CLI params.
    """
    ui.print_banner("")

    ui.console.print(
        "[bold cyan]🎯 Welcome! No arguments detected — launching interactive setup.[/bold cyan]\n"
    )

    # 1. Topic / Prompt
    prompt = questionary.text(
        "📝 What topic or project should the agents debate?",
        instruction="(e.g., 'Design a high-performance caching layer with Redis')",
        validate=lambda t: len(t.strip()) > 0 or "Please enter a non-empty topic",
    ).ask()

    if prompt is None:
        raise SystemExit(0)
    prompt = prompt.strip()

    # 2. Provider
    provider = questionary.select(
        "🤖 Which LLM provider would you like to use?",
        choices=[
            questionary.Choice("Google Gemini (default: gemini-3.6-flash)", value="gemini"),
            questionary.Choice("OpenAI (default: gpt-4o)", value="openai"),
            questionary.Choice("GitHub Copilot / Azure AI Models", value="copilot"),
        ],
        default="gemini",
    ).ask()

    if provider is None:
        raise SystemExit(0)

    # 3. Rounds
    rounds_choice = questionary.select(
        "🔄 How many debate rounds?",
        choices=[
            questionary.Choice("1 round (quick audit & synthesis)", value="1"),
            questionary.Choice("2 rounds (recommended: thesis, critique & counter-proposal)", value="2"),
            questionary.Choice("3 rounds (in-depth deep dive)", value="3"),
            questionary.Choice("Custom number of rounds...", value="custom"),
        ],
        default="1",
    ).ask()

    if rounds_choice is None:
        raise SystemExit(0)

    if rounds_choice == "custom":
        custom_rounds = questionary.text(
            "Enter number of rounds:",
            validate=lambda t: (t.strip().isdigit() and int(t.strip()) > 0) or "Please enter a positive integer",
        ).ask()
        if custom_rounds is None:
            raise SystemExit(0)
        rounds = int(custom_rounds.strip())
    else:
        rounds = int(rounds_choice)

    # 4. Interactive mode
    interactive = questionary.confirm(
        "💬 Enable interactive mode? (pause between rounds to inject notes or /round)",
        default=False,
    ).ask()

    if interactive is None:
        raise SystemExit(0)

    # 5. Project path (optional)
    scan_project = questionary.confirm(
        "📁 Would you like to scan an existing project directory?",
        default=False,
    ).ask()

    if scan_project is None:
        raise SystemExit(0)

    project_path: str | None = None
    if scan_project:
        project_path = questionary.path(
            "Enter the directory path to scan:",
            only_directories=True,
            validate=lambda p: (Path(p).expanduser().is_dir() if p else False) or "Directory does not exist",
        ).ask()
        if project_path is None:
            raise SystemExit(0)
        project_path = str(Path(project_path).expanduser().resolve())

    # 6. Config file (optional)
    use_custom_config = questionary.confirm(
        "⚙️  Use a custom YAML config file? (define custom personas or model overrides)",
        default=False,
    ).ask()

    if use_custom_config is None:
        raise SystemExit(0)

    config_path: str | None = None
    if use_custom_config:
        config_path = questionary.path(
            "Enter path to your config YAML:",
            validate=lambda p: (Path(p).expanduser().is_file() if p else False) or "File does not exist",
        ).ask()
        if config_path is None:
            raise SystemExit(0)
        config_path = str(Path(config_path).expanduser().resolve())

    # 7. Output file
    output_file = questionary.text(
        "📄 Output file path for the implementation plan:",
        default="plan_v2.md",
        validate=lambda t: len(t.strip()) > 0 or "Please enter a valid file path",
    ).ask()

    if output_file is None:
        raise SystemExit(0)
    output_file = output_file.strip()

    # Summary confirmation
    summary_lines = [
        f"  📝 [bold]Topic:[/bold]       [cyan]{prompt}[/cyan]",
        f"  🤖 [bold]Provider:[/bold]    [yellow]{provider.upper()}[/yellow]",
        f"  🔄 [bold]Rounds:[/bold]      [cyan]{rounds}[/cyan]",
        f"  💬 [bold]Interactive:[/bold] [{'green' if interactive else 'dim'}]{interactive}[/]",
    ]
    if project_path:
        summary_lines.append(f"  📁 [bold]Project:[/bold]     [yellow]{project_path}[/yellow]")
    if config_path:
        summary_lines.append(f"  ⚙️  [bold]Config:[/bold]      [dim]{config_path}[/dim]")
    summary_lines.append(f"  📄 [bold]Output:[/bold]      [white]{output_file}[/white]")

    ui.console.print()
    ui.console.print(
        Panel(
            "\n".join(summary_lines),
            title="📋 [bold]Session Launch Summary[/bold]",
            border_style="cyan",
            padding=(0, 1),
        )
    )
    ui.console.print()

    confirmed = questionary.confirm("🚀 Launch the debate now?", default=True).ask()

    if not confirmed:
        ui.console.print("[dim]Aborted by user.[/dim]")
        raise SystemExit(0)

    return {
        "prompt": prompt,
        "provider": provider,
        "rounds": rounds,
        "interactive": interactive,
        "path": project_path,
        "config_file": config_path,
        "output": output_file,
    }
