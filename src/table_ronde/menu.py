"""Interactive setup menu for table-ronde using Textual."""

from typing import Any

from textual.app import App, ComposeResult
from textual.containers import Horizontal, VerticalScroll
from textual.widgets import Button, Checkbox, Footer, Header, Input, Label, Select

from table_ronde import ui


class SetupMenuApp(App[dict[str, Any] | None]):
    """Textual App to configure the Table-Ronde launch parameters."""

    CSS = """
    Screen {
        align: center middle;
    }
    #form-container {
        width: 80%;
        height: 80%;
        border: solid cyan;
        padding: 1 2;
    }
    Horizontal {
        height: auto;
        margin-top: 1;
        align: center middle;
    }
    Button {
        margin: 0 2;
    }
    """

    def compose(self) -> ComposeResult:
        """Compose the UI layout."""
        yield Header()
        with VerticalScroll(id="form-container"):
            yield Label("📝 Topic or project description:")
            yield Input(
                placeholder="e.g. 'Design a high-performance caching layer'",
                id="prompt",
            )

            yield Label("🤖 LLM Provider:")
            yield Select(
                (
                    ("Google Gemini (default)", "gemini"),
                    ("OpenAI", "openai"),
                    ("GitHub Copilot", "copilot"),
                ),
                value="gemini",
                id="provider",
            )

            yield Label("🔄 Number of debate rounds:")
            yield Input(value="1", type="integer", id="rounds")

            yield Checkbox(
                "💬 Enable interactive mode (pause between rounds)", id="interactive"
            )

            yield Label("📁 Project path to scan (optional):")
            yield Input(placeholder="/path/to/project", id="path")

            yield Label("⚙️  Custom Config YAML (optional):")
            yield Input(placeholder="/path/to/config.yml", id="config_file")

            yield Label("📄 Output file path:")
            yield Input(value="plan_v2.md", id="output")

            with Horizontal():
                yield Button("🚀 Launch Debate", variant="success", id="launch")
                yield Button("❌ Cancel", variant="error", id="cancel")
        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "cancel":
            self.exit(None)
        elif event.button.id == "launch":
            prompt = self.query_one("#prompt", Input).value.strip()
            if not prompt:
                self.query_one("#prompt", Input).focus()
                # A quick textual 'toast' or simply focus is fine for validation
                self.notify("Please enter a topic", severity="error")
                return

            self.exit(
                {
                    "prompt": prompt,
                    "provider": self.query_one("#provider", Select).value,
                    "rounds": int(self.query_one("#rounds", Input).value or 1),
                    "interactive": self.query_one("#interactive", Checkbox).value,
                    "path": self.query_one("#path", Input).value.strip() or None,
                    "config_file": self.query_one("#config_file", Input).value.strip()
                    or None,
                    "output": self.query_one("#output", Input).value.strip()
                    or "plan_v2.md",
                }
            )


def run_interactive_menu() -> dict[str, Any]:
    """Launch the interactive Textual TUI.

    Returns a dict with resolved options matching CLI params.
    """
    ui.print_banner("")

    app = SetupMenuApp()
    result = app.run()

    if not result:
        ui.console.print("[dim]Aborted by user.[/dim]")
        raise SystemExit(0)

    # Print summary of choices
    summary_lines = [
        f"  📝 [bold]Topic:[/bold]       [cyan]{result['prompt']}[/cyan]",
        f"  🤖 [bold]Provider:[/bold]    [yellow]{result['provider'].upper()}[/yellow]",
        f"  🔄 [bold]Rounds:[/bold]      [cyan]{result['rounds']}[/cyan]",
        f"  💬 [bold]Interactive:[/bold] [{'green' if result['interactive'] else 'dim'}]{result['interactive']}[/]",
    ]
    if result.get("path"):
        summary_lines.append(
            f"  📁 [bold]Project:[/bold]     [yellow]{result['path']}[/yellow]"
        )
    if result.get("config_file"):
        summary_lines.append(
            f"  ⚙️  [bold]Config:[/bold]      [dim]{result['config_file']}[/dim]"
        )
    summary_lines.append(
        f"  📄 [bold]Output:[/bold]      [white]{result['output']}[/white]"
    )

    from rich.panel import Panel

    ui.console.print()
    ui.console.print(
        Panel(
            "\\n".join(summary_lines),
            title="📋 [bold]Session Launch Summary[/bold]",
            border_style="cyan",
            padding=(0, 1),
        )
    )
    ui.console.print()

    return result
