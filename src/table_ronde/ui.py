"""Centralized Rich UI components for table-ronde."""

from typing import Any

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text
from rich.tree import Tree

console = Console()

# ─── Agent color mapping ───────────────────────────────────
AGENT_COLORS: dict[str, str] = {
    "architect": "bold blue",
    "skeptic": "bold red",
    "enthusiast": "bold green",
}
FALLBACK_COLORS = ["bold magenta", "bold yellow", "bold cyan", "bold white"]


def get_agent_color(role: str, index: int = 0) -> str:
    """Get a consistent color for an agent role."""
    role_lower = role.lower()
    if role_lower in AGENT_COLORS:
        return AGENT_COLORS[role_lower]
    return FALLBACK_COLORS[index % len(FALLBACK_COLORS)]


# ─── Banner ────────────────────────────────────────────────
def print_banner(provider: str) -> None:
    """Print a styled ASCII art banner."""
    try:
        import pyfiglet

        ascii_art = pyfiglet.figlet_format("Table-Ronde", font="slant").rstrip()
    except Exception:
        ascii_art = "  TABLE-RONDE"

    banner_text = Text(ascii_art, style="bold cyan")
    provider_display = provider.upper() if provider else "UNSPECIFIED"
    subtitle = Text.assemble(
        ("Multi-Agent Debate Orchestrator", "dim"),
        " • ",
        (f"Provider: {provider_display}", "bold yellow"),
        " • ",
        ("v0.2.0", "dim green"),
    )

    panel = Panel(
        Text.assemble(banner_text, "\n\n", subtitle),
        border_style="cyan",
        padding=(0, 2),
    )
    console.print(panel)
    console.print()


# ─── Agents Table ──────────────────────────────────────────
def print_agents_table(
    personas: list[dict[str, Any]], architect_cfg: dict[str, Any]
) -> None:
    """Print a rich table showing all participating agents."""
    table = Table(
        title="🎭 Roundtable Participants",
        title_style="bold white",
        border_style="dim",
        show_lines=True,
        padding=(0, 1),
    )
    table.add_column("", justify="center", width=4)
    table.add_column("Agent", style="bold", min_width=18)
    table.add_column("Role", min_width=12)
    table.add_column("Model", style="dim")
    table.add_column("Temp", justify="center", width=6)

    # Architect first
    arch_role = str(architect_cfg.get("role", "architect"))
    arch_emoji = str(architect_cfg.get("emoji", "🏛️"))
    arch_title = str(architect_cfg.get("title", "Architect"))
    arch_model = str(architect_cfg.get("model", "default"))
    arch_temp = str(architect_cfg.get("temperature", 0.3))
    color = get_agent_color(arch_role)
    table.add_row(
        arch_emoji,
        Text(arch_title, style=color),
        Text("Moderator", style="italic"),
        arch_model,
        arch_temp,
    )

    # Personas
    for idx, p in enumerate(personas):
        role_name = str(p.get("role", f"persona_{idx}"))
        p_title = str(p.get("title", role_name))
        p_emoji = str(p.get("emoji", "🤖"))
        p_model = str(p.get("model", "default"))
        p_temp = str(p.get("temperature", 0.7))
        color = get_agent_color(role_name, idx)
        table.add_row(
            p_emoji,
            Text(p_title, style=color),
            Text(role_name.capitalize(), style="italic"),
            p_model,
            p_temp,
        )

    console.print(table)
    console.print()


# ─── Config Summary ───────────────────────────────────────
def print_config_summary(
    rounds: int,
    interactive: bool,
    project_path: str | None,
    output_path: str,
) -> None:
    """Print a compact config summary panel."""
    tree = Tree("⚙️  [bold]Session Configuration[/bold]")
    tree.add(f"🔄 Rounds: [cyan]{rounds}[/cyan]")
    interactive_status = (
        "[green]Enabled[/green]" if interactive else "[dim]Disabled[/dim]"
    )
    tree.add(f"💬 Interactive: {interactive_status}")
    if project_path:
        tree.add(f"📁 Project: [yellow]{project_path}[/yellow]")
    tree.add(f"📄 Output: [dim]{output_path}[/dim]")
    console.print(Panel(tree, border_style="dim", padding=(0, 1)))
    console.print()


# ─── Phase Headers ─────────────────────────────────────────
def print_phase_header(phase: str, description: str = "") -> None:
    """Print a visual phase separator."""
    console.print()
    console.print(Rule(f" {phase} ", style="bold cyan"))
    if description:
        console.print(Text(f"  {description}", style="dim italic"))
    console.print()


# ─── Round Indicator ───────────────────────────────────────
def print_round_header(current: int, total: int) -> None:
    """Print a round progress indicator."""
    total_safe = max(total, 1)
    current_safe = min(current, total_safe)
    bar = "█" * current_safe + "░" * (total_safe - current_safe)
    console.print(
        Panel(
            Text.assemble(
                ("ROUND ", "bold"),
                (f"{current}/{total}", "bold cyan"),
                ("  ", ""),
                (f"[{bar}]", "cyan"),
            ),
            border_style="cyan",
            padding=(0, 2),
        )
    )
    console.print()


# ─── Tool Call Indicator ───────────────────────────────────
def print_tool_call(tool_name: str, args_preview: str = "") -> None:
    """Display a tool invocation indicator."""
    parts = [
        ("  🔧 ", ""),
        ("Tool Call", "bold yellow"),
        (f" → {tool_name}", "yellow"),
    ]
    if args_preview:
        parts.append((f" ({args_preview})", "dim"))
    console.print(Text.assemble(*parts))


# ─── Streaming Panel (per-agent colored) ──────────────────
class AgentStreamPanel:
    """Manages a Live-updating panel for a single agent's streamed response."""

    def __init__(self, title: str, emoji: str, role: str, agent_index: int = 0):
        self.title = title
        self.emoji = emoji
        self.color = get_agent_color(role, agent_index)
        self.full_text = ""
        self.live: Live | None = None

    def _build_panel(self, done: bool = False) -> Panel:
        content: Markdown | Text
        if self.full_text.strip():
            content = Markdown(self.full_text)
        else:
            content = Text("⏳ Thinking...", style="dim italic")

        subtitle = "[dim]✓ done[/dim]" if done else "[dim]streaming...[/dim]"
        return Panel(
            content,
            title=f"{self.emoji} {self.title}",
            title_align="left",
            subtitle=subtitle,
            border_style=self.color,
            padding=(1, 2),
        )

    def start(self) -> None:
        self.live = Live(self._build_panel(), refresh_per_second=10, console=console)
        self.live.start()

    def update(self, text: str) -> None:
        self.full_text = text
        if self.live:
            self.live.update(self._build_panel())

    def finish(self) -> None:
        if self.live:
            self.live.update(self._build_panel(done=True))
            self.live.stop()


# ─── Final Summary ─────────────────────────────────────────
def print_final_summary(
    output_path: str,
    transcript_path: str | None = None,
    session_path: str | None = None,
    duration_secs: float | None = None,
) -> None:
    """Print the final success summary."""
    parts = [
        "[bold green]✨ Implementation Plan successfully generated![/bold green]",
        f"📄 Plan saved to: [bold white]{output_path}[/bold white]",
    ]

    if transcript_path:
        parts.append(f"📝 Transcript exported to: [dim]{transcript_path}[/dim]")
    if session_path:
        parts.append(f"💾 Session state saved to: [dim]{session_path}[/dim]")
    if duration_secs is not None:
        mins, secs = divmod(int(duration_secs), 60)
        parts.append(f"⏱️  Duration: [cyan]{mins}m {secs:02d}s[/cyan]")

    console.print(
        Panel(
            "\n".join(parts),
            title="🎉 [bold]Roundtable Complete[/bold]",
            border_style="green",
            padding=(1, 2),
        )
    )
