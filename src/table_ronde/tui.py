"""Full-screen Textual TUI for table-ronde.

Provides an interactive terminal UI that covers the entire lifecycle:
setup → live debate streaming → plan viewing.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Generator
from pathlib import Path
from typing import Any, ClassVar
from uuid import uuid4

from rich.markdown import Markdown as RichMarkdown
from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, ScrollableContainer, Vertical
from textual.reactive import reactive
from textual.screen import ModalScreen, Screen
from textual.widgets import (
    Button,
    Checkbox,
    Footer,
    Header,
    Input,
    Label,
    Markdown,
    Select,
    Static,
    TextArea,
)
from textual.worker import Worker, WorkerState

# ─── Custom Widgets ────────────────────────────────────────


class AgentMessageWidget(Static):
    """A single agent message that renders Markdown and updates live as tokens stream in."""

    text = reactive("", layout=False)

    def __init__(
        self,
        title: str,
        emoji: str,
        color: str,
        *,
        widget_id: str,
    ) -> None:
        super().__init__(id=widget_id)
        self.border_title = f"{emoji} {title}"
        self._agent_color = color

    def on_mount(self) -> None:
        # Rich style strings look like "bold cyan" — extract just the color name
        color = self._agent_color.split()[-1] if self._agent_color else "white"
        self.styles.border_title_color = color

    def watch_text(self, new_text: str) -> None:
        if new_text.strip():
            self.update(RichMarkdown(new_text))
        else:
            self.update(Text("⏳ Thinking...", style="dim italic"))

    def mark_done(self) -> None:
        self.border_subtitle = "✓ done"
        self.add_class("done")

    def mark_errored(self, error_msg: str = "") -> None:
        self.border_subtitle = f"✗ error: {error_msg}" if error_msg else "✗ error"
        self.add_class("errored")

    def mark_cancelled(self) -> None:
        self.border_subtitle = "⊘ cancelled"
        self.add_class("cancelled")


class SidebarInfo(Static):
    """Sidebar widget showing agent list, round progress, phase, and elapsed time."""

    phase = reactive("SETUP")
    current_round = reactive(0)
    total_rounds = reactive(1)
    elapsed_secs = reactive(0.0)

    def __init__(self, agents: list[dict[str, Any]], architect: dict[str, Any]) -> None:
        super().__init__()
        self._agents = agents
        self._architect = architect

    def render(self) -> Text:
        from table_ronde.ui import get_agent_color

        parts: list[tuple[str, str]] = []

        # Title
        parts.append(("👥 AGENTS\n", "bold"))
        # Architect
        arch_emoji = str(self._architect.get("emoji", "🏛️"))
        arch_title = str(self._architect.get("title", "Architect"))
        arch_role = str(self._architect.get("role", "architect"))
        parts.append((f"  {arch_emoji} {arch_title}\n", get_agent_color(arch_role)))
        # Personas
        for idx, p in enumerate(self._agents):
            p_emoji = str(p.get("emoji", "🤖"))
            p_title = str(p.get("title", p.get("role", "agent")))
            p_role = str(p.get("role", "agent"))
            parts.append((f"  {p_emoji} {p_title}\n", get_agent_color(p_role, idx)))

        parts.append(("\n", ""))

        # Round progress
        total_safe = max(self.total_rounds, 1)
        current_safe = min(self.current_round, total_safe)
        bar = "█" * current_safe + "░" * (total_safe - current_safe)
        parts.append(("🔄 ROUND\n", "bold"))
        parts.append((f"  {current_safe}/{total_safe} [{bar}]\n\n", "cyan"))

        # Phase
        parts.append(("📊 PHASE\n", "bold"))
        parts.append((f"  {self.phase}\n\n", "cyan"))

        # Elapsed
        mins, secs = divmod(int(self.elapsed_secs), 60)
        parts.append(("⏱️  ELAPSED\n", "bold"))
        parts.append((f"  {mins}m {secs:02d}s\n", "cyan"))

        return Text.assemble(*parts)


# ─── Screens ───────────────────────────────────────────────


class SetupScreen(Screen):
    """Configuration screen — collects all parameters before launching the debate."""

    def compose(self) -> ComposeResult:
        yield Header()
        with Container(id="setup-container"):
            yield Label("🛠️  Table-Ronde Setup", id="setup-title")

            yield Label("📝 Topic or project description:")
            yield TextArea(id="prompt", language=None)

            with Horizontal(classes="input-row"):
                with Vertical(classes="input-col"):
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
                with Vertical(classes="input-col"):
                    yield Label("🔗 Model (leave empty for default):")
                    yield Input(placeholder="e.g. gemini-3.6-flash, gpt-4o", id="model")

            with Horizontal(classes="input-row"):
                with Vertical(classes="input-col"):
                    yield Label("🔄 Number of debate rounds:")
                    yield Input(value="1", type="integer", id="rounds")
                with Vertical(classes="input-col"):
                    yield Label(" ")  # spacer for alignment
                    yield Checkbox(
                        "💬 Enable interactive mode (pause between rounds)",
                        id="interactive",
                    )

            with Horizontal(classes="input-row"):
                with Vertical(classes="input-col"):
                    yield Label("📁 Project path to scan (optional):")
                    yield Input(placeholder="/path/to/project", id="path")
                with Vertical(classes="input-col"):
                    yield Label("⚙️  Custom Config YAML (optional):")
                    yield Input(placeholder="/path/to/config.yml", id="config_file")

            with Horizontal(classes="input-row"):  # noqa: SIM117
                with Vertical(classes="input-col"):
                    yield Label("📄 Output file path:")
                    yield Input(value="plan_v2.md", id="output")

            with Horizontal(id="setup-buttons"):
                yield Button("🚀 Launch Debate", variant="success", id="launch")
                yield Button("❌ Cancel", variant="error", id="cancel")
        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "cancel":
            self.app.exit()
        elif event.button.id == "launch":
            prompt_area = self.query_one("#prompt", TextArea)
            prompt_text = prompt_area.text.strip()
            if not prompt_text:
                prompt_area.focus()
                self.notify("Please enter a topic", severity="error")
                return

            config = {
                "prompt": prompt_text,
                "provider": self.query_one("#provider", Select).value,
                "model": self.query_one("#model", Input).value.strip() or None,
                "rounds": int(self.query_one("#rounds", Input).value or 1),
                "interactive": self.query_one("#interactive", Checkbox).value,
                "path": self.query_one("#path", Input).value.strip() or None,
                "config_file": self.query_one("#config_file", Input).value.strip()
                or None,
                "output": self.query_one("#output", Input).value.strip()
                or "plan_v2.md",
            }
            self.app.push_screen(DebateScreen(config))


class DebateScreen(Screen):
    """Main debate screen with sidebar + scrollable arena."""

    BINDINGS: ClassVar = [
        ("ctrl+x", "cancel_debate", "Cancel Debate"),
        ("ctrl+s", "save_transcript", "Save Transcript"),
        ("ctrl+p", "view_plan", "View Plan"),
    ]

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self._config = config
        self._debate_worker: Worker | None = None
        self._start_time: float = 0.0
        self._timer = None
        self._result: dict[str, str] | None = None

        # For human-in-the-loop: worker blocks on this event
        self._human_input_event = threading.Event()
        self._human_input_value: str | None = None

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal():
            with Vertical(id="sidebar"):
                yield Label("TABLE RONDE", id="sidebar-title")
                # SidebarInfo will be mounted dynamically once agents are created
                yield Static("⏳ Initializing...", id="sidebar-placeholder")
            with Vertical(id="arena-wrapper"):
                with ScrollableContainer(id="arena"):
                    yield Static(
                        Text("🏛️ Waiting for debate to start...", style="dim italic"),
                        id="arena-empty",
                    )
                yield Static("", id="status-bar")
        yield Footer()

    def on_mount(self) -> None:
        self._start_time = time.monotonic()
        self._timer = self.set_interval(1.0, self._update_elapsed)
        # Start the debate in a background thread
        self._debate_worker = self.run_worker(
            self._run_debate,
            thread=True,
            name="debate",
            group="debate",
        )

    def _update_elapsed(self) -> None:
        elapsed = time.monotonic() - self._start_time
        try:
            sidebar = self.query_one(SidebarInfo)
            sidebar.elapsed_secs = elapsed
        except Exception:
            pass

    # ── Worker: runs the entire debate in a background thread ──

    def _run_debate(self) -> None:
        """Blocking function that runs in a worker thread."""
        import os

        import yaml

        from table_ronde.agents import TableRondeAgents
        from table_ronde.orchestrator import Orchestrator

        config = self._config
        provider = config["provider"]
        model = config.get("model")
        prompt = config["prompt"]
        path = config.get("path")
        interactive = config.get("interactive", False)
        rounds = config.get("rounds", 1)
        config_file = config.get("config_file")

        # Load YAML config if provided
        yaml_config = None
        if config_file:
            try:
                with open(config_file, "r", encoding="utf-8") as f:
                    yaml_config = yaml.safe_load(f)
            except Exception as e:
                self.app.call_from_thread(
                    self.notify, f"Error loading config: {e}", severity="error"
                )
                return

        if rounds is not None and yaml_config:
            yaml_config.setdefault("orchestrator", {})["rounds"] = rounds
        elif rounds is not None:
            if yaml_config is None:
                yaml_config = {"orchestrator": {"rounds": rounds}}

        # Check API keys
        prov_clean = provider.lower()
        if prov_clean in ("copilot", "github"):
            if not (os.getenv("GITHUB_TOKEN") or os.getenv("COPILOT_API_KEY")):
                self.app.call_from_thread(
                    self.notify,
                    "GITHUB_TOKEN or COPILOT_API_KEY not set",
                    severity="warning",
                )
        elif prov_clean == "openai":
            if not os.getenv("OPENAI_API_KEY"):
                self.app.call_from_thread(
                    self.notify, "OPENAI_API_KEY not set", severity="warning"
                )
        elif prov_clean == "gemini":
            if not os.getenv("GEMINI_API_KEY"):
                self.app.call_from_thread(
                    self.notify, "GEMINI_API_KEY not set", severity="warning"
                )

        # Create agents
        try:
            agents = TableRondeAgents(
                config=yaml_config, provider=provider, model_name=model
            )
        except Exception as e:
            self.app.call_from_thread(
                self.notify, f"Failed to initialize agents: {e}", severity="error"
            )
            self.app.call_from_thread(self._update_status, "Error: agent init failed")
            return

        # Mount sidebar info on main thread
        self.app.call_from_thread(
            self._mount_sidebar, agents.personas, agents.architect_cfg
        )

        # Set total rounds in sidebar
        orch_cfg = (
            agents.config.get("orchestrator", {})
            if isinstance(agents.config, dict)
            else {}
        )
        total_rounds = orch_cfg.get("rounds", 1)
        self.app.call_from_thread(self._set_total_rounds, total_rounds)

        # Build callbacks
        stream_cb = self._make_stream_callback(agents)
        phase_cb = self._make_phase_callback()
        human_cb = self._make_human_input_callback() if interactive else None

        orchestrator = Orchestrator(
            agents,
            on_message_callback=stream_cb,
            human_input_callback=human_cb,
            on_phase_callback=phase_cb,
        )

        user_prompt = prompt or "Analysis and improvement of the provided project."

        try:
            result = orchestrator.run_simulation(
                user_prompt,
                project_path=path,
            )
            self._result = result

            # Save plan to file
            output_path = Path(config.get("output", "plan_v2.md"))
            output_path.write_text(result["final_plan"], encoding="utf-8")

            self.app.call_from_thread(
                self._update_status,
                f"✨ Done! Plan saved to {output_path.resolve()}",
            )
            self.app.call_from_thread(
                self.notify,
                f"Plan saved to {output_path.resolve()}",
                severity="information",
            )
        except Exception as e:
            self.app.call_from_thread(
                self.notify, f"Debate error: {e}", severity="error"
            )
            self.app.call_from_thread(self._update_status, f"Error: {e}")

    # ── Callback factories (run from worker thread) ──

    def _make_stream_callback(self, agents: Any) -> Callable:
        from langchain_core.messages import BaseMessageChunk

        from table_ronde.ui import get_agent_color

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

            color = get_agent_color(role, agent_index)
            widget_id = f"msg-{uuid4().hex[:8]}"

            # Mount widget on main thread
            self.app.call_from_thread(
                self._mount_agent_widget, title, emoji, color, widget_id
            )

            full_text = ""
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
                        self.app.call_from_thread(
                            self._update_agent_widget, widget_id, full_text
                        )
            except Exception as e:
                self.app.call_from_thread(self._error_agent_widget, widget_id, str(e))
                raise

            self.app.call_from_thread(self._finish_agent_widget, widget_id)
            return full_text

        return callback

    def _make_phase_callback(self) -> Callable[[str, dict[str, Any]], None]:
        def callback(phase: str, data: dict[str, Any]) -> None:
            if phase == "opening":
                self.app.call_from_thread(self._update_phase, "OPENING")
                self.app.call_from_thread(
                    self._update_status,
                    f"⚡ Opening — {data.get('architect', 'Architect')} sets the stage",
                )
            elif phase == "round":
                self.app.call_from_thread(
                    self._update_round, data["current"], data["total"]
                )
                self.app.call_from_thread(self._update_phase, "DEBATE")
                self.app.call_from_thread(
                    self._update_status,
                    f"🔄 Round {data['current']}/{data['total']}",
                )
            elif phase == "resolution":
                self.app.call_from_thread(self._update_phase, "RESOLUTION")
                self.app.call_from_thread(
                    self._update_status,
                    f"🏁 Resolution — {data.get('architect', 'Architect')} synthesizes",
                )
            elif phase == "tool_call":
                self.app.call_from_thread(
                    self._update_status,
                    f"🔧 Tool: {data['name']}",
                )

        return callback

    def _make_human_input_callback(self) -> Callable[[], str | None]:
        def callback() -> str | None:
            self._human_input_event.clear()
            self._human_input_value = None
            # Push modal on main thread
            self.app.call_from_thread(self._push_human_input_modal)
            # Block worker thread until modal dismisses
            self._human_input_event.wait()
            return self._human_input_value

        return callback

    # ── Main-thread UI update methods (called via call_from_thread) ──

    def _mount_sidebar(self, personas: list[dict], architect: dict) -> None:
        try:
            placeholder = self.query_one("#sidebar-placeholder")
            placeholder.remove()
        except Exception:
            pass
        sidebar = self.query_one("#sidebar")
        sidebar_info = SidebarInfo(personas, architect)
        sidebar.mount(sidebar_info)

    def _set_total_rounds(self, total: int) -> None:
        try:
            sidebar = self.query_one(SidebarInfo)
            sidebar.total_rounds = total
        except Exception:
            pass

    def _update_phase(self, phase: str) -> None:
        try:
            sidebar = self.query_one(SidebarInfo)
            sidebar.phase = phase
        except Exception:
            pass

    def _update_round(self, current: int, total: int) -> None:
        try:
            sidebar = self.query_one(SidebarInfo)
            sidebar.current_round = current
            sidebar.total_rounds = total
        except Exception:
            pass

    def _update_status(self, text: str) -> None:
        try:
            status = self.query_one("#status-bar", Static)
            status.update(text)
        except Exception:
            pass

    def _mount_agent_widget(
        self, title: str, emoji: str, color: str, widget_id: str
    ) -> None:
        # Remove the placeholder if it exists
        try:
            placeholder = self.query_one("#arena-empty")
            placeholder.remove()
        except Exception:
            pass

        arena = self.query_one("#arena")
        widget = AgentMessageWidget(
            title=title, emoji=emoji, color=color, widget_id=widget_id
        )
        arena.mount(widget)
        widget.scroll_visible()

    def _update_agent_widget(self, widget_id: str, text: str) -> None:
        try:
            widget = self.query_one(f"#{widget_id}", AgentMessageWidget)
            widget.text = text
            widget.scroll_visible()
        except Exception:
            pass

    def _finish_agent_widget(self, widget_id: str) -> None:
        try:
            widget = self.query_one(f"#{widget_id}", AgentMessageWidget)
            widget.mark_done()
        except Exception:
            pass

    def _error_agent_widget(self, widget_id: str, error_msg: str) -> None:
        try:
            widget = self.query_one(f"#{widget_id}", AgentMessageWidget)
            widget.mark_errored(error_msg)
        except Exception:
            pass

    def _push_human_input_modal(self) -> None:
        def on_dismiss(value: str | None) -> None:
            self._human_input_value = value
            self._human_input_event.set()

        self.app.push_screen(HumanInputModal(), callback=on_dismiss)

    # ── Actions ──

    def action_cancel_debate(self) -> None:
        if self._debate_worker and not self._debate_worker.is_finished:
            self._debate_worker.cancel()
            self.notify("Debate cancelled", severity="warning")
            self._update_status("⊘ Debate cancelled by user")
            self._update_phase("CANCELLED")

    def action_save_transcript(self) -> None:
        if self._result:
            transcript = self._result.get("full_transcript", "")
            if transcript:
                path = Path("transcript.md")
                path.write_text(transcript, encoding="utf-8")
                self.notify(
                    f"Transcript saved to {path.resolve()}", severity="information"
                )
            else:
                self.notify("No transcript available yet", severity="warning")
        else:
            self.notify("Debate not finished yet", severity="warning")

    def action_view_plan(self) -> None:
        if self._result:
            self.app.push_screen(
                PlanViewScreen(
                    self._result["final_plan"], self._config.get("output", "plan_v2.md")
                )
            )
        else:
            self.notify(
                "Plan not available yet — debate still running", severity="warning"
            )

    # ── Worker lifecycle ──

    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        if event.worker.name != "debate":
            return
        if event.state == WorkerState.ERROR:
            self._update_status(f"✗ Worker error: {event.worker.error}")
            self.notify(f"Debate failed: {event.worker.error}", severity="error")
        elif event.state == WorkerState.CANCELLED:
            self._update_status("⊘ Debate cancelled")
            # Unblock human input if waiting
            self._human_input_event.set()
        elif event.state == WorkerState.SUCCESS:
            self._update_phase("DONE")


class HumanInputModal(ModalScreen[str | None]):
    """Modal for human-in-the-loop input between rounds."""

    def compose(self) -> ComposeResult:
        with Container(id="modal-container"):
            yield Label("💬 Human-in-the-Loop", id="modal-title")
            yield Label("Enter your note for the Architect, or request a new round:")
            yield TextArea(id="human-input")
            with Horizontal(id="modal-buttons"):
                yield Button("📝 Submit Note", variant="primary", id="submit")
                yield Button("🔄 New Round", variant="warning", id="round")
                yield Button("⏭️ Skip", variant="default", id="skip")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "submit":
            text = self.query_one("#human-input", TextArea).text.strip()
            if text:
                self.dismiss(text)
            else:
                self.notify("Please enter a note or click Skip", severity="warning")
        elif event.button.id == "round":
            self.dismiss("/round")
        elif event.button.id == "skip":
            self.dismiss(None)


class PlanViewScreen(Screen):
    """Full-screen Markdown viewer for the final implementation plan."""

    BINDINGS: ClassVar = [
        ("escape", "go_back", "Back"),
    ]

    def __init__(self, plan_text: str, output_path: str) -> None:
        super().__init__()
        self._plan_text = plan_text
        self._output_path = output_path

    def compose(self) -> ComposeResult:
        yield Header()
        with ScrollableContainer(id="plan-container"):
            yield Markdown(self._plan_text)
        with Horizontal(id="plan-buttons"):
            yield Button("💾 Export Plan", variant="success", id="export")
            yield Button("🔙 Back to Debate", variant="default", id="back")
        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "back":
            self.app.pop_screen()
        elif event.button.id == "export":
            path = Path(self._output_path)
            path.write_text(self._plan_text, encoding="utf-8")
            self.notify(f"Plan exported to {path.resolve()}", severity="information")

    def action_go_back(self) -> None:
        self.app.pop_screen()


# ─── Main App ──────────────────────────────────────────────


class TableRondeTUI(App):
    """The Table-Ronde Textual TUI application."""

    TITLE = "🏛️ Table-Ronde"
    SUB_TITLE = "Multi-Agent Debate Orchestrator"

    CSS_PATH = "theme.tcss"

    BINDINGS: ClassVar = [
        ("ctrl+q", "quit", "Quit"),
        ("ctrl+d", "toggle_dark", "Toggle Dark"),
    ]

    def __init__(self, launch_config: dict[str, Any] | None = None) -> None:
        super().__init__()
        self._launch_config = launch_config

    def on_mount(self) -> None:
        if self._launch_config:
            # Skip setup, go directly to debate
            self.push_screen(DebateScreen(self._launch_config))
        else:
            self.push_screen(SetupScreen())

    def action_toggle_dark(self) -> None:
        self.dark = not self.dark


def run_tui(config: dict[str, Any] | None = None) -> None:
    """Entry point to launch the TUI.

    Args:
        config: If provided, skips the SetupScreen and launches directly
                into the DebateScreen with these parameters.
    """
    app = TableRondeTUI(launch_config=config)
    app.run()
