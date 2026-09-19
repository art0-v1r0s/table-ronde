"""Tests for the Textual TUI components."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from textual.widgets import Button, Checkbox, Input, Select, TextArea

from table_ronde.tui import (
    AgentMessageWidget,
    DebateScreen,
    HumanInputModal,
    PlanViewScreen,
    SetupScreen,
    TableRondeTUI,
)


@pytest.mark.asyncio
async def test_setup_screen_renders() -> None:
    """SetupScreen should mount with all form inputs visible."""
    app = TableRondeTUI()
    async with app.run_test():
        # The app should push SetupScreen on mount
        assert isinstance(app.screen, SetupScreen)

        # Check all form elements are present
        assert app.screen.query_one("#prompt", TextArea) is not None
        assert app.screen.query_one("#provider", Select) is not None
        assert app.screen.query_one("#model", Input) is not None
        assert app.screen.query_one("#rounds", Input) is not None
        assert app.screen.query_one("#interactive", Checkbox) is not None
        assert app.screen.query_one("#path", Input) is not None
        assert app.screen.query_one("#config_file", Input) is not None
        assert app.screen.query_one("#output", Input) is not None
        assert app.screen.query_one("#launch", Button) is not None
        assert app.screen.query_one("#cancel", Button) is not None


@pytest.mark.asyncio
async def test_setup_validation_empty_prompt() -> None:
    """Clicking Launch with an empty prompt should not push a new screen."""
    app = TableRondeTUI()
    async with app.run_test(size=(120, 50)) as pilot:
        assert isinstance(app.screen, SetupScreen)

        # Directly press the launch button by finding it
        launch_btn = app.screen.query_one("#launch", Button)
        launch_btn.press()
        await pilot.pause()

        # Should still be on SetupScreen (no screen push happened)
        assert isinstance(app.screen, SetupScreen)


@pytest.mark.asyncio
async def test_setup_cancel_exits() -> None:
    """Clicking Cancel on SetupScreen should exit the app."""
    app = TableRondeTUI()
    async with app.run_test(size=(120, 50)) as pilot:
        assert isinstance(app.screen, SetupScreen)

        cancel_btn = app.screen.query_one("#cancel", Button)
        cancel_btn.press()
        await pilot.pause()


@pytest.mark.asyncio
async def test_agent_message_widget_reactive() -> None:
    """AgentMessageWidget should update its display when text changes."""

    class _TestApp(TableRondeTUI):
        """Minimal app to test AgentMessageWidget in isolation."""

        CSS_PATH = "../src/table_ronde/theme.tcss"

        def on_mount(self) -> None:
            # Don't push any screens — stay on default
            pass

    app = _TestApp()
    async with app.run_test(size=(120, 40)) as pilot:
        widget = AgentMessageWidget(
            title="Test Agent",
            emoji="🤖",
            color="bold cyan",
            widget_id="test-msg",
        )
        await app.screen.mount(widget)
        await pilot.pause()

        # Initially should be empty
        assert widget.text == ""

        # Set text via reactive
        widget.text = "Hello, world!"
        assert widget.text == "Hello, world!"

        # Mark done
        widget.mark_done()
        assert widget.has_class("done")

        # Mark errored
        widget.mark_errored("timeout")
        assert widget.has_class("errored")


@pytest.mark.asyncio
async def test_human_input_modal_submit() -> None:
    """HumanInputModal should return the entered text on submit."""
    app = TableRondeTUI()
    async with app.run_test(size=(120, 40)) as pilot:
        results: list[str | None] = []

        def on_dismiss(value: str | None) -> None:
            results.append(value)

        app.push_screen(HumanInputModal(), callback=on_dismiss)
        await pilot.pause()

        # Type into the text area
        text_area = app.screen.query_one("#human-input", TextArea)
        text_area.load_text("My feedback note")

        # Click submit
        submit_btn = app.screen.query_one("#submit", Button)
        submit_btn.press()
        await pilot.pause()

        assert len(results) == 1
        assert results[0] == "My feedback note"


@pytest.mark.asyncio
async def test_human_input_modal_skip() -> None:
    """HumanInputModal skip button should return None."""
    app = TableRondeTUI()
    async with app.run_test(size=(120, 40)) as pilot:
        results: list[str | None] = []

        def on_dismiss(value: str | None) -> None:
            results.append(value)

        app.push_screen(HumanInputModal(), callback=on_dismiss)
        await pilot.pause()

        skip_btn = app.screen.query_one("#skip", Button)
        skip_btn.press()
        await pilot.pause()

        assert len(results) == 1
        assert results[0] is None


@pytest.mark.asyncio
async def test_human_input_modal_round() -> None:
    """HumanInputModal round button should return '/round'."""
    app = TableRondeTUI()
    async with app.run_test(size=(120, 40)) as pilot:
        results: list[str | None] = []

        def on_dismiss(value: str | None) -> None:
            results.append(value)

        app.push_screen(HumanInputModal(), callback=on_dismiss)
        await pilot.pause()

        round_btn = app.screen.query_one("#round", Button)
        round_btn.press()
        await pilot.pause()

        assert len(results) == 1
        assert results[0] == "/round"


@pytest.mark.asyncio
async def test_plan_view_screen_renders() -> None:
    """PlanViewScreen should render the markdown content."""
    app = TableRondeTUI()
    async with app.run_test(size=(120, 40)) as pilot:
        plan_text = "# Final Plan\n\nThis is the plan content."
        app.push_screen(PlanViewScreen(plan_text, "test_plan.md"))
        await pilot.pause()

        assert isinstance(app.screen, PlanViewScreen)
        assert app.screen.query_one("#export", Button) is not None
        assert app.screen.query_one("#back", Button) is not None


@pytest.mark.asyncio
async def test_plan_view_back_button() -> None:
    """PlanViewScreen back button should pop the screen."""
    app = TableRondeTUI()
    async with app.run_test(size=(120, 40)) as pilot:
        # We're on SetupScreen initially
        assert isinstance(app.screen, SetupScreen)

        plan_text = "# Plan"
        app.push_screen(PlanViewScreen(plan_text, "test.md"))
        await pilot.pause()
        assert isinstance(app.screen, PlanViewScreen)

        back_btn = app.screen.query_one("#back", Button)
        back_btn.press()
        await pilot.pause()
        # Should be back on SetupScreen
        assert isinstance(app.screen, SetupScreen)


@pytest.mark.asyncio
async def test_direct_launch_pushes_debate_screen() -> None:
    """Providing a launch_config should push DebateScreen (not SetupScreen)."""
    config = {
        "prompt": "Test topic",
        "provider": "gemini",
        "model": None,
        "rounds": 1,
        "interactive": False,
        "path": None,
        "config_file": None,
        "output": "test_plan.md",
    }

    # Patch run_worker so the debate thread doesn't actually start (avoids API calls)
    with patch.object(DebateScreen, "on_mount", lambda self: None):
        app = TableRondeTUI(launch_config=config)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            # Should be on DebateScreen, not SetupScreen
            assert isinstance(app.screen, DebateScreen)
