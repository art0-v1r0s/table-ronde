from unittest.mock import patch

import pytest

from table_ronde.menu import run_interactive_menu


def test_menu_standard_flow():
    expected_result = {
        "prompt": "Build a GraphQL gateway",
        "provider": "gemini",
        "rounds": 2,
        "interactive": False,
        "path": None,
        "config_file": None,
        "output": "output.md",
    }

    with patch("table_ronde.menu.SetupMenuApp.run", return_value=expected_result):
        result = run_interactive_menu()

    assert result == expected_result


def test_menu_abort_on_cancel():
    with (
        patch("table_ronde.menu.SetupMenuApp.run", return_value=None),
        pytest.raises(SystemExit),
    ):
        run_interactive_menu()


def test_menu_custom_rounds():
    expected_result = {
        "prompt": "Test topic",
        "provider": "openai",
        "rounds": 5,
        "interactive": True,
        "path": None,
        "config_file": None,
        "output": "plan.md",
    }

    with patch("table_ronde.menu.SetupMenuApp.run", return_value=expected_result):
        result = run_interactive_menu()

    assert result["rounds"] == 5
    assert result["provider"] == "openai"
    assert result["interactive"] is True
