from unittest.mock import MagicMock, patch

import pytest

from table_ronde.menu import run_interactive_menu


def test_menu_standard_flow():
    with patch("questionary.text") as mock_text, \
         patch("questionary.select") as mock_select, \
         patch("questionary.confirm") as mock_confirm:

        # 1. prompt: text
        # 2. output: text
        mock_text_prompt = MagicMock()
        mock_text_prompt.ask.side_effect = ["Build a GraphQL gateway", "output.md"]
        mock_text.side_effect = lambda *args, **kwargs: mock_text_prompt

        # 1. provider: select -> gemini
        # 2. rounds: select -> 2
        mock_select_prompt = MagicMock()
        mock_select_prompt.ask.side_effect = ["gemini", "2"]
        mock_select.side_effect = lambda *args, **kwargs: mock_select_prompt

        # 1. interactive: confirm -> False
        # 2. scan_project: confirm -> False
        # 3. use_custom_config: confirm -> False
        # 4. launch: confirm -> True
        mock_confirm_prompt = MagicMock()
        mock_confirm_prompt.ask.side_effect = [False, False, False, True]
        mock_confirm.side_effect = lambda *args, **kwargs: mock_confirm_prompt

        result = run_interactive_menu()

        assert result["prompt"] == "Build a GraphQL gateway"
        assert result["provider"] == "gemini"
        assert result["rounds"] == 2
        assert result["interactive"] is False
        assert result["path"] is None
        assert result["config_file"] is None
        assert result["output"] == "output.md"


def test_menu_abort_on_cancel():
    with patch("questionary.text") as mock_text:
        mock_text_prompt = MagicMock()
        mock_text_prompt.ask.return_value = None  # User hit Ctrl+C
        mock_text.return_value = mock_text_prompt

        with pytest.raises(SystemExit):
            run_interactive_menu()


def test_menu_custom_rounds():
    with patch("questionary.text") as mock_text, \
         patch("questionary.select") as mock_select, \
         patch("questionary.confirm") as mock_confirm:

        # Prompt text, then custom rounds text, then output file text
        mock_text_prompt = MagicMock()
        mock_text_prompt.ask.side_effect = ["Test topic", "5", "plan.md"]
        mock_text.side_effect = lambda *args, **kwargs: mock_text_prompt

        mock_select_prompt = MagicMock()
        mock_select_prompt.ask.side_effect = ["openai", "custom"]
        mock_select.side_effect = lambda *args, **kwargs: mock_select_prompt

        mock_confirm_prompt = MagicMock()
        mock_confirm_prompt.ask.side_effect = [True, False, False, True]
        mock_confirm.side_effect = lambda *args, **kwargs: mock_confirm_prompt

        result = run_interactive_menu()
        assert result["rounds"] == 5
        assert result["provider"] == "openai"
        assert result["interactive"] is True
