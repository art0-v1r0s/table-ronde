from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from table_ronde.cli import app

runner = CliRunner()


def _setup_mock_agents(mock_agents):
    instance = MagicMock()
    instance.personas = [
        {
            "role": "skeptic",
            "title": "The Skeptic",
            "emoji": "😈",
            "model": "gemini-3.6-flash",
            "temperature": 0.6,
        }
    ]
    instance.architect_cfg = {
        "role": "architect",
        "title": "The Architect",
        "emoji": "🏛️",
        "model": "gemini-3.6-flash",
        "temperature": 0.3,
    }
    instance.config = {"orchestrator": {"rounds": 1}}
    mock_agents.return_value = instance
    return instance


@patch("table_ronde.cli.TableRondeAgents")
@patch("table_ronde.cli.Orchestrator")
def test_cli_direct_with_prompt(mock_orchestrator, mock_agents, tmp_path):
    _setup_mock_agents(mock_agents)
    mock_orch_instance = MagicMock()
    mock_orch_instance.run_simulation.return_value = {
        "final_plan": "# Final Plan",
        "full_transcript": "# Full Transcript",
    }
    mock_orchestrator.return_value = mock_orch_instance

    out_file = tmp_path / "plan.md"
    result = runner.invoke(
        app,
        [
            "Architecture microservices",
            "--rounds",
            "1",
            "-o",
            str(out_file),
            "--no-tui",
        ],
    )
    assert result.exit_code == 0
    assert mock_orchestrator.called
    assert mock_orch_instance.run_simulation.called
    assert out_file.exists()


@patch("table_ronde.cli.run_interactive_menu")
@patch("table_ronde.cli.TableRondeAgents")
@patch("table_ronde.cli.Orchestrator")
def test_cli_zero_args_triggers_menu(
    mock_orchestrator, mock_agents, mock_menu, tmp_path
):
    _setup_mock_agents(mock_agents)
    out_file = tmp_path / "plan_menu.md"
    mock_menu.return_value = {
        "prompt": "Test from menu",
        "provider": "gemini",
        "rounds": 1,
        "interactive": False,
        "path": None,
        "config_file": None,
        "output": str(out_file),
    }
    mock_orch_instance = MagicMock()
    mock_orch_instance.run_simulation.return_value = {
        "final_plan": "# Plan from menu",
        "full_transcript": "",
    }
    mock_orchestrator.return_value = mock_orch_instance

    result = runner.invoke(app, ["--no-tui"])
    assert result.exit_code == 0
    assert mock_menu.called
    assert mock_orchestrator.called
    assert out_file.exists()


@patch("table_ronde.cli.run_interactive_menu")
def test_cli_menu_abort(mock_menu):
    mock_menu.side_effect = SystemExit(0)
    result = runner.invoke(app, ["--no-tui"])
    assert result.exit_code == 0
