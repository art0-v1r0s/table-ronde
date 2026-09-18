import os
from unittest.mock import patch

from table_ronde.agents import TableRondeAgents


@patch("table_ronde.agents.ChatGoogleGenerativeAI")
@patch("table_ronde.agents.ChatOpenAI")
def test_default_config_loading(mock_openai, mock_gemini):
    os.environ["GEMINI_API_KEY"] = "fake-key"
    agents = TableRondeAgents()
    assert len(agents.personas) == 2
    assert agents.architect_cfg["role"] == "architect"
    llm = agents._get_llm_for_role("architect")
    assert llm is not None

@patch("table_ronde.agents.ChatGoogleGenerativeAI")
@patch("table_ronde.agents.ChatOpenAI")
def test_custom_config_loading(mock_openai, mock_gemini):
    os.environ["OPENAI_API_KEY"] = "fake-key"
    config = {
        "orchestrator": {"rounds": 1},
        "architect": {
            "role": "arch",
            "title": "Arch",
            "provider": "openai",
            "model": "gpt-4o"
        },
        "personas": []
    }
    agents = TableRondeAgents(config=config)
    assert len(agents.personas) == 0
    assert agents.architect_cfg["role"] == "arch"
