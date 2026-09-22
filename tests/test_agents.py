import os
from unittest.mock import MagicMock, patch

import pytest

from table_ronde.agents import TableRondeAgents, get_llm


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
            "model": "gpt-4o",
        },
        "personas": [],
    }
    agents = TableRondeAgents(config=config)
    assert len(agents.personas) == 0
    assert agents.architect_cfg["role"] == "arch"


# ── Ollama provider tests ──


@patch("table_ronde.agents.AVAILABLE_TOOLS", [])
def test_get_llm_ollama_default_model():
    """Ollama provider should use default model 'llama3.1' and no API key."""
    with patch("langchain_ollama.ChatOllama") as mock_ollama:
        mock_instance = MagicMock()
        mock_ollama.return_value = mock_instance
        mock_instance.bind_tools.return_value = mock_instance

        result = get_llm(provider="ollama")

        mock_ollama.assert_called_once_with(
            model="llama3.1",
            temperature=0.7,
            base_url="http://localhost:11434",
        )
        mock_instance.bind_tools.assert_called_once()
        assert result is mock_instance


@patch("table_ronde.agents.AVAILABLE_TOOLS", [])
def test_get_llm_ollama_custom_model():
    """Ollama provider should accept custom model name and base_url."""
    with patch("langchain_ollama.ChatOllama") as mock_ollama:
        mock_instance = MagicMock()
        mock_ollama.return_value = mock_instance
        mock_instance.bind_tools.return_value = mock_instance

        result = get_llm(
            provider="ollama",
            model_name="mistral",
            temperature=0.5,
            base_url="http://my-server:11434",
        )

        mock_ollama.assert_called_once_with(
            model="mistral",
            temperature=0.5,
            base_url="http://my-server:11434",
        )
        assert result is mock_instance


# ── Claude provider tests ──


@patch("table_ronde.agents.AVAILABLE_TOOLS", [])
def test_get_llm_claude_with_api_key():
    """Claude provider should work with an explicit API key."""
    with patch("langchain_anthropic.ChatAnthropic") as mock_anthropic:
        mock_instance = MagicMock()
        mock_anthropic.return_value = mock_instance
        mock_instance.bind_tools.return_value = mock_instance

        result = get_llm(provider="claude", api_key="sk-ant-fake")

        mock_anthropic.assert_called_once_with(
            model="claude-sonnet-4-20250514",
            temperature=0.7,
            api_key="sk-ant-fake",
        )
        mock_instance.bind_tools.assert_called_once()
        assert result is mock_instance


@patch("table_ronde.agents.AVAILABLE_TOOLS", [])
def test_get_llm_claude_from_env():
    """Claude provider should read ANTHROPIC_API_KEY from env."""
    with patch("langchain_anthropic.ChatAnthropic") as mock_anthropic:
        mock_instance = MagicMock()
        mock_anthropic.return_value = mock_instance
        mock_instance.bind_tools.return_value = mock_instance

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-env"}):
            result = get_llm(provider="claude")

        mock_anthropic.assert_called_once_with(
            model="claude-sonnet-4-20250514",
            temperature=0.7,
            api_key="sk-ant-env",
        )
        assert result is mock_instance


@patch("table_ronde.agents.AVAILABLE_TOOLS", [])
def test_get_llm_claude_custom_model():
    """Claude provider should accept custom model names."""
    with patch("langchain_anthropic.ChatAnthropic") as mock_anthropic:
        mock_instance = MagicMock()
        mock_anthropic.return_value = mock_instance
        mock_instance.bind_tools.return_value = mock_instance

        result = get_llm(
            provider="claude",
            model_name="claude-opus-4-20250514",
            temperature=0.3,
            api_key="sk-ant-fake",
        )

        mock_anthropic.assert_called_once_with(
            model="claude-opus-4-20250514",
            temperature=0.3,
            api_key="sk-ant-fake",
        )
        assert result is mock_instance


@patch("table_ronde.agents.AVAILABLE_TOOLS", [])
def test_get_llm_claude_missing_key_raises():
    """Claude provider should raise ValueError when no API key is available."""
    with patch.dict(os.environ, {}, clear=True):
        # Remove ANTHROPIC_API_KEY if set
        os.environ.pop("ANTHROPIC_API_KEY", None)
        with pytest.raises(ValueError, match="Anthropic API key not found"):
            get_llm(provider="claude")


def test_get_llm_unsupported_provider_raises():
    """Unsupported provider should raise ValueError with all valid options listed."""
    with pytest.raises(ValueError, match="Unsupported provider"):
        get_llm(provider="nonexistent")


# ── TableRondeAgents with new providers ──


@patch("table_ronde.agents.AVAILABLE_TOOLS", [])
def test_agents_ollama_provider():
    """TableRondeAgents should initialize with ollama provider."""
    with patch("langchain_ollama.ChatOllama") as mock_ollama:
        mock_instance = MagicMock()
        mock_ollama.return_value = mock_instance
        mock_instance.bind_tools.return_value = mock_instance

        config = {
            "orchestrator": {"rounds": 1, "default_provider": "ollama"},
            "architect": {"role": "architect", "title": "Arch"},
            "personas": [],
        }
        agents = TableRondeAgents(config=config, provider="ollama")
        assert agents._get_llm_for_role("architect") is not None


@patch("table_ronde.agents.AVAILABLE_TOOLS", [])
def test_agents_claude_provider():
    """TableRondeAgents should initialize with claude provider."""
    with patch("langchain_anthropic.ChatAnthropic") as mock_anthropic:
        mock_instance = MagicMock()
        mock_anthropic.return_value = mock_instance
        mock_instance.bind_tools.return_value = mock_instance

        config = {
            "orchestrator": {"rounds": 1, "default_provider": "claude"},
            "architect": {"role": "architect", "title": "Arch"},
            "personas": [],
        }
        agents = TableRondeAgents(
            config=config, provider="claude", api_key="sk-ant-fake"
        )
        assert agents._get_llm_for_role("architect") is not None
