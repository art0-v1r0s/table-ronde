from unittest.mock import MagicMock, patch

import pytest

from table_ronde.agents import TableRondeAgents, get_llm


def test_get_llm_gemini():
    with patch("table_ronde.agents.ChatGoogleGenerativeAI") as mock_gemini:
        get_llm(provider="gemini", api_key="fake-gemini-key")
        mock_gemini.assert_called_once()
        assert mock_gemini.call_args.kwargs["google_api_key"] == "fake-gemini-key"
        assert mock_gemini.call_args.kwargs["model"] == "gemini-2.5-flash"


def test_get_llm_copilot():
    with patch("table_ronde.agents.ChatOpenAI") as mock_openai:
        get_llm(provider="copilot", api_key="fake-github-token")
        mock_openai.assert_called_once()
        assert mock_openai.call_args.kwargs["api_key"] == "fake-github-token"
        assert mock_openai.call_args.kwargs["base_url"] == "https://models.inference.ai.azure.com"
        assert mock_openai.call_args.kwargs["model"] == "gpt-4o"


def test_get_llm_unsupported():
    with pytest.raises(ValueError, match="Fournisseur"):
        get_llm(provider="invalid-provider")


def test_table_ronde_agents_init():
    with patch("table_ronde.agents.get_llm") as mock_get_llm:
        agents = TableRondeAgents(provider="copilot", api_key="test-key")
        assert mock_get_llm.call_count == 3
        assert agents.provider == "copilot"


def test_stream_agent_yields_chunks():
    with patch("table_ronde.agents.get_llm") as mock_get_llm:
        mock_llm = MagicMock()
        mock_llm.stream.return_value = iter([
            MagicMock(content="Hello "),
            MagicMock(content="World"),
        ])
        mock_get_llm.return_value = mock_llm

        agents = TableRondeAgents(provider="gemini", api_key="fake-key")
        chunks = list(agents.stream_agent("skeptic", [], "Test"))
        assert len(chunks) == 2
        assert chunks[0].content == "Hello "
        assert chunks[1].content == "World"
