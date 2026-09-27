"""Tests for context_pruner module."""

from unittest.mock import MagicMock

from langchain_core.messages import AIMessage, HumanMessage

from table_ronde.context_pruner import (
    PRUNE_THRESHOLD_CHARS,
    estimate_history_size,
    prune_history,
)


def test_estimate_history_size_strings():
    history = [
        HumanMessage(content="Hello world"),
        AIMessage(content="Hi there"),
    ]
    assert estimate_history_size(history) == len("Hello world") + len("Hi there")


def test_estimate_history_size_list_content():
    history = [
        AIMessage(content=[{"text": "abc"}, {"text": "def"}]),
    ]
    assert estimate_history_size(history) == 6


def test_prune_history_under_threshold():
    """History under threshold should be returned unchanged."""
    history = [HumanMessage(content="short")]
    mock_llm = MagicMock()

    result = prune_history(mock_llm, history, round_number=1)

    assert result is history  # Same object, not pruned
    mock_llm.invoke.assert_not_called()


def test_prune_history_over_threshold():
    """History over threshold should be summarized and pruned."""
    # Create a large history
    big_msg = "x" * (PRUNE_THRESHOLD_CHARS + 1000)
    history = [
        HumanMessage(content=big_msg),
        AIMessage(content="response 1"),
        AIMessage(content="response 2"),
        AIMessage(content="response 3"),
        AIMessage(content="response 4"),
        AIMessage(content="response 5"),
    ]

    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = "Summary of the debate so far."
    mock_llm.invoke.return_value = mock_response

    result = prune_history(mock_llm, history, round_number=2, keep_last=4)

    # Should have 1 summary + 4 recent = 5 messages
    assert len(result) == 5
    assert "CONTEXT SUMMARY" in result[0].content
    mock_llm.invoke.assert_called_once()


def test_prune_history_preserves_recent_messages():
    """The last N messages should be preserved after pruning."""
    big_msg = "x" * (PRUNE_THRESHOLD_CHARS + 1000)
    history = [
        HumanMessage(content=big_msg),
        AIMessage(content="old message"),
        AIMessage(content="recent 1"),
        AIMessage(content="recent 2"),
    ]

    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = "Summary"
    mock_llm.invoke.return_value = mock_response

    result = prune_history(mock_llm, history, round_number=1, keep_last=2)

    assert len(result) == 3  # 1 summary + 2 recent
    assert result[1].content == "recent 1"
    assert result[2].content == "recent 2"
