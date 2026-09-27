"""Context pruning: summarize and compact conversation history between rounds."""

import logging

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.runnables import Runnable

logger = logging.getLogger(__name__)

# Maximum number of raw messages to keep after pruning
KEEP_LAST_MESSAGES = 4

# Maximum characters in messages before pruning kicks in
PRUNE_THRESHOLD_CHARS = 30_000


def estimate_history_size(history: list[BaseMessage]) -> int:
    """Estimate the total character count of the history."""
    total = 0
    for msg in history:
        if isinstance(msg.content, str):
            total += len(msg.content)
        elif isinstance(msg.content, list):
            for part in msg.content:
                if isinstance(part, str):
                    total += len(part)
                elif isinstance(part, dict) and "text" in part:
                    total += len(part["text"])
    return total


def summarize_history(
    llm: Runnable,
    history: list[BaseMessage],
    round_number: int,
) -> str:
    """Ask the LLM to produce a concise summary of the debate so far."""
    debate_text = []
    for msg in history:
        role = msg.__class__.__name__
        if isinstance(msg.content, str) and msg.content.strip():
            debate_text.append(f"[{role}]: {msg.content[:2000]}")
        elif isinstance(msg.content, list):
            text_parts = []
            for part in msg.content:
                if isinstance(part, str):
                    text_parts.append(part)
                elif isinstance(part, dict) and "text" in part:
                    text_parts.append(part["text"])
            if text_parts:
                debate_text.append(f"[{role}]: {''.join(text_parts)[:2000]}")

    combined = "\n".join(debate_text[-20:])  # Last 20 messages max for summarization

    summary_prompt = (
        f"You are a technical debate summarizer. Summarize rounds 1-{round_number} "
        f"of this architecture debate in 300 words maximum.\n"
        f"Keep ONLY:\n"
        f"- Key technical decisions made\n"
        f"- Unresolved disagreements\n"
        f"- Critical arguments from each participant\n"
        f"- Action items or constraints identified\n\n"
        f"Debate transcript:\n{combined}"
    )

    try:
        response = llm.invoke([HumanMessage(content=summary_prompt)])
        if isinstance(response.content, str):
            return response.content
        return str(response.content)
    except Exception as e:
        logger.warning("Failed to summarize history: %s", e)
        return f"[Summary of rounds 1-{round_number} unavailable due to error: {e}]"


def prune_history(
    llm: Runnable,
    history: list[BaseMessage],
    round_number: int,
    keep_last: int = KEEP_LAST_MESSAGES,
    threshold_chars: int = PRUNE_THRESHOLD_CHARS,
) -> list[BaseMessage]:
    """Prune the history if it exceeds the threshold.

    Returns a new list with a summary message followed by the last `keep_last` messages.
    If the history is under the threshold, returns it unchanged.
    """
    current_size = estimate_history_size(history)

    if current_size < threshold_chars:
        logger.debug(
            "History size %d chars is under threshold %d, skipping pruning",
            current_size,
            threshold_chars,
        )
        return history

    logger.info(
        "History size %d chars exceeds threshold %d. Pruning after round %d...",
        current_size,
        threshold_chars,
        round_number,
    )

    summary = summarize_history(llm, history, round_number)

    summary_msg = HumanMessage(
        content=(
            f"[CONTEXT SUMMARY — Rounds 1-{round_number}]\n\n"
            f"{summary}\n\n"
            f"[End of summary. The debate continues below.]"
        )
    )

    # Keep the last N messages for continuity
    recent = history[-keep_last:] if len(history) > keep_last else history

    pruned = [summary_msg] + recent

    new_size = estimate_history_size(pruned)
    logger.info(
        "Pruned history from %d to %d chars (%d messages → %d)",
        current_size,
        new_size,
        len(history),
        len(pruned),
    )

    return pruned
