import logging
import os
from typing import Any

try:
    from typesafe_sdk import TypeSafeClient, Choice, Noul, Score
except ImportError:
    TypeSafeClient = None


class JevEngine:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.enabled = config.get("jev_enabled", False)
        self.consensus_threshold = config.get("consensus_threshold", 0.85)
        self.routing_threshold = config.get("routing_threshold", 0.75)

        self.client = None
        if self.enabled:
            if TypeSafeClient is None:
                logging.warning("Jev is enabled but typesafe-sdk is not installed.")
                self.enabled = False
            elif not os.environ.get("TYPESAFE_API_KEY"):
                logging.warning("Jev is enabled but TYPESAFE_API_KEY is not set.")
                self.enabled = False
            else:
                self.client = TypeSafeClient()

    def evaluate_consensus(self, history: list[Any]) -> bool:
        if not self.enabled or not self.client:
            return False

        try:
            recent_msgs = history[-4:]
            formatted_history = ""
            for msg in recent_msgs:
                if hasattr(msg, "content"):
                    if isinstance(msg.content, str):
                        content = msg.content
                    elif isinstance(msg.content, list):
                        content = "".join(
                            [
                                part["text"]
                                if isinstance(part, dict) and "text" in part
                                else str(part)
                                for part in msg.content
                            ]
                        )
                    else:
                        content = str(msg.content)

                    role = msg.__class__.__name__
                    formatted_history += f"{role}: {content}\n"

            response = self.client.system_one(
                state=f"Conversation:\n{formatted_history}",
                questions={
                    "consensus": Noul(instructions="The participants have reached a clear, explicit technical agreement on the architecture and tools.")
                }
            )
            score = response.answers["consensus"].noul
            return score >= self.consensus_threshold
        except Exception as e:
            logging.error(f"JevEngine consensus evaluation failed: {e}")
            return False

    def evaluate_user_note(self, note: str) -> str:
        if not self.enabled or not self.client:
            return "NEW_ROUND"

        try:
            response = self.client.system_one(
                state=f"User Note:\n{note}",
                questions={
                    "routing": Choice(
                        instructions="Based on the user note, what should happen next?",
                        criteria={
                            "SYNTHESIS": "The note is a minor tweak or confirmation. Proceed to final synthesis.",
                            "NEW_ROUND": "The note introduces major architectural changes, paradigms, or asks for debate."
                        }
                    )
                }
            )
            ans = response.answers["routing"]
            if ans.confidence >= self.routing_threshold:
                return ans.choice
            return "NEW_ROUND"
        except Exception as e:
            logging.error(f"JevEngine user note evaluation failed: {e}")
            return "NEW_ROUND"
