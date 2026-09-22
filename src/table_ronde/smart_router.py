import logging
from typing import Any
from pydantic import BaseModel, Field

from table_ronde.agents import get_llm

class ConsensusDecision(BaseModel):
    consensus_reached: bool = Field(
        description="True if participants have reached a clear technical agreement, False otherwise."
    )
    confidence: float = Field(description="Confidence score between 0.0 and 1.0")


class RoutingDecision(BaseModel):
    action: str = Field(
        description="Must be exactly 'SYNTHESIS' (minor tweak/agreement) or 'NEW_ROUND' (major change/debate)."
    )


class SmartRouterEngine:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.enabled = config.get("smart_routing_enabled", False)
        self.provider = config.get("router_provider", "openai")
        self.model_name = config.get("router_model", "gpt-4o-mini")
        self.consensus_threshold = config.get("consensus_threshold", 0.85)

        self.consensus_evaluator = None
        self.routing_evaluator = None

        if self.enabled:
            try:
                # Use the project's get_llm factory to support any provider
                llm = get_llm(
                    provider=self.provider,
                    model_name=self.model_name,
                    temperature=0.0
                )
                self.consensus_evaluator = llm.with_structured_output(ConsensusDecision)
                self.routing_evaluator = llm.with_structured_output(RoutingDecision)
            except Exception as e:
                logging.error(f"Failed to initialize SmartRouterEngine with provider {self.provider} and model {self.model_name}: {e}")
                self.enabled = False

    def evaluate_consensus(self, history: list[Any]) -> bool:
        if not self.enabled or not self.consensus_evaluator:
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

            prompt = f"Analyze this architecture debate and decide if they explicitly agree on the solution.\n\n{formatted_history}"
            result = self.consensus_evaluator.invoke(prompt)
            if hasattr(result, "consensus_reached") and hasattr(result, "confidence"):
                return result.consensus_reached and result.confidence >= self.consensus_threshold
            return False
        except Exception as e:
            logging.error(f"SmartRouterEngine consensus evaluation failed: {e}")
            return False

    def evaluate_user_note(self, note: str) -> str:
        if not self.enabled or not self.routing_evaluator:
            return "NEW_ROUND"

        try:
            prompt = f"User note: '{note}'. Decide if this introduces a major change requiring a 'NEW_ROUND', or if it's a minor tweak that goes straight to 'SYNTHESIS'."
            result = self.routing_evaluator.invoke(prompt)
            if hasattr(result, "action") and result.action in ["SYNTHESIS", "NEW_ROUND"]:
                return result.action
            return "NEW_ROUND"
        except Exception as e:
            logging.error(f"SmartRouterEngine user note evaluation failed: {e}")
            return "NEW_ROUND"
