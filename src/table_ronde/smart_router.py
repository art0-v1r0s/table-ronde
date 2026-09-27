import logging
from typing import Any

from pydantic import BaseModel, Field

from table_ronde.agents import get_llm

logger = logging.getLogger(__name__)

class TaskRoutingDecision(BaseModel):
    rounds: int = Field(description="Recommended number of rounds (1 to 5) based on task complexity.")
    architect_model_tier: str = Field(description="'pro' for complex tasks requiring deep reasoning, 'flash' for simple tasks.")
    expert_model_tier: str = Field(description="'pro' or 'flash'.")
    temperature: float = Field(description="Between 0.0 (strict) and 0.8 (creative).")
    complexity: int = Field(description="Estimated complexity score (1-10).")
    domain: str = Field(description="Main domain (e.g., 'Frontend', 'DevOps', 'Security', 'Architecture').")

class ConsensusDecision(BaseModel):
    consensus_reached: bool = Field(
        description="True if participants have reached a clear technical agreement, False otherwise."
    )
    confidence: float = Field(description="Confidence score between 0.0 and 1.0")


class RoutingDecision(BaseModel):
    action: str = Field(
        description="Must be exactly 'SYNTHESIS' (minor tweak/agreement) or 'NEW_ROUND' (major change/debate)."
    )


FAST_MODELS: dict[str, str] = {
    "gemini": "gemini-2.0-flash",
    "openai": "gpt-4o-mini",
    "copilot": "gpt-4o-mini",
    "github": "gpt-4o-mini",
    "claude": "claude-haiku-4-20250414",
    "ollama": "llama3.1",
}


class SmartRouterEngine:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.enabled = config.get("smart_routing_enabled", False)
        self.provider = config.get(
            "router_provider", config.get("default_provider", "gemini")
        )
        self.model_name = config.get(
            "router_model", FAST_MODELS.get(self.provider, "gemini-2.0-flash")
        )
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
                logger.error(
                    "Failed to initialize SmartRouterEngine with provider %s and model %s: %s",
                    self.provider,
                    self.model_name,
                    e,
                )
                self.enabled = False

    def analyze_task(self, prompt: str) -> TaskRoutingDecision | None:
        if not self.enabled:
            return None
        try:
            llm = get_llm(self.provider, self.model_name, temperature=0.0)
            analyzer = llm.with_structured_output(TaskRoutingDecision)
            return analyzer.invoke(f"Analyze this task and determine the optimal debate configuration:\n\n{prompt}")
        except Exception as e:
            logger.error("SmartRouter analyze_task failed: %s", e)
            return None

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
            logger.error("SmartRouterEngine consensus evaluation failed: %s", e)
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
            logger.error("SmartRouterEngine user note evaluation failed: %s", e)
            return "NEW_ROUND"
