import os
from collections.abc import Generator
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    BaseMessage,
    BaseMessageChunk,
    HumanMessage,
    SystemMessage,
)
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

DEFAULT_CONFIG = {
    "orchestrator": {
        "rounds": 1,
        "default_provider": "gemini",
        "default_model": "gemini-3.6-flash",
    },
    "architect": {
        "role": "architect",
        "title": "The Architect",
        "emoji": "🏛️",
        "temperature": 0.3,
        "prompt": (
            "You are The Architect (Lead Agent & Moderator).\n\n"
            "MINDSET:\n"
            "You are a pragmatic, objective, and highly structured Tech Lead. "
            "Your goal is not just to write code, but to design a comprehensive, robust, and deliverable vision. "
            "You listen to the experts (Skeptic, Enthusiast, etc.) with neutrality, extract the best of their ideas, "
            "and settle technical debates to make final decisions.\n\n"
            "ROLE & FORMAT:\n"
            "1. During the debate: Moderate, refocus the discussion if needed, and ask the right architectural questions.\n"
            "2. At the end (Final Resolution): You have the absolute authority to generate the **Implementation Plan v2.0**. "
            "This plan must be a highly structured Markdown document containing:\n"
            "   - An executive summary of the decisions made.\n"
            "   - The finalized and justified technology stack.\n"
            "   - The target architecture (ideally with a `mermaid` diagram if relevant).\n"
            "   - The phased implementation steps (Step-by-step action plan).\n"
            "   - Residual risks and mitigation strategies."
        )
    },
    "personas": [
        {
            "role": "skeptic",
            "title": "The Skeptic",
            "emoji": "😈",
            "temperature": 0.6,
            "prompt": (
                "You are The Skeptic (Devil's Advocate & Cyber/Perf Expert).\n\n"
                "MINDSET:\n"
                "You are brilliant, cynical, and obsessed with stability, security, and maintainability. "
                "You are a fierce advocate of the KISS (Keep It Simple, Stupid) and YAGNI principles. "
                "You despise tech 'hype', bloated frameworks, accidental complexity, and empty marketing promises.\n\n"
                "ROLE & COMMUNICATION:\n"
                "Your sole purpose is to crash the project on paper before it crashes in production. "
                "- Actively hunt for security flaws, bottlenecks, memory leaks, and race conditions.\n"
                "- If code or an architecture diagram is provided, point out exactly what will fail (e.g., 'This loop will OOM the server', 'This is an obvious SQL injection').\n"
                "- Be sharp, cynical, highly technical, and uncompromising. Use heavy technical jargon (O(n), SPOF, race condition, etc.).\n"
                "- Do not propose building a massive over-engineered system to fix a problem: always propose to simplify or remove what is unnecessary."
            )
        },
        {
            "role": "enthusiast",
            "title": "The Enthusiast",
            "emoji": "🚀",
            "temperature": 0.8,
            "prompt": (
                "You are The Enthusiast (10x Visionary & Productivity Expert).\n\n"
                "MINDSET:\n"
                "You are optimistic, heavily focused on Developer Experience (DX) and Time-to-Market. "
                "You love modern technologies, extreme automation, bleeding-edge open-source, and cloud-native solutions. "
                "Where the Skeptic sees risks, you see opportunities to build a scalable and incredible product.\n\n"
                "ROLE & COMMUNICATION:\n"
                "Your goal is to push the project towards modern excellence.\n"
                "- Propose innovative paradigms (Serverless, Event-Driven, Edge Computing, AI) if it drastically accelerates or improves the project.\n"
                "- Counter-attack the Skeptic by proving your solutions are viable (e.g., 'No, we are not going to write an HTTP server in C, we will use FastAPI and save 3 months').\n"
                "- Speak with passion, use an energetic tone, and highlight very specific libraries or tools (e.g., Docker, GitHub Actions, Redis, Tailwind, etc.).\n"
                "- Your improvements must be ambitious but must lead to concrete code or a realistic architecture."
            )
        }
    ]
}


def get_llm(
    provider: str = "gemini",
    model_name: str | None = None,
    temperature: float = 0.7,
    api_key: str | None = None,
    base_url: str | None = None,
) -> BaseChatModel:
    prov = provider.lower()
    if prov in ("copilot", "github", "openai"):
        model = model_name or "gpt-4o"
        key = (
            api_key
            or os.getenv("GITHUB_TOKEN")
            or os.getenv("COPILOT_API_KEY")
            or os.getenv("OPENAI_API_KEY")
        )
        if not key:
            raise ValueError(
                "API key not found for GitHub Copilot / OpenAI. "
                "Please set GITHUB_TOKEN or COPILOT_API_KEY in your environment variables."
            )
        endpoint = base_url or (
            "https://models.inference.ai.azure.com" if prov in ("copilot", "github") else None
        )
        kwargs: dict = {"model": model, "temperature": temperature, "api_key": key}
        if endpoint:
            kwargs["base_url"] = endpoint
        return ChatOpenAI(**kwargs)
    elif prov == "gemini":
        model = model_name or "gemini-3.6-flash"
        key = api_key or os.getenv("GEMINI_API_KEY")
        if not key:
            raise ValueError(
                "Gemini API key not found. Please set the GEMINI_API_KEY environment variable."
            )
        return ChatGoogleGenerativeAI(model=model, temperature=temperature, google_api_key=key)
    else:
        raise ValueError(f"Unsupported provider: '{provider}'. Choose 'gemini' or 'copilot'.")


class TableRondeAgents:
    def __init__(
        self,
        config: dict[str, Any] | None = None,
        provider: str | None = None,
        model_name: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        self.config = config or DEFAULT_CONFIG
        orch_config = self.config.get("orchestrator", {})
        
        self.default_provider = provider or orch_config.get("default_provider", "gemini")
        self.default_model = model_name or orch_config.get("default_model", "gemini-3.6-flash")
        self.api_key = api_key
        self.base_url = base_url
        
        self.llms: dict[str, BaseChatModel] = {}
        self.prompts: dict[str, str] = {}
        self.personas: list[dict[str, Any]] = self.config.get("personas", [])
        self.architect_cfg: dict[str, Any] = self.config.get("architect", DEFAULT_CONFIG["architect"])
        
        # Initialize LLMs and prompts for each persona
        for p in self.personas:
            self._init_agent(p)
        
        # Initialize the Architect
        self._init_agent(self.architect_cfg)

    def _init_agent(self, agent_cfg: dict[str, Any]):
        role = agent_cfg["role"]
        prov = agent_cfg.get("provider", self.default_provider)
        model = agent_cfg.get("model", self.default_model)
        temp = agent_cfg.get("temperature", 0.7)
        self.prompts[role] = agent_cfg.get("prompt", "")
        self.llms[role] = get_llm(
            provider=prov,
            model_name=model,
            temperature=temp,
            api_key=self.api_key,
            base_url=self.base_url
        )

    def _get_llm_for_role(self, role: str) -> BaseChatModel:
        if role not in self.llms:
            raise ValueError(f"Unknown role: {role}")
        return self.llms[role]

    def _get_system_prompt_for_role(self, role: str) -> str:
        if role not in self.prompts:
            raise ValueError(f"Unknown role: {role}")
        return self.prompts[role]

    def _build_messages(self, role: str, history: list, new_instruction: str) -> list[BaseMessage]:
        messages: list[BaseMessage] = [SystemMessage(content=self._get_system_prompt_for_role(role))]
        messages.extend(history)
        messages.append(HumanMessage(content=new_instruction))
        return messages

    def invoke_agent(self, role: str, history: list, new_instruction: str) -> str:
        messages = self._build_messages(role, history, new_instruction)
        response = self._get_llm_for_role(role).invoke(messages)
        return str(response.content)

    def stream_agent(
        self, role: str, history: list, new_instruction: str
    ) -> Generator[BaseMessageChunk, None, None]:
        messages = self._build_messages(role, history, new_instruction)
        yield from self._get_llm_for_role(role).stream(messages)
