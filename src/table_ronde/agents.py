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
        "title": "L'Architecte",
        "emoji": "🏛️",
        "temperature": 0.3,
        "prompt": "Tu es l'Agent : L'Architecte (Le Modérateur / Orienteur).\nTon rôle est de cadrer la discussion, de modérer les débats entre les experts, et de trancher avec pragmatisme.\nTu retiens les critiques valides pour éliminer les risques, et tu conserves les meilleures idées.\nQuand on te demande la résolution finale, tu génères un **Plan d'Implémentation v2.0** ultra-structuré en Markdown."
    },
    "personas": [
        {
            "role": "skeptic",
            "title": "Le Sceptique",
            "emoji": "😈",
            "temperature": 0.6,
            "prompt": "Tu es l'Agent : Le Sceptique (L'Avocat du Diable).\nTon rôle est de trouver les failles, les problèmes de scalabilité, les risques de sécurité, le manque de rigueur et la dette technique dans le sujet ou le projet présenté.\nTu détestes la complexité inutile, les frameworks surdimensionnés et les promesses irréalistes.\nRéfère-toi explicitement au code ou à la structure si un projet existant est fourni.\nSois incisif, direct, technique et sans concession, mais toujours professionnel."
        },
        {
            "role": "enthusiast",
            "title": "L'Enthousiaste",
            "emoji": "🚀",
            "temperature": 0.8,
            "prompt": "Tu es l'Agent : L'Enthousiaste (Le Visionnaire).\nTon rôle est de voir le potentiel, proposer des solutions modernes, accélérer le développement, intégrer des bibliothèques efficaces et automatiser au maximum.\nTu défends les idées innovantes face aux critiques et tu cherches le chemin le plus rapide pour délivrer de la valeur.\nPropose des améliorations ambitieuses mais concrètes."
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
                "Clé d'API introuvable pour GitHub Copilot / OpenAI. "
                "Veuillez définir GITHUB_TOKEN ou COPILOT_API_KEY dans vos variables d'environnement."
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
                "La clé d'API Gemini est introuvable. Veuillez définir la variable d'environnement GEMINI_API_KEY."
            )
        return ChatGoogleGenerativeAI(model=model, temperature=temperature, google_api_key=key)
    else:
        raise ValueError(f"Fournisseur (provider) non pris en charge : '{provider}'. Choisissez 'gemini' ou 'copilot'.")


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
        
        # Initialiser les LLMs et les prompts pour chaque persona
        for p in self.personas:
            self._init_agent(p)
        
        # Initialiser l'Architecte
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
            raise ValueError(f"Rôle inconnu : {role}")
        return self.llms[role]

    def _get_system_prompt_for_role(self, role: str) -> str:
        if role not in self.prompts:
            raise ValueError(f"Rôle inconnu : {role}")
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
