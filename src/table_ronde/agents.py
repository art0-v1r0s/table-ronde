import os
from collections.abc import Generator

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    BaseMessage,
    BaseMessageChunk,
    HumanMessage,
    SystemMessage,
)
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

SKEPTIC_SYSTEM_PROMPT = """Tu es l'Agent 1 : Le Sceptique (L'Avocat du Diable).
Ton rôle est de trouver les failles, les problèmes de scalabilité, les risques de sécurité, le manque de rigueur et la dette technique dans le sujet ou le projet présenté.
Tu détestes la complexité inutile, les frameworks surdimensionnés et les promesses irréalistes.
Réfère-toi explicitement au code ou à la structure si un projet existant est fourni.
Sois incisif, direct, technique et sans concession, mais toujours professionnel."""

ENTHUSIAST_SYSTEM_PROMPT = """Tu es l'Agent 2 : L'Enthousiaste (Le Visionnaire).
Ton rôle est de voir le potentiel, proposer des solutions modernes, accélérer le développement, intégrer des bibliothèques efficaces et automatiser au maximum.
Tu défends les idées innovantes face aux attaques du Sceptique et tu cherches le chemin le plus rapide pour délivrer de la valeur.
Propose des améliorations ambitieuses mais concrètes."""

ARCHITECT_SYSTEM_PROMPT = """Tu es l'Agent 3 : L'Architecte (Le Modérateur / Orienteur).
Ton rôle est de cadrer la discussion, de modérer les débats entre le Sceptique et l'Enthousiaste, et de trancher avec pragmatisme.
Tu retiens les critiques valides du Sceptique pour éliminer les risques, et tu conserves les meilleures idées de l'Enthousiaste.
Quand on te demande la résolution finale, tu génères un **Plan d'Implémentation v2.0** ultra-structuré en Markdown."""


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
        model = model_name or "gemini-2.5-flash"
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
        provider: str = "gemini",
        model_name: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        self.provider = provider
        self.skeptic_llm = get_llm(
            provider=provider, model_name=model_name, temperature=0.6, api_key=api_key, base_url=base_url
        )
        self.enthusiast_llm = get_llm(
            provider=provider, model_name=model_name, temperature=0.8, api_key=api_key, base_url=base_url
        )
        self.architect_llm = get_llm(
            provider=provider, model_name=model_name, temperature=0.3, api_key=api_key, base_url=base_url
        )

    def _get_llm_for_role(self, role: str) -> BaseChatModel:
        mapping = {
            "skeptic": self.skeptic_llm,
            "enthusiast": self.enthusiast_llm,
            "architect": self.architect_llm,
        }
        if role not in mapping:
            raise ValueError(f"Rôle inconnu : {role}")
        return mapping[role]

    def _get_system_prompt_for_role(self, role: str) -> str:
        mapping = {
            "skeptic": SKEPTIC_SYSTEM_PROMPT,
            "enthusiast": ENTHUSIAST_SYSTEM_PROMPT,
            "architect": ARCHITECT_SYSTEM_PROMPT,
        }
        if role not in mapping:
            raise ValueError(f"Rôle inconnu : {role}")
        return mapping[role]

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

