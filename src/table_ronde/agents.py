import os

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

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


def get_llm(model_name: str = "gemini-2.5-flash", temperature: float = 0.7, api_key: str | None = None) -> ChatGoogleGenerativeAI:
    key = api_key or os.getenv("GEMINI_API_KEY")
    if not key:
        raise ValueError(
            "La clé d'API Gemini est introuvable. Veuillez définir la variable d'environnement GEMINI_API_KEY."
        )
    return ChatGoogleGenerativeAI(model=model_name, temperature=temperature, google_api_key=key)


class TableRondeAgents:
    def __init__(self, model_name: str = "gemini-2.5-flash", api_key: str | None = None):
        self.skeptic_llm = get_llm(model_name, temperature=0.6, api_key=api_key)
        self.enthusiast_llm = get_llm(model_name, temperature=0.8, api_key=api_key)
        self.architect_llm = get_llm(model_name, temperature=0.3, api_key=api_key)

    def invoke_agent(self, role: str, history: list, new_instruction: str) -> str:
        if role == "skeptic":
            system_prompt = SKEPTIC_SYSTEM_PROMPT
            llm = self.skeptic_llm
        elif role == "enthusiast":
            system_prompt = ENTHUSIAST_SYSTEM_PROMPT
            llm = self.enthusiast_llm
        elif role == "architect":
            system_prompt = ARCHITECT_SYSTEM_PROMPT
            llm = self.architect_llm
        else:
            raise ValueError(f"Rôle inconnu : {role}")

        messages = [SystemMessage(content=system_prompt)]
        messages.extend(history)
        messages.append(HumanMessage(content=new_instruction))

        response = llm.invoke(messages)
        return str(response.content)
