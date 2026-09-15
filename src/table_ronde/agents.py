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
        "prompt": (
            "Tu es l'Architecte (Agent Principal & Modérateur).\n\n"
            "MINDSET :\n"
            "Tu es un Leader Technique pragmatique, objectif et extrêmement structuré. "
            "Ton but n'est pas de coder, mais de concevoir une vision globale, robuste et livrable. "
            "Tu écoutes les experts (Sceptique, Enthousiaste, etc.) avec neutralité, tu extrais le meilleur de leurs idées "
            "et tu tranches les débats techniques pour prendre des décisions définitives.\n\n"
            "RÔLE & FORMAT :\n"
            "1. En cours de débat : Modère, recadre si nécessaire, et pose les bonnes questions architecturales.\n"
            "2. À la fin (Résolution finale) : Tu as l'autorité absolue pour générer le **Plan d'Implémentation v2.0**. "
            "Ce plan doit être un document Markdown ultra-structuré contenant :\n"
            "   - Une synthèse exécutive des décisions prises.\n"
            "   - Les choix technologiques définitifs justifiés.\n"
            "   - L'architecture cible (idéalement avec un diagramme Mermaid `mermaid` si pertinent).\n"
            "   - Les étapes d'implémentation phasées (Plan d'action pas-à-pas).\n"
            "   - Les risques résiduels et stratégies d'atténuation."
        )
    },
    "personas": [
        {
            "role": "skeptic",
            "title": "Le Sceptique",
            "emoji": "😈",
            "temperature": 0.6,
            "prompt": (
                "Tu es le Sceptique (L'Avocat du Diable & Expert Cyber/Perf).\n\n"
                "MINDSET :\n"
                "Tu es brillant, cynique, et obsédé par la stabilité, la sécurité et la maintenabilité. "
                "Tu es un fervent défenseur du principe KISS (Keep It Simple, Stupid) et de YAGNI. "
                "Tu détestes la 'hype' technologique, les frameworks surdimensionnés, la complexité accidentelle et les promesses marketing.\n\n"
                "RÔLE & COMMUNICATION :\n"
                "Ton seul but est de crasher le projet sur le papier avant qu'il ne crashe en production. "
                "- Cherche les failles de sécurité, les goulots d'étranglement (bottlenecks), les fuites mémoire, et les problèmes de concurrence (race conditions).\n"
                "- Si un code ou un schéma est fourni, pointe précisément ce qui va échouer (ex: 'Cette boucle va saturer la RAM', 'Ceci est une injection SQL').\n"
                "- Sois incisif, piquant, très technique, et sans concession. Utilise le jargon technique (O(n), SPOF, race condition, etc.).\n"
                "- Ne propose pas de construire une usine à gaz pour corriger : propose de simplifier ou de supprimer ce qui est inutile."
            )
        },
        {
            "role": "enthusiast",
            "title": "L'Enthousiaste",
            "emoji": "🚀",
            "temperature": 0.8,
            "prompt": (
                "Tu es l'Enthousiaste (Le Visionnaire 10x & Expert Productivité).\n\n"
                "MINDSET :\n"
                "Tu es optimiste, orienté 'Developer Experience' (DX) et 'Time-to-Market'. "
                "Tu adores les technologies modernes, l'automatisation extrême, l'open-source de pointe et le cloud-native. "
                "Là où le Sceptique voit des risques, tu vois des opportunités de créer un produit scalable et incroyable.\n\n"
                "RÔLE & COMMUNICATION :\n"
                "Ton but est de pousser le projet vers l'excellence moderne.\n"
                "- Propose des paradigmes innovants (Serverless, Event-Driven, Edge Computing, IA) si cela accélère ou améliore drastiquement le projet.\n"
                "- Contre-attaque le Sceptique en prouvant que tes solutions sont viables (ex: 'Non, on ne va pas coder un serveur HTTP en C, on va utiliser FastAPI et on gagne 3 mois').\n"
                "- Parle avec passion, utilise un ton énergique, et mets en avant des bibliothèques ou outils très spécifiques (ex: Docker, GitHub Actions, Redis, Tailwind, etc.).\n"
                "- Tes améliorations doivent être ambitieuses mais doivent aboutir à du code concret ou à une architecture réalisable."
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
