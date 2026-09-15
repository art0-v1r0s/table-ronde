from collections.abc import Callable
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

from table_ronde.agents import TableRondeAgents
from table_ronde.scanner import scan_project


class Orchestrator:
    def __init__(
        self,
        agents: TableRondeAgents,
        on_message_callback: Callable[[str, str], None] | None = None,
    ):
        """
        :param agents: Instance de TableRondeAgents
        :param on_message_callback: Callback(role, text) appelé à chaque intervention d'un agent.
        """
        self.agents = agents
        self.on_message_callback = on_message_callback
        self.history: list[Any] = []

    def _notify(self, role: str, message: str):
        if self.on_message_callback:
            self.on_message_callback(role, message)

    def run_simulation(
        self, prompt_user: str, project_path: str | None = None
    ) -> dict[str, str]:
        # 1. Préparation du contexte initial
        context = f"Sujet / Demande initiale de l'utilisateur :\n{prompt_user}\n"
        if project_path:
            scanned_data = scan_project(project_path)
            context += f"\n\nContextuel du projet existant ({project_path}) :\n{scanned_data}"

        # --- PHASE 1 : L'AUDIT ---
        architect_intro = self.agents.invoke_agent(
            "architect",
            self.history,
            f"Présente l'ouverture de la séance d'audit basée sur ce contexte :\n{context}",
        )
        self._notify("architect", architect_intro)
        self.history.append(HumanMessage(content=f"Contexte du projet :\n{context}"))
        self.history.append(AIMessage(content=f"[Architecte] {architect_intro}"))

        skeptic_audit = self.agents.invoke_agent(
            "skeptic",
            self.history,
            "Fais une critique incisive et identifie les failles majeures du projet présenté.",
        )
        self._notify("skeptic", skeptic_audit)
        self.history.append(AIMessage(content=f"[Sceptique] {skeptic_audit}"))

        enthusiast_audit = self.agents.invoke_agent(
            "enthusiast",
            self.history,
            "Réponds aux attaques du Sceptique, défends la vision et propose des ajouts innovants.",
        )
        self._notify("enthusiast", enthusiast_audit)
        self.history.append(AIMessage(content=f"[Enthousiaste] {enthusiast_audit}"))

        # --- PHASE 2 : LE CHOC DES IDÉES ---
        skeptic_rebuttal = self.agents.invoke_agent(
            "skeptic",
            self.history,
            "Attaque spécifiquement les propositions de l'Enthousiaste et pointe du doigt les risques techniques/complexité.",
        )
        self._notify("skeptic", skeptic_rebuttal)
        self.history.append(AIMessage(content=f"[Sceptique] {skeptic_rebuttal}"))

        enthusiast_rebuttal = self.agents.invoke_agent(
            "enthusiast",
            self.history,
            "Propose des solutions aux réserves du Sceptique et montre le chemin le plus court vers la livraison.",
        )
        self._notify("enthusiast", enthusiast_rebuttal)
        self.history.append(AIMessage(content=f"[Enthousiaste] {enthusiast_rebuttal}"))

        # --- PHASE 3 : LA RÉSOLUTION ---
        architect_final = self.agents.invoke_agent(
            "architect",
            self.history,
            "Fais la synthèse du débat et génère le 'Plan d'Implémentation v2.0' complet en Markdown.",
        )
        self._notify("architect", architect_final)
        self.history.append(AIMessage(content=f"[Architecte - Plan Final] {architect_final}"))

        return {
            "transcript_summary": "Simulation terminée avec succès.",
            "final_plan": architect_final,
        }
