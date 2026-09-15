from collections.abc import Callable, Generator
from typing import Any

from langchain_core.messages import AIMessage, BaseMessageChunk, HumanMessage

from table_ronde.agents import TableRondeAgents
from table_ronde.scanner import scan_project

StreamCallback = Callable[[str, Generator[BaseMessageChunk, None, None]], str]


class Orchestrator:
    def __init__(
        self,
        agents: TableRondeAgents,
        on_message_callback: StreamCallback | None = None,
        human_input_callback: Callable[[], str | None] | None = None,
    ):
        """
        :param agents: Instance de TableRondeAgents
        :param on_message_callback: Callback(role, generator) appelé à chaque intervention d'un agent pour streamer la réponse.
        :param human_input_callback: Callback() appelé avant la résolution pour recueillir la note de l'utilisateur.
        """
        self.agents = agents
        self.on_message_callback = on_message_callback
        self.human_input_callback = human_input_callback
        self.history: list[Any] = []
        self.transcript_entries: list[dict[str, str]] = []

    def _stream_and_record(self, role: str, instruction: str, title: str) -> str:
        """Lance stream_agent(), délègue le rendu au callback, enregistre dans le transcript."""
        gen = self.agents.stream_agent(role, self.history, instruction)
        if self.on_message_callback:
            full_text = self.on_message_callback(role, gen)
        else:
            full_text = "".join(
                chunk.content for chunk in gen if hasattr(chunk, "content")
            )
        self.transcript_entries.append({"role": role, "title": title, "content": full_text})
        return full_text

    def run_simulation(
        self, prompt_user: str, project_path: str | None = None
    ) -> dict[str, str]:
        self.history.clear()
        self.transcript_entries.clear()

        # 1. Préparation du contexte initial
        context = f"Sujet / Demande initiale de l'utilisateur :\n{prompt_user}\n"
        if project_path:
            scanned_data = scan_project(project_path)
            context += f"\n\nContextuel du projet existant ({project_path}) :\n{scanned_data}"

        # --- PHASE 1 : L'AUDIT ---
        architect_intro = self._stream_and_record(
            "architect",
            f"Présente l'ouverture de la séance d'audit basée sur ce contexte :\n{context}",
            "Ouverture de l'Architecte",
        )
        self.history.append(HumanMessage(content=f"Contexte du projet :\n{context}"))
        self.history.append(AIMessage(content=f"[Architecte] {architect_intro}"))

        skeptic_audit = self._stream_and_record(
            "skeptic",
            "Fais une critique incisive et identifie les failles majeures du projet présenté.",
            "Audit du Sceptique",
        )
        self.history.append(AIMessage(content=f"[Sceptique] {skeptic_audit}"))

        enthusiast_audit = self._stream_and_record(
            "enthusiast",
            "Réponds aux attaques du Sceptique, défends la vision et propose des ajouts innovants.",
            "Vision de l'Enthousiaste",
        )
        self.history.append(AIMessage(content=f"[Enthousiaste] {enthusiast_audit}"))

        # --- PHASE 2 : LE CHOC DES IDÉES ---
        skeptic_rebuttal = self._stream_and_record(
            "skeptic",
            "Attaque spécifiquement les propositions de l'Enthousiaste et pointe du doigt les risques techniques/complexité.",
            "Réfutation du Sceptique",
        )
        self.history.append(AIMessage(content=f"[Sceptique] {skeptic_rebuttal}"))

        enthusiast_rebuttal = self._stream_and_record(
            "enthusiast",
            "Propose des solutions aux réserves du Sceptique et montre le chemin le plus court vers la livraison.",
            "Contre-propositions de l'Enthousiaste",
        )
        self.history.append(AIMessage(content=f"[Enthousiaste] {enthusiast_rebuttal}"))

        # --- INTERVENTION UTILISATEUR (MODE INTERACTIF) ---
        if self.human_input_callback:
            user_note = self.human_input_callback()
            if user_note:
                self.history.append(
                    HumanMessage(content=f"[Note de l'utilisateur] : {user_note}")
                )

        # --- PHASE 3 : LA RÉSOLUTION ---
        architect_final = self._stream_and_record(
            "architect",
            "Fais la synthèse du débat et génère le 'Plan d'Implémentation v2.0' complet en Markdown.",
            "Plan d'Implémentation Final (Architecte)",
        )
        self.history.append(AIMessage(content=f"[Architecte - Plan Final] {architect_final}"))

        # Construction du transcript complet
        transcript_lines = [
            "# Transcript Complet de la Table-Ronde",
            "",
            "## 🎯 Sujet initial",
            prompt_user,
            "",
            "---",
            "",
            "## 💬 Débat entre les Agents",
            "",
        ]
        for entry in self.transcript_entries:
            role_emoji = {
                "architect": "🏛️",
                "skeptic": "😈",
                "enthusiast": "🚀",
            }.get(entry["role"], "🤖")
            transcript_lines.append(f"### {role_emoji} {entry['title']}")
            transcript_lines.append(entry["content"])
            transcript_lines.append("")

        full_transcript = "\n".join(transcript_lines)

        return {
            "transcript_summary": "Simulation terminée avec succès.",
            "final_plan": architect_final,
            "full_transcript": full_transcript,
        }

