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
        :param on_message_callback: Callback(role, generator) appelé à chaque intervention.
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

        architect_role = self.agents.architect_cfg["role"]
        architect_title = self.agents.architect_cfg["title"]
        
        # --- PHASE 1 : OUVERTURE ---
        architect_intro = self._stream_and_record(
            architect_role,
            f"Présente l'ouverture de la séance d'audit basée sur ce contexte :\n{context}",
            f"Ouverture de {architect_title}",
        )
        self.history.append(HumanMessage(content=f"Contexte du projet :\n{context}"))
        self.history.append(AIMessage(content=f"[{architect_title}] {architect_intro}"))

        # --- PHASE 2 : LES TOURS DE DÉBAT ---
        num_rounds = self.agents.config.get("orchestrator", {}).get("rounds", 1)
        
        current_round = 1
        while current_round <= num_rounds:
            for p in self.agents.personas:
                role = p["role"]
                title = p["title"]
                
                # Instruction générique pour les débats dynamiques
                instruction = (
                    "À ton tour de prendre la parole dans ce débat. "
                    "Exprime tes arguments en fonction de ton rôle et rebondis sur ce qui vient d'être dit par les autres."
                )
                
                resp = self._stream_and_record(
                    role,
                    instruction,
                    f"Intervention de {title} (Tour {current_round})",
                )
                self.history.append(AIMessage(content=f"[{title}] {resp}"))

            # --- INTERVENTION UTILISATEUR (MODE INTERACTIF) ---
            if self.human_input_callback:
                user_note = self.human_input_callback()
                if user_note:
                    if user_note.strip().lower() == "/tour":
                        num_rounds += 1
                    else:
                        self.history.append(
                            HumanMessage(content=f"[Note de l'utilisateur] : {user_note}")
                        )
            
            current_round += 1

        # --- PHASE 3 : LA RÉSOLUTION ---
        architect_final = self._stream_and_record(
            architect_role,
            "Fais la synthèse du débat en intégrant les notes éventuelles de l'utilisateur et génère le 'Plan d'Implémentation v2.0' complet en Markdown.",
            f"Plan d'Implémentation Final ({architect_title})",
        )
        self.history.append(AIMessage(content=f"[{architect_title} - Plan Final] {architect_final}"))

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
        
        # Trouver les emojis
        role_emojis = {p["role"]: p.get("emoji", "🤖") for p in self.agents.personas}
        role_emojis[architect_role] = self.agents.architect_cfg.get("emoji", "🏛️")
        
        for entry in self.transcript_entries:
            role_emoji = role_emojis.get(entry["role"], "🤖")
            transcript_lines.append(f"### {role_emoji} {entry['title']}")
            transcript_lines.append(entry["content"])
            transcript_lines.append("")

        full_transcript = "\n".join(transcript_lines)

        return {
            "transcript_summary": "Simulation terminée avec succès.",
            "final_plan": architect_final,
            "full_transcript": full_transcript,
        }
