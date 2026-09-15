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
        :param agents: Instance of TableRondeAgents
        :param on_message_callback: Callback(role, generator) called at each intervention.
        :param human_input_callback: Callback() called before resolution to gather user input.
        """
        self.agents = agents
        self.on_message_callback = on_message_callback
        self.human_input_callback = human_input_callback
        self.history: list[Any] = []
        self.transcript_entries: list[dict[str, str]] = []

    def _stream_and_record(self, role: str, instruction: str, title: str) -> str:
        """Launches stream_agent(), delegates rendering to callback, and records in transcript."""
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

        # 1. Initial context setup
        context = f"Topic / Initial user request:\n{prompt_user}\n"
        if project_path:
            scanned_data = scan_project(project_path)
            context += f"\n\nExisting project context ({project_path}):\n{scanned_data}"

        architect_role = self.agents.architect_cfg["role"]
        architect_title = self.agents.architect_cfg["title"]
        
        # --- PHASE 1 : OPENING ---
        architect_intro = self._stream_and_record(
            architect_role,
            f"Present the opening of the audit session based on this context:\n{context}",
            f"Opening by {architect_title}",
        )
        self.history.append(HumanMessage(content=f"Project Context:\n{context}"))
        self.history.append(AIMessage(content=f"[{architect_title}] {architect_intro}"))

        # --- PHASE 2 : DEBATE ROUNDS ---
        num_rounds = self.agents.config.get("orchestrator", {}).get("rounds", 1)
        
        current_round = 1
        while current_round <= num_rounds:
            for p in self.agents.personas:
                role = p["role"]
                title = p["title"]
                
                # Generic instruction for dynamic debate
                instruction = (
                    "It is your turn to speak in this debate. "
                    "Express your arguments based on your role and bounce back on what was just said by the others."
                )
                
                resp = self._stream_and_record(
                    role,
                    instruction,
                    f"Intervention by {title} (Round {current_round})",
                )
                self.history.append(AIMessage(content=f"[{title}] {resp}"))

            # --- USER INTERVENTION (INTERACTIVE MODE) ---
            if self.human_input_callback:
                user_note = self.human_input_callback()
                if user_note:
                    if user_note.strip().lower() == "/round":
                        num_rounds += 1
                    else:
                        self.history.append(
                            HumanMessage(content=f"[User Note] : {user_note}")
                        )
            
            current_round += 1

        # --- PHASE 3 : RESOLUTION ---
        architect_final = self._stream_and_record(
            architect_role,
            "Synthesize the debate by integrating any potential user notes and generate the complete 'Implementation Plan v2.0' in Markdown.",
            f"Final Implementation Plan ({architect_title})",
        )
        self.history.append(AIMessage(content=f"[{architect_title} - Final Plan] {architect_final}"))

        # Build the full transcript
        transcript_lines = [
            "# Complete Roundtable Transcript",
            "",
            "## 🎯 Initial Topic",
            prompt_user,
            "",
            "---",
            "",
            "## 💬 Agent Debate",
            "",
        ]
        
        # Find emojis
        role_emojis = {p["role"]: p.get("emoji", "🤖") for p in self.agents.personas}
        role_emojis[architect_role] = self.agents.architect_cfg.get("emoji", "🏛️")
        
        for entry in self.transcript_entries:
            role_emoji = role_emojis.get(entry["role"], "🤖")
            transcript_lines.append(f"### {role_emoji} {entry['title']}")
            transcript_lines.append(entry["content"])
            transcript_lines.append("")

        full_transcript = "\n".join(transcript_lines)

        return {
            "transcript_summary": "Simulation finished successfully.",
            "final_plan": architect_final,
            "full_transcript": full_transcript,
        }
