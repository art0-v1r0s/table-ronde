import json
from collections.abc import Callable, Generator
from typing import Any

from langchain_core.messages import (
    AIMessage,
    BaseMessageChunk,
    HumanMessage,
    ToolMessage,
    messages_from_dict,
    messages_to_dict,
)

from table_ronde.agents import TableRondeAgents
from table_ronde.scanner import scan_project
from table_ronde.tools import AVAILABLE_TOOLS

StreamCallback = Callable[[str, Generator[BaseMessageChunk, None, None]], str]


class Orchestrator:
    def __init__(
        self,
        agents: TableRondeAgents,
        on_message_callback: StreamCallback | None = None,
        human_input_callback: Callable[[], str | None] | None = None,
        on_phase_callback: Callable[[str, dict[str, Any]], None] | None = None,
    ):
        """
        :param agents: Instance of TableRondeAgents
        :param on_message_callback: Callback(role, generator) called at each intervention.
        :param human_input_callback: Callback() called before resolution to gather user input.
        :param on_phase_callback: Callback(phase_name, data_dict) called during phase transitions.
        """
        self.agents = agents
        self.on_message_callback = on_message_callback
        self.human_input_callback = human_input_callback
        self.on_phase_callback = on_phase_callback
        self.history: list[Any] = []
        self.transcript_entries: list[dict[str, str]] = []
        self.tools_by_name = {t.name: t for t in AVAILABLE_TOOLS}

    def save_session(self, filepath: str) -> None:
        """Serializes the history to a JSON file."""
        data = {
            "history": messages_to_dict(self.history),
            "transcript_entries": self.transcript_entries,
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load_session(self, filepath: str) -> None:
        """Deserializes the history from a JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.history = messages_from_dict(data.get("history", []))
        self.transcript_entries = data.get("transcript_entries", [])

    def _stream_and_record(self, role: str, instruction: str | None, title: str) -> str:
        """Launches stream_agent(), handles tool calls, delegates rendering to callback, and records in transcript."""
        if instruction:
            self.history.append(HumanMessage(content=instruction))

        while True:
            # We stream without passing new_instruction since it's already in history
            gen = self.agents.stream_agent(role, self.history)
            
            tool_call_chunks: list[Any] = []
            final_content = ""
            
            def chunk_interceptor(generator, tc_chunks) -> Generator[BaseMessageChunk, None, None]:
                nonlocal final_content
                for chunk in generator:
                    if chunk.tool_call_chunks:
                        tc_chunks.extend(chunk.tool_call_chunks)
                    if hasattr(chunk, "content") and chunk.content:
                        if isinstance(chunk.content, str):
                            final_content += chunk.content
                        elif isinstance(chunk.content, list):
                            for part in chunk.content:
                                if isinstance(part, str):
                                    final_content += part
                                elif isinstance(part, dict) and "text" in part:
                                    final_content += part["text"]
                    yield chunk

            intercepted_gen = chunk_interceptor(gen, tool_call_chunks)
            
            if self.on_message_callback:
                self.on_message_callback(role, intercepted_gen)
            else:
                for _ in intercepted_gen:
                    pass
                    
            if not tool_call_chunks:
                self.transcript_entries.append({"role": role, "title": title, "content": final_content})
                self.history.append(AIMessage(content=final_content))
                return final_content
                
            # If we get here, the model wanted to call tools
            import json as json_lib
            tool_calls = []
            
            # Naive merging of tool call chunks
            calls_by_index = {}
            for chunk in tool_call_chunks:
                idx = chunk.get("index")
                if idx not in calls_by_index:
                    calls_by_index[idx] = {"name": "", "args": "", "id": chunk.get("id")}
                if chunk.get("name"):
                    calls_by_index[idx]["name"] += chunk.get("name")
                if chunk.get("args"):
                    calls_by_index[idx]["args"] += chunk.get("args")
                    
            # Add AIMessage with tool calls to history
            ai_message = AIMessage(content="", tool_calls=[])
            for idx, call_data in calls_by_index.items():
                try:
                    args_dict = json_lib.loads(call_data["args"])
                except Exception:
                    args_dict = {}
                tool_call_dict = {
                    "name": call_data["name"],
                    "args": args_dict,
                    "id": call_data["id"] or f"call_{idx}"
                }
                # Use ToolCall cast or just dict append. Actually `ToolCall` is a TypedDict.
                ai_message.tool_calls.append(tool_call_dict) # type: ignore
                tool_calls.append(tool_call_dict)
                
            self.history.append(ai_message)
            
            # Execute tools
            for tc in tool_calls:
                tool_name = tc["name"]
                tool_args = tc["args"]
                if self.on_phase_callback:
                    self.on_phase_callback("tool_call", {"name": tool_name, "args": str(tool_args)[:80]})
                if tool_name in self.tools_by_name:
                    tool_instance = self.tools_by_name[tool_name]
                    try:
                        result = tool_instance.invoke(tool_args)
                    except Exception as e:
                        result = f"Error executing tool: {e}"
                else:
                    result = f"Tool {tool_name} not found."
                    
                self.history.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
                
            # The loop will continue and stream again using the new history with tool results
            if self.on_message_callback:
                # Need a little separator if we re-stream in the same UI?
                # Actually, the loop just calls stream_agent again, triggering a new UI block.
                # To keep it clean, we might just loop.
                pass

    def run_simulation(
        self, prompt_user: str, project_path: str | None = None, is_resume: bool = False, save_path: str | None = None
    ) -> dict[str, str]:
        if not is_resume:
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
            if self.on_phase_callback:
                self.on_phase_callback("opening", {"architect": architect_title})
            self._stream_and_record(
                architect_role,
                f"Present the opening of the audit session based on this context:\n{context}",
                f"Opening by {architect_title}",
            )
        else:
            # We already have history, but we need variables
            architect_role = self.agents.architect_cfg["role"]
            architect_title = self.agents.architect_cfg["title"]
            
        # --- PHASE 2 : DEBATE ROUNDS ---
        orch_config = self.agents.config.get("orchestrator", {}) if isinstance(self.agents.config, dict) else {}
        num_rounds = orch_config.get("rounds", 1)
        
        current_round = 1
        while current_round <= num_rounds:
            if self.on_phase_callback:
                self.on_phase_callback("round", {"current": current_round, "total": num_rounds})
            for p in self.agents.personas:
                role = p["role"]
                title = p["title"]
                
                # Generic instruction for dynamic debate
                instruction = (
                    "It is your turn to speak in this debate. "
                    "Express your arguments based on your role and bounce back on what was just said by the others. "
                    "Use tools if you need to verify claims or search code/web."
                )
                
                self._stream_and_record(
                    role,
                    instruction,
                    f"Intervention by {title} (Round {current_round})",
                )

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
            
            if save_path:
                self.save_session(save_path)
            
            current_round += 1

        # --- PHASE 3 : RESOLUTION ---
        if self.on_phase_callback:
            self.on_phase_callback("resolution", {"architect": architect_title})
        architect_final = self._stream_and_record(
            architect_role,
            "Synthesize the debate by integrating any potential user notes and generate the complete 'Implementation Plan v2.0' in Markdown.",
            f"Final Implementation Plan ({architect_title})",
        )
        
        if save_path:
            self.save_session(save_path)

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
