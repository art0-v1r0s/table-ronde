from unittest.mock import MagicMock

from table_ronde.orchestrator import Orchestrator


def test_orchestrator_flow():
    mock_agents = MagicMock()
    mock_agents.stream_agent.side_effect = lambda role, history, instr: iter(
        [MagicMock(content=f"Réponse fictive de {role}")]
    )

    messages_received = []

    def stream_callback(role, gen):
        text = "".join(c.content for c in gen if hasattr(c, "content"))
        messages_received.append((role, text))
        return text

    orchestrator = Orchestrator(mock_agents, on_message_callback=stream_callback)
    res = orchestrator.run_simulation("Tester l'orchestrateur")

    assert "final_plan" in res
    assert res["final_plan"] == "Réponse fictive de architect"
    assert "full_transcript" in res
    assert "# Transcript Complet" in res["full_transcript"]
    assert len(messages_received) == 6
    roles = [m[0] for m in messages_received]
    assert roles == ["architect", "skeptic", "enthusiast", "skeptic", "enthusiast", "architect"]


def test_human_input_injected_in_history():
    mock_agents = MagicMock()
    mock_agents.stream_agent.side_effect = lambda r, h, i: iter([MagicMock(content="ok")])

    orchestrator = Orchestrator(
        mock_agents,
        on_message_callback=lambda role, gen: "".join(c.content for c in gen if hasattr(c, "content")),
        human_input_callback=lambda: "Focus sur la sécurité",
    )
    orchestrator.run_simulation("Test interactif")

    human_msgs = [
        m for m in orchestrator.history
        if hasattr(m, "content") and "Focus sur la sécurité" in m.content
    ]
    assert len(human_msgs) == 1

