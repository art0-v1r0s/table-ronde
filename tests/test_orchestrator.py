from unittest.mock import MagicMock
from table_ronde.orchestrator import Orchestrator


def test_orchestrator_flow():
    mock_agents = MagicMock()
    mock_agents.invoke_agent.side_effect = lambda role, history, instr: f"Réponse fictive de {role}"

    messages_received = []

    def callback(role, msg):
        messages_received.append((role, msg))

    orchestrator = Orchestrator(mock_agents, on_message_callback=callback)
    res = orchestrator.run_simulation("Tester l'orchestrateur")

    assert "final_plan" in res
    assert res["final_plan"] == "Réponse fictive de architect"
    assert len(messages_received) == 6
    roles = [m[0] for m in messages_received]
    assert roles == ["architect", "skeptic", "enthusiast", "skeptic", "enthusiast", "architect"]
