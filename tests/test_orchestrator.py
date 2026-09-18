from unittest.mock import MagicMock

from table_ronde.orchestrator import Orchestrator


def test_orchestrator_flow():
    mock_agents = MagicMock()
    mock_agents.config = {"orchestrator": {"rounds": 1}}
    mock_agents.architect_cfg = {"role": "architect", "title": "Architect", "emoji": "🏛️"}
    mock_agents.personas = [{"role": "skeptic", "title": "Skeptic", "emoji": "🤔"}]
    mock_agents.stream_agent.side_effect = lambda role, history, instr=None: iter(
        [MagicMock(content=f"Réponse fictive de {role}", tool_call_chunks=[])]
    )

    messages_received = []

    def stream_callback(role, gen):
        text = "".join(c.content for c in gen if hasattr(c, "content"))
        messages_received.append((role, text))
        return text

    orchestrator = Orchestrator(mock_agents, on_message_callback=stream_callback)
    res = orchestrator.run_simulation("Tester l'orchestrateur")
    
    assert "Simulation finished" in res["transcript_summary"]
    assert "Réponse fictive de architect" in res["final_plan"]


def test_human_input_injected_in_history():
    mock_agents = MagicMock()
    mock_agents.config = {"orchestrator": {"rounds": 1}}
    mock_agents.architect_cfg = {"role": "architect", "title": "Architect", "emoji": "🏛️"}
    mock_agents.personas = [{"role": "skeptic", "title": "Skeptic", "emoji": "🤔"}]
    mock_agents.stream_agent.side_effect = lambda r, h, i=None: iter([MagicMock(content="ok", tool_call_chunks=[])])

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


def test_orchestrator_phase_callbacks():
    mock_agents = MagicMock()
    mock_agents.config = {"orchestrator": {"rounds": 2}}
    mock_agents.architect_cfg = {"role": "architect", "title": "Architect", "emoji": "🏛️"}
    mock_agents.personas = [{"role": "skeptic", "title": "Skeptic", "emoji": "🤔"}]
    mock_agents.stream_agent.side_effect = lambda role, history, instr=None: iter(
        [MagicMock(content="Mock answer", tool_call_chunks=[])]
    )

    phases_called = []

    def phase_cb(phase, data):
        phases_called.append((phase, data))

    orchestrator = Orchestrator(mock_agents, on_phase_callback=phase_cb)
    orchestrator.run_simulation("Test phases")

    phase_names = [p[0] for p in phases_called]
    assert "opening" in phase_names
    assert phase_names.count("round") == 2
    assert "resolution" in phase_names

