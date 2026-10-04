from unittest.mock import MagicMock, patch

from table_ronde.smart_router import (
    ConsensusDecision,
    RoutingDecision,
    SmartRouterEngine,
    TaskRoutingDecision,
)


def test_analyze_task_returns_decision():
    config = {"smart_routing_enabled": True, "router_provider": "gemini"}
    
    mock_decision = TaskRoutingDecision(
        rounds=3,
        architect_model_tier="pro",
        expert_model_tier="flash",
        temperature=0.4,
        complexity=8,
        domain="Backend"
    )
    
    with patch("table_ronde.smart_router._get_raw_llm") as mock_get_llm:
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm
        
        mock_analyzer = MagicMock()
        mock_analyzer.invoke.return_value = mock_decision
        mock_llm.with_structured_output.return_value = mock_analyzer
        
        router = SmartRouterEngine(config)
        decision = router.analyze_task("Create a scalable backend")
        
        assert decision.rounds == 3
        assert decision.architect_model_tier == "pro"
        assert decision.expert_model_tier == "flash"
        assert decision.temperature == 0.4
        assert decision.complexity == 8
        assert decision.domain == "Backend"
        
        # Test that analyze_task reuses the analyzer
        assert mock_llm.with_structured_output.call_count == 3
        
def test_evaluate_consensus_empty_history():
    config = {"smart_routing_enabled": True, "router_provider": "gemini"}
    with patch("table_ronde.smart_router._get_raw_llm"):
        router = SmartRouterEngine(config)
        assert router.evaluate_consensus([]) is False
        assert router.evaluate_consensus([MagicMock()]) is False
        
def test_evaluate_consensus_reached():
    config = {"smart_routing_enabled": True, "router_provider": "gemini"}
    
    mock_decision = ConsensusDecision(
        consensus_reached=True,
        confidence=0.9
    )
    
    with patch("table_ronde.smart_router._get_raw_llm") as mock_get_llm:
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm
        mock_evaluator = MagicMock()
        mock_evaluator.invoke.return_value = mock_decision
        
        # Mocking the 3 structured output calls
        mock_llm.with_structured_output.side_effect = [MagicMock(), mock_evaluator, MagicMock()]
        
        router = SmartRouterEngine(config)
        
        msg1 = MagicMock()
        msg1.content = "I agree."
        msg2 = MagicMock()
        msg2.content = "Me too."
        
        assert router.evaluate_consensus([msg1, msg2]) is True

def test_evaluate_user_note():
    config = {"smart_routing_enabled": True, "router_provider": "gemini"}
    
    mock_decision = RoutingDecision(
        action="SYNTHESIS"
    )
    
    with patch("table_ronde.smart_router._get_raw_llm") as mock_get_llm:
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm
        mock_evaluator = MagicMock()
        mock_evaluator.invoke.return_value = mock_decision
        
        # Mocking the 3 structured output calls (Task, Consensus, Routing)
        mock_llm.with_structured_output.side_effect = [MagicMock(), MagicMock(), mock_evaluator]
        
        router = SmartRouterEngine(config)
        assert router.evaluate_user_note("looks good") == "SYNTHESIS"

def test_initialization_failure():
    config = {"smart_routing_enabled": True, "router_provider": "gemini"}
    with patch("table_ronde.smart_router._get_raw_llm", side_effect=Exception("API Error")):
        router = SmartRouterEngine(config)
        assert router.enabled is False
        assert router.analyze_task("test") is None
