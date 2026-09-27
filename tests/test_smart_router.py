import pytest
from unittest.mock import MagicMock, patch
from table_ronde.smart_router import SmartRouterEngine, TaskRoutingDecision

def test_analyze_task_returns_decision():
    config = {"smart_routing_enabled": True, "router_provider": "gemini"}
    router = SmartRouterEngine(config)
    
    mock_decision = TaskRoutingDecision(
        rounds=3,
        architect_model_tier="pro",
        expert_model_tier="flash",
        temperature=0.4,
        complexity=8,
        domain="Backend"
    )
    
    with patch("table_ronde.smart_router.get_llm") as mock_get_llm:
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm
        
        mock_analyzer = MagicMock()
        mock_analyzer.invoke.return_value = mock_decision
        mock_llm.with_structured_output.return_value = mock_analyzer
        
        decision = router.analyze_task("Create a scalable backend")
        assert decision.rounds == 3
        assert decision.architect_model_tier == "pro"
        assert decision.expert_model_tier == "flash"
        assert decision.temperature == 0.4
        assert decision.complexity == 8
        assert decision.domain == "Backend"
