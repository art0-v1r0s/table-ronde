"""
Tests pour LayaSmartRouter et SmartRouterV2 — mocks complets de laya.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

# ── Tests LayaSmartRouter ─────────────────────────────────────────────────────

class TestLayaSmartRouterUnavailable:
    """Vérifie le comportement gracieux quand laya n'est pas installé."""

    def _make_unavailable_router(self):
        from table_ronde.laya_router import LayaSmartRouter
        r = LayaSmartRouter.__new__(LayaSmartRouter)
        r._router = None
        r.available = False
        r.effective_device = None
        return r

    def test_available_false_when_no_laya(self):
        router = self._make_unavailable_router()
        assert router.available is False

    def test_analyze_task_returns_none(self):
        router = self._make_unavailable_router()
        assert router.analyze_task("some prompt") is None

    def test_evaluate_consensus_returns_none(self):
        router = self._make_unavailable_router()
        assert router.evaluate_consensus("text") is None

    def test_evaluate_user_note_returns_none(self):
        router = self._make_unavailable_router()
        assert router.evaluate_user_note("note") is None


class TestLayaSmartRouterWithMock:
    """Vérifie le comportement quand Laya est disponible (mock)."""

    def _make_available_router(self, predict_return: dict):
        from table_ronde.laya_router import LayaSmartRouter
        r = LayaSmartRouter.__new__(LayaSmartRouter)
        mock_inner = MagicMock()
        mock_inner.predict.return_value = predict_return
        r._router = mock_inner
        r.available = True
        r.effective_device = "cuda"
        return r

    def test_analyze_task_returns_dict(self):
        expected = {"domain": {"security": 0.9}, "complexity": {"expected_score": 4.0},
                    "needs_multi_round_debate": {"p": 0.85}, "needs_frontier_model": {"p": 0.8}}
        router = self._make_available_router(expected)
        result = router.analyze_task("prevent SQL injection")
        assert result is not None
        assert "domain" in result

    def test_evaluate_consensus_returns_probability(self):
        router = self._make_available_router({"consensus_reached": {"p": 0.88}})
        p = router.evaluate_consensus("agents agree on solution")
        assert p == pytest.approx(0.88)

    def test_evaluate_user_note_returns_probability(self):
        router = self._make_available_router({"requires_new_round": {"p": 0.3}})
        p = router.evaluate_user_note("looks good, let's finalize")
        assert p == pytest.approx(0.3)

    def test_analyze_task_returns_none_on_exception(self):
        from table_ronde.laya_router import LayaSmartRouter
        r = LayaSmartRouter.__new__(LayaSmartRouter)
        mock_inner = MagicMock()
        mock_inner.predict.side_effect = RuntimeError("GPU OOM")
        r._router = mock_inner
        r.available = True
        r.effective_device = "cuda"
        result = r.analyze_task("test")
        assert result is None

    def test_auto_device_detection_cuda(self):
        from table_ronde.laya_router import LayaSmartRouter
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_laya = MagicMock()
        with patch.dict("sys.modules", {"torch": mock_torch, "laya": mock_laya}):
            router = LayaSmartRouter(device="auto")
            assert router.effective_device == "cuda"
            assert router.available is True

    def test_auto_device_detection_mps(self):
        from table_ronde.laya_router import LayaSmartRouter
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = True
        mock_laya = MagicMock()
        with patch.dict("sys.modules", {"torch": mock_torch, "laya": mock_laya}):
            router = LayaSmartRouter(device="auto")
            assert router.effective_device == "mps"
            assert router.available is True

    def test_auto_device_detection_cpu(self):
        from table_ronde.laya_router import LayaSmartRouter
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False
        mock_laya = MagicMock()
        with patch.dict("sys.modules", {"torch": mock_torch, "laya": mock_laya}):
            router = LayaSmartRouter(device="auto")
            assert router.effective_device == "cpu"
            assert router.available is True



# ── Tests SmartRouterV2 (heuristic strategy) ─────────────────────────────────

class TestSmartRouterV2Heuristic:
    """Tests la stratégie heuristic (aucune dépendance externe)."""

    def _make_router(self, strategy: str = "heuristic"):
        from table_ronde.smart_router import SmartRouterV2
        config = {
            "smart_routing_enabled": True,
            "smart_router": {"strategy": strategy},
        }
        with patch("table_ronde.smart_router._get_raw_llm"):
            return SmartRouterV2(config)

    def test_security_domain_detected(self):
        router = self._make_router()
        decision = router.analyze_task("How to prevent SQL injection in my API?")
        assert decision is not None
        assert decision.route_name == "security"
        assert decision.routing_method == "heuristic"

    def test_devops_domain_detected(self):
        router = self._make_router()
        decision = router.analyze_task("Setup CI/CD pipeline with Docker and Kubernetes")
        assert decision is not None
        assert decision.route_name == "devops"

    def test_simple_task_one_round(self):
        router = self._make_router()
        decision = router.analyze_task("rename this variable")
        assert decision is not None
        assert decision.rounds == 1
        assert decision.route_name == "simple"

    def test_complex_architecture_three_rounds(self):
        router = self._make_router()
        decision = router.analyze_task(
            "Design a distributed microservices architecture with Kafka, "
            "including security controls and zero-trust network."
        )
        assert decision is not None
        assert decision.rounds == 3

    def test_routing_method_is_heuristic(self):
        router = self._make_router()
        decision = router.analyze_task("any prompt")
        assert decision is not None
        assert decision.routing_method == "heuristic"

    def test_current_feedback_created_after_analyze(self):
        router = self._make_router()
        router.analyze_task("design a system")
        assert router.current_feedback is not None
        assert router.current_feedback.routing_method == "heuristic"

    def test_invalid_strategy_defaults_to_auto(self):
        from table_ronde.smart_router import SmartRouterV2
        config = {
            "smart_routing_enabled": True,
            "smart_router": {"strategy": "invalid_value"},
        }
        with patch("table_ronde.smart_router._get_raw_llm"):
            router = SmartRouterV2(config)
        assert router.strategy == "auto"


# ── Tests heuristiques autonomes ──────────────────────────────────────────────

class TestHeuristicFunctions:
    def test_complexity_simple(self):
        from table_ronde.smart_router import _score_complexity_heuristic
        assert _score_complexity_heuristic("rename this variable") < 0.3

    def test_complexity_complex(self):
        from table_ronde.smart_router import _score_complexity_heuristic
        score = _score_complexity_heuristic(
            "Design a zero-trust microservices architecture with Kafka "
            "for event-driven distributed security systems."
        )
        assert score > 0.5

    def test_domain_security(self):
        from table_ronde.smart_router import _detect_domain_heuristic
        assert _detect_domain_heuristic("prevent SQL injection and XSS attacks") == "security"

    def test_domain_devops(self):
        from table_ronde.smart_router import _detect_domain_heuristic
        assert _detect_domain_heuristic("setup CI/CD with Docker and kubernetes") == "devops"

    def test_domain_data(self):
        from table_ronde.smart_router import _detect_domain_heuristic
        assert _detect_domain_heuristic("optimize slow SQL queries on postgres") == "data"

    def test_domain_simple(self):
        from table_ronde.smart_router import _detect_domain_heuristic
        assert _detect_domain_heuristic("rename this variable to user_count") == "simple"

    def test_consensus_with_agreement_words(self):
        from table_ronde.smart_router import _evaluate_consensus_heuristic
        msgs = [MagicMock(content="I agree with this approach."),
                MagicMock(content="Good point, solid approach overall.")]
        reached, confidence = _evaluate_consensus_heuristic(msgs)
        assert reached is True
        assert confidence > 0.3

    def test_no_consensus_with_disagreement(self):
        from table_ronde.smart_router import _evaluate_consensus_heuristic
        msgs = [MagicMock(content="However, I see a risk here."),
                MagicMock(content="This is a problem, concern remains.")]
        reached, _ = _evaluate_consensus_heuristic(msgs)
        assert reached is False


# ── Tests FeedbackStore ───────────────────────────────────────────────────────

class TestFeedbackStore:
    def test_record_creates_file(self, tmp_path, monkeypatch):
        from table_ronde.laya_feedback import FeedbackStore, RoutingFeedback
        fake_path = tmp_path / "feedback.jsonl"
        monkeypatch.setattr("table_ronde.laya_feedback.FEEDBACK_PATH", fake_path)
        store = FeedbackStore()
        fb = RoutingFeedback(
            timestamp="2026-10-01T00:00:00Z", prompt="test", predicted_domain="security",
            predicted_rounds=3, predicted_complexity=8, routing_method="laya",
            confidence_score=0.9,
        )
        store.record(fb)
        assert fake_path.exists()
        import json
        data = json.loads(fake_path.read_text().strip())
        assert data["predicted_domain"] == "security"

    def test_stats_empty(self, tmp_path, monkeypatch):
        fake_path = tmp_path / "nonexistent.jsonl"
        monkeypatch.setattr("table_ronde.laya_feedback.FEEDBACK_PATH", fake_path)
        from table_ronde.laya_feedback import FeedbackStore
        stats = FeedbackStore().stats()
        assert stats["total"] == 0
