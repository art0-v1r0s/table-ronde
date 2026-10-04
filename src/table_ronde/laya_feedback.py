"""
laya_feedback.py — Collecte les décisions de routing et leur outcome réel.
Ces données alimentent le pipeline de fine-tuning Laya.
Stockage : ~/.table_ronde/feedback_store.jsonl (append-only, thread-safe)
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

DATA_DIR_ENV = os.getenv("TABLE_RONDE_DATA_DIR")
if DATA_DIR_ENV:
    FEEDBACK_PATH = Path(DATA_DIR_ENV) / "feedback_store.jsonl"
else:
    FEEDBACK_PATH = Path.home() / ".table_ronde" / "feedback_store.jsonl"


@dataclass
class RoutingFeedback:
    """Enregistrement d'une décision de routing + résultat observé après le débat."""
    timestamp: str
    prompt: str                  # Prompt tronqué à 500 chars
    predicted_domain: str        # Domaine prédit par le router
    predicted_rounds: int        # Rounds recommandés
    predicted_complexity: int    # Complexité prédite (1-10)
    routing_method: str          # "laya" | "heuristic" | "llm"
    confidence_score: float      # Score de confiance (0.0-1.0)
    # Champs remplis après le débat par l'Orchestrator
    actual_rounds_used: int = 0
    consensus_reached: bool = False


class FeedbackStore:
    """Stockage append-only thread-safe des feedbacks de routing."""

    def record(self, feedback: RoutingFeedback) -> None:
        """Écrit un feedback en mode append. Crée le fichier si nécessaire."""
        import time
        try:
            FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)
            for _ in range(3):
                try:
                    with FEEDBACK_PATH.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(asdict(feedback)) + "\n")
                    break
                except PermissionError:
                    time.sleep(0.1)
        except Exception as e:
            logger.warning("Could not save routing feedback: %s", e)

    def stats(self) -> dict:
        """Retourne des statistiques sur les feedbacks accumulés."""
        if not FEEDBACK_PATH.exists():
            return {"total": 0, "domains": {}}
        total = 0
        domains: dict[str, int] = {}
        methods: dict[str, int] = {}
        try:
            with FEEDBACK_PATH.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        total += 1
                        d = data.get("predicted_domain", "unknown")
                        domains[d] = domains.get(d, 0) + 1
                        m = data.get("routing_method", "unknown")
                        methods[m] = methods.get(m, 0) + 1
                    except json.JSONDecodeError:
                        continue
        except OSError as e:
            logger.warning("Could not read feedback store: %s", e)
        return {"total": total, "domains": domains, "methods": methods}
