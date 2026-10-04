"""
laya_router.py — Interface avec le modèle Laya (convaiinnovations/laya).
Laya est un moteur de décision non-autoregressif (ModernBERT-large, 421M params).
Inference en ~33ms GPU. Aucune génération de texte = zéro hallucinations.
Licence Apache 2.0. Installation : uv add laya
Réf : https://huggingface.co/convaiinnovations/laya
"""
from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

# ── Schéma de questions pour l'analyse de tâche ──────────────────────────────
TASK_ANALYSIS_QUESTIONS: dict[str, Any] = {
    "domain": {
        "type": "choice",
        "instructions": "What is the primary technical domain of this task?",
        "criteria": {
            "architecture": "system design, microservices, distributed systems, scalability, event-driven patterns",
            "security": "authentication, vulnerabilities, SQL injection, XSS, cryptography, OAuth, zero-trust",
            "devops": "CI/CD, Docker, Kubernetes, deployment, monitoring, SLO, infrastructure as code",
            "frontend": "UI, React, CSS, user interface, components, accessibility",
            "data": "databases, ETL, data pipelines, SQL optimization, vector search, analytics",
            "simple": "typos, variable renaming, minor fixes, quick questions",
        },
    },
    "complexity": {
        "type": "score",
        "instructions": "How complex is this task? Consider required expertise depth.",
        "criteria": {
            "1": "trivial — one-line fix or well-known answer",
            "2": "simple — standard solution, minimal design",
            "3": "moderate — some design decisions needed",
            "4": "complex — significant trade-offs, expert knowledge",
            "5": "very complex — deep architectural expertise required",
        },
    },
    "needs_multi_round_debate": {
        "type": "noul",
        "instructions": (
            "This task requires multiple rounds of debate between expert agents "
            "because there are significant architectural trade-offs or conflicting "
            "best practices that benefit from adversarial review."
        ),
    },
    "needs_frontier_model": {
        "type": "noul",
        "instructions": (
            "This task requires a frontier-level AI model (GPT-4o, Gemini Pro, Claude Sonnet) "
            "due to deep reasoning requirements. A smaller flash model would produce insufficient results."
        ),
    },
}

# ── Schéma de questions pour la détection de consensus ───────────────────────
CONSENSUS_QUESTIONS: dict[str, Any] = {
    "consensus_reached": {
        "type": "noul",
        "instructions": (
            "In this technical debate transcript, the participants have explicitly converged "
            "on a mutually agreed solution. There are no remaining open objections or competing proposals."
        ),
    }
}

# ── Schéma de questions pour l'évaluation des notes utilisateur ──────────────
USER_NOTE_QUESTIONS: dict[str, Any] = {
    "requires_new_round": {
        "type": "noul",
        "instructions": (
            "This user note introduces a significant new requirement or fundamentally "
            "redirects the technical debate, requiring the agents to debate again."
        ),
    }
}


class LayaSmartRouter:
    """
    Interface avec le package `laya` pour toutes les décisions de routing.
    Gestion gracieuse de l'absence du package (fallback automatique).
    """

    def __init__(
        self,
        checkpoint: str = "convaiinnovations/laya",
        device: str = "auto",
    ):
        self.checkpoint = os.path.expanduser(checkpoint)
        self.device = device
        self._router: Any = None
        self.available: bool = False
        self.effective_device: str | None = None
        self._load()

    def _load(self) -> None:
        """Charge Laya. En cas d'échec, `available` reste False (fallback gracieux)."""
        try:
            import torch
            from laya import Router  # type: ignore[import-not-found]

            if self.device == "auto":
                if torch.cuda.is_available():
                    self.effective_device = "cuda"
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    self.effective_device = "mps"
                else:
                    self.effective_device = "cpu"
            else:
                self.effective_device = self.device

            self._router = Router(
                checkpoint=self.checkpoint,
                device=self.effective_device,
            )
            self.available = True
            logger.info(
                "Laya loaded: checkpoint=%s device=%s",
                self.checkpoint,
                self.effective_device,
            )
        except ImportError:
            logger.warning(
                "laya package not installed. Install with: uv add laya\n"
                "Falling back to heuristic/LLM routing."
            )
        except Exception as e:
            logger.warning("Laya init failed: %s. Falling back to next strategy.", e)

    def analyze_task(self, prompt: str) -> dict[str, Any] | None:
        """
        Analyse le prompt. Retourne le dict Laya brut ou None si échec.
        Le prompt est tronqué à 1024 chars pour respecter la fenêtre contextuelle.
        """
        if not self.available or self._router is None:
            return None
        try:
            state = {"task_description": prompt[:1024]}
            return self._router.predict(state, TASK_ANALYSIS_QUESTIONS)
        except Exception as e:
            logger.warning("Laya analyze_task failed: %s", e)
            return None

    def evaluate_consensus(self, debate_text: str) -> float | None:
        """
        Évalue si le consensus est atteint.
        Retourne P(consensus) ∈ [0.0, 1.0] ou None si Laya indisponible.
        Le texte est tronqué aux 3000 derniers chars.
        """
        if not self.available or self._router is None:
            return None
        try:
            state = {"debate_transcript": debate_text[-3000:]}
            result = self._router.predict(state, CONSENSUS_QUESTIONS)
            return result.get("consensus_reached", {}).get("p")
        except Exception as e:
            logger.warning("Laya evaluate_consensus failed: %s", e)
            return None

    def evaluate_user_note(self, note: str) -> float | None:
        """
        Évalue si la note nécessite un NEW_ROUND.
        Retourne P(new_round) ∈ [0.0, 1.0] ou None si Laya indisponible.
        """
        if not self.available or self._router is None:
            return None
        try:
            state = {"user_note": note}
            result = self._router.predict(state, USER_NOTE_QUESTIONS)
            return result.get("requires_new_round", {}).get("p")
        except Exception as e:
            logger.warning("Laya evaluate_user_note failed: %s", e)
            return None
