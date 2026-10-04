"""
finetune_laya.py — Fine-tuning de Laya sur les données de routing Table Ronde.
Utilise RLCD (Reinforcement Learning for Calibrated Decisions).

Usage :
    uv run python scripts/finetune_laya.py [--output PATH] [--epochs N]
    # Ou via la commande CLI :
    table-ronde finetune-router

Références :
    - https://github.com/NandhaKishorM/laya
    - Notebook : laya_finetune_typed_decisions_2xT4_kaggle.ipynb
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Dataset synthétique de base ───────────────────────────────────────────────
# Couvre les 6 domaines avec des exemples annotés (labels de vérité terrain).
SYNTHETIC_DATASET: list[dict] = [
    # security
    {"state": {"task_description": "How do I prevent SQL injection in my FastAPI endpoints?"},
     "labels": {"domain": "security", "complexity": 4, "needs_multi_round_debate": True, "needs_frontier_model": True}},
    {"state": {"task_description": "Implement OAuth2 authentication with JWT refresh tokens"},
     "labels": {"domain": "security", "complexity": 4, "needs_multi_round_debate": True, "needs_frontier_model": True}},
    {"state": {"task_description": "Review this code for XSS vulnerabilities"},
     "labels": {"domain": "security", "complexity": 3, "needs_multi_round_debate": True, "needs_frontier_model": False}},
    {"state": {"task_description": "Setup zero-trust network architecture with mTLS"},
     "labels": {"domain": "security", "complexity": 5, "needs_multi_round_debate": True, "needs_frontier_model": True}},
    # architecture
    {"state": {"task_description": "Design a microservices architecture for an e-commerce platform with 1M users"},
     "labels": {"domain": "architecture", "complexity": 5, "needs_multi_round_debate": True, "needs_frontier_model": True}},
    {"state": {"task_description": "Should I use CQRS and Event Sourcing for my notification system?"},
     "labels": {"domain": "architecture", "complexity": 4, "needs_multi_round_debate": True, "needs_frontier_model": True}},
    {"state": {"task_description": "How to implement the Saga pattern for distributed transactions?"},
     "labels": {"domain": "architecture", "complexity": 5, "needs_multi_round_debate": True, "needs_frontier_model": True}},
    # devops
    {"state": {"task_description": "Create a CI/CD pipeline with GitHub Actions for a Python monorepo"},
     "labels": {"domain": "devops", "complexity": 3, "needs_multi_round_debate": False, "needs_frontier_model": False}},
    {"state": {"task_description": "Setup Kubernetes horizontal pod autoscaling with custom metrics"},
     "labels": {"domain": "devops", "complexity": 4, "needs_multi_round_debate": True, "needs_frontier_model": False}},
    {"state": {"task_description": "Implement blue-green deployment with Terraform on AWS"},
     "labels": {"domain": "devops", "complexity": 4, "needs_multi_round_debate": True, "needs_frontier_model": False}},
    # frontend
    {"state": {"task_description": "Build a real-time dashboard with React and WebSockets"},
     "labels": {"domain": "frontend", "complexity": 3, "needs_multi_round_debate": False, "needs_frontier_model": False}},
    {"state": {"task_description": "Design a component library with Tailwind CSS and Storybook"},
     "labels": {"domain": "frontend", "complexity": 2, "needs_multi_round_debate": False, "needs_frontier_model": False}},
    # data
    {"state": {"task_description": "Design a real-time ETL pipeline from Kafka to PostgreSQL with pgvector"},
     "labels": {"domain": "data", "complexity": 4, "needs_multi_round_debate": True, "needs_frontier_model": False}},
    {"state": {"task_description": "Optimize slow SQL queries on a 100M rows table"},
     "labels": {"domain": "data", "complexity": 3, "needs_multi_round_debate": False, "needs_frontier_model": False}},
    # simple
    {"state": {"task_description": "Rename the variable 'x' to 'user_count' in this function"},
     "labels": {"domain": "simple", "complexity": 1, "needs_multi_round_debate": False, "needs_frontier_model": False}},
    {"state": {"task_description": "Fix the typo in this docstring"},
     "labels": {"domain": "simple", "complexity": 1, "needs_multi_round_debate": False, "needs_frontier_model": False}},
    {"state": {"task_description": "What does the 'yield' keyword do in Python?"},
     "labels": {"domain": "simple", "complexity": 1, "needs_multi_round_debate": False, "needs_frontier_model": False}},
]


def build_dataset(feedback_path: Path) -> list[dict]:
    """
    Combine le dataset synthétique avec les feedbacks réels validés.
    Un feedback est "validé" si |rounds_prédits - rounds_réels| <= 1.
    """
    dataset = list(SYNTHETIC_DATASET)
    if feedback_path.exists():
        with feedback_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    pred = record.get("predicted_rounds", 0)
                    actual = record.get("actual_rounds_used", 0)
                    if actual > 0 and abs(pred - actual) <= 1:
                        dataset.append({
                            "state": {"task_description": record["prompt"]},
                            "labels": {
                                "domain": record["predicted_domain"],
                                "needs_multi_round_debate": actual > 1,
                            },
                        })
                except (json.JSONDecodeError, KeyError):
                    continue
    logger.info("Dataset: %d samples (%d synthetic, %d from feedback)",
                len(dataset), len(SYNTHETIC_DATASET), len(dataset) - len(SYNTHETIC_DATASET))
    return dataset


def finetune(
    output_dir: str = "~/.table_ronde/laya_custom/",
    feedback_path: str = "~/.table_ronde/feedback_store.jsonl",
    base_checkpoint: str = "convaiinnovations/laya",
    epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 2e-5,
) -> dict:
    """Lance le fine-tuning RLCD."""
    from table_ronde.laya_router import TASK_ANALYSIS_QUESTIONS

    out_path = Path(output_dir).expanduser()
    fb_path = Path(feedback_path).expanduser()

    dataset = build_dataset(fb_path)

    try:
        import torch
        from laya import Trainer  # type: ignore[import-not-found]
    except ImportError:
        logger.error("laya or torch not installed.")
        sys.exit(1)

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"

    trainer = Trainer(
        base_checkpoint=base_checkpoint,
        questions=TASK_ANALYSIS_QUESTIONS,
        dataset=dataset,
        output_dir=str(out_path),
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        calibrate=True,
        device=device,
    )

    logger.info("Starting RLCD fine-tuning on %d samples...", len(dataset))
    metrics = trainer.train()

    logger.info("Fine-tuning complete. Checkpoint: %s", out_path)
    logger.info(
        "Update your config.yaml to use the fine-tuned model:\n"
        "  smart_router:\n"
        "    laya:\n"
        "      checkpoint: %s", out_path
    )
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune Laya for Table Ronde routing")
    parser.add_argument("--output", default="~/.table_ronde/laya_custom/")
    parser.add_argument("--feedback", default="~/.table_ronde/feedback_store.jsonl")
    parser.add_argument("--base-checkpoint", default="convaiinnovations/laya")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    metrics = finetune(args.output, args.feedback, args.base_checkpoint, args.epochs, args.batch_size)
    logger.info("Metrics: %s", metrics)
