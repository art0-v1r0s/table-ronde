"""
smart_router.py — Smart Router hybride v2.0
Architecture à 3 niveaux :
  1. Laya         (~33ms GPU, local, zéro hallucinations) — si disponible
  2. Heuristiques (0ms, déterministe, offline)            — fallback niveau 1
  3. LLM          (3-5s, qualité maximale)                — fallback niveau 2

Stratégies configurables : auto | laya | heuristic | llm
"""
from __future__ import annotations

import logging
import os
import re
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ── Utilitaire LLM ────────────────────────────────────────────────────────────

def _get_raw_llm(provider: str, model_name: str, temperature: float = 0.0):
    """Get a raw LLM without tool bindings, for structured output use."""
    from table_ronde.agents import get_llm as _get_llm_with_tools
    prov = provider.lower()
    if prov in ("copilot", "github", "openai"):
        from langchain_openai import ChatOpenAI
        key = os.getenv("GITHUB_TOKEN") or os.getenv("COPILOT_API_KEY") or os.getenv("OPENAI_API_KEY")
        endpoint = "https://models.inference.ai.azure.com" if prov in ("copilot", "github") else None
        kwargs: dict[str, Any] = {"model": model_name, "temperature": temperature, "api_key": key, "max_retries": 3}
        if endpoint:
            kwargs["base_url"] = endpoint
        return ChatOpenAI(**kwargs)
    elif prov == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        return ChatGoogleGenerativeAI(model=model_name, temperature=temperature, google_api_key=key, max_retries=3)
    elif prov == "claude":
        from langchain_anthropic import ChatAnthropic
        key = os.getenv("ANTHROPIC_API_KEY")
        return ChatAnthropic(model=model_name, temperature=temperature, api_key=key, max_retries=3)
    elif prov == "ollama":
        from langchain_ollama import ChatOllama
        ollama_url = os.getenv("OLLAMA_HOST") or "http://localhost:11434"
        if not ollama_url.startswith("http"):
            ollama_url = f"http://{ollama_url}"
        return ChatOllama(model=model_name, temperature=temperature, base_url=ollama_url)
    else:
        return _get_llm_with_tools(provider, model_name, temperature=temperature)


# ── Modèles Pydantic ──────────────────────────────────────────────────────────

class TaskRoutingDecision(BaseModel):
    rounds: int = Field(ge=1, le=5, description="Recommended number of rounds (1 to 5).")
    architect_model_tier: str = Field(pattern="^(pro|flash)$", description="'pro' or 'flash'.")
    expert_model_tier: str = Field(pattern="^(pro|flash)$", description="'pro' or 'flash'.")
    temperature: float = Field(ge=0.0, le=1.0, description="Between 0.0 (strict) and 0.8 (creative).")
    complexity: int = Field(ge=1, le=10, description="Complexity score 1-10.")
    domain: str = Field(description="Main domain: 'Frontend', 'DevOps', 'Security', 'Architecture', 'Data', 'Simple'.")
    # Champs v2 — valeurs par défaut pour compatibilité ascendante
    route_name: str = Field(default="unknown", description="Detected route identifier.")
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Router confidence.")
    routing_method: str = Field(default="llm", description="'laya' | 'heuristic' | 'llm'.")


class ConsensusDecision(BaseModel):
    consensus_reached: bool = Field(description="True if participants reached clear technical agreement.")
    confidence: float = Field(description="Confidence score between 0.0 and 1.0.")


class RoutingDecision(BaseModel):
    action: str = Field(description="Must be 'SYNTHESIS' or 'NEW_ROUND'.")


# ── Configuration des modèles rapides (LLM fallback) ─────────────────────────

FAST_MODELS: dict[str, str] = {
    "gemini": "gemini-3.5-flash-lite",
    "openai": "gpt-4o-mini",
    "copilot": "gpt-4o-mini",
    "github": "gpt-4o-mini",
    "claude": "claude-haiku-4-20250414",
    "ollama": "llama3.1",
}

VALID_STRATEGIES = ("auto", "laya", "heuristic", "llm")


# ── Heuristiques (Niveau 2 du fallback) ──────────────────────────────────────

_DOMAIN_KEYWORDS: dict[str, list[str]] = {
    "security": ["security", "injection", "xss", "csrf", "authentication", "oauth",
                 "jwt", "zero-trust", "encryption", "vulnerability", "exploit"],
    "devops":   ["docker", "kubernetes", "k8s", "ci/cd", "pipeline", "deploy",
                 "helm", "terraform", "ansible", "monitoring", "prometheus"],
    "data":     ["database", "sql", "postgres", "redis", "elasticsearch", "etl",
                 "pipeline", "vector", "embedding", "query", "index", "shard"],
    "frontend": ["react", "vue", "angular", "css", "tailwind", "ui", "ux",
                 "component", "responsive", "accessibility"],
    "simple":   ["rename", "typo", "comment", "fix this", "what is"],
}

_COMPLEXITY_KEYWORDS: dict[str, int] = {
    "architecture": 2, "distributed": 2, "microservice": 2,
    "kafka": 2, "kubernetes": 2, "security": 2, "vulnerability": 2,
    "sql injection": 3, "zero-trust": 3, "algorithm": 2,
    "optimization": 2, "machine learning": 2, "scalable": 1,
    "rename": -2, "typo": -2, "color": -1, "comment": -1,
}

_AGREE_PATTERNS = [
    r"\bi agree\b", r"\bi concur\b", r"\bexcellent point\b",
    r"\bgood point\b", r"\bsolid approach\b", r"\bthis is the right\b",
    r"\blet's go with\b", r"\bwe should adopt\b", r"\bfull agreement\b",
]
_DISAGREE_PATTERNS = [
    r"\bhowever\b", r"\bi disagree\b", r"\brisk\b", r"\bdanger\b",
    r"\bproblem\b", r"\bconcern\b",
]


def _score_complexity_heuristic(prompt: str) -> float:
    """Score de complexité ∈ [0.0, 1.0] basé sur heuristiques textuelles."""
    text = prompt.lower()
    score = min(len(text.split()) / 200.0, 0.3)
    score += min(text.count("?") * 0.05, 0.15)
    kw_score = sum(
        weight * text.count(kw)
        for kw, weight in _COMPLEXITY_KEYWORDS.items()
        if kw in text
    )
    score += min(kw_score * 0.05, 0.50)
    if "```" in prompt or "def " in prompt or "class " in prompt:
        score += 0.10
    return min(max(score, 0.0), 1.0)


def _detect_domain_heuristic(prompt: str) -> str:
    """Détecte le domaine le plus probable via correspondance de mots-clés."""
    text = prompt.lower()
    scores = {domain: sum(1 for kw in keywords if kw in text)
              for domain, keywords in _DOMAIN_KEYWORDS.items()}
    best = max(scores, key=lambda k: scores[k])
    return best if scores[best] > 0 else "architecture"


def _evaluate_consensus_heuristic(history: list) -> tuple[bool, float]:
    """Détecte le consensus via patterns textuels. Retourne (atteint, confiance)."""
    if len(history) < 2:
        return False, 0.0
    texts = []
    for msg in history[-6:]:
        content = getattr(msg, "content", "")
        if isinstance(content, list):
            content = " ".join(
                part["text"] if isinstance(part, dict) and "text" in part else str(part)
                for part in content
            )
        texts.append(str(content).lower())
    combined = " ".join(texts)
    agree_count = sum(1 for p in _AGREE_PATTERNS if re.search(p, combined))
    disagree_count = sum(1 for p in _DISAGREE_PATTERNS if re.search(p, combined))
    if agree_count == 0:
        return False, 0.0
    ratio = agree_count / (agree_count + disagree_count + 1)
    return ratio >= 0.5, ratio


def _format_history_text(history: list) -> str:
    """Formate l'historique de messages en texte pour Laya."""
    lines = []
    for msg in history:
        content = getattr(msg, "content", "")
        if isinstance(content, list):
            content = " ".join(
                part["text"] if isinstance(part, dict) and "text" in part else str(part)
                for part in content
            )
        role = msg.__class__.__name__.replace("Message", "")
        lines.append(f"{role}: {str(content)[:500]}")
    return "\n".join(lines)


# ── LLM Router (ancien SmartRouterEngine, renommé) ───────────────────────────

class LLMSmartRouterEngine:
    """
    Router basé sur LLM avec structured output.
    C'est l'ancien SmartRouterEngine, conservé intégralement et renommé.
    Utilisé comme niveau 3 (fallback final) par SmartRouterV2.
    """

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.enabled = config.get("smart_routing_enabled", False)
        self.provider = config.get("router_provider", config.get("default_provider", "gemini"))
        self.model_name = config.get("router_model", FAST_MODELS.get(self.provider, "gemini-3.5-flash-lite"))
        self.consensus_threshold = config.get("consensus_threshold", 0.85)

        self.task_analyzer = None
        self.consensus_evaluator = None
        self.routing_evaluator = None

        if self.enabled:
            try:
                llm = _get_raw_llm(provider=self.provider, model_name=self.model_name, temperature=0.0)
                self.task_analyzer = llm.with_structured_output(TaskRoutingDecision)
                self.consensus_evaluator = llm.with_structured_output(ConsensusDecision)
                self.routing_evaluator = llm.with_structured_output(RoutingDecision)
            except Exception as e:
                logger.error(
                    "Failed to initialize LLMSmartRouterEngine with %s/%s: %s",
                    self.provider, self.model_name, e,
                )
                self.enabled = False

    def analyze_task(self, prompt: str) -> TaskRoutingDecision | None:
        if not self.enabled or not self.task_analyzer:
            return None
        try:
            decision = self.task_analyzer.invoke(
                f"Analyze this task and determine the optimal debate configuration:\n\n{prompt}"
            )
            # Assurer que les champs v2 ont des valeurs par défaut
            if decision and not decision.route_name:
                decision.route_name = decision.domain.lower()
            if decision:
                decision.routing_method = "llm"
            return decision
        except Exception as e:
            logger.error("LLMSmartRouterEngine analyze_task failed: %s", e)
            return None

    def evaluate_consensus(self, history: list[Any]) -> bool:
        if not self.enabled or not self.consensus_evaluator:
            return False
        if len(history) < 2:
            return False
        try:
            recent_msgs = history[-4:]
            formatted_history = ""
            for msg in recent_msgs:
                if hasattr(msg, "content"):
                    content = msg.content
                    if isinstance(content, list):
                        content = "".join(
                            part["text"] if isinstance(part, dict) and "text" in part else str(part)
                            for part in content
                        )
                    else:
                        content = str(content)
                    role = msg.__class__.__name__
                    formatted_history += f"{role}: {content}\n"
            prompt = f"Analyze this architecture debate and decide if they explicitly agree on the solution.\n\n{formatted_history}"
            result = self.consensus_evaluator.invoke(prompt)
            if hasattr(result, "consensus_reached") and hasattr(result, "confidence"):
                return result.consensus_reached and result.confidence >= self.consensus_threshold
            return False
        except Exception as e:
            logger.error("LLMSmartRouterEngine consensus evaluation failed: %s", e)
            return False

    def evaluate_user_note(self, note: str) -> str:
        if not self.enabled or not self.routing_evaluator:
            return "NEW_ROUND"
        try:
            prompt = f"User note: '{note}'. Decide if this introduces a major change requiring 'NEW_ROUND', or a minor tweak for 'SYNTHESIS'."
            result = self.routing_evaluator.invoke(prompt)
            if hasattr(result, "action") and result.action in ("SYNTHESIS", "NEW_ROUND"):
                return result.action
            return "NEW_ROUND"
        except Exception as e:
            logger.error("LLMSmartRouterEngine user note evaluation failed: %s", e)
            return "NEW_ROUND"


# ── Smart Router V2 (Orchestrateur des 3 niveaux) ────────────────────────────

class SmartRouterV2:
    """
    Smart Router hybride v2.0.
    Stratégie configurable via config['smart_router']['strategy'] :
      - 'auto'      : cascade Laya → Heuristique → LLM (recommandé)
      - 'laya'      : Laya strict (RuntimeError si Laya non disponible)
      - 'heuristic' : Heuristiques uniquement (0ms, offline)
      - 'llm'       : LLM uniquement (comportement pré-v2)
    """

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.enabled = config.get("smart_routing_enabled", False)

        sr_cfg = config.get("smart_router", {})
        if "strategy" in sr_cfg:
            self.strategy: str = sr_cfg["strategy"]
        elif "router_strategy" in config:
            self.strategy = config["router_strategy"]
        elif "smart_router" in config:
            self.strategy = "auto"
        else:
            self.strategy = "llm"

        if self.strategy not in VALID_STRATEGIES:
            logger.warning(
                "Invalid router_strategy '%s'. Falling back to 'auto'. Valid: %s",
                self.strategy, VALID_STRATEGIES,
            )
            self.strategy = "auto"

        laya_cfg = sr_cfg.get("laya", {})
        self.confidence_threshold: float = laya_cfg.get("confidence_threshold", 0.65)
        self.consensus_threshold: float = laya_cfg.get("consensus_threshold", 0.75)

        # Niveau 3 : LLM (l'ancien comportement)
        self._llm_router = LLMSmartRouterEngine(config)

        # Niveau 1 : Laya (chargé seulement si la stratégie le demande)
        self._laya: Any = None
        if self.enabled and self.strategy in ("auto", "laya"):
            try:
                from table_ronde.laya_router import LayaSmartRouter
                checkpoint = laya_cfg.get("checkpoint", "convaiinnovations/laya")
                device = laya_cfg.get("device", "auto")
                self._laya = LayaSmartRouter(checkpoint=checkpoint, device=device)
            except Exception as e:
                logger.warning("Could not initialize Laya: %s", e)

        # Si le routeur LLM a échoué et que Laya n'est pas dispo, désactiver le routeur
        if not self._llm_router.enabled and (not self._laya or not self._laya.available):
            self.enabled = False

        # Feedback (gardé en mémoire, écrit à la fin de la session)
        self.current_feedback: Any = None

    @property
    def consensus_threshold_llm(self) -> float:
        """Seuil de consensus pour le LLM fallback (legacy)."""
        return self.config.get("consensus_threshold", 0.85)

    def analyze_task(self, prompt: str) -> TaskRoutingDecision | None:
        """
        Analyse le prompt selon la stratégie configurée.
        Crée aussi self.current_feedback pour collecte ultérieure.
        """
        if not self.enabled:
            return None

        decision: TaskRoutingDecision | None = None

        if self.strategy == "laya":
            # Mode strict : Laya ou erreur explicite
            if not (self._laya and self._laya.available):
                raise RuntimeError(
                    "router_strategy='laya' but Laya is not available. "
                    "Install with: uv add laya"
                )
            raw = self._laya.analyze_task(prompt)
            if raw:
                decision = self._parse_laya_result(raw)

        elif self.strategy == "heuristic":
            decision = self._analyze_heuristic(prompt)

        elif self.strategy == "llm":
            decision = self._llm_router.analyze_task(prompt)

        else:  # auto
            # Niveau 1 : Laya
            if self._laya and self._laya.available:
                raw = self._laya.analyze_task(prompt)
                if raw:
                    candidate = self._parse_laya_result(raw)
                    if candidate.confidence_score >= self.confidence_threshold:
                        decision = candidate
                    else:
                        logger.info(
                            "Laya confidence %.2f < threshold %.2f → heuristic fallback",
                            candidate.confidence_score, self.confidence_threshold,
                        )
            # Niveau 2 : Heuristiques
            if decision is None:
                decision = self._analyze_heuristic(prompt)
                # Niveau 3 : LLM si heuristique peu confiante
                if decision.confidence_score < 0.4:
                    llm_dec = self._llm_router.analyze_task(prompt)
                    if llm_dec:
                        decision = llm_dec

        # Enregistrement en mémoire pour la collecte du feedback post-débat
        if decision:
            from datetime import UTC, datetime

            from table_ronde.laya_feedback import RoutingFeedback

            self.current_feedback = RoutingFeedback(
                timestamp=datetime.now(UTC).isoformat(),
                prompt=prompt[:500],
                predicted_domain=decision.route_name,
                predicted_rounds=decision.rounds,
                predicted_complexity=decision.complexity,
                routing_method=decision.routing_method,
                confidence_score=decision.confidence_score,
            )

        return decision

    def _parse_laya_result(self, raw: dict[str, Any]) -> TaskRoutingDecision:
        """Convertit le résultat brut Laya en TaskRoutingDecision."""
        # Domain
        domain_probs: dict[str, float] = raw.get("domain", {})
        domain = max(domain_probs, key=lambda k: domain_probs[k]) if domain_probs else "architecture"
        confidence = float(domain_probs.get(domain, 0.5))

        # Complexity: expected_score ∈ [1,5] → scaled to [1,10]
        complexity_data = raw.get("complexity", {})
        expected_score = float(complexity_data.get("expected_score", 2.5))
        complexity_10 = max(1, min(10, round(expected_score * 2)))

        # Rounds
        debate_p = float(raw.get("needs_multi_round_debate", {}).get("p", 0.5))
        rounds = 3 if debate_p > 0.7 else 2 if debate_p > 0.4 else 1

        # Model tiers
        pro_p = float(raw.get("needs_frontier_model", {}).get("p", 0.5))
        architect_tier = "pro" if pro_p > 0.6 else "flash"
        expert_tier = "pro" if pro_p > 0.75 else "flash"

        # Temperature par domaine
        _DOMAIN_TEMPS = {"security": 0.2, "architecture": 0.3, "devops": 0.3,
                         "data": 0.3, "frontend": 0.5, "simple": 0.5}
        temperature = _DOMAIN_TEMPS.get(domain, 0.4)

        return TaskRoutingDecision(
            rounds=rounds,
            architect_model_tier=architect_tier,
            expert_model_tier=expert_tier,
            temperature=temperature,
            complexity=complexity_10,
            domain=domain.capitalize(),
            route_name=domain,
            confidence_score=confidence,
            routing_method="laya",
        )

    def _analyze_heuristic(self, prompt: str) -> TaskRoutingDecision:
        """Analyse heuristique. Utilisé comme niveau 2 ou stratégie exclusive."""
        domain = _detect_domain_heuristic(prompt)
        complexity = max(1, min(10, int(_score_complexity_heuristic(prompt) * 10)))
        # Configuration par domaine
        _DOMAIN_CONFIGS: dict[str, tuple[int, str, str, float]] = {
            # (rounds, arch_tier, expert_tier, temperature)
            "security":     (3, "pro",   "pro",   0.2),
            "architecture": (3, "pro",   "flash", 0.3),
            "devops":       (2, "flash", "pro",   0.3),
            "data":         (2, "pro",   "flash", 0.3),
            "frontend":     (2, "flash", "flash", 0.5),
            "simple":       (1, "flash", "flash", 0.5),
        }
        rounds, arch, expert, temp = _DOMAIN_CONFIGS.get(domain, (2, "flash", "flash", 0.4))
        return TaskRoutingDecision(
            rounds=rounds,
            architect_model_tier=arch,
            expert_model_tier=expert,
            temperature=temp,
            complexity=complexity,
            domain=domain.capitalize(),
            route_name=domain,
            confidence_score=0.6,
            routing_method="heuristic",
        )

    def evaluate_consensus(self, history: list[Any]) -> bool:
        if not self.enabled or len(history) < 2:
            return False

        # Niveau 1 : Laya
        if self.strategy in ("auto", "laya") and self._laya and self._laya.available:
            debate_text = _format_history_text(history[-8:])
            p = self._laya.evaluate_consensus(debate_text)
            if p is not None:
                if p >= self.consensus_threshold:
                    logger.info("Consensus detected by Laya (p=%.2f)", p)
                    return True
                return False  # Laya a répondu → pas besoin de fallback

        # Niveau 2 : Heuristiques
        if self.strategy in ("auto", "heuristic"):
            reached, confidence = _evaluate_consensus_heuristic(history)
            if confidence > 0.3:
                return reached

        # Niveau 3 : LLM
        return self._llm_router.evaluate_consensus(history)

    def evaluate_user_note(self, note: str) -> str:
        if not self.enabled:
            return "NEW_ROUND"

        # Niveau 1 : Laya
        if self.strategy in ("auto", "laya") and self._laya and self._laya.available:
            p = self._laya.evaluate_user_note(note)
            if p is not None:
                result = "NEW_ROUND" if p >= 0.5 else "SYNTHESIS"
                logger.info("User note routing by Laya: %s (p=%.2f)", result, p)
                return result

        # Niveau 3 : LLM (pas de niveau 2 pour les notes utilisateur)
        return self._llm_router.evaluate_user_note(note)


# ── Alias de compatibilité (NE PAS SUPPRIMER — requis par orchestrator.py) ───

SmartRouterEngine = SmartRouterV2


# ── Fonction apply_task_routing (mise à jour pour utiliser SmartRouterV2) ────

def apply_task_routing(
    provider: str,
    model: str | None,
    prompt: str,
    config: dict,
    router_provider: str | None = None,
    router_model: str | None = None,
) -> tuple[int, TaskRoutingDecision | None]:
    """Runs the Smart Router analysis and updates the config in place."""
    router_cfg: dict[str, Any] = {
        "router_provider": router_provider or provider,
        "smart_routing_enabled": True,
        "smart_router": config.get("smart_router", {}),
    }
    if router_model:
        router_cfg["router_model"] = router_model

    router = SmartRouterV2(router_cfg)
    decision = router.analyze_task(prompt)
    rounds = 1

    if decision:
        rounds = decision.rounds
        config.setdefault("orchestrator", {})["rounds"] = rounds
        config["orchestrator"]["default_provider"] = provider
        config["orchestrator"]["smart_routing_enabled"] = True

        from table_ronde.agents import resolve_model_tier
        arch_model = resolve_model_tier(provider, decision.architect_model_tier)
        expert_model = resolve_model_tier(provider, decision.expert_model_tier)

        if arch_model:
            config.setdefault("architect", {})["model"] = arch_model
            config["architect"]["temperature"] = decision.temperature
        if expert_model:
            config.setdefault("orchestrator", {})["default_model"] = expert_model

    return rounds, decision
