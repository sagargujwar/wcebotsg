"""
app/intent.py
─────────────
Hybrid query classifier: deterministic lexicon rules fused with embedding
similarity against per-category seed prototypes.

Design (why hybrid):
  • Rules are free, instant and explainable, but miss paraphrases
    ("how much do I pay" contains no "fee").
  • MiniLM prototypes catch paraphrases, but a cosine of ~0.35 is ambiguous
    on its own for short queries.
  • Fusing both keeps every request at zero extra API cost and < ~5 ms
    (one short-query encode, batched with nothing else).

Confidence gate: a category only wins when its fused score clears MIN_SCORE
and beats the runner-up by MIN_MARGIN.  Otherwise the query is routed to
Category.GENERAL — the exact baseline pipeline (no boost, no threshold
override, no rubric).  A misclassification therefore can only ever *equal*
baseline behaviour, never silently rewrite it.

The classifier never predicts OUT_OF_SCOPE: the similarity gate and the
grounded prompt already own the refusal path.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from app.categories import (
    Category,
    PREDICTABLE_CATEGORIES,
    get_spec,
    rule_scores,
)

logger = logging.getLogger(__name__)

# Fusion weights (tuned against tests/eval/queries.jsonl — see
# scripts/eval_accuracy.py).  Embeddings weigh more because they generalise
# to paraphrases; rules act as a strong deterministic prior.
RULE_WEIGHT = 0.45
PROTO_WEIGHT = 0.55

# Confidence gates: winner must clear MIN_SCORE *and* beat the runner-up by
# MIN_MARGIN, else → GENERAL (baseline behaviour).  MIN_SCORE is calibrated so
# a clear lexicon hit (0.5 rules) plus weak embedding support (~0.15 cosine)
# routes, while embedding-only noise (≤0.33 fused) falls through to baseline.
# Tuned against tests/eval/queries.jsonl via scripts/eval_accuracy.py.
MIN_SCORE = 0.34
MIN_MARGIN = 0.05


@dataclass(frozen=True)
class IntentResult:
    """Outcome of classification for one query."""

    category: Category
    confidence: float                       # fused score of the winning category
    method: str                             # "hybrid" | "rules" | "none"
    scores: Dict[str, float] = field(default_factory=dict)  # top fused scores

    @property
    def is_specific(self) -> bool:
        """True when a concrete category (not the GENERAL fallback) won."""
        return self.category not in (Category.GENERAL, Category.OUT_OF_SCOPE)


class QueryClassifier:
    """
    Hybrid classifier.

    Parameters
    ----------
    encoder:
        Anything exposing `encode(list[str], normalize_embeddings=True)`
        (a SentenceTransformer).  Normally the retriever's already-loaded
        model — no second model instance.  ``None`` degrades to rules-only
        classification (used by tests and degenerate deployments).
    """

    def __init__(self, encoder=None) -> None:
        self._encoder = encoder
        self._centroids: Optional[Dict[Category, List[float]]] = None
        if encoder is not None:
            try:
                self._centroids = self._build_centroids()
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Prototype centroid build failed (%s) — rules only", exc)
                self._centroids = None

    # ── Prototypes ────────────────────────────────────────────────────────────

    def _build_centroids(self) -> Dict[Category, List[float]]:
        """Mean of normalised seed-query embeddings per category."""
        import numpy as np

        texts: List[str] = []
        owner: List[Category] = []
        for category in PREDICTABLE_CATEGORIES:
            for seed in get_spec(category).seed_queries:
                texts.append(seed)
                owner.append(category)

        matrix = self._encoder.encode(texts, normalize_embeddings=True)
        matrix = np.asarray(matrix, dtype="float32")

        centroids: Dict[Category, List[float]] = {}
        for category in PREDICTABLE_CATEGORIES:
            rows = matrix[[i for i, c in enumerate(owner) if c == category]]
            mean = rows.mean(axis=0)
            norm = float(np.linalg.norm(mean))
            if norm > 0:
                mean = mean / norm
            centroids[category] = mean.tolist()
        logger.info(
            "Intent prototypes ready — %d categories × %d seed queries",
            len(centroids),
            len(texts),
        )
        return centroids

    # ── Scoring layers ────────────────────────────────────────────────────────

    def _rule_scores(self, query: str) -> Dict[Category, float]:
        return rule_scores(query)

    def _prototype_scores(self, query: str) -> Dict[Category, float]:
        """Cosine similarity of the query to each category centroid, in [0, 1]."""
        if not self._centroids:
            return {}
        import numpy as np

        q = np.asarray(
            self._encoder.encode([query], normalize_embeddings=True), dtype="float32"
        )[0]
        scores: Dict[Category, float] = {}
        for category, centroid in self._centroids.items():
            sim = float(np.dot(q, np.asarray(centroid, dtype="float32")))
            scores[category] = max(0.0, min(1.0, sim))
        return scores

    # ── Public API ────────────────────────────────────────────────────────────

    def classify(self, query: str) -> IntentResult:
        """
        Assign *query* to a category.

        Always returns — worst case ``Category.GENERAL`` with method "none",
        which downstream treats as "no category signal".
        """
        rules = self._rule_scores(query)
        protos = self._prototype_scores(query)

        if protos:
            method = "hybrid"
            fused = {
                category: RULE_WEIGHT * rules.get(category, 0.0)
                + PROTO_WEIGHT * protos.get(category, 0.0)
                for category in PREDICTABLE_CATEGORIES
            }
        else:
            method = "rules"
            fused = {c: s for c, s in rules.items()}

        if not fused or max(fused.values(), default=0.0) <= 0.0:
            return IntentResult(category=Category.GENERAL, confidence=0.0, method="none")

        ranked = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
        winner, score = ranked[0]
        runner_up = ranked[1][1] if len(ranked) > 1 else 0.0

        if score < MIN_SCORE or (score - runner_up) < MIN_MARGIN:
            logger.debug(
                "Low confidence (%.3f, margin %.3f) → general: %.80s",
                score,
                score - runner_up,
                query,
            )
            return IntentResult(
                category=Category.GENERAL,
                confidence=round(score, 3),
                method=method,
                scores={k: round(v, 3) for k, v in ranked[:3]},
            )

        return IntentResult(
            category=winner,
            confidence=round(score, 3),
            method=method,
            scores={k: round(v, 3) for k, v in ranked[:3]},
        )
