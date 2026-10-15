"""
tests/test_intent.py
────────────────────
Tests for the query taxonomy (app/categories.py) and the hybrid classifier
(app/intent.py).  Fully offline: the prototype layer is exercised with a
deterministic stub encoder, never with network weights.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.categories import (  # noqa: E402
    ABBREVIATIONS,
    CATEGORY_SPECS,
    Category,
    UI_LABELS,
    boost_for,
    expand_query,
    get_spec,
    label_for,
    rule_scores,
    threshold_for,
    url_to_category,
)
from app.intent import (  # noqa: E402
    MIN_MARGIN,
    MIN_SCORE,
    IntentResult,
    QueryClassifier,
)


# ─────────────────────────────────────────────────────────────────────────────
# Stub encoder: seeds of a category collapse onto that category's axis, and
# tests inject exact query vectors to control prototype cosine similarities.
# ─────────────────────────────────────────────────────────────────────────────

CATS = [
    Category.ADMISSIONS_FEES,
    Category.ACADEMICS,
    Category.PLACEMENTS_CAREERS,
    Category.PEOPLE_ADMIN,
    Category.INFRA_CAMPUS,
    Category.NOTICES_EVENTS,
    Category.RESEARCH,
    Category.GENERAL_INFO,
]


def axis(category: Category, dim: int = len(CATS)) -> np.ndarray:
    v = np.zeros(dim, dtype="float32")
    v[CATS.index(category)] = 1.0
    return v


def vector_at(category: Category, cosine: float) -> np.ndarray:
    """
    Unit vector with cosine *cosine* to *category*'s axis; the residual is
    spread evenly over the remaining axes so it can't crown a rival category.
    """
    e = axis(category)
    residual = np.zeros(len(CATS), dtype="float32")
    for i, other in enumerate(CATS):
        if other != category:
            residual += axis(other)
    residual /= np.linalg.norm(residual)
    v = cosine * e + float(np.sqrt(max(0.0, 1.0 - cosine * cosine))) * residual
    return (v / np.linalg.norm(v)).astype("float32")


def vector_between(first: Category, second: Category) -> np.ndarray:
    """Unit vector equidistant from two category axes (a deliberate tie)."""
    v = axis(first) + axis(second)
    return (v / np.linalg.norm(v)).astype("float32")


class StubEncoder:
    """Deterministic encoder: seeds → category axis, known queries → table."""

    def __init__(self):
        self.table = {}
        for i, category in enumerate(CATS):
            for seed in get_spec(category).seed_queries:
                self.table[seed] = axis(category)

    def encode(self, texts, normalize_embeddings=True):
        rows = []
        for text in texts:
            if text in self.table:
                rows.append(np.array(self.table[text], dtype="float32"))
            else:
                rows.append(np.full(len(CATS), 0.02, dtype="float32"))
        matrix = np.vstack(rows)
        if normalize_embeddings:
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            matrix = matrix / norms
        return matrix


@pytest.fixture
def stub_encoder():
    return StubEncoder()


@pytest.fixture
def hybrid(stub_encoder):
    return QueryClassifier(encoder=stub_encoder)


@pytest.fixture
def rules_only():
    return QueryClassifier(encoder=None)


# ─────────────────────────────────────────────────────────────────────────────
# Taxonomy integrity
# ─────────────────────────────────────────────────────────────────────────────

class TestTaxonomy:
    def test_every_category_has_a_spec(self):
        for category in Category:
            assert category in CATEGORY_SPECS
            assert get_spec(category).id == category

    def test_every_predictable_category_has_seeds_and_urls(self):
        for category, spec in CATEGORY_SPECS.items():
            if category in (
                Category.GENERAL,
                Category.OUT_OF_SCOPE,
            ):
                continue
            assert len(spec.seed_queries) >= 8, f"{category} needs ≥8 seeds"
            assert len(spec.url_prefixes) >= 1, f"{category} needs URL prefixes"
            assert spec.rubric, f"{category} needs an answer rubric"

    def test_labels_cover_content_categories(self):
        assert label_for(Category.ADMISSIONS_FEES) == "Admissions & Fees"
        assert label_for(Category.GENERAL) == ""
        assert UI_LABELS[Category.OUT_OF_SCOPE] == ""

    def test_threshold_and_boost_helpers(self):
        assert threshold_for(Category.NOTICES_EVENTS, 0.35) == 0.30
        assert threshold_for(Category.ACADEMICS, 0.35) == 0.35  # no override
        assert threshold_for(Category.GENERAL, 0.35) == 0.35
        assert boost_for(Category.ADMISSIONS_FEES) > 0
        assert boost_for(Category.GENERAL) == 0.0
        assert boost_for(Category.OUT_OF_SCOPE) == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# URL → category tagging (ingestion side)
# ─────────────────────────────────────────────────────────────────────────────

class TestUrlToCategory:
    def test_admissions_pages(self):
        assert url_to_category("https://x.edu/admissions/first-year-btech/") == Category.ADMISSIONS_FEES
        assert url_to_category("https://x.edu/admissions/fees-as-per-authorities-dte-fra/") == Category.ADMISSIONS_FEES

    def test_longest_prefix_wins(self):
        # PhD admission page is a research page, not an admissions page
        assert url_to_category("https://x.edu/admissions/phd/") == Category.RESEARCH
        # Scholarship page under campus-life is an admissions/fees page
        assert url_to_category("https://x.edu/campus-life/student-scholarships/") == Category.ADMISSIONS_FEES
        # Other campus-life pages stay infrastructure
        assert url_to_category("https://x.edu/campus-life/hostels-cl/") == Category.INFRA_CAMPUS

    def test_about_pages_split_people_vs_general_info(self):
        assert url_to_category("https://x.edu/about/director/") == Category.PEOPLE_ADMIN
        assert url_to_category("https://x.edu/about/heads-and-deans/") == Category.PEOPLE_ADMIN
        assert url_to_category("https://x.edu/about/vision-mission/") == Category.GENERAL_INFO
        assert url_to_category("https://x.edu/about/nirf-nba-and-naac/") == Category.GENERAL_INFO

    def test_notice_board_overrides(self):
        assert url_to_category("https://x.edu/notice_board/some-event/") == Category.NOTICES_EVENTS
        assert url_to_category(
            "https://x.edu/notice_board/service-rules-regulation-faculty_staff/"
        ) == Category.PEOPLE_ADMIN
        assert url_to_category(
            "https://x.edu/notice_board/apply-for-faculty-recruitment-x/"
        ) == Category.PEOPLE_ADMIN

    def test_communication_page_is_people(self):
        assert url_to_category("https://x.edu/communication/") == Category.PEOPLE_ADMIN

    def test_trust_pages_are_general_info(self):
        assert url_to_category("https://x.edu/about/walchand-trust/") == Category.GENERAL_INFO
        assert url_to_category("https://x.edu/about/mte-society/") == Category.GENERAL_INFO
        # …but other /about/ pages remain people
        assert url_to_category("https://x.edu/about/director/") == Category.PEOPLE_ADMIN

    def test_pdf_title_heuristic(self):
        # Opaque /wp-content/uploads/ URLs fall back to title + filename
        assert (
            url_to_category(
                "https://x.edu/wp-content/uploads/abc123.pdf",
                title="WCE Fee Structure 2026-27",
            )
            == Category.ADMISSIONS_FEES
        )
        assert (
            url_to_category(
                "https://x.edu/wp-content/uploads/Combined_Recruitment_Advertisement.pdf",
                title="Combined Recruitment Advertisement",
            )
            == Category.PEOPLE_ADMIN
        )

    def test_department_and_campus_urls(self):
        assert (
            url_to_category("https://x.edu/department-of-computer-engineering/overview/")
            == Category.ACADEMICS
        )
        assert url_to_category("https://x.edu/student-careers/placement-in-wce/") == Category.PLACEMENTS_CAREERS
        assert url_to_category("https://x.edu/research-consultancy/overview/") == Category.RESEARCH

    def test_unknown_url_is_general(self):
        assert url_to_category("https://x.edu/wp-content/uploads/2026-3-10.pdf") == Category.GENERAL


# ─────────────────────────────────────────────────────────────────────────────
# Rule layer
# ─────────────────────────────────────────────────────────────────────────────

class TestRuleLayer:
    def test_fee_query_scores_admissions(self):
        scores = rule_scores("What is the fee structure for first year?")
        assert scores[Category.ADMISSIONS_FEES] >= 0.5

    def test_plural_and_singular_both_match(self):
        assert Category.ADMISSIONS_FEES in rule_scores("admissions process")
        assert Category.ADMISSIONS_FEES in rule_scores("admission process")

    def test_regex_patterns_beat_plain_keywords(self):
        # The ₹ pattern weighs 1.5 raw points → 1.5 / 2 = 0.75 normalised
        # (a lone keyword would be only 0.5).
        scores = rule_scores("What is ₹1,09,500 for general category?")
        assert scores[Category.ADMISSIONS_FEES] == 0.75

    def test_scores_are_normalised(self):
        for score in rule_scores("hostel fees wifi director notice research placement").values():
            assert 0.0 < score <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Hybrid classifier
# ─────────────────────────────────────────────────────────────────────────────

class TestRulesOnlyClassifier:
    def test_fee_query_is_admissions(self, rules_only):
        result = rules_only.classify("What is the fee structure for B.E. first year?")
        assert result.category == Category.ADMISSIONS_FEES
        assert result.method == "rules"
        assert result.is_specific

    def test_people_query(self, rules_only):
        result = rules_only.classify("Who is the Director of the college?")
        assert result.category == Category.PEOPLE_ADMIN

    def test_infrastructure_query(self, rules_only):
        result = rules_only.classify("Is Wi-Fi available inside the hostel?")
        assert result.category == Category.INFRA_CAMPUS

    def test_notices_query(self, rules_only):
        result = rules_only.classify("Show me the latest notices on the notice board")
        assert result.category == Category.NOTICES_EVENTS

    def test_no_hits_is_general(self, rules_only):
        result = rules_only.classify("capital of France xyzzy")
        assert result.category == Category.GENERAL
        assert not result.is_specific
        assert result.method == "none"

    def test_tied_categories_fall_back_to_general(self, rules_only):
        # "scholarship" → admissions, "committee" → people: exact tie, no winner
        result = rules_only.classify("scholarship committee")
        assert result.category == Category.GENERAL


class TestHybridClassifier:
    def test_prototype_centroids_built(self, hybrid):
        assert hybrid._centroids is not None
        assert set(hybrid._centroids) == set(CATS)

    def test_embedding_support_decides_tie(self, hybrid, stub_encoder):
        # Rules alone would tie (both 0.5); prototypes break it towards notices.
        query = "scholarship committee weekly bulletin"
        stub_encoder.table[query] = vector_at(Category.NOTICES_EVENTS, 0.85)
        result = hybrid.classify(query)
        assert result.method == "hybrid"
        assert result.category == Category.NOTICES_EVENTS

    def test_clear_rule_hit_survives_fusion(self, hybrid, stub_encoder):
        query = "fee structure please"
        stub_encoder.table[query] = vector_at(Category.ADMISSIONS_FEES, 0.45)
        result = hybrid.classify(query)
        assert result.category == Category.ADMISSIONS_FEES
        assert result.confidence >= MIN_SCORE

    def test_weak_signals_fall_back_to_general(self, hybrid, stub_encoder):
        query = "tell me something vague"
        stub_encoder.table[query] = vector_at(Category.GENERAL_INFO, 0.30)
        result = hybrid.classify(query)
        assert result.category == Category.GENERAL
        assert not result.is_specific

    def test_small_margin_falls_back_to_general(self, hybrid, stub_encoder):
        # Equal rules (hostel → infra, scholarship → admissions) AND equal
        # prototype support → the margin gate must refuse to pick a winner.
        query = "hostel scholarship"
        stub_encoder.table[query] = vector_between(
            Category.ADMISSIONS_FEES, Category.INFRA_CAMPUS
        )
        result = hybrid.classify(query)
        assert result.category == Category.GENERAL
        assert not result.is_specific

    def test_scores_are_reported(self, hybrid, stub_encoder):
        query = "What is the syllabus of computer engineering?"
        stub_encoder.table[query] = vector_at(Category.ACADEMICS, 0.55)
        result = hybrid.classify(query)
        assert isinstance(result, IntentResult)
        assert result.scores
        assert max(result.scores.values()) == pytest.approx(result.confidence, abs=0.01)

    def test_seed_queries_classify_to_their_own_category(self, hybrid):
        """Every seed query must round-trip to its own category."""
        for category, spec in CATEGORY_SPECS.items():
            if category in (Category.GENERAL, Category.OUT_OF_SCOPE):
                continue
            for seed in spec.seed_queries:
                result = hybrid.classify(seed)
                assert result.category == category, (
                    f"{seed!r} → {result.category} (expected {category}); "
                    f"scores={result.scores}"
                )


# ─────────────────────────────────────────────────────────────────────────────
# Query rewriting
# ─────────────────────────────────────────────────────────────────────────────

class TestExpandQuery:
    def test_abbreviation_expanded(self):
        expanded = expand_query("Who is the HOD of IT?")
        assert "head of department" in expanded
        assert "HOD" in expanded  # original wording preserved

    def test_multiple_abbreviations(self):
        expanded = expand_query("CDC placement drive after CAP round for PG students")
        assert "career development cell" in expanded
        assert "centralised admission process" in expanded
        assert "post graduate" in expanded

    def test_plain_query_unchanged(self):
        q = "What is the fee structure?"
        assert expand_query(q) == q

    def test_no_partial_word_matches(self):
        assert expand_query("I need a CHOD boss") == "I need a CHOD boss"

    def test_capped_expansions(self):
        query = " ".join(ABBREVIATIONS)  # every abbreviation at once
        expanded = expand_query(query, max_expansions=2)
        assert expanded.count("(") == 1
        assert len(expanded.split("; ")) <= 3
