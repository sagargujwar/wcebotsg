"""
tests/test_senses.py
────────────────────
Tests for the context-meaning (word-sense) layer: the CONTEXT_SENSES table
with active_senses()/sense_matches() in app/categories.py, and the re-rank
that consumes them (app/retrieval.py: apply_sense_rerank).

The layer answers "what does this word actually mean *here*": a student's
"syllabus" is the document built from course contents, course codes and
credits — not the word's dictionary sense.  The meaning is applied to the
chunks retrieval already found, and only to their *order*, so these tests pin
both halves of the contract:

  • the right document moves to the front (the LLM reads and cites it first), and
  • nothing else can move at all — scores, membership and refusal gate stay
    exactly as they were.

Fully offline: no encoder, no Chroma, no network.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.categories import (  # noqa: E402
    CONTEXT_SENSES,
    active_senses,
    expand_query,
    sense_matches,
)
from app.retrieval import apply_sense_rerank  # noqa: E402
from app.syllabus import parse_syllabus_query  # noqa: E402


def _chunk(text: str, score: float, url: str = "http://test.edu/doc") -> dict:
    return {
        "text": text,
        "source_url": url,
        "doc_type": "html",
        "title": "",
        "category": "",
        "score": score,
    }


# A chunk that *is* a syllabus document (the words such PDFs are built from) …
SYLLABUS_TEXT = (
    "Course Code 7CS301 Course Name Compiler Design Credits 4 "
    "Course Contents: parsing, lexing and code generation."
)
# … and a chunk that merely repeats the trigger word, as big reports do.
REPORT_TEXT = (
    "The Institute follows a semester based syllabus. The syllabus is "
    "approved by the academic council and reviewed every year."
)


# ─────────────────────────────────────────────────────────────────────────────
# Reading the meaning out of a query
# ─────────────────────────────────────────────────────────────────────────────

class TestActiveSenses:
    def test_bare_trigger(self):
        assert active_senses("What is the syllabus?") == ["syllabus"]

    @pytest.mark.parametrize("query", ["latest notice", "latest notices"])
    def test_singular_and_plural_both_trigger(self, query):
        assert active_senses(query) == ["notice"]

    @pytest.mark.parametrize(
        "query",
        [
            "feedback about the faculty",   # "fee" inside "feedback"
            "examination schedule please",  # "exam" inside "examination"
            "What is the capital of France?",
            "",
        ],
    )
    def test_no_false_positives(self, query):
        assert active_senses(query) == []

    def test_several_triggers_in_table_order(self):
        assert active_senses("hostel fees and placement") == ["fee", "placement", "hostel"]

    def test_composes_with_abbreviation_expansion(self):
        # Abbreviation full forms never invent a sense of their own.
        expanded = expand_query("hod of it")
        assert active_senses(expanded) == []
        assert active_senses(expand_query("ty syllabus")) == ["syllabus"]


# ─────────────────────────────────────────────────────────────────────────────
# Recognising the meaning in a document
# ─────────────────────────────────────────────────────────────────────────────

class TestSenseMatches:
    def test_document_that_is_the_thing_scores_high(self):
        assert sense_matches(SYLLABUS_TEXT, ["syllabus"]) >= 3

    def test_document_that_only_repeats_the_word_scores_low(self):
        # Repeating "syllabus" is not what syllabus means here.
        assert sense_matches(REPORT_TEXT, ["syllabus"]) == 0

    def test_case_insensitive(self):
        assert sense_matches("COURSE CONTENTS and COURSE CODE", ["syllabus"]) == 2

    def test_shared_phrases_count_once(self):
        # "cutoff" and "cut off" carry the same phrase list.
        assert sense_matches("cutoff", ["cutoff", "cut off"]) == 1

    def test_unknown_senses_and_empty_text(self):
        assert sense_matches("Course Code 7CS301", []) == 0
        assert sense_matches("", ["syllabus"]) == 0


# ─────────────────────────────────────────────────────────────────────────────
# The re-rank
# ─────────────────────────────────────────────────────────────────────────────

class TestSenseRerank:
    def test_promotes_the_chunk_that_is_the_thing(self):
        results = [_chunk(REPORT_TEXT, 0.80, "http://test.edu/annual-report"),
                   _chunk(SYLLABUS_TEXT, 0.78, "http://test.edu/syllabus.pdf")]
        out = apply_sense_rerank(results, "syllabus")
        assert out[0]["source_url"] == "http://test.edu/syllabus.pdf"

    def test_membership_is_untouched(self):
        # Answer coverage (hit@6) cannot change: nothing enters or leaves.
        results = [_chunk(REPORT_TEXT, 0.80, "http://a"),
                   _chunk(SYLLABUS_TEXT, 0.78, "http://b"),
                   _chunk("Library reading room timings 9 to 5.", 0.70, "http://c")]
        out = apply_sense_rerank(results, "syllabus")
        assert sorted(c["source_url"] for c in out) == ["http://a", "http://b", "http://c"]
        assert len(out) == 3

    def test_scores_are_untouched(self):
        # The similarity gate reads the best score; it must not move.
        results = [_chunk(REPORT_TEXT, 0.80, "http://a"),
                   _chunk(SYLLABUS_TEXT, 0.78, "http://b")]
        out = apply_sense_rerank(results, "syllabus")
        assert sorted(c["score"] for c in out) == [0.78, 0.80]

    def test_query_without_ambiguous_word_is_untouched(self):
        results = [_chunk(REPORT_TEXT, 0.80, "http://a"),
                   _chunk(SYLLABUS_TEXT, 0.78, "http://b")]
        assert apply_sense_rerank(results, "Who is the principal?") == results

    def test_empty_results(self):
        assert apply_sense_rerank([], "syllabus") == []

    def test_too_far_from_the_best_is_not_promoted(self):
        # Outside _BOOST_BAND the meaning must not reach: the prior breaks
        # near-ties only, it cannot drag a topically distant chunk forward.
        results = [_chunk(REPORT_TEXT, 0.80, "http://a"),
                   _chunk(SYLLABUS_TEXT, 0.50, "http://b")]
        out = apply_sense_rerank(results, "syllabus")
        assert out[0]["source_url"] == "http://a"

    def test_one_phrase_is_not_enough(self):
        # A passing mention does not make a document the thing.
        results = [_chunk(REPORT_TEXT, 0.80, "http://a"),
                   _chunk("Syllabus contents for the semester.", 0.78, "http://b")]
        out = apply_sense_rerank(results, "syllabus")
        assert out[0]["source_url"] == "http://a"

    def test_targeted_chunks_stay_in_front(self):
        # Person / syllabus document targeting lands chunks at score 1.0;
        # a sense bonus is clamped so it can tie them but never pass them.
        results = [
            _chunk("Name : Pujari Siddharaj Dattatray", 1.0, "http://a/profile.pdf"),
            _chunk("Fee structure tuition instalment", 0.98, "http://b/fees"),
        ]
        out = apply_sense_rerank(results, "which fee applies")
        assert out[0]["source_url"] == "http://a/profile.pdf"

    def test_stable_for_ties(self):
        first = _chunk("Fee structure tuition instalment", 0.78, "http://a")
        second = _chunk("Fee structure tuition instalment", 0.78, "http://b")
        out = apply_sense_rerank([first, second], "fees")
        assert [c["source_url"] for c in out] == ["http://a", "http://b"]


# ─────────────────────────────────────────────────────────────────────────────
# Safety: a meaning may only reorder — it can never invent a constraint
# ─────────────────────────────────────────────────────────────────────────────

class TestMeaningTableSafety:
    @pytest.mark.parametrize("key", sorted(CONTEXT_SENSES))
    def test_phrases_never_target_a_branch_document(self, key):
        # A meaning must not name a branch, or a branch-agnostic query
        # ("syllabus") would be steered at one department's document.
        assert parse_syllabus_query(" ".join(CONTEXT_SENSES[key])) is None

    @pytest.mark.parametrize("key", sorted(CONTEXT_SENSES))
    def test_phrases_have_no_year_or_digits(self, key):
        for phrase in CONTEXT_SENSES[key]:
            assert not any(ch.isdigit() for ch in phrase), phrase  # no "2025-26"

    @pytest.mark.parametrize("key", sorted(CONTEXT_SENSES))
    def test_phrases_are_not_proper_nouns(self, key):
        for phrase in CONTEXT_SENSES[key]:
            for word in phrase.split():
                # No Title-case word ("Walchand") — acronyms are fine.
                assert not (word[:1].isupper() and word[1:].islower()), word

    @pytest.mark.parametrize("key", sorted(CONTEXT_SENSES))
    def test_every_meaning_has_phrases(self, key):
        assert CONTEXT_SENSES[key], key
