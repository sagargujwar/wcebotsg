"""
tests/test_syllabus.py
──────────────────────
Unit tests for syllabus-aware retrieval (app/syllabus.py).

Covers the query side (branch / year-of-study / academic-year parsing, and
the non-syllabus queries that must stay silent), the document side (filename
parsing incl. the terse exports and the "TY-…-final" adjective trap), and
find_syllabus_chunks end-to-end against a stub collection — including the
regression where shared assessment boilerplate outranked the named document.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.syllabus import (  # noqa: E402
    _parse_ay,
    _parse_yos,
    find_syllabus_chunks,
    is_syllabus_doc,
    parse_syllabus_query,
    reset_syllabus_index_cache,
)


# ── Stub collection ──────────────────────────────────────────────────────────

def _chunk(text: str, url: str, **meta) -> dict:
    base = {
        "source_url": url,
        "doc_type": "pdf",
        "title": "",
        "category": "academics",
    }
    base.update(meta)
    return {"text": text, "metadata": base}


class StubCollection:
    """Minimal Chroma stand-in: find_syllabus_chunks only calls .get()."""

    def __init__(self, chunks: list[dict]):
        self._chunks = chunks

    def get(self, include=None, **_kwargs):
        return {
            "documents": [c["text"] for c in self._chunks],
            "metadatas": [c["metadata"] for c in self._chunks],
        }


BASE = "https://walchandsangli.ac.in/wp-content/uploads/"
TY_CSE = BASE + "5.-TY-CSE-Syllabus-25-26-CC.pdf"
SY_CSE = BASE + "4.-SY-Btech-CSE-25-26-CC.pdf"
FY_CSE = BASE + "5.-FY-CSE-26_27.pdf"
FINAL_CSE = BASE + "5.-CSE-Final-Year-CC-AY-26-27.pdf"
TY_MTECH_CSE = BASE + "FY-MTech_CSE_CC-2026-2027-onwards-01_7_26.pdf"
GENERIC = BASE + "Course_Contents-19-to-23.pdf"

TY_HEADER = (
    "Course Information\nProgramme B.Tech. (Computer Science and Engineering)\n"
    "Class, Semester Third Year B. Tech., Sem V\nCourse Code 7CS301\n"
    "Course Name Compiler Design\nCredits 4\nCourse Outcomes: describe phases "
    "of compilation, design lexer and parser algorithms for a compiler."
)
TY_BODY = (
    "Course Code 7CS302 Course Name Operating Systems Credits 4. Scheduling "
    "algorithms, deadlock handling, memory management and file systems."
)
SY_HEADER = (
    "Course Information Programme B.Tech. (Computer Science and Engineering) "
    "Class, Semester Second Year B. Tech., Sem III Course Code 5CS301 Course "
    "Name Data Structures Credits 4 Course Outcomes: analyse linear and "
    "non-linear data structures."
)
GENERIC_BOILERPLATE = (
    "Assessment The assessment of every course is based on MSE, ISE and ESE. "
    "Minimum passing marks in each head of examination as per university norms."
)


@pytest.fixture(autouse=True)
def _fresh_syllabus_cache():
    """The syllabus-doc index is process-global; isolate every test."""
    reset_syllabus_index_cache()
    yield
    reset_syllabus_index_cache()


@pytest.fixture
def kb_stub() -> StubCollection:
    """A tiny KB reproducing the live layout: real branch syllabus PDFs plus
    the branch-less boilerplate doc that outranks them on raw similarity."""
    return StubCollection(
        [
            _chunk(
                "3/2/26, 3:15 PM " + TY_HEADER,
                TY_CSE,
                title="5. Ty Cse Syllabus 25 26 Cc",
                chunk_index=0,
            ),
            _chunk(TY_BODY, TY_CSE, title="5. Ty Cse Syllabus 25 26 Cc", chunk_index=1),
            _chunk(
                SY_HEADER,
                SY_CSE,
                title="4. Sy Btech Cse 25 26 Cc",
                chunk_index=0,
            ),
            _chunk(
                "Course Information Programme B.Tech. Computer Science and "
                "Engineering First Year Sem I Course Code 1CS301 Course Name "
                "Engineering Graphics Credits 3 Course Outcomes: draw orthographic "
                "views.",
                FY_CSE,
                title="5. Fy Cse 26 27",
                chunk_index=0,
            ),
            _chunk(
                "Programme B.Tech. Computer Science and Engineering Final Year "
                "Sem VII Course Code 8CS301 Course Name Machine Learning Credits "
                "4 Course Outcomes: implement supervised learning algorithms.",
                FINAL_CSE,
                title="5. Cse Final Year Cc Ay 26 27",
                chunk_index=0,
            ),
            _chunk(
                "Programme M.Tech. Computer Science and Engineering Sem I Course "
                "Code MCSE301 Course Name Advanced Machine Learning Credits 4 "
                "Course Outcomes: critique research papers in machine learning.",
                TY_MTECH_CSE,
                title="Fy Mtech Cse Cc 2026 2027 Onwards",
                chunk_index=0,
            ),
            _chunk(GENERIC_BOILERPLATE, GENERIC, title="Course Contents 19 To 23"),
            _chunk(
                "The annual tuition fee for General category B.E. students at WCE "
                "is ₹1,09,500.",
                BASE + "fees",
                doc_type="html",
                title="Fee Structure",
                category="admissions_fees",
            ),
        ]
    )


# ── Query parsing ────────────────────────────────────────────────────────────

class TestParseSyllabusQuery:
    def test_branch_year_and_level(self):
        cons = parse_syllabus_query("current third year syllabus for cse")
        assert cons is not None
        assert "cse" in cons["dept"]
        assert cons["yos"] == 3
        assert cons["level"] == "btech"

    def test_academic_year_parsed(self):
        cons = parse_syllabus_query("cse syllabus for 2026-2027")
        assert cons is not None
        assert "cse" in cons["dept"]
        assert cons["ay"] == (26, 27)
        assert cons["yos"] is None

    def test_short_academic_year(self):
        cons = parse_syllabus_query("sy cse syllabus 25-26")
        assert cons is not None
        assert cons["yos"] == 2
        assert cons["ay"] == (25, 26)

    def test_computer_engineering_alias(self):
        cons = parse_syllabus_query("What is the syllabus of Computer Engineering?")
        assert cons is not None
        assert "cse" in cons["dept"]

    def test_mechanical_second_year(self):
        cons = parse_syllabus_query(
            "List the subjects in second year mechanical engineering"
        )
        assert cons is not None
        assert "mech" in cons["dept"]
        assert cons["yos"] == 2

    def test_mtech_level_detected(self):
        cons = parse_syllabus_query("mtech cse syllabus")
        assert cons is not None
        assert cons["level"] == "mtech"

    def test_pronoun_it_does_not_target_it_branch(self):
        # The English pronoun must not select the I.T. department.
        assert parse_syllabus_query("is it possible to get the syllabus") is None

    def test_it_branch_with_context_still_works(self):
        cons = parse_syllabus_query("syllabus for it")
        assert cons is not None
        assert "it" in cons["dept"]

    def test_non_syllabus_query_is_silent(self):
        assert parse_syllabus_query("who is the principal") is None
        assert parse_syllabus_query("what are the admission fees") is None

    def test_branchless_syllabus_query_is_silent(self):
        # Ambiguous across departments → baseline retrieval owns it.
        assert parse_syllabus_query("What subjects are taught in B.Tech first year?") is None

    def test_syllabus_word_without_branch_is_silent(self):
        assert parse_syllabus_query("tell me about the syllabus") is None


class TestParsers:
    def test_ay_long_form(self):
        assert _parse_ay("5.-TY-CSE-Syllabus-AY-2025-26-CC.pdf") == (25, 26)

    def test_ay_short_form(self):
        assert _parse_ay("5.-TY-CSE-Syllabus-25-26-CC.pdf") == (25, 26)

    def test_ay_underscore_form(self):
        assert _parse_ay("5.-FY-CSE-26_27.pdf") == (26, 27)

    def test_ay_span_is_not_an_academic_year(self):
        assert _parse_ay("Course_Contents-19-to-23.pdf") is None

    def test_ay_ignores_trailing_date(self):
        assert _parse_ay("5.-Final-year-CSE-Syllabus-AY-2024-25-18_7_24.pdf") == (24, 25)

    def test_yos_first_occurrence_wins(self):
        # "final" here is a filename adjective; the doc is T.Y.
        assert _parse_yos("ty aiml syllabus with codes final") == 3

    def test_yos_final_year(self):
        assert _parse_yos("final year cse 24 25 cc") == 4

    def test_yos_dotted_forms(self):
        assert _parse_yos("t y b tech mech 2025 26 cc") == 3
        assert _parse_yos("s y m tech str syllabus") == 2


# ── Document predicate ───────────────────────────────────────────────────────

class TestIsSyllabusDoc:
    def test_named_syllabus_doc(self):
        assert is_syllabus_doc(TY_CSE, "5. Ty Cse Syllabus 25 26 Cc")

    def test_terse_export_with_branch_and_year(self):
        assert is_syllabus_doc(FY_CSE, "5. Fy Cse 26 27")

    def test_course_content_underscored(self):
        assert is_syllabus_doc(
            BASE + "4.-SY_B.Tech_.-Electrical-WP-_Course-content_AY-2026-27.pdf"
        )

    def test_branchless_boilerplate_rejected(self):
        assert not is_syllabus_doc(GENERIC, "Course Contents 19 To 23")
        assert not is_syllabus_doc(BASE + "MDM-CC-25-26.pdf", "Mdm Cc 25 26")
        assert not is_syllabus_doc(BASE + "Honors-2526-CC.pdf", "Honors 2526 Cc")

    def test_non_syllabus_docs_rejected(self):
        assert not is_syllabus_doc(BASE + "SDP1.pdf", "Sdp1")
        assert not is_syllabus_doc(BASE + "fees", "Fee Structure")
        assert not is_syllabus_doc("" )


# ── find_syllabus_chunks ─────────────────────────────────────────────────────

class TestFindSyllabusChunks:
    def test_third_year_cse_query_targets_ty_doc(self, kb_stub):
        hits = find_syllabus_chunks(kb_stub, "current third year syllabus for cse")
        assert hits, "expected targeted chunks"
        assert all(h["source_url"] == TY_CSE for h in hits)
        assert all(h["score"] == 1.0 for h in hits)
        assert any("Compiler Design" in h["text"] for h in hits)

    def test_header_chunk_leads(self, kb_stub):
        hits = find_syllabus_chunks(kb_stub, "third year cse syllabus")
        assert "Course Name Compiler Design" in hits[0]["text"]

    def test_caps_injected_chunks(self, kb_stub):
        hits = find_syllabus_chunks(kb_stub, "third year cse syllabus", max_chunks=1)
        assert len(hits) == 1

    def test_academic_year_query_picks_matching_doc(self, kb_stub):
        hits = find_syllabus_chunks(kb_stub, "cse syllabus for 2026-2027")
        assert hits
        # FY-CSE-26_27 and CSE-Final-Year-CC-AY-26-27 both match 26-27;
        # deterministic tie-break must pick one of them — never another year.
        assert hits[0]["source_url"] in {FY_CSE, FINAL_CSE}

    def test_second_year_query_picks_sy_cse(self, kb_stub):
        hits = find_syllabus_chunks(kb_stub, "second year cse syllabus 25-26")
        assert hits
        assert hits[0]["source_url"] == SY_CSE
        assert "Data Structures" in hits[0]["text"]

    def test_year_of_study_outranks_academic_year(self, kb_stub):
        # T.Y. requested; only TY_CSE carries yos=3 even though its AY (25-26)
        # differs from nothing the query stated.
        hits = find_syllabus_chunks(kb_stub, "third year cse syllabus")
        assert hits[0]["source_url"] == TY_CSE

    def test_mtech_query_excludes_btech_docs(self, kb_stub):
        hits = find_syllabus_chunks(kb_stub, "mtech cse syllabus")
        assert hits
        assert hits[0]["source_url"] == TY_MTECH_CSE

    def test_btech_query_excludes_mtech_docs(self, kb_stub):
        hits = find_syllabus_chunks(kb_stub, "cse syllabus for 2026-2027")
        assert hits
        assert hits[0]["source_url"] != TY_MTECH_CSE

    def test_boilerplate_doc_never_targeted(self, kb_stub):
        hits = find_syllabus_chunks(kb_stub, "cse syllabus for 2026-2027")
        assert all(h["source_url"] != GENERIC for h in hits)

    def test_non_syllabus_query_returns_empty(self, kb_stub):
        assert find_syllabus_chunks(kb_stub, "who is the principal") == []

    def test_branchless_query_returns_empty(self, kb_stub):
        assert (
            find_syllabus_chunks(
                kb_stub, "What subjects are taught in B.Tech first year?"
            )
            == []
        )

    def test_pronoun_it_returns_empty(self, kb_stub):
        assert find_syllabus_chunks(kb_stub, "is it possible to get the syllabus") == []

    def test_unknown_branch_returns_empty(self, kb_stub):
        assert find_syllabus_chunks(kb_stub, "astronomy syllabus") == []

    def test_result_shape(self, kb_stub):
        hits = find_syllabus_chunks(kb_stub, "third year cse syllabus")
        for h in hits:
            assert set(h) >= {"text", "source_url", "doc_type", "title", "score"}
            assert isinstance(h["score"], float)

    def test_empty_collection(self):
        assert find_syllabus_chunks(StubCollection([]), "third year cse syllabus") == []

    def test_index_is_cached_across_calls(self, kb_stub):
        find_syllabus_chunks(kb_stub, "third year cse syllabus")
        # Second call must reuse the cached index (same object identity).
        from app.syllabus import _SYLLABUS_DOCS_CACHE

        first = _SYLLABUS_DOCS_CACHE
        assert first is not None
        find_syllabus_chunks(kb_stub, "second year cse syllabus")
        assert _SYLLABUS_DOCS_CACHE is first
