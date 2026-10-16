"""
tests/test_people.py
────────────────────
Unit tests for person-aware retrieval (app/people.py).

Covers the query side (name terms vs role queries), the document side
(person-doc detection, subject extraction incl. the PDF quirks seen in this
KB), and find_person_chunks end-to-end against a stub collection — including
the regression where a co-author mention in someone else's CV out-ranked the
queried person's own profile.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.people import (  # noqa: E402
    _covers,
    _is_person_doc,
    _subject_terms,
    find_person_chunks,
    name_terms,
    reset_person_doc_cache,
)


# ── Stub collection ──────────────────────────────────────────────────────────

def _chunk(text: str, url: str, **meta) -> dict:
    base = {
        "source_url": url,
        "doc_type": "pdf",
        "title": "",
        "category": "general",
    }
    base.update(meta)
    return {"text": text, "metadata": base}


class StubCollection:
    """Minimal Chroma stand-in: find_person_chunks only calls .get()."""

    def __init__(self, chunks: list[dict]):
        self._chunks = chunks

    def get(self, include=None, **_kwargs):
        return {
            "documents": [c["text"] for c in self._chunks],
            "metadatas": [c["metadata"] for c in self._chunks],
        }


SDP1 = "https://walchandsangli.ac.in/wp-content/uploads/SDP1.pdf"
ATUL = "https://walchandsangli.ac.in/wp-content/uploads/Mr.-Atul-Patane-Resume.pdf"
AISHE = "https://walchandsangli.ac.in/wp-content/uploads/AISHE-2024-25-Report.pdf"


@pytest.fixture(autouse=True)
def _fresh_person_cache():
    """The person-doc index is process-global; isolate every test."""
    reset_person_doc_cache()
    yield
    reset_person_doc_cache()


@pytest.fixture
def kb_stub() -> StubCollection:
    """A tiny KB reproducing the live layout: one person's profile, one
    colleague's CV that *cites* them as a co-author, and an aggregate report."""
    return StubCollection(
        [
            _chunk(
                "3/2/26, 3:15 PM Faculty_Profile\nProfile of Mr. Pujari Siddharaj "
                "Dattatray\nName : Mr. Pujari Siddharaj Dattatray\nDepartment : "
                "Computer Engineering\nDate of Birth : 12-05-1988\nDesignation : "
                "Assistant Professor\nOfficial E-mail : sdp@walchandsangli.ac.in",
                SDP1,
            ),
            _chunk(
                "Publications\n1. Pujari S. D., \"Detection of Autism Spectrum "
                "Disorder Using Machine Learning\", 2024.",
                SDP1,
            ),
            _chunk(
                "ID : 579562\nPrevious Experience : 1. Organization Name :- Tata "
                "Consultancy Services, Designation :- System Engineer",
                SDP1,
            ),
            _chunk(
                ":: Personal Details:\nName Atul Siddheshwar Patane\nDate of Birth : "
                "15th October 1990\nObjective: To work with a reputed institution",
                ATUL,
            ),
            _chunk(
                "Publications\n1. Patane A. S., Bhosale S., Pujari S. D., \"Edge "
                "computing for IoT\", 2023.",
                ATUL,
            ),
            _chunk(
                "AISHE-2024-25 Report\nFaculty counts by department\nComputer "
                "Engineering : 42\nTotal students : 3100",
                AISHE,
            ),
        ]
    )


# ── name_terms ───────────────────────────────────────────────────────────────

class TestNameTerms:
    def test_full_name_yields_name_words(self):
        assert name_terms("who is siddharaj pujari") == ["siddharaj", "pujari"]

    def test_middle_initial_dropped(self):
        assert name_terms("tell me about siddharaj d pujari") == ["siddharaj", "pujari"]

    def test_role_query_yields_nothing(self):
        assert name_terms("who is the principal of wce") == []

    def test_role_query_hod_yields_nothing(self):
        assert name_terms("who is the head of computer engineering department") == []

    def test_department_words_are_stopped(self):
        assert name_terms("who is professor patane from mechanical") == ["patane"]

    def test_titles_stripped(self):
        assert name_terms("who is dr rashmi jadhav") == ["rashmi", "jadhav"]

    def test_dedupes_repeats(self):
        assert name_terms("who is pujari siddharaj pujari") == ["pujari", "siddharaj"]

    def test_empty_query(self):
        assert name_terms("") == []


# ── Document-side detection ──────────────────────────────────────────────────

class TestIsPersonDoc:
    @pytest.mark.parametrize(
        "url",
        [
            SDP1,
            "https://x.ac.in/uploads/Mr.-Atul-Patane-Resume.pdf",
            "https://x.ac.in/uploads/CV-SDR.pdf",
            "https://x.ac.in/uploads/Faculty_Profile_Anant-J-Umbarkar.pdf",
            "https://x.ac.in/uploads/Profile_NLG.pdf",
            "https://x.ac.in/uploads/Souvik-Barat-CV-2025.pdf",
        ],
    )
    def test_person_files(self, url):
        assert _is_person_doc(url)

    @pytest.mark.parametrize(
        "url",
        [
            AISHE,
            "https://x.ac.in/faculty/",
            "https://x.ac.in/uploads/NIRF-22-2023.pdf",
            "",
        ],
    )
    def test_non_person_files(self, url):
        assert not _is_person_doc(url)

    def test_text_guard_rejects_profile_named_non_resume(self):
        # Filename says profile, but the body is a syllabus: no bio markers.
        assert not _is_person_doc(
            "https://x.ac.in/Profile_CS.pdf",
            text="Course outcome 1: apply linear algebra to solve problems.",
        )

    def test_text_guard_accepts_real_bio(self):
        assert _is_person_doc(
            "https://x.ac.in/Profile_CS.pdf",
            text="Designation: Assistant Professor. Date of Birth: 01-01-1990.",
        )


# ── Subject extraction ───────────────────────────────────────────────────────

class TestSubjectTerms:
    def test_profile_of_line(self):
        text = "Faculty_Profile\nProfile of Mr. Pujari Siddharaj Dattatray\nDept"
        terms = _subject_terms("https://x/SDP1.pdf", text)
        assert {"pujari", "siddharaj", "dattatray"} <= set(terms)

    def test_name_without_colon(self):
        # Atul's CV: "Name Atul Siddheshwar Patane" (no colon after Name).
        text = ":: Personal Details:\nName Atul Siddheshwar Patane\nCity : Sangli"
        terms = _subject_terms("https://x/Mr.-Atul-Patane-Resume.pdf", text)
        assert {"atul", "siddheshwar", "patane"} <= set(terms)

    def test_camel_joined_name_split(self):
        # Rashmi's PDF extracted as "RashmiJadhav" (space lost).
        text = "PERSONAL PARTICULARS:\nName : Dr. RashmiJadhav\nAddress : Kolhapur"
        terms = _subject_terms("https://x/Dr.-Rashmi-Jadhav-Resume.pdf", text)
        assert {"rashmi", "jadhav"} <= set(terms)

    def test_colon_on_following_line(self):
        text = "PERSONAL PARTICULARS:\nName\n: Dr. RashmiJadhav\nPresent Address"
        terms = _subject_terms("https://x/Dr.-Rashmi-Jadhav-Resume.pdf", text)
        assert {"rashmi", "jadhav"} <= set(terms)

    def test_name_of_board_header_ignored(self):
        text = "Name of Board/Institution/Year of passing\nShivaji University 2011"
        terms = _subject_terms("https://x/Profile_XYZ.pdf", text)
        assert "board" not in terms
        assert "institution" not in terms

    def test_filename_contributes_terms(self):
        terms = _subject_terms(
            "https://x/Faculty_Profile_Anant-J-Umbarkar.pdf", text=""
        )
        assert {"anant", "umbarkar"} <= set(terms)

    def test_filename_doc_words_excluded(self):
        terms = _subject_terms("https://x/Mr.-Atul-Patane-Resume.pdf", text="")
        assert "resume" not in terms
        assert "cv" not in terms

    def test_stopwords_excluded(self):
        text = "Profile of Dr. College of Engineering"
        terms = _subject_terms("https://x/Profile_X.pdf", text)
        assert "college" not in terms


class TestCovers:
    def test_exact(self):
        assert _covers(["siddharaj", "pujari"], "pujari")

    def test_query_prefix_of_subject(self):
        # query "anant" vs subject word "anantkumar"
        assert _covers(["umbarkar", "anantkumar", "janardan"], "anant")

    def test_subject_prefix_of_query(self):
        assert _covers(["anant"], "anantkumar")

    def test_no_match(self):
        assert not _covers(["atul", "siddheshwar", "patane"], "siddharaj")

    def test_empty_subject(self):
        assert not _covers([], "pujari")


# ── find_person_chunks ───────────────────────────────────────────────────────

class TestFindPersonChunks:
    def test_profile_wins_over_citing_colleague(self, kb_stub):
        """Regression: Atul's CV cites Pujari as co-author; the query must
        still resolve to Pujari's own profile (SDP1.pdf)."""
        hits = find_person_chunks(kb_stub, "who is siddharaj pujari")
        assert hits, "expected person hits"
        assert hits[0]["source_url"] == SDP1

    def test_query_for_the_citing_colleague(self, kb_stub):
        hits = find_person_chunks(kb_stub, "who is atul patane")
        assert hits
        assert hits[0]["source_url"] == ATUL

    def test_role_query_returns_empty(self, kb_stub):
        assert find_person_chunks(kb_stub, "who is the principal of wce") == []

    def test_unknown_person_returns_empty(self, kb_stub):
        assert find_person_chunks(kb_stub, "who is sachin tendulkar") == []

    def test_aggregate_report_never_returned(self, kb_stub):
        hits = find_person_chunks(kb_stub, "who is siddharaj pujari")
        assert all(h["source_url"] != AISHE for h in hits)

    def test_chunks_capped_at_max(self, kb_stub):
        hits = find_person_chunks(kb_stub, "who is siddharaj pujari", max_chunks=2)
        assert len(hits) == 2

    def test_max_chunks_one_returns_bio_only(self, kb_stub):
        hits = find_person_chunks(kb_stub, "who is siddharaj pujari", max_chunks=1)
        assert len(hits) == 1
        assert "Profile of" in hits[0]["text"]

    def test_injected_chunks_score_one(self, kb_stub):
        hits = find_person_chunks(kb_stub, "who is siddharaj pujari")
        assert all(h["score"] == 1.0 for h in hits)

    def test_result_shape_matches_retriever(self, kb_stub):
        hits = find_person_chunks(kb_stub, "who is atul patane")
        for h in hits:
            assert set(h) >= {"text", "source_url", "doc_type", "title", "category", "score"}

    def test_markerless_cv_still_returned(self, kb_stub):
        # A resume without "Designation/Date of Birth" markers must not be
        # dropped once its subject matches (Souvik Barat / CV-AAA cases).
        stub = StubCollection(
            [
                _chunk(
                    "Dr Souvik Barat\nDistinguished Principal Scientist, Tata "
                    "Consultancy Services\nWork Experience\n1. Scientist",
                    "https://x/Souvik-Barat-CV-2025.pdf",
                ),
                _chunk(
                    "Publications\n1. Barat, S., Kulkarni, V., 2020.",
                    "https://x/Souvik-Barat-CV-2025.pdf",
                ),
            ]
        )
        hits = find_person_chunks(stub, "who is souvik barat")
        assert hits
        assert hits[0]["source_url"].endswith("Souvik-Barat-CV-2025.pdf")

    def test_empty_kb(self):
        assert find_person_chunks(StubCollection([]), "who is siddharaj pujari") == []

    def test_collection_get_failure_is_swallowed(self):
        class Boom:
            def get(self, **_kwargs):
                raise RuntimeError("chroma down")

        assert find_person_chunks(Boom(), "who is siddharaj pujari") == []

    def test_name_only_in_filename_not_returned(self, kb_stub):
        # Profile file whose body never mentions the queried person: the
        # filename supplies the subject, but we must not serve chunks that
        # contain no trace of them.
        stub = StubCollection(
            [
                _chunk(
                    "Fee structure for AY 2025-26\nTuition fee : 1,00,000",
                    "https://x/Profile_ABC.pdf",
                ),
            ]
        )
        assert find_person_chunks(stub, "who is anita deshpande") == []

    def test_cache_reused_across_calls(self, kb_stub):
        first = find_person_chunks(kb_stub, "who is siddharaj pujari")
        # Mutate the underlying store; a second call must not see it.
        kb_stub._chunks.clear()
        second = find_person_chunks(kb_stub, "who is siddharaj pujari")
        assert len(first) == len(second) > 0
