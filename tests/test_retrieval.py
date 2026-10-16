"""
tests/test_retrieval.py
───────────────────────
Tests for ChromaRetriever using an in-memory (ephemeral) Chroma collection.
No real ChromaDB on disk is created; no network or API calls are made.
"""
from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# ── Fixtures ──────────────────────────────────────────────────────────────────

SAMPLE_CHUNKS = [
    {
        "id": "chunk-001",
        "text": "The annual tuition fee for General category B.E. students at WCE is ₹1,09,500.",
        "source_url": "http://test.edu/fees",
        "doc_type": "html",
        "title": "Fee Structure",
    },
    {
        "id": "chunk-002",
        "text": "Data Structures and Algorithms is a core subject in Semester III of B.E. CSE at WCE.",
        "source_url": "http://test.edu/syllabus",
        "doc_type": "html",
        "title": "CSE Syllabus",
    },
    {
        "id": "chunk-003",
        "text": "Admissions are conducted through MHT-CET and JEE Main via the CAP process.",
        "source_url": "http://test.edu/admissions",
        "doc_type": "html",
        "title": "Admissions",
    },
    {
        "id": "chunk-004",
        "text": "SC/ST students are eligible for a full fee waiver under the Government scholarship.",
        "source_url": "http://test.edu/fees",
        "doc_type": "html",
        "title": "Fee Structure",
    },
]


@pytest.fixture(scope="module")
def retriever():
    """
    Build a ChromaRetriever backed by an in-memory (ephemeral) Chroma instance
    pre-seeded with SAMPLE_CHUNKS.
    """
    import chromadb
    from chromadb.config import Settings as ChromaSettings
    from sentence_transformers import SentenceTransformer

    # Use in-memory Chroma (no disk I/O)
    client = chromadb.EphemeralClient(
        settings=ChromaSettings(anonymized_telemetry=False)
    )
    collection = client.get_or_create_collection(
        name="test_kb",
        metadata={"hnsw:space": "cosine"},
    )

    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    texts = [c["text"] for c in SAMPLE_CHUNKS]
    embeddings = encoder.encode(texts, normalize_embeddings=True).tolist()

    collection.upsert(
        ids=[c["id"] for c in SAMPLE_CHUNKS],
        documents=texts,
        embeddings=embeddings,
        metadatas=[
            {
                "source_url": c["source_url"],
                "doc_type": c["doc_type"],
                "title": c["title"],
                "last_scraped_at": "2025-01-01T00:00:00+00:00",
                "chunk_index": i,
            }
            for i, c in enumerate(SAMPLE_CHUNKS)
        ],
    )

    # Patch ChromaRetriever to use our ephemeral collection
    from app import retrieval as retrieval_module

    mock_retriever = MagicMock(spec=retrieval_module.ChromaRetriever)
    mock_retriever._collection = collection
    mock_retriever._encoder = encoder

    # Wire real search logic
    def real_search(query: str, top_k: int):
        qe = encoder.encode(query, normalize_embeddings=True).tolist()
        results = collection.query(
            query_embeddings=[qe],
            n_results=min(top_k, collection.count()),
            include=["documents", "metadatas", "distances"],
        )
        chunks = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            score = max(0.0, 1.0 - dist / 2.0)
            chunks.append(
                {
                    "text": doc,
                    "source_url": meta.get("source_url", ""),
                    "doc_type": meta.get("doc_type", "html"),
                    "title": meta.get("title", ""),
                    "score": score,
                }
            )
        return chunks

    mock_retriever.search = real_search
    mock_retriever.ping.return_value = True
    return mock_retriever


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestChromaRetriever:
    def test_search_returns_results(self, retriever):
        results = retriever.search("What is the tuition fee?", top_k=4)
        assert len(results) > 0

    def test_search_top_k_respected(self, retriever):
        results = retriever.search("fee", top_k=2)
        assert len(results) <= 2

    def test_search_result_shape(self, retriever):
        results = retriever.search("admissions MHT-CET", top_k=4)
        assert len(results) > 0
        for r in results:
            assert "text" in r
            assert "source_url" in r
            assert "doc_type" in r
            assert "title" in r
            assert "score" in r
            assert 0.0 <= r["score"] <= 1.0

    def test_relevant_result_for_fee_query(self, retriever):
        results = retriever.search("annual tuition fee General category", top_k=4)
        top = results[0]
        # The fee chunk should surface at the top
        assert "fee" in top["text"].lower() or "₹" in top["text"]
        assert top["score"] > 0.3

    def test_relevant_result_for_syllabus_query(self, retriever):
        results = retriever.search("Data Structures semester", top_k=4)
        texts = [r["text"] for r in results]
        assert any("Data Structures" in t for t in texts)

    def test_metadata_source_url_present(self, retriever):
        results = retriever.search("admissions CAP process", top_k=2)
        for r in results:
            assert r["source_url"].startswith("http")

    def test_ping_returns_true(self, retriever):
        assert retriever.ping() is True


class TestUpsertBatching:
    """Regression: Chroma rejects a single upsert above max batch size (≈5461)."""

    @staticmethod
    def _fake_retriever():
        from app.retrieval import ChromaRetriever

        fake = MagicMock(spec=ChromaRetriever)
        fake._collection = MagicMock()
        return fake

    def test_large_upsert_is_sliced(self):
        from app.retrieval import ChromaRetriever

        fake = self._fake_retriever()
        n = 10_001
        ChromaRetriever.upsert(
            fake,
            ids=[f"id-{i}" for i in range(n)],
            documents=["text"] * n,
            embeddings=[[0.1, 0.2]] * n,
            metadatas=[{"source_url": "http://test.edu"}] * n,
        )
        calls = fake._collection.upsert.call_args_list
        sizes = [len(c.kwargs["ids"]) for c in calls]
        assert sizes == [4000, 4000, 2001]
        assert all(s <= 5461 for s in sizes)
        # All parallel lists stay aligned within each batch
        for c in calls:
            assert len(c.kwargs["documents"]) == len(c.kwargs["ids"])
            assert len(c.kwargs["embeddings"]) == len(c.kwargs["ids"])
            assert len(c.kwargs["metadatas"]) == len(c.kwargs["ids"])

    def test_small_upsert_is_single_call(self):
        from app.retrieval import ChromaRetriever

        fake = self._fake_retriever()
        ChromaRetriever.upsert(
            fake,
            ids=["a", "b", "c"],
            documents=["x", "y", "z"],
            embeddings=[[0.0]] * 3,
            metadatas=[{}, {}, {}],
        )
        fake._collection.upsert.assert_called_once()


class TestRerankChunks:
    """Query-time reranking: drop PDF header/footer stubs, cap per-source repeats."""

    @staticmethod
    def _chunk(url: str, text: str, score: float):
        return {
            "text": text,
            "source_url": url,
            "doc_type": "pdf",
            "title": "t",
            "score": score,
        }

    def test_is_stub_detects_header_fragments(self):
        from app.retrieval import _is_stub

        assert _is_stub("Syllabus Prepared By Dr. X | Syllabus Checked By HOD")  # 53 chars
        assert _is_stub(
            "Syllabus Prepared By Dr. A. A. Mohite and Dr. B. N. Jamadar. "
            "Syllabus Checked By Head of Department. "  # < 300 with attrs
        )
        assert not _is_stub("A" * 150)  # long, no attrs → content
        assert not _is_stub(
            "Detailed syllabus with course codes, credits and assessment " * 4
        )

    def test_stubs_dropped_in_favour_of_content(self):
        from app.retrieval import _rerank_chunks

        stub = self._chunk("http://t.edu/a.pdf", "Syllabus Prepared By X", 0.90)
        content = self._chunk(
            "http://t.edu/b.pdf",
            "Detailed course contents with subject names, credits and " * 5,
            0.80,
        )
        out = _rerank_chunks([stub, content], top_k=1)
        assert out == [content]

    def test_per_source_cap(self):
        from app.retrieval import _rerank_chunks

        chunks = [
            self._chunk("http://t.edu/a.pdf", f"Real content chunk {i} " * 20, 0.9 - i * 0.01)
            for i in range(3)
        ] + [
            self._chunk("http://t.edu/b.pdf", "Other content chunk x " * 20, 0.85),
            self._chunk("http://t.edu/b.pdf", "Other content chunk y " * 20, 0.84),
        ]
        out = _rerank_chunks(chunks, top_k=4)
        urls = [c["source_url"] for c in out]
        assert urls.count("http://t.edu/a.pdf") == 2
        assert urls.count("http://t.edu/b.pdf") == 2

    def test_backfill_when_pool_is_all_stubs(self):
        from app.retrieval import _rerank_chunks

        chunks = [
            self._chunk(f"http://t.edu/{i}.pdf", "Syllabus Checked By Y", 0.9 - i * 0.01)
            for i in range(8)
        ]
        out = _rerank_chunks(chunks, top_k=4)
        assert len(out) == 4
        assert out[0]["score"] >= out[-1]["score"]  # score order kept

    def test_short_pool_returned_unchanged(self):
        from app.retrieval import _rerank_chunks

        chunks = [self._chunk("http://t.edu/a.pdf", "Syllabus Prepared By", 0.9)]
        assert _rerank_chunks(chunks, top_k=6) == chunks

    def test_duplicate_text_from_different_urls_dropped(self):
        from app.retrieval import _rerank_chunks

        body = "Identical NAAC report paragraph about courses and credits. " * 10
        chunks = [
            self._chunk("http://t.edu/a.pdf", body, 0.90),
            self._chunk("http://t.edu/b.pdf", body, 0.89),  # same text, other URL
            self._chunk("http://t.edu/c.pdf", body + " extra content here " * 10, 0.88),
        ]
        out = _rerank_chunks(chunks, top_k=2)
        urls = [c["source_url"] for c in out]
        assert "http://t.edu/b.pdf" not in urls
        assert len(set(urls)) == len(urls)


class TestAssembleResults:
    """Category-aware assembly: banded boost + uncorroborated fallback."""

    @staticmethod
    def _chunk(n: int, score: float, category: str = "") -> dict:
        return {
            "text": f"Detailed college document content number {n} with plenty of body text. " * 4,
            "source_url": f"http://t.edu/page-{n}",
            "doc_type": "html",
            "title": f"Page {n}",
            "category": category,
            "score": score,
        }

    def test_empty_pool(self):
        from app.retrieval import assemble_results

        assert assemble_results([], top_k=4, category="admissions_fees") == []

    def test_general_or_none_is_baseline(self):
        from app.retrieval import _rerank_chunks, assemble_results

        pool = [self._chunk(i, 0.9 - i * 0.01, "infra_campus") for i in range(8)]
        baseline = _rerank_chunks(list(pool), top_k=4)
        for cat in (None, "", "general"):
            out = assemble_results(list(pool), top_k=4, category=cat)
            assert [c["source_url"] for c in out] == [c["source_url"] for c in baseline]
            assert [c["score"] for c in out] == [c["score"] for c in baseline]

    def test_boost_promotes_matching_category_within_band(self):
        from app.retrieval import assemble_results

        pool = [
            self._chunk(0, 0.50, "academics"),          # pool best
            self._chunk(1, 0.47, "admissions_fees"),    # within 0.15 band
            self._chunk(2, 0.46, "academics"),
        ]
        out = assemble_results(pool, top_k=3, category="admissions_fees")
        assert out[0]["source_url"] == "http://t.edu/page-1"
        assert out[0]["score"] == pytest.approx(0.47 + 0.06)

    def test_boost_ignored_outside_band(self):
        from app.retrieval import assemble_results

        pool = [
            self._chunk(0, 0.60, "academics"),
            self._chunk(1, 0.30, "admissions_fees"),  # 0.30 gap > 0.15 band
        ]
        out = assemble_results(pool, top_k=2, category="admissions_fees")
        assert out[0]["source_url"] == "http://t.edu/page-0"
        assert out[1]["score"] == pytest.approx(0.30)  # untouched

    def test_uncorroborated_category_falls_back_to_baseline(self):
        from app.retrieval import _rerank_chunks, assemble_results

        pool = [self._chunk(i, 0.5 - i * 0.01, "academics") for i in range(5)]
        baseline = _rerank_chunks(list(pool), top_k=3)
        out = assemble_results(list(pool), top_k=3, category="people_admin")
        assert [c["source_url"] for c in out] == [c["source_url"] for c in baseline]
        assert [c["score"] for c in out] == [c["score"] for c in baseline]

    def test_untagged_chunks_never_boosted(self):
        from app.retrieval import assemble_results

        pool = [self._chunk(i, 0.4, "") for i in range(4)]
        out = assemble_results(pool, top_k=4, category="notices_events")
        assert all(c["score"] == pytest.approx(0.4) for c in out)

    def test_boost_clamped_to_one(self):
        from app.retrieval import assemble_results

        pool = [
            self._chunk(0, 0.99, "admissions_fees"),
            self._chunk(1, 0.98, "admissions_fees"),
        ]
        out = assemble_results(pool, top_k=2, category="admissions_fees")
        assert all(c["score"] <= 1.0 for c in out)


# ── Person-doc targeting (real search() path) ────────────────────────────────

PERSON_CHUNKS = [
    {
        "id": "sdp-0",
        "text": (
            "Faculty_Profile\nProfile of Mr. Pujari Siddharaj Dattatray\n"
            "Name : Mr. Pujari Siddharaj Dattatray\nDepartment : Computer "
            "Engineering\nDate of Birth : 12-05-1988\nDesignation : Assistant "
            "Professor\nOfficial E-mail : sdp@walchandsangli.ac.in"
        ),
        "source_url": "http://test.edu/uploads/SDP1.pdf",
        "title": "Sdp1",
    },
    {
        "id": "sdp-1",
        "text": (
            "Publications\n1. Pujari S. D., \"Detection of Autism Spectrum "
            "Disorder Using Machine Learning\", International Journal, 2024. "
            "The method achieves state-of-the-art accuracy on the benchmark."
        ),
        "source_url": "http://test.edu/uploads/SDP1.pdf",
        "title": "Sdp1",
    },
    {
        "id": "aishe-0",
        "text": (
            "AISHE-2024-25 Report\nFaculty counts by department\nComputer "
            "Engineering : 42 teachers sanctioned, 38 in position. Total "
            "students enrolled across all programmes : 3100."
        ),
        "source_url": "http://test.edu/uploads/AISHE-2024-25-Report.pdf",
        "title": "AISHE",
    },
    {
        "id": "fees-0",
        "text": (
            "The annual tuition fee for General category B.E. students at WCE "
            "is ₹1,09,500. Tuition fee is payable in two instalments per year."
        ),
        "source_url": "http://test.edu/fees",
        "title": "Fee Structure",
    },
]


@pytest.fixture(scope="module")
def person_retriever():
    """A real ChromaRetriever.search backed by an in-memory collection that
    contains a person's profile plus an aggregate report outranking it."""
    import chromadb
    from chromadb.config import Settings as ChromaSettings
    from sentence_transformers import SentenceTransformer

    from app import people as people_module
    from app.retrieval import ChromaRetriever

    client = chromadb.EphemeralClient(
        settings=ChromaSettings(anonymized_telemetry=False)
    )
    collection = client.get_or_create_collection(
        name="test_person_kb", metadata={"hnsw:space": "cosine"}
    )
    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    texts = [c["text"] for c in PERSON_CHUNKS]
    embeddings = encoder.encode(texts, normalize_embeddings=True).tolist()
    collection.upsert(
        ids=[c["id"] for c in PERSON_CHUNKS],
        documents=texts,
        embeddings=embeddings,
        metadatas=[
            {
                "source_url": c["source_url"],
                "doc_type": "pdf" if c["source_url"].endswith(".pdf") else "html",
                "title": c["title"],
                "category": "",
                "chunk_index": i,
            }
            for i, c in enumerate(PERSON_CHUNKS)
        ],
    )

    # Bypass __init__ (which opens the on-disk KB) and wire the test store.
    retriever = ChromaRetriever.__new__(ChromaRetriever)
    retriever._encoder = encoder
    retriever._collection = collection

    people_module.reset_person_doc_cache()
    yield retriever
    people_module.reset_person_doc_cache()


class TestPersonDocTargeting:
    def test_name_query_surfaces_own_profile_first(self, person_retriever):
        results = person_retriever.search("who is siddharaj pujari", top_k=3)
        assert results, "expected results"
        assert results[0]["source_url"].endswith("SDP1.pdf")

    def test_profile_beats_aggregate_report(self, person_retriever):
        # The AISHE report outranks the profile on raw embedding similarity;
        # person targeting must still put the profile in front.
        results = person_retriever.search("tell me about siddharaj d pujari", top_k=4)
        urls = [c["source_url"] for c in results]
        assert any(u.endswith("SDP1.pdf") for u in urls)
        assert urls.index(next(u for u in urls if u.endswith("SDP1.pdf"))) == 0

    def test_role_query_untargeted(self, person_retriever):
        # No name → person path must stay silent (baseline owns the ranking).
        from app.people import find_person_chunks

        assert find_person_chunks(person_retriever._collection, "what is the tuition fee") == []
        results = person_retriever.search("what is the tuition fee", top_k=3)
        assert results

    def test_person_chunks_lead_even_at_top_k_one(self, person_retriever):
        results = person_retriever.search("who is siddharaj pujari", top_k=1)
        assert len(results) == 1
        assert results[0]["source_url"].endswith("SDP1.pdf")

    def test_result_shape_intact(self, person_retriever):
        results = person_retriever.search("who is siddharaj pujari", top_k=3)
        for r in results:
            assert set(r) >= {"text", "source_url", "doc_type", "title", "score"}
            assert isinstance(r["score"], float)


# ── Syllabus-doc targeting (real search() path) ──────────────────────────────

SYLLABUS_CHUNKS = [
    {
        "id": "ty-cse-0",
        "text": (
            "3/2/26, 3:15 PM Course Information\nProgramme B.Tech. (Computer "
            "Science and Engineering)\nClass, Semester Third Year B. Tech., "
            "Sem V\nCourse Code 7CS301\nCourse Name Compiler Design\n"
            "Credits 4\nCourse Outcomes: describe phases of compilation, "
            "design lexer and parser algorithms for a compiler."
        ),
        "source_url": "http://test.edu/uploads/5.-TY-CSE-Syllabus-25-26-CC.pdf",
        "title": "5. Ty Cse Syllabus 25 26 Cc",
    },
    {
        "id": "ty-cse-1",
        "text": (
            "Course Code 7CS302 Course Name Operating Systems Credits 4. "
            "Scheduling algorithms, deadlock handling, memory management and "
            "file systems covered over the semester."
        ),
        "source_url": "http://test.edu/uploads/5.-TY-CSE-Syllabus-25-26-CC.pdf",
        "title": "5. Ty Cse Syllabus 25 26 Cc",
    },
    {
        # Shared assessment boilerplate: near-identical text lives in every
        # syllabus PDF, so raw similarity floods with it — exactly the failure
        # mode this targeting exists to beat.
        "id": "boilerplate-0",
        "text": (
            "Assessment The assessment of every course is based on MSE, ISE "
            "and ESE. Minimum passing marks in each head of examination as "
            "per university norms. Attendance requirement is 75 percent."
        ),
        "source_url": "http://test.edu/uploads/Course_Contents-19-to-23.pdf",
        "title": "Course Contents 19 To 23",
    },
    {
        "id": "fees-0",
        "text": (
            "The annual tuition fee for General category B.E. students at WCE "
            "is ₹1,09,500. Tuition fee is payable in two instalments per year."
        ),
        "source_url": "http://test.edu/fees",
        "title": "Fee Structure",
    },
]


@pytest.fixture(scope="module")
def syllabus_retriever():
    """A real ChromaRetriever.search backed by an in-memory collection whose
    boilerplate doc outranks the named syllabus doc on raw similarity."""
    import chromadb
    from chromadb.config import Settings as ChromaSettings
    from sentence_transformers import SentenceTransformer

    from app import syllabus as syllabus_module
    from app.retrieval import ChromaRetriever

    client = chromadb.EphemeralClient(
        settings=ChromaSettings(anonymized_telemetry=False)
    )
    collection = client.get_or_create_collection(
        name="test_syllabus_kb", metadata={"hnsw:space": "cosine"}
    )
    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    texts = [c["text"] for c in SYLLABUS_CHUNKS]
    embeddings = encoder.encode(texts, normalize_embeddings=True).tolist()
    collection.upsert(
        ids=[c["id"] for c in SYLLABUS_CHUNKS],
        documents=texts,
        embeddings=embeddings,
        metadatas=[
            {
                "source_url": c["source_url"],
                "doc_type": "pdf" if c["source_url"].endswith(".pdf") else "html",
                "title": c["title"],
                "category": "",
                "chunk_index": i,
            }
            for i, c in enumerate(SYLLABUS_CHUNKS)
        ],
    )

    retriever = ChromaRetriever.__new__(ChromaRetriever)
    retriever._encoder = encoder
    retriever._collection = collection

    syllabus_module.reset_syllabus_index_cache()
    yield retriever
    syllabus_module.reset_syllabus_index_cache()


class TestSyllabusDocTargeting:
    def test_branch_year_query_surfaces_syllabus_first(self, syllabus_retriever):
        results = syllabus_retriever.search(
            "current third year syllabus for cse", top_k=3
        )
        assert results, "expected results"
        assert results[0]["source_url"].endswith("5.-TY-CSE-Syllabus-25-26-CC.pdf")

    def test_course_content_beats_boilerplate(self, syllabus_retriever):
        # The boilerplate doc is the only CSE-ish context the pool would have
        # ranked highly; targeting must put the real course content in front.
        results = syllabus_retriever.search(
            "cse syllabus for 25-26 third year", top_k=4
        )
        urls = [c["source_url"] for c in results]
        assert any(u.endswith("5.-TY-CSE-Syllabus-25-26-CC.pdf") for u in urls)
        assert urls[0].endswith("5.-TY-CSE-Syllabus-25-26-CC.pdf")
        assert "Course Name Compiler Design" in results[0]["text"]

    def test_named_subject_actually_present(self, syllabus_retriever):
        results = syllabus_retriever.search(
            "what subjects are in the third year cse syllabus", top_k=3
        )
        joined = " ".join(r["text"] for r in results)
        assert "Compiler Design" in joined

    def test_non_syllabus_query_untargeted(self, syllabus_retriever):
        from app.syllabus import find_syllabus_chunks

        assert find_syllabus_chunks(
            syllabus_retriever._collection, "what is the tuition fee"
        ) == []
        results = syllabus_retriever.search("what is the tuition fee", top_k=3)
        assert results

    def test_pronoun_it_untargeted(self, syllabus_retriever):
        from app.syllabus import find_syllabus_chunks

        assert find_syllabus_chunks(
            syllabus_retriever._collection, "is it possible to get the syllabus"
        ) == []

    def test_syllabus_chunks_lead_even_at_top_k_one(self, syllabus_retriever):
        results = syllabus_retriever.search(
            "third year cse syllabus", top_k=1
        )
        assert len(results) == 1
        assert results[0]["source_url"].endswith("5.-TY-CSE-Syllabus-25-26-CC.pdf")

    def test_result_shape_intact(self, syllabus_retriever):
        results = syllabus_retriever.search("third year cse syllabus", top_k=3)
        for r in results:
            assert set(r) >= {"text", "source_url", "doc_type", "title", "score"}
            assert isinstance(r["score"], float)
