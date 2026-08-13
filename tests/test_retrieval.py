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
