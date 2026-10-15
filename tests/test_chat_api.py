"""
tests/test_chat_api.py
──────────────────────
Integration tests for the FastAPI /chat and /health endpoints.

All LLM calls, Chroma searches, and Redis operations are mocked.
No real API keys, no Redis instance, and no ChromaDB on disk are needed.
"""
from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.constants import NO_INFO_MESSAGE  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Mock singletons injected before the app is imported
# ─────────────────────────────────────────────────────────────────────────────

GOOD_CHUNKS = [
    {
        "text": "The annual tuition fee for General category B.E. students at WCE is ₹1,09,500.",
        "source_url": "http://test.edu/fees",
        "doc_type": "html",
        "title": "Fee Structure",
        "score": 0.82,
    }
]

LOW_SCORE_CHUNKS = [
    {
        "text": "Some unrelated content about gardening.",
        "source_url": "http://test.edu/garden",
        "doc_type": "html",
        "title": "Garden Blog",
        "score": 0.10,
    }
]

GENERATED_ANSWER = (
    "The annual tuition fee for General category B.E. students at WCE is ₹1,09,500 per year. "
    "Source: Fee Structure (http://test.edu/fees)"
)


def _make_mock_cache(*, cached_response=None, rate_limit_ok=True):
    mock = MagicMock()
    mock.get.return_value = cached_response
    mock.set.return_value = None
    mock.ping.return_value = True
    mock.check_rate_limit.return_value = rate_limit_ok
    return mock


def _make_mock_retriever(chunks=None):
    mock = MagicMock()
    mock.search.return_value = chunks if chunks is not None else GOOD_CHUNKS
    mock.ping.return_value = True
    return mock


def _make_mock_llm(answer=GENERATED_ANSWER):
    mock = MagicMock()
    mock.generate.return_value = answer
    return mock


def _make_mock_social(chunks=None, intent=False):
    mock = MagicMock()
    mock.enabled = True
    mock.context_chunks.return_value = chunks if chunks is not None else []
    mock.matches_intent.return_value = intent
    return mock


SOCIAL_CHUNKS = [
    {
        "text": (
            "Latest videos from the official WCE Sangli YouTube channel:\n"
            "[1] Campus tour 2026 — https://www.youtube.com/watch?v=abc123 "
            "(published 2026-01-05)"
        ),
        "source_url": "https://www.youtube.com/channel/UCtest",
        "doc_type": "social",
        "title": "WCE YouTube — latest videos",
        "score": 1.0,
    }
]


# ─────────────────────────────────────────────────────────────────────────────
# Client factory
# ─────────────────────────────────────────────────────────────────────────────

def _make_client(
    *,
    cache=None,
    retriever=None,
    llm=None,
    social=None,
):
    """
    Create a TestClient with mocked singletons injected into app.main's globals.
    """
    import app.main as main_module

    cache = cache or _make_mock_cache()
    retriever = retriever or _make_mock_retriever()
    llm = llm or _make_mock_llm()
    social = social or _make_mock_social()

    main_module._cache = cache
    main_module._retriever = retriever
    main_module._llm = llm
    main_module._social = social

    return TestClient(main_module.app, raise_server_exceptions=False)


# ─────────────────────────────────────────────────────────────────────────────
# /chat — happy path
# ─────────────────────────────────────────────────────────────────────────────

class TestChatEndpoint:
    def test_valid_query_returns_200(self):
        client = _make_client()
        resp = client.post("/chat", json={"query": "What is the fee for General category?"})
        assert resp.status_code == 200
        data = resp.json()
        assert "answer" in data
        assert "cached" in data
        assert data["cached"] is False

    def test_valid_query_returns_source_url(self):
        client = _make_client()
        resp = client.post("/chat", json={"query": "What is the tuition fee?"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["source_url"] == "http://test.edu/fees"

    def test_llm_answer_in_response(self):
        client = _make_client()
        resp = client.post("/chat", json={"query": "fee structure"})
        assert resp.status_code == 200
        assert GENERATED_ANSWER in resp.json()["answer"]

    # ── Input validation ──────────────────────────────────────────────────────

    def test_empty_query_returns_400(self):
        client = _make_client()
        resp = client.post("/chat", json={"query": ""})
        assert resp.status_code == 422  # Pydantic validation error

    def test_whitespace_only_query_returns_422(self):
        client = _make_client()
        resp = client.post("/chat", json={"query": "   "})
        assert resp.status_code == 422

    def test_query_too_long_returns_422(self):
        client = _make_client()
        long_query = "a" * 501
        resp = client.post("/chat", json={"query": long_query})
        assert resp.status_code == 422

    def test_missing_body_returns_422(self):
        client = _make_client()
        resp = client.post("/chat", json={})
        assert resp.status_code == 422

    # ── Similarity threshold gate / out-of-scope ─────────────────────────────

    def test_low_similarity_returns_fallback_no_llm(self):
        mock_llm = _make_mock_llm()
        client = _make_client(
            retriever=_make_mock_retriever(chunks=LOW_SCORE_CHUNKS),
            llm=mock_llm,
        )
        resp = client.post("/chat", json={"query": "nonsense xyzzy 12345"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["answer"] == NO_INFO_MESSAGE
        assert data["source_url"] is None
        # LLM must NOT have been called
        mock_llm.generate.assert_not_called()

    def test_empty_chunks_returns_fallback(self):
        mock_llm = _make_mock_llm()
        client = _make_client(
            retriever=_make_mock_retriever(chunks=[]),
            llm=mock_llm,
        )
        resp = client.post("/chat", json={"query": "anything"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["answer"] == NO_INFO_MESSAGE
        assert data["source_url"] is None
        mock_llm.generate.assert_not_called()

    def test_out_of_scope_llm_refusal_has_no_source_url(self):
        """Even when retrieval passes the gate, a refusal must hide View Document."""
        mock_llm = _make_mock_llm(answer=NO_INFO_MESSAGE)
        client = _make_client(llm=mock_llm)
        resp = client.post("/chat", json={"query": "capital of France"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["answer"] == NO_INFO_MESSAGE
        assert data["source_url"] is None

    def test_wrapped_refusal_still_hides_source_url(self):
        """LLM refusals wrapped in quotes/dashes still suppress the link."""
        mock_llm = _make_mock_llm(answer=f'"{NO_INFO_MESSAGE}"')
        client = _make_client(llm=mock_llm)
        resp = client.post("/chat", json={"query": "tell me a joke"})
        assert resp.status_code == 200
        assert resp.json()["source_url"] is None

    def test_llm_refusal_is_not_cached(self):
        """Refusals must not be cached — an improved KB/LLM applies immediately."""
        mock_llm = _make_mock_llm(answer=NO_INFO_MESSAGE)
        mock_cache = _make_mock_cache()
        client = _make_client(llm=mock_llm, cache=mock_cache)
        resp = client.post("/chat", json={"query": "capital of France"})
        assert resp.status_code == 200
        assert resp.json()["answer"] == NO_INFO_MESSAGE
        mock_cache.set.assert_not_called()

    def test_cached_refusal_still_hides_source_url(self):
        """A cached refusal never resurrects an old source link."""
        cached_payload = {
            "answer": NO_INFO_MESSAGE,
            "source_url": "http://test.edu/old-doc",
        }
        client = _make_client(cache=_make_mock_cache(cached_response=cached_payload))
        resp = client.post("/chat", json={"query": "who is the PM"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["cached"] is True
        assert data["answer"] == NO_INFO_MESSAGE
        assert data["source_url"] is None

    # ── Live social (YouTube) context ────────────────────────────────────────

    def test_social_chunks_injected_for_video_query(self):
        """Live YouTube chunks reach the LLM and win the source URL."""
        mock_llm = _make_mock_llm()
        client = _make_client(
            retriever=_make_mock_retriever(chunks=[]),  # nothing in the KB
            llm=mock_llm,
            social=_make_mock_social(chunks=SOCIAL_CHUNKS, intent=True),
        )
        resp = client.post("/chat", json={"query": "latest youtube videos"})
        assert resp.status_code == 200
        data = resp.json()

        # LLM was called with the social chunk in context
        mock_llm.generate.assert_called_once()
        passed_chunks = mock_llm.generate.call_args[0][1]
        assert any("Campus tour 2026" in c["text"] for c in passed_chunks)
        # The social chunk supplies the source link
        assert data["source_url"] == "https://www.youtube.com/channel/UCtest"

    def test_no_social_chunks_when_service_returns_none(self):
        client = _make_client(social=_make_mock_social(chunks=[]))
        resp = client.post("/chat", json={"query": "latest youtube videos"})
        assert resp.status_code == 200
        assert resp.json()["answer"] == GENERATED_ANSWER

    # ── Widget endpoints ─────────────────────────────────────────────────────

    def test_widget_panel_served(self):
        client = _make_client()
        resp = client.get("/widget")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
        assert "chat-container" in resp.text

    def test_widget_script_served(self):
        client = _make_client()
        resp = client.get("/widget.js")
        assert resp.status_code == 200
        assert "javascript" in resp.headers["content-type"]
        assert "wcebot-fab" in resp.text
        assert "/widget" in resp.text

    # ── Cache ─────────────────────────────────────────────────────────────────

    def test_cache_hit_returns_cached_true(self):
        cached_payload = {
            "answer": "Cached answer about fees.",
            "source_url": "http://test.edu/fees",
        }
        mock_cache = _make_mock_cache(cached_response=cached_payload)
        mock_llm = _make_mock_llm()

        client = _make_client(cache=mock_cache, llm=mock_llm)
        resp = client.post("/chat", json={"query": "What is the fee?"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["cached"] is True
        assert data["answer"] == "Cached answer about fees."
        # LLM should NOT be called on cache hit
        mock_llm.generate.assert_not_called()

    def test_cache_miss_sets_cache(self):
        mock_cache = _make_mock_cache(cached_response=None)
        client = _make_client(cache=mock_cache)
        resp = client.post("/chat", json={"query": "fee structure please"})
        assert resp.status_code == 200
        mock_cache.set.assert_called_once()

    # ── Rate limiting ─────────────────────────────────────────────────────────

    def test_rate_limit_exceeded_returns_429(self):
        mock_cache = _make_mock_cache(rate_limit_ok=False)
        client = _make_client(cache=mock_cache)
        resp = client.post("/chat", json={"query": "fee?"})
        assert resp.status_code == 429
        assert "Rate limit" in resp.json().get("detail", "")

    def test_rate_limit_ok_passes_through(self):
        mock_cache = _make_mock_cache(rate_limit_ok=True)
        client = _make_client(cache=mock_cache)
        resp = client.post("/chat", json={"query": "What is the fee?"})
        assert resp.status_code == 200

    # ── LLM failure ───────────────────────────────────────────────────────────

    def test_llm_failure_returns_graceful_response(self):
        unavailable_msg = (
            "I'm sorry, the WCE Bot service is temporarily unavailable. "
            "Please try again in a few moments."
        )
        mock_llm = _make_mock_llm(answer=unavailable_msg)
        client = _make_client(llm=mock_llm)
        resp = client.post("/chat", json={"query": "fee?"})
        assert resp.status_code == 200
        # Should return the graceful message, not a 500
        assert "unavailable" in resp.json()["answer"].lower()

    # ── PDF source URL ────────────────────────────────────────────────────────

    def test_pdf_chunk_returns_source_url(self):
        pdf_chunks = [
            {
                "text": "Fee structure table from official PDF.",
                "source_url": "http://test.edu/fee_structure.pdf",
                "doc_type": "pdf",
                "title": "Fee Structure PDF",
                "score": 0.88,
            }
        ]
        client = _make_client(retriever=_make_mock_retriever(chunks=pdf_chunks))
        resp = client.post("/chat", json={"query": "fee PDF"})
        assert resp.status_code == 200
        assert resp.json()["source_url"] == "http://test.edu/fee_structure.pdf"


# ─────────────────────────────────────────────────────────────────────────────
# /health
# ─────────────────────────────────────────────────────────────────────────────

class TestHealthEndpoint:
    def test_health_ok(self):
        client = _make_client()
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["chroma_ok"] is True
        assert data["redis_ok"] is True
        assert data["status"] == "ok"

    def test_health_degraded_when_redis_down(self):
        mock_cache = _make_mock_cache()
        mock_cache.ping.return_value = False

        client = _make_client(cache=mock_cache)
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["redis_ok"] is False
        assert data["status"] == "degraded"

    def test_health_degraded_when_chroma_down(self):
        mock_retriever = _make_mock_retriever()
        mock_retriever.ping.return_value = False

        client = _make_client(retriever=mock_retriever)
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["chroma_ok"] is False
        assert data["status"] == "degraded"
