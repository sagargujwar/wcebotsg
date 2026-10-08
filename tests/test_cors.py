"""
tests/test_cors.py
──────────────────
CORS policy tests: the frontend may be opened as a file:// page or served
from any localhost port (Live Server, python -m http.server, etc.).
"""
from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import app.main as main_module  # noqa: E402


def _make_client():
    cache = MagicMock()
    cache.get.return_value = None
    cache.set.return_value = None
    cache.ping.return_value = True
    cache.check_rate_limit.return_value = True

    retriever = MagicMock()
    retriever.search.return_value = [
        {
            "text": "The annual tuition fee is ₹1,09,500.",
            "source_url": "http://test.edu/fees",
            "doc_type": "html",
            "title": "Fee Structure",
            "score": 0.82,
        }
    ]
    retriever.ping.return_value = True

    llm = MagicMock()
    llm.generate.return_value = "The fee is ₹1,09,500 per year."

    main_module._cache = cache
    main_module._retriever = retriever
    main_module._llm = llm

    return TestClient(main_module.app, raise_server_exceptions=False)


def _post_with_origin(origin: str):
    client = _make_client()
    return client.post(
        "/chat",
        json={"query": "What is the fee?"},
        headers={"Origin": origin},
    )


class TestCORS:
    def test_allows_localhost_with_arbitrary_port(self):
        """Live Server / http.server on any port must not be blocked."""
        resp = _post_with_origin("http://localhost:5500")
        assert resp.status_code == 200
        assert resp.headers.get("access-control-allow-origin") == "http://localhost:5500"

    def test_allows_loopback_ip_with_arbitrary_port(self):
        resp = _post_with_origin("http://127.0.0.1:3000")
        assert resp.status_code == 200
        assert resp.headers.get("access-control-allow-origin") == "http://127.0.0.1:3000"

    def test_allows_file_origin_null(self):
        """Opening frontend/index.html directly sends Origin: null."""
        resp = _post_with_origin("null")
        assert resp.status_code == 200
        assert resp.headers.get("access-control-allow-origin") == "null"

    def test_allows_explicit_configured_origin(self):
        resp = _post_with_origin("http://localhost:8000")
        assert resp.status_code == 200
        assert resp.headers.get("access-control-allow-origin") == "http://localhost:8000"

    def test_rejects_foreign_origin(self):
        resp = _post_with_origin("https://evil.example.com")
        assert resp.status_code == 200
        assert "access-control-allow-origin" not in resp.headers

    def test_preflight_allowed_from_localhost_port(self):
        client = _make_client()
        resp = client.options(
            "/chat",
            headers={
                "Origin": "http://localhost:5500",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type",
            },
        )
        assert resp.status_code == 200
        assert resp.headers.get("access-control-allow-origin") == "http://localhost:5500"
