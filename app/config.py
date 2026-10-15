"""
app/config.py
─────────────
Single source of truth for all configuration. Loaded from a .env file
(or environment variables) via pydantic-settings.
"""
from __future__ import annotations

from functools import lru_cache
from typing import List

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── LLM API keys ──────────────────────────────────────────────────────────
    gemini_api_key: SecretStr = SecretStr("")
    groq_api_key: SecretStr = SecretStr("")

    # ── Redis ─────────────────────────────────────────────────────────────────
    redis_url: str = "redis://localhost:6379/0"

    # ── Scraper ───────────────────────────────────────────────────────────────
    base_url: str = "https://your-college-website.edu"
    max_crawl_depth: int = 2

    # Safety cap for long crawls: stop after this many documents (0 = unlimited).
    # High by default so a normal depth-3 crawl is never truncated, but a
    # pathological link explosion (?page=… loops etc.) can't run for days.
    max_pages: int = 20000

    # Custom CA bundle (PEM) for scraping. Empty → certifi defaults.
    # The college site serves a certificate without its intermediate, so the
    # bundle (certifi + Sectigo intermediate) is required for TLS verification.
    ssl_ca_bundle: str = ""

    # ── Storage ───────────────────────────────────────────────────────────────
    chroma_path: str = "./data/chroma_db"
    pdf_download_path: str = "./data/pdfs"

    # ── Retrieval ─────────────────────────────────────────────────────────────
    similarity_threshold: float = 0.35
    top_k: int = 6

    # ── Rate limiting ─────────────────────────────────────────────────────────
    rate_limit_requests: int = 20
    rate_limit_window_seconds: int = 60

    # ── Cache ─────────────────────────────────────────────────────────────────
    cache_ttl_seconds: int = 86400  # 24 hours

    # ── Input validation ──────────────────────────────────────────────────────
    max_query_length: int = 500

    # ── CORS ──────────────────────────────────────────────────────────────────
    # Stored as a comma-separated string in .env, parsed into a list.
    # "null" covers the frontend opened directly as a file:// URL.
    allowed_origins: str = (
        "http://localhost,http://localhost:8000,http://127.0.0.1:8000,null"
    )
    # Regex for additional allowed origins (any localhost port, e.g. Live Server).
    allowed_origin_regex: str = r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$"

    # ── Branding / fallback ───────────────────────────────────────────────────
    official_site_fallback_url: str = "https://your-college-website.edu"

    # ── Social media ──────────────────────────────────────────────────────────
    # YouTube channel ID (the "UC…" id) used for keyless RSS feed lookups.
    # Empty → YouTube feed integration is disabled (static links still work).
    youtube_channel_id: str = ""

    # ── Demo proxy ────────────────────────────────────────────────────────────
    # Port served by scripts/demo_proxy.py (local mirror of the college site
    # with the floating chat widget injected).
    demo_proxy_port: int = 9000

    # Origin of this backend used when building the widget snippet URL
    # (<script src="<widget_api_base>/widget.js">). Override if the API is
    # served behind another host/port.
    widget_api_base: str = "http://localhost:8000"

    @field_validator("similarity_threshold")
    @classmethod
    def _validate_threshold(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError("similarity_threshold must be between 0.0 and 1.0")
        return v

    def allowed_origins_list(self) -> List[str]:
        """Return CORS allowed origins as a Python list."""
        return [o.strip() for o in self.allowed_origins.split(",") if o.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached Settings instance (loaded once at startup)."""
    return Settings()
