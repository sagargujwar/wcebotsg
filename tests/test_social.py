"""
tests/test_social.py
────────────────────
Tests for the social-media features:
  - SocialService (YouTube RSS intent, parsing, caching, graceful failure)
  - scraper social-link extraction
  - refresh_kb static social chunk building

All HTTP calls are mocked — no internet access required.
"""
from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch

import pytest
from bs4 import BeautifulSoup

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.constants import NO_INFO_MESSAGE  # noqa: E402,F401  (re-export sanity)
from app.social import SocialService  # noqa: E402
from scraper.scraper import CollegeScraper, _platform_for  # noqa: E402
from scripts.refresh_kb import build_social_chunks  # noqa: E402

CHANNEL_ID = "UCabcdefABCDEFabcdefAB"

SAMPLE_FEED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns:yt="http://www.youtube.com/xml/schemas/2015"
      xmlns="http://www.w3.org/2005/Atom">
  <title>Media WCE</title>
  <entry>
    <id>yt:video:abc123</id>
    <yt:videoId>abc123</yt:videoId>
    <yt:channelId>UCtest</yt:channelId>
    <title>Campus Tour 2026</title>
    <link rel="alternate" href="https://www.youtube.com/watch?v=abc123"/>
    <published>2026-01-05T10:00:00+00:00</published>
  </entry>
  <entry>
    <id>yt:video:def456</id>
    <yt:videoId>def456</yt:videoId>
    <title>Placement Drive Highlights</title>
    <link rel="alternate" href="https://www.youtube.com/watch?v=def456"/>
    <published>2026-01-01T10:00:00+00:00</published>
  </entry>
</feed>
"""


def _make_service(channel_id: str = CHANNEL_ID) -> SocialService:
    svc = SocialService()
    svc._channel_id = channel_id  # settings-driven; force for the test
    return svc


def _mock_feed_response(xml: str = SAMPLE_FEED):
    resp = MagicMock()
    resp.text = xml
    resp.raise_for_status = MagicMock()
    return resp


# ─────────────────────────────────────────────────────────────────────────────
# Intent detection
# ─────────────────────────────────────────────────────────────────────────────

class TestIntent:
    @pytest.mark.parametrize(
        "query",
        [
            "latest youtube videos",
            "What videos has WCE uploaded?",
            "Show me the newest reel",
            "Any new posts on the channel?",
            "latest video from walchand college",
            "WCE shorts",
        ],
    )
    def test_matches_video_queries(self, query):
        assert _make_service().matches_intent(query) is True

    @pytest.mark.parametrize(
        "query",
        [
            "What is the fee structure?",
            "admission deadline for 2026-27",
            "Who is the principal of WCE?",
            "capital of France",
        ],
    )
    def test_ignores_ordinary_queries(self, query):
        assert _make_service().matches_intent(query) is False


# ─────────────────────────────────────────────────────────────────────────────
# Feed parsing / context chunks
# ─────────────────────────────────────────────────────────────────────────────

class TestFeed:
    def test_parse_feed_returns_items(self):
        items = SocialService._parse_feed(SAMPLE_FEED)
        assert len(items) == 2
        assert items[0]["title"] == "Campus Tour 2026"
        assert items[0]["link"] == "https://www.youtube.com/watch?v=abc123"
        assert items[0]["published"] == "2026-01-05"

    def test_context_chunks_for_video_query(self):
        svc = _make_service()
        with patch(
            "app.social.requests.get", return_value=_mock_feed_response()
        ) as mock_get:
            chunks = svc.context_chunks("latest youtube videos")

        mock_get.assert_called_once()
        assert len(chunks) == 1
        chunk = chunks[0]
        assert chunk["score"] == 1.0
        assert chunk["source_url"] == svc.channel_url
        assert "Campus Tour 2026" in chunk["text"]
        assert "Placement Drive Highlights" in chunk["text"]

    def test_no_chunks_for_non_video_query(self):
        svc = _make_service()
        with patch("app.social.requests.get") as mock_get:
            assert svc.context_chunks("fee structure") == []
        mock_get.assert_not_called()

    def test_disabled_without_channel_id(self):
        svc = _make_service(channel_id="")
        assert svc.enabled is False
        with patch("app.social.requests.get") as mock_get:
            assert svc.context_chunks("latest youtube videos") == []
        mock_get.assert_not_called()

    def test_fetch_failure_returns_no_chunks(self):
        svc = _make_service()
        with patch("app.social.requests.get", side_effect=OSError("network down")):
            assert svc.context_chunks("latest youtube videos") == []

    def test_invalid_xml_returns_no_chunks(self):
        svc = _make_service()
        with patch(
            "app.social.requests.get", return_value=_mock_feed_response("<not-xml")
        ):
            assert svc.context_chunks("latest youtube videos") == []

    def test_feed_is_cached(self):
        svc = _make_service()
        with patch(
            "app.social.requests.get", return_value=_mock_feed_response()
        ) as mock_get:
            svc.context_chunks("latest videos")
            svc.context_chunks("newest video")
        assert mock_get.call_count == 1  # second call hit the TTL cache


# ─────────────────────────────────────────────────────────────────────────────
# Scraper — social link capture
# ─────────────────────────────────────────────────────────────────────────────

class TestScraperSocialLinks:
    HTML = """
    <html><body>
      <footer>
        <a href="https://www.facebook.com/walchandcollegeofengineeringsangli">FB</a>
        <a href="https://www.instagram.com/wcesangli?utm_source=website">IG</a>
        <a href="https://www.linkedin.com/school/walchandcollegeofengineering/">LI</a>
        <a href="https://www.linkedin.com/in/some-personal-profile/">Personal</a>
        <a href="https://example.com/about">Not social</a>
        <a href="/contact">Relative</a>
      </footer>
    </body></html>
    """

    def _make_scraper(self, tmp_path) -> CollegeScraper:
        with patch.object(CollegeScraper, "_load_robots", return_value=None):
            return CollegeScraper(
                base_url="http://test-college.edu",
                pdf_download_path=str(tmp_path / "pdfs"),
            )

    @pytest.mark.parametrize(
        "url,expected",
        [
            ("https://www.instagram.com/wcesangli", "Instagram"),
            ("https://facebook.com/pages/x", "Facebook"),
            ("https://www.youtube.com/@mediawce", "YouTube"),
            ("https://x.com/somehandle", "X (Twitter)"),
            ("https://example.com/other", None),
        ],
    )
    def test_platform_for(self, url, expected):
        assert _platform_for(url) == expected

    def test_personal_linkedin_profiles_skipped(self):
        assert _platform_for("https://www.linkedin.com/in/john-doe/") is None
        assert (
            _platform_for("https://www.linkedin.com/school/some-college/")
            == "LinkedIn"
        )

    def test_extract_social_links(self, tmp_path):
        scraper = self._make_scraper(tmp_path)
        found = scraper._extract_social_links(BeautifulSoup(self.HTML, "html.parser"))

        assert found["Facebook"] == (
            "https://www.facebook.com/walchandcollegeofengineeringsangli"
        )
        # tracking query stripped
        assert found["Instagram"] == "https://www.instagram.com/wcesangli"
        assert found["LinkedIn"] == (
            "https://www.linkedin.com/school/walchandcollegeofengineering/"
        )
        assert "example.com" not in " ".join(found.values())

    def test_crawl_collects_social_links(self, tmp_path):
        scraper = self._make_scraper(tmp_path)
        scraper._robots = None
        scraper._session.get = lambda url, **kw: MagicMock(
            status_code=200,
            headers={"Content-Type": "text/html"},
            text=self.HTML,
            content=self.HTML.encode(),
            raise_for_status=MagicMock(),
        )

        docs = scraper.crawl(max_depth=0)
        assert len(docs) == 1
        assert "Instagram" in scraper.social_links
        assert scraper.social_links["Instagram"] == (
            "https://www.instagram.com/wcesangli"
        )


# ─────────────────────────────────────────────────────────────────────────────
# refresh_kb — static social chunks
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildSocialChunks:
    def test_curated_entries_become_chunks(self):
        chunks = build_social_chunks(discovered={})
        platforms = {c["title"] for c in chunks}
        assert "WCE Instagram" in platforms
        assert "WCE Facebook" in platforms
        assert "WCE LinkedIn" in platforms
        assert "WCE YouTube" in platforms

    def test_chunk_shape(self):
        chunks = build_social_chunks(discovered={})
        instagram = next(c for c in chunks if c["title"] == "WCE Instagram")
        assert instagram["doc_type"] == "social"
        assert instagram["source_url"] == "https://www.instagram.com/wcesangli/"
        assert "@wcesangli" in instagram["text"]
        assert instagram["id"]  # deterministic id present

    def test_ids_are_deterministic(self):
        ids_a = [c["id"] for c in build_social_chunks(discovered={})]
        ids_b = [c["id"] for c in build_social_chunks(discovered={})]
        assert ids_a == ids_b

    def test_discovered_platform_added_when_missing(self):
        chunks = build_social_chunks(
            discovered={"Telegram": "https://t.me/wcesangli"}
        )
        telegram = next(c for c in chunks if c["title"] == "WCE Telegram")
        assert telegram["source_url"] == "https://t.me/wcesangli"
        assert telegram["doc_type"] == "social"

    def test_entry_without_url_keeps_empty_source(self):
        chunks = build_social_chunks(discovered={})
        x_entry = next(c for c in chunks if c["title"] == "WCE X (Twitter)")
        assert x_entry["source_url"] == ""  # backend converts "" → no link
