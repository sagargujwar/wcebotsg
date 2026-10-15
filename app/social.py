"""
app/social.py
─────────────
Live social-media context for the chatbot.

Currently supports YouTube via its keyless RSS/Atom feed:
    https://www.youtube.com/feeds/videos.xml?channel_id=<UC…>

Instagram / Facebook / LinkedIn have no keyless feed, so their links live in
the knowledge base instead (static_docs/social_media.json → refresh_kb.py).

Design notes:
  - Feed responses are cached in-process for FEED_TTL_SECONDS so a burst of
    "latest video" questions costs at most one upstream request.
  - Every failure path degrades to "no extra context" — the retriever and the
    static KB chunks still answer, and the request never raises.
  - Injected chunks carry score 1.0 so they always pass the similarity gate
    and win the source-URL slot for video-related questions.
"""
from __future__ import annotations

import logging
import re
import threading
import time
from typing import List, Optional
from xml.etree import ElementTree

import requests

from app.config import get_settings

logger = logging.getLogger(__name__)

FEED_TTL_SECONDS = 900  # 15 minutes
FEED_TIMEOUT_SECONDS = 8
MAX_FEED_ITEMS = 5

_YOUTUBE_FEED_URL = "https://www.youtube.com/feeds/videos.xml?channel_id={channel_id}"
_YOUTUBE_CHANNEL_URL = "https://www.youtube.com/channel/{channel_id}"

# Queries that should pull in the live YouTube feed. Kept deliberately broad —
# an over-broad match only adds context; the grounding prompt still refuses
# anything unrelated to the college.
_YT_INTENT_RE = re.compile(
    r"\b(youtube|videos?|reels?|shorts?|channel|live stream|livestream|"
    r"latest\s+(?:video|update|post|upload)|new\s+(?:video|upload)|"
    r"uploaded|watch)\b",
    re.IGNORECASE,
)

_ATOM_NS = {"atom": "http://www.w3.org/2005/Atom", "yt": "http://www.youtube.com/xml/schemas/2015"}


class SocialService:
    """Provides optional live social context for a query."""

    def __init__(self) -> None:
        self._settings = get_settings()
        self._channel_id = self._settings.youtube_channel_id.strip()
        self._lock = threading.Lock()
        self._cached_items: Optional[List[dict]] = None
        self._cached_at: float = 0.0

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def enabled(self) -> bool:
        return bool(self._channel_id)

    def matches_intent(self, query: str) -> bool:
        """True when *query* plausibly asks about videos / the YouTube channel."""
        return bool(_YT_INTENT_RE.search(query))

    def context_chunks(self, query: str) -> List[dict]:
        """
        Return extra context chunks for *query*, or [] when unavailable.

        The caller appends these to the retrieved chunks before the LLM call.
        """
        if not self.enabled or not self.matches_intent(query):
            return []

        items = self._feed_items()
        if not items:
            return []

        lines = [
            f"[{i}] {item['title']} — {item['link']} (published {item['published']})"
            for i, item in enumerate(items, 1)
        ]
        text = (
            "Latest videos from the official WCE Sangli YouTube channel "
            f"({self.channel_url}):\n" + "\n".join(lines)
        )
        return [
            {
                "text": text,
                "source_url": self.channel_url,
                "doc_type": "social",
                "title": "WCE YouTube — latest videos",
                "score": 1.0,
            }
        ]

    @property
    def channel_url(self) -> str:
        return _YOUTUBE_CHANNEL_URL.format(channel_id=self._channel_id)

    # ── Feed fetching (cached) ────────────────────────────────────────────────

    def _feed_items(self) -> List[dict]:
        with self._lock:
            if (
                self._cached_items is not None
                and time.monotonic() - self._cached_at < FEED_TTL_SECONDS
            ):
                return self._cached_items

            items: List[dict] = []
            try:
                resp = requests.get(
                    _YOUTUBE_FEED_URL.format(channel_id=self._channel_id),
                    timeout=FEED_TIMEOUT_SECONDS,
                    headers={"User-Agent": "WCEBot/1.0 (college info assistant)"},
                )
                resp.raise_for_status()
                items = self._parse_feed(resp.text)
                logger.info("YouTube feed: %d items", len(items))
            except Exception as exc:
                # Keep serving a stale copy if we have one, else nothing.
                logger.warning("YouTube feed fetch failed: %s", exc)
                if self._cached_items is not None:
                    return self._cached_items

            self._cached_items = items
            self._cached_at = time.monotonic()
            return items

    @staticmethod
    def _parse_feed(xml_text: str) -> List[dict]:
        """Parse a YouTube Atom feed into {title, link, published} dicts."""
        root = ElementTree.fromstring(xml_text)
        items: List[dict] = []
        for entry in root.findall("atom:entry", _ATOM_NS):
            title = (entry.findtext("atom:title", default="", namespaces=_ATOM_NS) or "").strip()
            video_id = entry.findtext("yt:videoId", default="", namespaces=_ATOM_NS) or ""
            link_el = entry.find("atom:link[@rel='alternate']", _ATOM_NS)
            link = link_el.get("href", "") if link_el is not None else ""
            if not link and video_id:
                link = f"https://www.youtube.com/watch?v={video_id}"
            published = (
                entry.findtext("atom:published", default="", namespaces=_ATOM_NS) or ""
            ).strip()
            if title and link:
                items.append(
                    {"title": title, "link": link, "published": published[:10]}
                )
            if len(items) >= MAX_FEED_ITEMS:
                break
        return items
