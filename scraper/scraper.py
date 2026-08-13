"""
scraper/scraper.py
──────────────────
BFS web crawler + PDF downloader for college websites.

Usage (standalone):
    from scraper.scraper import CollegeScraper
    scraper = CollegeScraper(base_url="https://...", pdf_download_path="./data/pdfs")
    docs = scraper.crawl(max_depth=2)

To point at a real site, set BASE_URL in .env and run scripts/refresh_kb.py.
"""
from __future__ import annotations

import logging
import os
import re
import time
import urllib.robotparser
from collections import deque
from datetime import datetime, timezone
from typing import List, Optional
from urllib.parse import urljoin, urlparse

import pdfplumber
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# Tags whose content is stripped before text extraction
_NOISE_TAGS = [
    "script", "style", "nav", "footer", "header",
    "aside", "noscript", "iframe", "form",
]

_REQUEST_DELAY = 0.5  # seconds between requests (polite crawling)
_REQUEST_TIMEOUT = 10  # seconds

_PDF_MIME_TYPES = {"application/pdf", "application/x-pdf"}


class CollegeScraper:
    """BFS crawler that extracts clean text from HTML pages and PDFs."""

    def __init__(
        self,
        base_url: str,
        pdf_download_path: str = "./data/pdfs",
        session: Optional[requests.Session] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.pdf_download_path = pdf_download_path
        os.makedirs(pdf_download_path, exist_ok=True)

        self._session = session or requests.Session()
        self._session.headers.update(
            {"User-Agent": "WCEBot-Scraper/1.0 (college info RAG crawler)"}
        )

        self._parsed_base = urlparse(self.base_url)
        self._robots = self._load_robots()

    # ── Public API ────────────────────────────────────────────────────────────

    def crawl(self, max_depth: int = 2) -> List[dict]:
        """
        BFS-crawl pages reachable from *base_url* up to *max_depth* hops.

        Returns a list of document dicts:
            {url, text, doc_type, title, last_scraped_at}
        """
        visited: set[str] = set()
        queue: deque[tuple[str, int]] = deque([(self.base_url, 0)])
        docs: List[dict] = []

        while queue:
            url, depth = queue.popleft()
            if url in visited or depth > max_depth:
                continue
            visited.add(url)

            if not self._is_allowed(url):
                logger.info("robots.txt disallows: %s", url)
                continue

            logger.info("Crawling [depth=%d]: %s", depth, url)
            doc = self._fetch(url)
            if doc is None:
                continue
            docs.append(doc)

            # Discover links for further crawling (only for HTML pages)
            if doc["doc_type"] == "html":
                for link in self._extract_links(url, doc.get("_raw_html", "")):
                    if link not in visited:
                        if link.lower().endswith(".pdf"):
                            # Treat PDF links as documents, not crawl targets
                            if link not in visited:
                                queue.append((link, depth + 1))
                        elif depth < max_depth:
                            queue.append((link, depth + 1))

            time.sleep(_REQUEST_DELAY)

        logger.info("Crawl complete — %d documents collected", len(docs))
        return docs

    # ── Private helpers ───────────────────────────────────────────────────────

    def _fetch(self, url: str) -> Optional[dict]:
        """Download *url* and return a document dict, or None on failure."""
        try:
            resp = self._session.get(url, timeout=_REQUEST_TIMEOUT, allow_redirects=True)
            resp.raise_for_status()
        except requests.RequestException as exc:
            logger.warning("Failed to fetch %s: %s", url, exc)
            return None

        content_type = resp.headers.get("Content-Type", "").split(";")[0].strip()
        now = datetime.now(timezone.utc).isoformat()

        # ── PDF ──────────────────────────────────────────────────────────────
        if content_type in _PDF_MIME_TYPES or url.lower().endswith(".pdf"):
            return self._process_pdf(url, resp.content, now)

        # ── HTML ─────────────────────────────────────────────────────────────
        if "html" in content_type:
            return self._process_html(url, resp.text, now)

        logger.debug("Skipping unsupported content-type '%s' at %s", content_type, url)
        return None

    def _process_html(self, url: str, html: str, now: str) -> dict:
        soup = BeautifulSoup(html, "html.parser")

        # Extract title
        title_tag = soup.find("title")
        title = title_tag.get_text(strip=True) if title_tag else url

        # Strip noise
        for tag in soup(_NOISE_TAGS):
            tag.decompose()

        # Prefer <main> content; fall back to <body>
        content = soup.find("main") or soup.find("article") or soup.find("body")
        text = content.get_text(separator="\n", strip=True) if content else ""
        text = re.sub(r"\n{3,}", "\n\n", text).strip()

        return {
            "url": url,
            "text": text,
            "doc_type": "html",
            "title": title,
            "last_scraped_at": now,
            "_raw_html": html,  # kept temporarily for link extraction; not stored in Chroma
        }

    def _process_pdf(self, url: str, content: bytes, now: str) -> Optional[dict]:
        # Save PDF to disk
        filename = self._pdf_filename(url)
        filepath = os.path.join(self.pdf_download_path, filename)
        with open(filepath, "wb") as fh:
            fh.write(content)
        logger.info("Saved PDF: %s", filepath)

        # Extract text with pdfplumber
        pages_text: List[str] = []
        try:
            with pdfplumber.open(filepath) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        pages_text.append(page_text.strip())
        except Exception as exc:
            logger.warning("pdfplumber failed on %s: %s", filepath, exc)
            return None

        text = "\n\n".join(pages_text)
        # Use filename as title (will be overridden if a better title is found)
        title = os.path.splitext(filename)[0].replace("_", " ").replace("-", " ").title()

        return {
            "url": url,
            "text": text,
            "doc_type": "pdf",
            "title": title,
            "last_scraped_at": now,
        }

    def _extract_links(self, base: str, html: str) -> List[str]:
        """Return absolute URLs from <a href> tags that stay on the same domain."""
        soup = BeautifulSoup(html, "html.parser")
        links: List[str] = []
        for tag in soup.find_all("a", href=True):
            href = tag["href"].strip()
            if href.startswith("#") or href.startswith("mailto:") or href.startswith("tel:"):
                continue
            absolute = urljoin(base, href)
            parsed = urlparse(absolute)
            # Stay on the same domain
            if parsed.netloc == self._parsed_base.netloc:
                # Strip fragment and normalise
                clean = parsed._replace(fragment="").geturl()
                links.append(clean)
        return links

    def _is_allowed(self, url: str) -> bool:
        if self._robots is None:
            return True
        return self._robots.can_fetch("*", url)

    def _load_robots(self) -> Optional[urllib.robotparser.RobotFileParser]:
        robots_url = f"{self.base_url}/robots.txt"
        rp = urllib.robotparser.RobotFileParser()
        rp.set_url(robots_url)
        try:
            rp.read()
        except Exception as exc:
            logger.debug("Could not load robots.txt (%s): %s", robots_url, exc)
            return None
        return rp

    @staticmethod
    def _pdf_filename(url: str) -> str:
        """Derive a safe filename from a PDF URL."""
        path = urlparse(url).path
        name = os.path.basename(path) or "document.pdf"
        # Sanitise — keep only safe characters
        name = re.sub(r"[^\w\-.]", "_", name)
        if not name.lower().endswith(".pdf"):
            name += ".pdf"
        return name
