"""BFS web crawler with persistent document cache and PDF downloader."""
from __future__ import annotations

import json
import logging
import os
import re
import time
import urllib.robotparser
from collections import deque
from datetime import datetime, timezone
from typing import Any, Optional
from urllib.parse import parse_qsl, urlencode, urljoin, urlparse

import pdfplumber
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

_NOISE_TAGS = ["script", "style", "nav", "footer", "header", "aside", "noscript", "iframe", "form"]
_REQUEST_DELAY = 0.5
_REQUEST_TIMEOUT = 15
_RETRY_ATTEMPTS = 3
_RETRY_BACKOFF = 1.0
_PROGRESS_EVERY = 25
_PDF_MIME_TYPES = {"application/pdf", "application/x-pdf"}
_NON_PAGE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp", ".ico", ".bmp", ".tif", ".tiff",
    ".css", ".js", ".mjs", ".woff", ".woff2", ".ttf", ".eot", ".otf",
    ".mp4", ".mp3", ".wav", ".avi", ".mov", ".webm", ".ogg", ".zip", ".rar", ".7z",
    ".gz", ".tar", ".bz2", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx", ".odt",
    ".ods", ".odp", ".csv", ".apk", ".exe", ".dmg", ".iso", ".deb",
}
_TRACKING_PARAMS = {"fbclid", "gclid", "msclkid", "ref", "mc_cid", "mc_eid"}
_SOCIAL_DOMAINS = {
    "facebook.com": "Facebook", "instagram.com": "Instagram", "linkedin.com": "LinkedIn",
    "youtube.com": "YouTube", "youtu.be": "YouTube", "twitter.com": "X (Twitter)",
    "x.com": "X (Twitter)", "t.me": "Telegram",
}
_SKIP_URL_SUBSTRINGS = ("/in/", "/u/")


def _normalize_url(url: str) -> str:
    parsed = urlparse(url)
    scheme = (parsed.scheme or "https").lower()
    host = (parsed.netloc or "").lower()
    if host.startswith("www."):
        host = host[4:]
    path = parsed.path.rstrip("/")
    query_pairs = [
        (key, value) for key, value in parse_qsl(parsed.query, keep_blank_values=True)
        if not (key.lower().startswith("utm_") or key.lower() in _TRACKING_PARAMS)
    ]
    result = f"{scheme}://{host}{path}"
    query = urlencode(query_pairs)
    return f"{result}?{query}" if query else result


def _host_key(url: str) -> str:
    return (urlparse(url).netloc or "").lower().removeprefix("www.")


def _platform_for(url: str) -> Optional[str]:
    parsed = urlparse(url)
    host = (parsed.netloc or "").lower().removeprefix("www.")
    if host.endswith("linkedin.com") and any(part in parsed.path for part in _SKIP_URL_SUBSTRINGS):
        return None
    for domain, platform in _SOCIAL_DOMAINS.items():
        if host == domain or host.endswith("." + domain):
            return platform
    return None


class CollegeScraper:
    """BFS crawler. Successful documents are cached between runs.

    Cache entries expire after cache_ttl_days. Use force_refresh=True to ignore
    the cache and fetch all pages/PDFs again. Cache is written after each fetched
    document, so an interrupted run can reuse already completed work.
    """

    def __init__(
        self,
        base_url: str,
        pdf_download_path: str = "./data/pdfs",
        session: Optional[requests.Session] = None,
        ca_bundle: Optional[str] = None,
        max_pages: int = 0,
        cache_path: Optional[str] = None,
        cache_ttl_days: int = 7,
        force_refresh: bool = False,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.pdf_download_path = pdf_download_path
        self._max_pages = max(0, int(max_pages))
        self.cache_ttl_days = max(0, int(cache_ttl_days))
        self.force_refresh = force_refresh
        os.makedirs(self.pdf_download_path, exist_ok=True)
        self.cache_path = cache_path or os.path.join(os.path.dirname(self.pdf_download_path) or ".", "crawl_cache.json")
        os.makedirs(os.path.dirname(os.path.abspath(self.cache_path)), exist_ok=True)
        self._cache: dict[str, dict[str, Any]] = self._load_cache()
        self._session = session or requests.Session()
        if ca_bundle:
            self._session.verify = ca_bundle
        self._session.headers.update({"User-Agent": "WCEBot-Scraper/1.0 (college info RAG crawler)"})
        self._parsed_base = urlparse(self.base_url)
        self._robots = self._load_robots()
        self.social_links: dict[str, str] = {}
        self.cache_hits = 0
        self.network_fetches = 0

    def _load_cache(self) -> dict[str, dict[str, Any]]:
        try:
            with open(self.cache_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, dict):
                logger.info("Loaded %d cached URLs from %s", len(data), self.cache_path)
                return data
        except FileNotFoundError:
            pass
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Could not read crawl cache (%s); starting an empty cache", exc)
        return {}

    def _save_cache(self) -> None:
        temporary = self.cache_path + ".tmp"
        try:
            with open(temporary, "w", encoding="utf-8") as fh:
                json.dump(self._cache, fh, ensure_ascii=False)
            os.replace(temporary, self.cache_path)
        except OSError as exc:
            logger.warning("Could not save crawl cache: %s", exc)
            try:
                if os.path.exists(temporary):
                    os.remove(temporary)
            except OSError:
                pass

    def _cache_entry_is_fresh(self, key: str) -> bool:
        if self.force_refresh or self.cache_ttl_days <= 0:
            return False
        entry = self._cache.get(key)
        if not entry:
            return False
        try:
            saved = datetime.fromisoformat(entry["cached_at"])
            age_seconds = (datetime.now(timezone.utc) - saved).total_seconds()
            if age_seconds > self.cache_ttl_days * 86400:
                return False
            doc = entry.get("doc", {})
            if doc.get("doc_type") == "pdf":
                pdf_path = entry.get("pdf_path")
                if not pdf_path or not os.path.isfile(pdf_path):
                    return False
            return bool(doc.get("text"))
        except (KeyError, TypeError, ValueError):
            return False

    def _get_cached_doc(self, key: str) -> Optional[dict[str, Any]]:
        if not self._cache_entry_is_fresh(key):
            return None
        self.cache_hits += 1
        return dict(self._cache[key]["doc"])

    def _cache_doc(self, requested_url: str, doc: dict[str, Any]) -> None:
        key = _normalize_url(requested_url)
        clean_doc = dict(doc)
        links = clean_doc.pop("links", [])
        social_links = clean_doc.pop("social_links", {})
        entry: dict[str, Any] = {
            "cached_at": datetime.now(timezone.utc).isoformat(),
            "doc": clean_doc,
            "links": links,
            "social_links": social_links,
        }
        if clean_doc.get("doc_type") == "pdf":
            entry["pdf_path"] = os.path.join(self.pdf_download_path, self._pdf_filename(clean_doc["url"]))
        self._cache[key] = entry
        # Also index the final URL if redirects changed the URL.
        self._cache[_normalize_url(clean_doc["url"])] = entry
        self._save_cache()

    def crawl(self, max_depth: int = 2) -> list[dict[str, Any]]:
        visited: set[str] = set()
        collected: set[str] = set()
        queue: deque[tuple[str, int]] = deque([(self.base_url, 0)])
        docs: list[dict[str, Any]] = []
        started = time.monotonic()

        while queue:
            url, depth = queue.popleft()
            key = _normalize_url(url)
            if key in visited or depth > max_depth:
                continue
            visited.add(key)
            if not self._is_allowed(url):
                logger.info("robots.txt disallows: %s", url)
                continue
            if self._max_pages and len(docs) >= self._max_pages:
                logger.warning("Reached MAX_PAGES=%d — stopping (%d URLs still queued)", self._max_pages, len(queue))
                break

            doc = self._get_cached_doc(key)
            if doc is not None:
                entry = self._cache.get(key, {})
                links = entry.get("links", [])
                social_links = entry.get("social_links", {})
                logger.info("Cache hit [depth=%d]: %s", depth, url)
            else:
                logger.info("Crawling [depth=%d]: %s", depth, url)
                time.sleep(_REQUEST_DELAY)
                self.network_fetches += 1
                fetched = self._fetch(url)
                if fetched is None:
                    continue
                doc = fetched
                links = doc.pop("links", [])
                social_links = doc.pop("social_links", {})
                # Restore transient fields in cache, but keep returned document shape clean.
                cache_doc = dict(doc)
                cache_doc["links"] = links
                cache_doc["social_links"] = social_links
                self._cache_doc(url, cache_doc)

            doc_key = _normalize_url(doc["url"])
            if doc_key in collected:
                visited.add(doc_key)
                continue
            collected.add(doc_key)
            visited.add(doc_key)
            docs.append(doc)
            for platform, link in social_links.items():
                self.social_links.setdefault(platform, link)
            if doc.get("doc_type") == "html":
                for link in links:
                    if _normalize_url(link) not in visited and depth < max_depth:
                        queue.append((link, depth + 1))
            if len(docs) % _PROGRESS_EVERY == 0:
                logger.info("Progress: %d documents, %d queued, %d cache hits, %.0fs elapsed", len(docs), len(queue), self.cache_hits, time.monotonic() - started)

        logger.info("Crawl complete — %d documents, %d cache hits, %d network fetches", len(docs), self.cache_hits, self.network_fetches)
        return docs

    def _get_with_retries(self, url: str) -> Optional[requests.Response]:
        last_exc: Optional[requests.RequestException] = None
        for attempt in range(1, _RETRY_ATTEMPTS + 1):
            try:
                resp = self._session.get(url, timeout=_REQUEST_TIMEOUT, allow_redirects=True)
                resp.raise_for_status()
                return resp
            except requests.HTTPError as exc:
                status = exc.response.status_code if exc.response is not None else None
                if status is not None and status < 500:
                    logger.warning("Failed to fetch %s: %s", url, exc)
                    return None
                last_exc = exc
            except requests.RequestException as exc:
                last_exc = exc
            if attempt < _RETRY_ATTEMPTS:
                wait = _RETRY_BACKOFF * attempt
                logger.warning("Fetch %s failed (%s) — retry %d/%d in %.0fs", url, last_exc, attempt, _RETRY_ATTEMPTS, wait)
                time.sleep(wait)
        logger.warning("Failed to fetch %s: %s", url, last_exc)
        return None

    def _fetch(self, url: str) -> Optional[dict[str, Any]]:
        resp = self._get_with_retries(url)
        if resp is None:
            return None
        content_type = resp.headers.get("Content-Type", "").split(";")[0].strip().lower()
        final_url = resp.url if isinstance(getattr(resp, "url", None), str) and resp.url else url
        now = datetime.now(timezone.utc).isoformat()
        if content_type in _PDF_MIME_TYPES or urlparse(final_url).path.lower().endswith(".pdf"):
            return self._process_pdf(final_url, resp.content, now)
        if "html" in content_type:
            return self._process_html(final_url, resp.text, now)
        if not content_type:
            head = resp.content[:600].lstrip().lower()
            if head.startswith(b"%pdf"):
                return self._process_pdf(final_url, resp.content, now)
            if head.startswith((b"<!doctype html", b"<html", b"<?xml")):
                return self._process_html(final_url, resp.text, now)
        logger.debug("Skipping unsupported content-type '%s' at %s", content_type or "?", url)
        return None

    def _process_html(self, url: str, html: str, now: str) -> dict[str, Any]:
        soup = BeautifulSoup(html, "html.parser")
        title_tag = soup.find("title")
        title = title_tag.get_text(strip=True) if title_tag else url
        links = self._extract_links(url, soup)
        social_links = self._extract_social_links(soup)
        for tag in soup(_NOISE_TAGS):
            tag.decompose()
        content = soup.find("main") or soup.find("article") or soup.find("body")
        text = content.get_text(separator="\n", strip=True) if content else ""
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        return {"url": url, "text": text, "doc_type": "html", "title": title, "last_scraped_at": now, "links": links, "social_links": social_links}

    def _process_pdf(self, url: str, content: bytes, now: str) -> Optional[dict[str, Any]]:
        filename = self._pdf_filename(url)
        filepath = os.path.join(self.pdf_download_path, filename)
        with open(filepath, "wb") as fh:
            fh.write(content)
        logger.info("Saved PDF: %s", filepath)
        pages_text: list[str] = []
        try:
            with pdfplumber.open(filepath) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        pages_text.append(page_text.strip())
        except Exception as exc:
            logger.warning("pdfplumber failed on %s: %s", filepath, exc)
            return None
        text = "\n\n".join(pages_text).strip()
        title = os.path.splitext(filename)[0].replace("_", " ").replace("-", " ").title()
        return {"url": url, "text": text, "doc_type": "pdf", "title": title, "last_scraped_at": now}

    def _extract_links(self, base: str, soup: BeautifulSoup) -> list[str]:
        links: list[str] = []
        seen: set[str] = set()
        base_host = _host_key(self.base_url)
        for tag in soup.find_all("a", href=True):
            href = tag["href"].strip()
            if href.startswith("#") or href.startswith(("mailto:", "tel:", "javascript:", "data:")):
                continue
            absolute = urljoin(base, href)
            parsed = urlparse(absolute)
            if parsed.scheme not in ("http", "https") or _host_key(absolute) != base_host:
                continue
            if os.path.splitext(parsed.path)[1].lower() in _NON_PAGE_EXTENSIONS:
                continue
            clean = _normalize_url(absolute)
            if clean not in seen:
                seen.add(clean)
                links.append(clean)
        return links

    def _extract_social_links(self, soup: BeautifulSoup) -> dict[str, str]:
        found: dict[str, str] = {}
        for tag in soup.find_all("a", href=True):
            href = tag["href"].strip()
            if not href.startswith(("http://", "https://")):
                continue
            platform = _platform_for(href)
            if platform is None or platform in found:
                continue
            clean = href.split("#", 1)[0]
            if "?" in clean:
                base, _, query = clean.partition("?")
                params = [part for part in query.split("&") if part]
                keys = [part.split("=", 1)[0].lower() for part in params]
                tracking = {"fbclid", "gclid", "msclkid", "ref"}
                if keys and all(key.startswith("utm_") or key in tracking for key in keys):
                    clean = base
            found[platform] = clean
        return found

    def _is_allowed(self, url: str) -> bool:
        if self._robots is None:
            return True
        if getattr(self._robots, "disallow_all", False):
            return False
        if getattr(self._robots, "allow_all", False):
            return True
        return self._robots.can_fetch("*", url)

    def _load_robots(self) -> Optional[urllib.robotparser.RobotFileParser]:
        robots_url = f"{self.base_url}/robots.txt"
        parser = urllib.robotparser.RobotFileParser()
        parser.set_url(robots_url)
        try:
            resp = self._session.get(robots_url, timeout=_REQUEST_TIMEOUT)
        except requests.RequestException as exc:
            logger.warning("Could not fetch robots.txt (%s): %s", robots_url, exc)
            return None
        if resp.status_code in (401, 403):
            logger.warning("robots.txt returned %d — treating as disallow-all", resp.status_code)
            parser.disallow_all = True
            return parser
        if 400 <= resp.status_code < 500:
            parser.allow_all = True
            return parser
        if resp.status_code >= 500:
            logger.warning("robots.txt returned %d — failing open", resp.status_code)
            return None
        parser.parse(resp.text.splitlines())
        return parser

    @staticmethod
    def _pdf_filename(url: str) -> str:
        name = os.path.basename(urlparse(url).path) or "document.pdf"
        name = re.sub(r"[^\w\-.]", "_", name)
        if not name.lower().endswith(".pdf"):
            name += ".pdf"
        return name
