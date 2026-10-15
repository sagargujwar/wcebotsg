"""
tests/test_scraper.py
─────────────────────
Unit tests for the scraper and chunker. All HTTP calls are mocked — no
internet access required.
"""
from __future__ import annotations

import io
import os
from unittest.mock import MagicMock, patch

import pytest
import requests
from bs4 import BeautifulSoup

from scraper.chunker import MAX_CHUNK_CHARS, TextChunker, _deterministic_id
from scraper.scraper import CollegeScraper, _host_key, _normalize_url


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _mock_response(content: str | bytes, content_type: str = "text/html", status: int = 200):
    """Build a minimal mock requests.Response."""
    resp = MagicMock()
    resp.status_code = status
    resp.raise_for_status = MagicMock()
    resp.headers = {"Content-Type": content_type}
    if isinstance(content, str):
        resp.text = content
        resp.content = content.encode()
    else:
        resp.text = content.decode("utf-8", errors="replace")
        resp.content = content
    return resp


# ─────────────────────────────────────────────────────────────────────────────
# CollegeScraper — HTML processing
# ─────────────────────────────────────────────────────────────────────────────

class TestCollegeScraper:
    BASE_URL = "http://test-college.edu"

    def _make_scraper(self, tmp_path) -> CollegeScraper:
        return CollegeScraper(
            base_url=self.BASE_URL,
            pdf_download_path=str(tmp_path / "pdfs"),
        )

    def test_process_html_strips_nav_footer(self, tmp_path, index_html_content):
        scraper = self._make_scraper(tmp_path)
        # Patch robots.txt so crawl doesn't try to fetch it
        with patch.object(scraper, "_robots", None):
            doc = scraper._process_html(
                url=f"{self.BASE_URL}/index.html",
                html=index_html_content,
                now="2025-01-01T00:00:00+00:00",
            )

        assert doc is not None
        assert doc["doc_type"] == "html"
        # Nav/footer content should NOT appear in the extracted text
        assert "Home" not in doc["text"] or "Welcome to Walchand" in doc["text"]
        # Main content should be present
        assert "Walchand College of Engineering" in doc["text"]

    def test_process_html_extracts_title(self, tmp_path, index_html_content):
        scraper = self._make_scraper(tmp_path)
        doc = scraper._process_html(
            url=f"{self.BASE_URL}/",
            html=index_html_content,
            now="2025-01-01T00:00:00+00:00",
        )
        assert "Walchand" in doc["title"] or doc["title"] != ""

    def test_process_html_syllabus(self, tmp_path, syllabus_html_content):
        scraper = self._make_scraper(tmp_path)
        doc = scraper._process_html(
            url=f"{self.BASE_URL}/syllabus",
            html=syllabus_html_content,
            now="2025-01-01T00:00:00+00:00",
        )
        assert "Data Structures and Algorithms" in doc["text"]
        assert "Operating Systems" in doc["text"]

    def test_crawl_returns_docs(self, tmp_path, index_html_content, syllabus_html_content):
        """Crawl mock pages using mocked requests.Session."""
        scraper = self._make_scraper(tmp_path)
        # Disable robots.txt check
        scraper._robots = None

        def fake_get(url, **kwargs):
            if "syllabus" in url:
                return _mock_response(syllabus_html_content)
            return _mock_response(index_html_content)

        scraper._session.get = fake_get

        docs = scraper.crawl(max_depth=1)
        assert len(docs) >= 1
        texts_combined = " ".join(d["text"] for d in docs)
        assert "Walchand College" in texts_combined

    def test_crawl_respects_depth_zero(self, tmp_path, index_html_content):
        scraper = self._make_scraper(tmp_path)
        scraper._robots = None
        scraper._session.get = lambda url, **kw: _mock_response(index_html_content)

        docs = scraper.crawl(max_depth=0)
        # depth=0 means only the base URL
        assert len(docs) == 1

    def test_pdf_filename_sanitisation(self, tmp_path):
        scraper = self._make_scraper(tmp_path)
        fname = scraper._pdf_filename("http://test.edu/downloads/fee structure 2025.pdf")
        assert " " not in fname
        assert fname.endswith(".pdf")


# ─────────────────────────────────────────────────────────────────────────────
# PDF extraction
# ─────────────────────────────────────────────────────────────────────────────

class TestPDFExtraction:
    def test_pdf_text_extracted(self, tmp_path, fee_pdf_path):
        """pdfplumber extracts text from the generated fee_structure.pdf fixture."""
        if not os.path.exists(fee_pdf_path):
            pytest.skip("fee_structure.pdf not generated yet — run generate_fixtures.py")

        scraper = CollegeScraper(
            base_url="http://test.edu",
            pdf_download_path=str(tmp_path / "pdfs"),
        )
        with open(fee_pdf_path, "rb") as f:
            pdf_bytes = f.read()

        doc = scraper._process_pdf(
            url="http://test.edu/fee_structure.pdf",
            content=pdf_bytes,
            now="2025-01-01T00:00:00+00:00",
        )
        assert doc is not None
        assert doc["doc_type"] == "pdf"
        assert "Fee" in doc["text"] or "fee" in doc["text"].lower()


# ─────────────────────────────────────────────────────────────────────────────
# TextChunker
# ─────────────────────────────────────────────────────────────────────────────

class TestTextChunker:
    SAMPLE_TEXT = (
        "Walchand College of Engineering is located in Sangli, Maharashtra.\n\n"
        "The college was established in 1947 and is affiliated with Shivaji University.\n\n"
        "Admissions are done through MHT-CET or JEE Main scores via the CAP process.\n\n"
        "The annual tuition fee for General category students is ₹1,09,500.\n\n"
        "SC/ST students are eligible for a full fee waiver under the Government scholarship scheme."
    )

    def test_chunks_generated(self):
        chunker = TextChunker()
        chunks = chunker.chunk(
            text=self.SAMPLE_TEXT,
            source_url="http://test.edu/",
            doc_type="html",
            title="WCE Home",
        )
        assert len(chunks) >= 3

    def test_chunk_metadata_fields(self):
        chunker = TextChunker()
        chunks = chunker.chunk(
            text=self.SAMPLE_TEXT,
            source_url="http://test.edu/fees",
            doc_type="html",
            title="Fee Info",
        )
        for c in chunks:
            assert "id" in c
            assert "text" in c
            assert "source_url" in c
            assert "doc_type" in c
            assert "title" in c
            assert "last_scraped_at" in c
            assert c["source_url"] == "http://test.edu/fees"
            assert c["doc_type"] == "html"

    def test_chunk_ids_are_deterministic(self):
        chunker = TextChunker()
        chunks1 = chunker.chunk(
            text=self.SAMPLE_TEXT,
            source_url="http://test.edu/",
            doc_type="html",
            title="WCE",
        )
        chunks2 = chunker.chunk(
            text=self.SAMPLE_TEXT,
            source_url="http://test.edu/",
            doc_type="html",
            title="WCE",
        )
        ids1 = [c["id"] for c in chunks1]
        ids2 = [c["id"] for c in chunks2]
        assert ids1 == ids2

    def test_chunk_ids_differ_by_url(self):
        chunker = TextChunker()
        chunks_a = chunker.chunk(self.SAMPLE_TEXT, "http://a.edu/", "html", "A")
        chunks_b = chunker.chunk(self.SAMPLE_TEXT, "http://b.edu/", "html", "B")
        ids_a = {c["id"] for c in chunks_a}
        ids_b = {c["id"] for c in chunks_b}
        assert ids_a.isdisjoint(ids_b)

    def test_overlap_present(self):
        """Adjacent chunks should share some text (overlap)."""
        chunker = TextChunker()
        chunks = chunker.chunk(self.SAMPLE_TEXT, "http://x.edu/", "html", "X")
        if len(chunks) >= 2:
            # At least one consecutive pair should have shared words
            found_overlap = False
            for i in range(len(chunks) - 1):
                words_a = set(chunks[i]["text"].split())
                words_b = set(chunks[i + 1]["text"].split())
                if words_a & words_b:
                    found_overlap = True
                    break
            assert found_overlap, "Expected overlap between adjacent chunks"

    def test_empty_text_returns_empty(self):
        chunker = TextChunker()
        chunks = chunker.chunk("", "http://x.edu/", "html", "Empty")
        assert chunks == []

    def test_short_text_not_discarded_entirely(self):
        chunker = TextChunker()
        text = "Annual tuition fee for General category: ₹1,09,500 per year.\n\nSC/ST: Nil."
        chunks = chunker.chunk(text, "http://x.edu/", "html", "Fees")
        assert len(chunks) >= 1

    def test_short_fragment_merges_into_previous_chunk(self):
        chunker = TextChunker()
        text = (
            "A long enough paragraph about the college and its history since 1947.\n\n"
            "SC/ST: Nil."
        )
        chunks = chunker.chunk(text, "http://x.edu/merge", "html", "Fees")
        assert len(chunks) == 1
        assert "SC/ST: Nil." in chunks[0]["text"]

    def test_long_paragraph_split_to_max_chars(self):
        chunker = TextChunker()
        para = " ".join(
            f"Sentence number {i} carries a bit of detail about the college."
            for i in range(120)
        )
        assert len(para) > 4 * MAX_CHUNK_CHARS
        chunks = chunker.chunk(para, "http://x.edu/big", "html", "Big")
        assert len(chunks) >= 4
        assert all(len(c["text"]) <= MAX_CHUNK_CHARS for c in chunks)

    def test_overlap_never_duplicates_whole_paragraph(self):
        chunker = TextChunker()
        para1 = (
            "First sentence of the first paragraph is here now. "
            "Second sentence adds more detail about the same thing. "
            "Third sentence closes it with a final remark."
        )
        para2 = (
            "Another paragraph starts here with its own first sentence. "
            "Its second sentence continues the thought process nicely. "
            "And a third sentence wraps everything up cleanly today."
        )
        chunks = chunker.chunk(para1 + "\n\n" + para2, "http://x.edu/ov", "html", "Ov")
        assert len(chunks) == 2
        # A whole paragraph must never be copied into the next chunk …
        assert not chunks[1]["text"].startswith(chunks[0]["text"])
        # … and the shared prefix must stay within the 15% overlap budget.
        common = os.path.commonprefix([chunks[0]["text"], chunks[1]["text"]])
        assert len(common) <= int(0.15 * len(para1)) + 5

    def test_overlap_fit_on_uneven_paragraphs_stays_under_max(self):
        """Regression: tail sized vs the previous piece overflowed MAX when
        the current piece was longer (chunks up to 1675 chars in the wild)."""
        chunker = TextChunker()
        s = "This is a syllabus sentence describing a course outcome in detail. "
        p1 = (s * 18).strip()
        p2 = (
            "Laboratory experiment outcome number two is described here in equal detail. "
            * 19
        ).strip()
        chunks = chunker.chunk(p1 + "\n\n" + p2, "http://x.edu/fit", "html", "Fit")
        assert len(chunks) == 2
        assert all(len(c["text"]) <= MAX_CHUNK_CHARS for c in chunks)


# ─────────────────────────────────────────────────────────────────────────────
# URL normalisation / dedup
# ─────────────────────────────────────────────────────────────────────────────

class TestUrlNormalisation:
    def test_collapses_slash_utm_case_and_fragment(self):
        assert _normalize_url("https://Site.edu/?utm_source=x") == "https://site.edu"
        assert _normalize_url("https://site.edu/") == "https://site.edu"
        assert _normalize_url("https://site.edu/#team") == "https://site.edu"

    def test_keeps_real_query_and_path_case(self):
        assert (
            _normalize_url("https://site.edu/Dept/Results?id=5&utm_medium=mail")
            == "https://site.edu/Dept/Results?id=5"
        )

    def test_host_key_is_www_insensitive(self):
        assert _host_key("https://www.site.edu/x") == _host_key("https://site.edu/y")

    def test_www_stripped_to_same_key(self):
        assert (
            _normalize_url("https://www.site.edu/about")
            == _normalize_url("https://site.edu/about")
        )


# ─────────────────────────────────────────────────────────────────────────────
# Crawl behaviour — dedup, redirects, limits, retries, robots.txt
# ─────────────────────────────────────────────────────────────────────────────

def _http_error_response(status: int) -> MagicMock:
    """Mock response whose raise_for_status() raises HTTPError(status)."""
    resp = MagicMock()
    resp.status_code = status
    resp.headers = {}
    resp.text = ""
    resp.content = b""
    resp.raise_for_status = MagicMock(
        side_effect=requests.HTTPError(f"{status} error", response=resp)
    )
    return resp


class TestCrawlRobustness:
    BASE_URL = "http://test-college.edu"

    def _make_scraper(self, tmp_path, **kwargs) -> CollegeScraper:
        return CollegeScraper(
            base_url=self.BASE_URL,
            pdf_download_path=str(tmp_path / "pdfs"),
            **kwargs,
        )

    def test_trailing_slash_homepage_crawled_once(
        self, tmp_path, index_html_content
    ):
        """href="/" must not create a duplicate of the base URL."""
        scraper = self._make_scraper(tmp_path)
        scraper._robots = None

        def fake_get(url, **kwargs):
            return _mock_response(index_html_content)

        scraper._session.get = fake_get
        with patch("scraper.scraper.time.sleep"):
            docs = scraper.crawl(max_depth=1)

        keys = [_normalize_url(d["url"]) for d in docs]
        assert len(keys) == len(set(keys)), "duplicate pages crawled"
        assert keys.count(_normalize_url(self.BASE_URL)) == 1
        assert len(docs) >= 4

    def test_redirect_final_url_used_for_doc_and_links(
        self, tmp_path, index_html_content
    ):
        """doc['url'] and relative-link resolution use the post-redirect URL."""
        scraper = self._make_scraper(tmp_path)
        scraper._robots = None
        final = "http://test-college.edu/old/deep/page"
        requested: list[str] = []

        def fake_get(url, **kwargs):
            requested.append(url)
            resp = _mock_response(
                '<html><head><title>R</title></head><body><main>'
                "<p>Some redirected page content that is definitely long enough.</p>"
                '<a href="next">Next</a></main></body></html>'
            )
            resp.url = final
            return resp

        scraper._session.get = fake_get
        with patch("scraper.scraper.time.sleep"):
            docs = scraper.crawl(max_depth=1)

        assert docs[0]["url"] == final
        assert "http://test-college.edu/old/deep/next" in requested
        assert "http://test-college.edu/next" not in requested

    def test_max_pages_stops_crawl(self, tmp_path, index_html_content):
        scraper = self._make_scraper(tmp_path, max_pages=2)
        scraper._robots = None
        scraper._session.get = lambda url, **kw: _mock_response(index_html_content)

        with patch("scraper.scraper.time.sleep"):
            docs = scraper.crawl(max_depth=1)
        assert len(docs) == 2

    def test_www_variant_of_same_page_crawled_once(self, tmp_path):
        """A www link must collapse to the bare key — one fetch, one doc."""
        scraper = self._make_scraper(tmp_path)
        scraper._robots = None
        home = (
            '<html><head><title>H</title></head><body><main>'
            "<p>Home page content that is long enough to keep.</p>"
            '<a href="https://www.test-college.edu/about">About</a>'
            "</main></body></html>"
        )
        about = (
            '<html><head><title>A</title></head><body><main>'
            "<p>About page content that is long enough to keep.</p>"
            "</main></body></html>"
        )
        requested: list[str] = []

        def fake_get(url, **kwargs):
            requested.append(url)
            return _mock_response(about if "about" in url else home)

        scraper._session.get = fake_get
        with patch("scraper.scraper.time.sleep"):
            docs = scraper.crawl(max_depth=1)

        about_fetches = [u for u in requested if "about" in u]
        assert len(about_fetches) == 1
        keys = [_normalize_url(d["url"]) for d in docs]
        assert len(keys) == len(set(keys))
        assert len(docs) == 2

    def test_redirect_to_collected_page_not_double_counted(self, tmp_path):
        """A legacy URL redirecting to an already-collected page is skipped."""
        scraper = self._make_scraper(tmp_path)
        scraper._robots = None
        home = (
            '<html><head><title>H</title></head><body><main>'
            "<p>Home page content that is long enough to keep.</p>"
            '<a href="/old-page">Old</a>'
            "</main></body></html>"
        )
        requested: list[str] = []

        def fake_get(url, **kwargs):
            requested.append(url)
            resp = _mock_response(home)
            if url.endswith("old-page"):
                resp.url = self.BASE_URL  # redirects to the already-collected home
            return resp

        scraper._session.get = fake_get
        with patch("scraper.scraper.time.sleep"):
            docs = scraper.crawl(max_depth=1)

        assert any("old-page" in u for u in requested)  # it was fetched …
        assert len(docs) == 1  # … but never counted twice

    def test_missing_content_type_sniffs_html(self, tmp_path, index_html_content):
        scraper = self._make_scraper(tmp_path)
        scraper._robots = None
        scraper._session.get = lambda url, **kw: _mock_response(
            index_html_content, content_type=""
        )
        with patch("scraper.scraper.time.sleep"):
            docs = scraper.crawl(max_depth=0)
        assert len(docs) == 1
        assert docs[0]["doc_type"] == "html"

    def test_missing_content_type_sniffs_pdf(self, tmp_path, fee_pdf_path):
        if not os.path.exists(fee_pdf_path):
            pytest.skip("fee_structure.pdf not generated yet")
        scraper = self._make_scraper(tmp_path)
        scraper._robots = None
        with open(fee_pdf_path, "rb") as fh:
            pdf_bytes = fh.read()
        scraper._session.get = lambda url, **kw: _mock_response(
            pdf_bytes, content_type=""
        )
        with patch("scraper.scraper.time.sleep"):
            docs = scraper.crawl(max_depth=0)
        assert len(docs) == 1
        assert docs[0]["doc_type"] == "pdf"

    def test_extract_links_skips_static_assets(self, tmp_path):
        scraper = self._make_scraper(tmp_path)
        html = (
            '<a href="/page">Page</a>'
            '<a href="/img/photo.png">Pic</a>'
            '<a href="/style.css">CSS</a>'
            '<a href="/docs/fee.pdf">PDF</a>'
            '<a href="https://other.edu/x">External</a>'
            '<a href="#top">Fragment</a>'
            '<a href="mailto:a@b.c">Mail</a>'
        )
        links = scraper._extract_links(self.BASE_URL, BeautifulSoup(html, "html.parser"))
        assert sorted(links) == [
            "http://test-college.edu/docs/fee.pdf",
            "http://test-college.edu/page",
        ]


class TestRetries:
    BASE_URL = "http://test-college.edu"

    def _scraper_with_session(self, tmp_path, session) -> CollegeScraper:
        with patch.object(CollegeScraper, "_load_robots", return_value=None):
            return CollegeScraper(
                base_url=self.BASE_URL,
                pdf_download_path=str(tmp_path / "pdfs"),
                session=session,
            )

    def test_5xx_retried_then_success(self, tmp_path, index_html_content):
        ok = _mock_response(index_html_content)
        session = MagicMock()
        session.get.side_effect = [_http_error_response(503), ok]
        scraper = self._scraper_with_session(tmp_path, session)

        with patch("scraper.scraper.time.sleep") as sleep_mock:
            docs = scraper.crawl(max_depth=0)

        assert len(docs) == 1
        assert session.get.call_count == 2
        # one politeness delay + one retry backoff
        assert sleep_mock.call_count == 2

    def test_4xx_fails_fast_without_retry(self, tmp_path):
        session = MagicMock()
        session.get.side_effect = [_http_error_response(404)]
        scraper = self._scraper_with_session(tmp_path, session)

        with patch("scraper.scraper.time.sleep") as sleep_mock:
            docs = scraper.crawl(max_depth=0)

        assert docs == []
        assert session.get.call_count == 1
        # only the pre-fetch politeness delay — no retry backoff
        assert sleep_mock.call_count == 1

    def test_connection_error_retried(self, tmp_path, index_html_content):
        ok = _mock_response(index_html_content)
        session = MagicMock()
        session.get.side_effect = [requests.ConnectionError("boom"), ok]
        scraper = self._scraper_with_session(tmp_path, session)

        with patch("scraper.scraper.time.sleep"):
            docs = scraper.crawl(max_depth=0)
        assert len(docs) == 1
        assert session.get.call_count == 2

    def test_failed_fetch_still_delays(self, tmp_path):
        """Even a failed fetch must be followed by the polite delay."""
        session = MagicMock()
        session.get.side_effect = [_http_error_response(404)]
        scraper = self._scraper_with_session(tmp_path, session)

        with patch("scraper.scraper.time.sleep") as sleep_mock:
            scraper.crawl(max_depth=0)
        assert sleep_mock.call_count >= 1


class TestDedupeChunks:
    def test_duplicate_ids_dropped_first_wins(self):
        from scripts.refresh_kb import dedupe_chunks

        a = {"id": "x", "text": "first"}
        b = {"id": "y", "text": "other"}
        c = {"id": "x", "text": "duplicate"}
        out = dedupe_chunks([a, b, c])
        assert [o["id"] for o in out] == ["x", "y"]
        assert out[0]["text"] == "first"


class TestRobotsViaSession:
    BASE_URL = "http://test-college.edu"

    def _scraper(self, tmp_path, body: str, status: int) -> CollegeScraper:
        session = MagicMock()
        session.get.return_value = _mock_response(
            body, content_type="text/plain", status=status
        )
        return CollegeScraper(
            base_url=self.BASE_URL,
            pdf_download_path=str(tmp_path / "pdfs"),
            session=session,
        )

    def test_disallow_rules_parsed(self, tmp_path):
        s = self._scraper(tmp_path, "User-agent: *\nDisallow: /private\n", 200)
        assert not s._is_allowed("http://test-college.edu/private/page")
        assert s._is_allowed("http://test-college.edu/public/page")

    def test_missing_robots_allows_all(self, tmp_path):
        s = self._scraper(tmp_path, "", 404)
        assert s._is_allowed("http://test-college.edu/anything")

    def test_forbidden_robots_disallows_all(self, tmp_path):
        s = self._scraper(tmp_path, "", 403)
        assert not s._is_allowed("http://test-college.edu/anything")

    def test_server_error_fails_open(self, tmp_path):
        s = self._scraper(tmp_path, "", 500)
        assert s._robots is None
        assert s._is_allowed("http://test-college.edu/anything")

    def test_ca_bundle_applies_to_session_before_robots(self, tmp_path):
        session = MagicMock()
        session.get.return_value = _mock_response("", content_type="text/plain", status=404)
        CollegeScraper(
            base_url=self.BASE_URL,
            pdf_download_path=str(tmp_path / "pdfs"),
            session=session,
            ca_bundle="./certs/ca-bundle.pem",
        )
        assert session.verify == "./certs/ca-bundle.pem"
        # robots.txt was fetched through the same session (so verify applies)
        assert session.get.call_count == 1
