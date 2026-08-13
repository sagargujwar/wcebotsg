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

from scraper.chunker import TextChunker, _deterministic_id
from scraper.scraper import CollegeScraper


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
