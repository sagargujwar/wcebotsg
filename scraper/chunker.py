"""
scraper/chunker.py
──────────────────
Paragraph-boundary text chunker with overlap.

Chunking strategy:
  - Split on double-newlines (paragraph boundaries) and Markdown-style headings.
  - Merge very short fragments (< MIN_CHUNK_CHARS) into the previous chunk.
  - Add ~10–15% overlap by prepending the last 1–2 sentences of the previous
    chunk to each new chunk.
  - Generate a deterministic SHA-256 chunk ID from (source_url + chunk_index)
    so re-running the scraper upserts rather than duplicates in Chroma.
"""
from __future__ import annotations

import hashlib
import logging
import re
from datetime import datetime, timezone
from typing import List

logger = logging.getLogger(__name__)

MIN_CHUNK_CHARS = 50   # discard very short fragments
MAX_CHUNK_CHARS = 1500  # soft cap — don't split mid-sentence, but warn

# Regex: split on blank lines OR lines that look like headings
_SPLIT_PATTERN = re.compile(r"\n\s*\n+|(?=\n#{1,6}\s)")

# Sentence boundary approximation for overlap extraction
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def _deterministic_id(source_url: str, chunk_index: int) -> str:
    """Return a stable SHA-256 hex string for (url, index)."""
    raw = f"{source_url}:::{chunk_index}"
    return hashlib.sha256(raw.encode()).hexdigest()


def _last_sentences(text: str, n: int = 2) -> str:
    """Return the last *n* sentences from *text* for overlap."""
    sentences = _SENTENCE_END.split(text.strip())
    tail = sentences[-n:] if len(sentences) >= n else sentences
    return " ".join(tail).strip()


class TextChunker:
    """Splits document text into overlapping paragraph-boundary chunks."""

    def chunk(
        self,
        text: str,
        source_url: str,
        doc_type: str,
        title: str,
        last_scraped_at: str = "",
    ) -> List[dict]:
        """
        Chunk *text* and return a list of chunk dicts ready for Chroma upsert:
            {id, text, source_url, doc_type, title, last_scraped_at, chunk_index}
        """
        if not last_scraped_at:
            last_scraped_at = datetime.now(timezone.utc).isoformat()

        raw_paragraphs = _SPLIT_PATTERN.split(text)
        paragraphs = [p.strip() for p in raw_paragraphs if p.strip()]

        # ── Merge short fragments ──────────────────────────────────────────
        merged: List[str] = []
        buffer = ""
        for para in paragraphs:
            if len(para) < MIN_CHUNK_CHARS:
                buffer = (buffer + "\n\n" + para).strip() if buffer else para
            else:
                if buffer:
                    merged.append(buffer)
                    buffer = ""
                merged.append(para)
        if buffer:
            merged.append(buffer)

        if not merged:
            logger.debug("No usable chunks extracted from %s", source_url)
            return []

        # ── Add overlap ────────────────────────────────────────────────────
        chunks: List[dict] = []
        previous_tail = ""

        for idx, para in enumerate(merged):
            chunk_text = (previous_tail + " " + para).strip() if previous_tail else para

            # Warn on very long chunks (don't force-split to avoid breaking tables)
            if len(chunk_text) > MAX_CHUNK_CHARS:
                logger.debug(
                    "Chunk %d from %s is long (%d chars)",
                    idx, source_url, len(chunk_text),
                )

            chunk_id = _deterministic_id(source_url, idx)
            chunks.append(
                {
                    "id": chunk_id,
                    "text": chunk_text,
                    "source_url": source_url,
                    "doc_type": doc_type,
                    "title": title,
                    "last_scraped_at": last_scraped_at,
                    "chunk_index": idx,
                }
            )

            # Tail for next chunk's overlap (~10–15%: last 2 sentences)
            overlap_candidate = _last_sentences(para, n=2)
            # Only use overlap if it's a meaningful fraction of the chunk
            if len(overlap_candidate) > 0.1 * len(para):
                previous_tail = overlap_candidate
            else:
                previous_tail = ""

        logger.debug(
            "Chunked '%s' into %d chunks (source: %s)", title, len(chunks), source_url
        )
        return chunks
