"""
scraper/chunker.py
──────────────────
Paragraph-boundary text chunker with overlap.

Chunking strategy:
  - Split on double-newlines (paragraph boundaries) and Markdown-style headings.
  - Merge very short fragments (< MIN_CHUNK_CHARS) into the previous chunk.
  - Split any paragraph longer than MAX_CHUNK_CHARS at sentence boundaries so
    the embedding model (256 wordpieces) never silently truncates a chunk.
  - Add up to ~15% overlap by prepending the tail (last 1–2 sentences) of the
    previous chunk — hard-capped so a whole paragraph is never duplicated.
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

MIN_CHUNK_CHARS = 50     # fragments shorter than this merge into their neighbour
MAX_CHUNK_CHARS = 1500   # hard cap — enforced at sentence/word boundaries
OVERLAP_RATIO = 0.15     # overlap tail is at most 15% of the paragraph
MIN_OVERLAP_CHARS = 20   # shorter tails aren't worth carrying forward

# Regex: split on blank lines OR lines that look like headings
_SPLIT_PATTERN = re.compile(r"\n\s*\n+|(?=\n#{1,6}\s)")

# Sentence boundary approximation for overlap extraction
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")

# Candidate cut points inside an oversized paragraph: sentence-ending
# punctuation followed by whitespace (supports closing quotes/brackets).
_SENTENCE_BREAK = re.compile(r"[.!?][\"')\]]?(?=\s)")


def _deterministic_id(source_url: str, chunk_index: int) -> str:
    """Return a stable SHA-256 hex string for (url, index)."""
    raw = f"{source_url}:::{chunk_index}"
    return hashlib.sha256(raw.encode()).hexdigest()


def _last_sentences(text: str, n: int = 2) -> str:
    """Return the last *n* sentences from *text* for overlap."""
    sentences = _SENTENCE_END.split(text.strip())
    tail = sentences[-n:] if len(sentences) >= n else sentences
    return " ".join(tail).strip()


def _word_break(text: str, start: int, limit: int) -> int:
    """Return the index to cut *text* at so text[start:end] <= limit chars."""
    end = min(start + limit, len(text))
    if end >= len(text):
        return len(text)
    pos = text.rfind(" ", start, end)
    if pos > start:
        return pos
    return end  # single unbroken run longer than the limit — force-cut


def _split_long(text: str, limit: int) -> List[str]:
    """
    Split *text* into pieces of at most *limit* characters.

    Prefers sentence boundaries; a stretch without sentence breaks (or a
    single sentence longer than *limit*) is hard-split at word boundaries.
    """
    if len(text) <= limit:
        return [text]

    breaks = [m.end() for m in _SENTENCE_BREAK.finditer(text)]
    pieces: List[str] = []
    start = 0
    idx = 0
    while start < len(text):
        # Push the split point as far as possible while staying within limit.
        last_good = -1
        while idx < len(breaks) and breaks[idx] - start <= limit:
            last_good = breaks[idx]
            idx += 1
        if last_good > start:
            piece = text[start:last_good].strip()
            start = last_good
        else:
            cut = _word_break(text, start, limit)
            piece = text[start:cut].strip()
            start = cut
        if piece:
            pieces.append(piece)
    return pieces or [text[:limit]]


def _trim_tail_to(tail: str, budget: int) -> str:
    """Trim *tail* from the front (word boundary) so it fits *budget* chars."""
    if budget <= 0:
        return ""
    if len(tail) <= budget:
        return tail
    window = tail[len(tail) - budget:]
    space = window.find(" ")
    if space == 0:
        window = window[1:]
    elif space > 0:
        window = window[space + 1:]
    return window


def _overlap_tail(para: str) -> str:
    """
    Return the tail of *para* to prepend to the next chunk.

    Last 1–2 sentences, trimmed to at most OVERLAP_RATIO of the paragraph and
    never more than fits alongside *para* in one MAX_CHUNK_CHARS chunk.
    Dropped entirely when it would duplicate the paragraph or is too small to
    help retrieval.
    """
    tail = _last_sentences(para, n=2)
    if not tail or len(tail) >= len(para):
        return ""
    budget = min(int(OVERLAP_RATIO * len(para)), MAX_CHUNK_CHARS - len(para) - 1)
    tail = _trim_tail_to(tail, budget)
    if len(tail) < MIN_OVERLAP_CHARS:
        return ""
    return tail


def _fit_overlap(tail: str, para: str) -> str:
    """
    Return *tail* trimmed so that ``tail + " " + para`` never exceeds
    MAX_CHUNK_CHARS.

    The ratio-capped tail from _overlap_tail is sized against the *previous*
    paragraph — when the current one is longer, that alone isn't enough, so
    the tail is re-trimmed here at emission time.
    """
    if not tail:
        return ""
    allowed = MAX_CHUNK_CHARS - len(para) - 1
    if allowed < MIN_OVERLAP_CHARS:
        return ""
    tail = _trim_tail_to(tail, allowed)
    return tail if len(tail) >= MIN_OVERLAP_CHARS else ""


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

        # ── Merge short fragments into the previous chunk ───────────────────
        merged: List[str] = []
        leading = ""  # fragments appearing before the first full paragraph
        for para in paragraphs:
            if len(para) < MIN_CHUNK_CHARS:
                if merged:
                    merged[-1] = merged[-1] + "\n\n" + para
                else:
                    leading = (leading + "\n\n" + para) if leading else para
                continue
            if leading:
                para = leading + "\n\n" + para
                leading = ""
            merged.append(para)
        if leading:  # document made only of short fragments
            merged.append(leading)

        if not merged:
            logger.debug("No usable chunks extracted from %s", source_url)
            return []

        # ── Enforce MAX_CHUNK_CHARS at sentence boundaries ──────────────────
        pieces: List[str] = []
        for para in merged:
            pieces.extend(_split_long(para, MAX_CHUNK_CHARS))

        # ── Add overlap ─────────────────────────────────────────────────────
        chunks: List[dict] = []
        previous_tail = ""
        idx = 0

        for para in pieces:
            tail = _fit_overlap(previous_tail, para)
            chunk_text = f"{tail} {para}".strip() if tail else para

            if len(chunk_text) > MAX_CHUNK_CHARS:
                logger.warning(
                    "Chunk %d from %s exceeds %d chars (%d)",
                    idx, source_url, MAX_CHUNK_CHARS, len(chunk_text),
                )

            chunks.append(
                {
                    "id": _deterministic_id(source_url, idx),
                    "text": chunk_text,
                    "source_url": source_url,
                    "doc_type": doc_type,
                    "title": title,
                    "last_scraped_at": last_scraped_at,
                    "chunk_index": idx,
                }
            )
            idx += 1

            # Tail for the next chunk's overlap (≤15%, never a whole paragraph)
            previous_tail = _overlap_tail(para)

        logger.debug(
            "Chunked '%s' into %d chunks (source: %s)", title, len(chunks), source_url
        )
        return chunks
