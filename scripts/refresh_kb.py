#!/usr/bin/env python
"""
scripts/refresh_kb.py
─────────────────────
Standalone knowledge-base refresh script.

Run manually or on a schedule (see README for cron instructions):
    python scripts/refresh_kb.py

What it does:
  1. Crawls BASE_URL up to MAX_CRAWL_DEPTH hops.
  2. Chunks all extracted text (paragraph boundaries + overlap).
  3. Embeds chunks locally with sentence-transformers.
  4. Upserts into ChromaDB (deterministic IDs → no duplicates on re-run).
  5. Logs a summary.
"""
from __future__ import annotations

import logging
import os
import sys
import time

# Ensure project root is on the path when run from scripts/ or project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.config import get_settings
from app.retrieval import ChromaRetriever
from scraper.chunker import TextChunker
from scraper.scraper import CollegeScraper

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger("refresh_kb")


def main() -> None:
    settings = get_settings()

    logger.info("=" * 60)
    logger.info("WCE Bot — Knowledge Base Refresh")
    logger.info("Base URL : %s", settings.base_url)
    logger.info("Max depth: %d", settings.max_crawl_depth)
    logger.info("=" * 60)

    t_start = time.perf_counter()

    # ── Step 1: Crawl ─────────────────────────────────────────────────────────
    scraper = CollegeScraper(
        base_url=settings.base_url,
        pdf_download_path=settings.pdf_download_path,
    )
    docs = scraper.crawl(max_depth=settings.max_crawl_depth)
    logger.info("Crawled %d documents.", len(docs))

    if not docs:
        logger.warning("No documents crawled — check BASE_URL and network connectivity.")
        sys.exit(1)

    # ── Step 2: Chunk ─────────────────────────────────────────────────────────
    chunker = TextChunker()
    all_chunks: list[dict] = []
    for doc in docs:
        chunks = chunker.chunk(
            text=doc["text"],
            source_url=doc["url"],
            doc_type=doc["doc_type"],
            title=doc["title"],
            last_scraped_at=doc["last_scraped_at"],
        )
        all_chunks.extend(chunks)

    logger.info("Generated %d chunks from %d documents.", len(all_chunks), len(docs))

    if not all_chunks:
        logger.warning("No chunks generated — documents may be empty or too short.")
        sys.exit(1)

    # ── Step 3 & 4: Embed + Upsert ────────────────────────────────────────────
    retriever = ChromaRetriever()
    encoder = retriever.get_encoder()

    logger.info("Embedding %d chunks …", len(all_chunks))
    texts = [c["text"] for c in all_chunks]
    embeddings = encoder.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=64,
    ).tolist()

    ids = [c["id"] for c in all_chunks]
    metadatas = [
        {
            "source_url": c["source_url"],
            "doc_type": c["doc_type"],
            "title": c["title"],
            "last_scraped_at": c["last_scraped_at"],
            "chunk_index": c["chunk_index"],
        }
        for c in all_chunks
    ]

    logger.info("Upserting into ChromaDB …")
    retriever.upsert(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    # ── Step 5: Summary ───────────────────────────────────────────────────────
    elapsed = time.perf_counter() - t_start
    logger.info("=" * 60)
    logger.info("Knowledge base refresh complete!")
    logger.info("  Documents crawled : %d", len(docs))
    logger.info("  Chunks upserted   : %d", len(all_chunks))
    logger.info("  Time elapsed      : %.1fs", elapsed)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
