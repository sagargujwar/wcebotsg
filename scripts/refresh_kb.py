#!/usr/bin/env python
"""Refresh the WCE Bot knowledge base using the persistent scraper cache."""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.categories import Category, url_to_category
from app.config import get_settings
from app.retrieval import ChromaRetriever
from scraper.chunker import TextChunker, _deterministic_id
from scraper.scraper import CollegeScraper

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s")
logger = logging.getLogger("refresh_kb")
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SOCIAL_DOCS_PATH = os.path.join(PROJECT_ROOT, "static_docs", "social_media.json")


def build_social_chunks(discovered: dict[str, str]) -> list[dict]:
    entries: list[dict] = []
    try:
        with open(SOCIAL_DOCS_PATH, encoding="utf-8") as fh:
            entries = json.load(fh).get("entries", [])
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not load %s: %s", SOCIAL_DOCS_PATH, exc)

    known = {entry.get("platform", "") for entry in entries}
    for platform, url in discovered.items():
        if platform not in known:
            entries.append({
                "platform": platform,
                "handle": "",
                "url": url,
                "description": f"Official {platform} profile of Walchand College of Engineering, Sangli, discovered on the college website.",
            })

    now = datetime.now(timezone.utc).isoformat()
    chunks: list[dict] = []
    for entry in entries:
        platform = (entry.get("platform") or "").strip()
        url = (entry.get("url") or "").strip()
        handle = (entry.get("handle") or "").strip()
        description = (entry.get("description") or "").strip()
        if not platform:
            continue
        text = f"{platform} account of Walchand College of Engineering, Sangli (WCE).\n"
        if handle:
            text += f"Handle/Name: {handle}\n"
        if url:
            text += f"Link: {url}\n"
        if description:
            text += description
        chunks.append({
            "id": _deterministic_id(f"static:social:{platform}", 0),
            "text": text.strip(),
            "source_url": url,
            "doc_type": "social",
            "title": f"WCE {platform}",
            "last_scraped_at": now,
            "chunk_index": 0,
            "category": Category.GENERAL_INFO.value,
        })
    return chunks


def dedupe_chunks(chunks: list[dict]) -> list[dict]:
    seen: set[str] = set()
    unique: list[dict] = []
    for chunk in chunks:
        if chunk["id"] not in seen:
            seen.add(chunk["id"])
            unique.append(chunk)
    return unique


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh the WCE Bot knowledge base.")
    parser.add_argument("--force-refresh", action="store_true", help="Ignore the crawl cache and fetch pages/PDFs again.")
    parser.add_argument("--cache-ttl-days", type=int, default=7, help="Reuse cached documents for this many days (default: 7; 0 disables cache).")
    args = parser.parse_args()

    settings = get_settings()
    pdf_path = os.path.abspath(settings.pdf_download_path)
    cache_path = os.path.join(os.path.dirname(pdf_path), "crawl_cache.json")

    logger.info("=" * 60)
    logger.info("WCE Bot — Knowledge Base Refresh")
    logger.info("Base URL : %s", settings.base_url)
    logger.info("Max depth: %d", settings.max_crawl_depth)
    logger.info("Max pages: %s", settings.max_pages or "unlimited")
    logger.info("Cache    : %s", cache_path)
    logger.info("Cache TTL: %d days%s", args.cache_ttl_days, " (disabled)" if args.cache_ttl_days <= 0 else "")
    logger.info("Mode     : %s", "FORCE REFRESH" if args.force_refresh else "cached refresh")
    logger.info("=" * 60)

    t_start = time.perf_counter()
    ca_bundle = ""
    if settings.ssl_ca_bundle.strip():
        candidate = os.path.abspath(settings.ssl_ca_bundle.strip())
        if os.path.isfile(candidate):
            ca_bundle = candidate
            logger.info("CA bundle: %s", ca_bundle)
        else:
            logger.warning("SSL_CA_BUNDLE not found (%s) — using certifi defaults.", candidate)

    scraper = CollegeScraper(
        base_url=settings.base_url,
        pdf_download_path=pdf_path,
        ca_bundle=ca_bundle or None,
        max_pages=settings.max_pages,
        cache_path=cache_path,
        cache_ttl_days=args.cache_ttl_days,
        force_refresh=args.force_refresh,
    )
    docs = scraper.crawl(max_depth=settings.max_crawl_depth)
    logger.info("Crawled/reused %d documents (%d cache hits, %d network fetches).", len(docs), scraper.cache_hits, scraper.network_fetches)

    if not docs:
        logger.warning("No documents crawled — check BASE_URL and network connectivity.")
        sys.exit(1)

    chunker = TextChunker()
    all_chunks: list[dict] = []
    for doc in docs:
        # Tag every chunk of a document with its taxonomy category (URL-prefix
        # match, falling back to a title lexicon — see app/categories.py).
        category = url_to_category(doc["url"], doc["title"]).value
        chunks = chunker.chunk(
            text=doc["text"],
            source_url=doc["url"],
            doc_type=doc["doc_type"],
            title=doc["title"],
            last_scraped_at=doc["last_scraped_at"],
        )
        for chunk in chunks:
            chunk["category"] = category
        all_chunks.extend(chunks)

    logger.info("Generated %d chunks from %d documents.", len(all_chunks), len(docs))
    if scraper.social_links:
        logger.info("Discovered social links: %s", dict(scraper.social_links))
    social_chunks = build_social_chunks(scraper.social_links)
    all_chunks.extend(social_chunks)
    logger.info("Added %d static social-media chunks.", len(social_chunks))

    if not all_chunks:
        logger.warning("No chunks generated — documents may be empty or too short.")
        sys.exit(1)

    unique_chunks = dedupe_chunks(all_chunks)
    if len(unique_chunks) != len(all_chunks):
        logger.warning("Dropped %d duplicate chunk IDs — %d unique remain.", len(all_chunks) - len(unique_chunks), len(unique_chunks))
    all_chunks = unique_chunks

    retriever = ChromaRetriever()
    encoder = retriever.get_encoder()
    logger.info("Embedding %d chunks …", len(all_chunks))
    texts = [chunk["text"] for chunk in all_chunks]
    embeddings = encoder.encode(texts, normalize_embeddings=True, show_progress_bar=True, batch_size=64).tolist()
    ids = [chunk["id"] for chunk in all_chunks]
    metadatas = [{
        "source_url": chunk["source_url"],
        "doc_type": chunk["doc_type"],
        "title": chunk["title"],
        "last_scraped_at": chunk["last_scraped_at"],
        "chunk_index": chunk["chunk_index"],
        "category": chunk.get("category", Category.GENERAL.value),
    } for chunk in all_chunks]

    logger.info("Upserting into ChromaDB …")
    retriever.upsert(ids=ids, documents=texts, embeddings=embeddings, metadatas=metadatas)

    dist = Counter(m["category"] for m in metadatas)
    logger.info(
        "Category distribution: %s",
        ", ".join(f"{c}={n}" for c, n in dist.most_common()),
    )

    elapsed = time.perf_counter() - t_start
    logger.info("=" * 60)
    logger.info("Knowledge base refresh complete!")
    logger.info("  Documents reused/crawled : %d", len(docs))
    logger.info("  Cache hits               : %d", scraper.cache_hits)
    logger.info("  Network fetches          : %d", scraper.network_fetches)
    logger.info("  Chunks upserted          : %d", len(all_chunks))
    logger.info("  Time elapsed             : %.1fs", elapsed)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
