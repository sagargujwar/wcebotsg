#!/usr/bin/env python
"""
Tag every existing Chroma chunk with its taxonomy category (in place).

The normal path — `scripts/refresh_kb.py` — computes `category` at ingestion.
This script is the one-shot migration for a knowledge base that was built
before the taxonomy existed: it rewrites ONLY the metadata, so embeddings are
untouched and no re-embedding (or network) is needed.

Usage:
    python scripts/tag_categories.py          # tag missing categories
    python scripts/tag_categories.py --force  # recompute every chunk
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.categories import Category, url_to_category  # noqa: E402
from app.retrieval import ChromaRetriever  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s")
logger = logging.getLogger("tag_categories")

_BATCH = 500


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill chunk categories in ChromaDB.")
    parser.add_argument("--force", action="store_true", help="Recompute even when a category already exists.")
    args = parser.parse_args()

    retriever = ChromaRetriever()
    collection = retriever._collection  # noqa: SLF001 — script-local use

    data = collection.get(include=["metadatas"])
    ids = data.get("ids", [])
    metadatas = data.get("metadatas", []) or []
    logger.info("Loaded %d chunks.", len(ids))

    updates: list[tuple[str, dict]] = []
    dist: Counter = Counter()
    for chunk_id, meta in zip(ids, metadatas):
        meta = dict(meta or {})
        existing = meta.get("category")
        if existing and not args.force:
            dist[existing] += 1
            continue
        category = url_to_category(meta.get("source_url", ""), meta.get("title", ""))
        meta["category"] = category.value
        updates.append((chunk_id, meta))
        dist[category.value] += 1

    if not updates:
        logger.info("Nothing to do — all chunks already tagged. Distribution: %s", dict(dist.most_common()))
        return

    logger.info("Updating %d chunk metadata records (no re-embedding) …", len(updates))
    for start in range(0, len(updates), _BATCH):
        batch = updates[start:start + _BATCH]
        collection.update(
            ids=[u[0] for u in batch],
            metadatas=[u[1] for u in batch],
        )

    logger.info("Done. Category distribution: %s",
                ", ".join(f"{c}={n}" for c, n in dist.most_common()))


if __name__ == "__main__":
    main()
