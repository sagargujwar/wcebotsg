"""
app/retrieval.py
────────────────
ChromaDB vector search backed by a local SentenceTransformer model.

The SentenceTransformer is loaded once at startup and reused across requests.
Chroma is opened in persistent mode, storing embeddings on disk so the
knowledge base survives restarts without re-embedding.
"""
from __future__ import annotations

import logging
import re
from typing import List, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer

from app.categories import Category, boost_for
from app.config import get_settings
from app.people import find_person_chunks
from app.syllabus import find_syllabus_chunks

logger = logging.getLogger(__name__)

COLLECTION_NAME = "college_kb"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Query-time reranking knobs (see _rerank_chunks).
_POOL_MULTIPLIER = 10      # fetch top_k * 10 candidates from Chroma …
_POOL_MAX = 100            # … but never more than this
_MAX_PER_SOURCE = 2        # max chunks per source_url in the final result
_STUB_MAX_CHARS = 150      # shorter than this = page-header/footer fragment
_STUB_WITH_ATTRS_MAX = 300  # … or short + signature boilerplate
_STUB_PATTERN = re.compile(r"prepared by|checked by", re.IGNORECASE)

# ── Category-aware retrieval (see assemble_results).
# A chunk of the predicted category only receives its score boost when it is
# within _BOOST_BAND of the pool's best raw score.  The prior therefore breaks
# ties among *plausibly relevant* candidates instead of dragging a
# same-category but topically distant chunk above genuinely relevant ones.
_BOOST_BAND = 0.15


def _is_stub(text: str) -> bool:
    """
    True for PDF page-header/footer fragments (e.g. "Syllabus Prepared By
    Dr. … | Syllabus Checked By HOD …").  They match keyword queries strongly
    but contain no answerable content, starving the LLM of real chunks.
    """
    if len(text) < _STUB_MAX_CHARS:
        return True
    return len(text) < _STUB_WITH_ATTRS_MAX and bool(_STUB_PATTERN.search(text))


def _rerank_chunks(chunks: List[dict], top_k: int) -> List[dict]:
    """
    Drop stub chunks and cap repeats per source URL, preserving score order.
    If fewer than top_k survive, backfill with the dropped stubs (by score) so
    the caller always gets up to top_k candidates.
    """
    if len(chunks) <= top_k:
        return chunks

    kept: List[dict] = []
    dropped: List[dict] = []
    per_source: dict = {}
    seen_text: set = set()
    for c in chunks:
        url = c.get("source_url", "")
        key = " ".join(c.get("text", "").split())  # whitespace-normalised
        is_dup_text = key in seen_text
        if (
            not is_dup_text
            and not _is_stub(c.get("text", ""))
            and per_source.get(url, 0) < _MAX_PER_SOURCE
        ):
            kept.append(c)
            per_source[url] = per_source.get(url, 0) + 1
            seen_text.add(key)
        else:
            dropped.append(c)

    for c in dropped:
        if len(kept) >= top_k:
            break
        url = c.get("source_url", "")
        if per_source.get(url, 0) >= _MAX_PER_SOURCE:
            continue
        kept.append(c)
        per_source[url] = per_source.get(url, 0) + 1
    return kept[:top_k]


def _boost_amount(category: str) -> float:
    """Boost for a category *value* string; unknown values boost nothing."""
    try:
        return boost_for(Category(category))
    except ValueError:
        return 0.0


def assemble_results(pool: List[dict], top_k: int, category: Optional[str] = None) -> List[dict]:
    """
    Turn a raw, score-sorted candidate pool into the final top-k result.

    Category-aware path (when *category* is a specific taxonomy value such as
    ``"admissions_fees"``):
      1. Chunks tagged with *category* get a small score boost — but only when
         they sit within _BOOST_BAND of the pool's best raw score (the prior
         breaks ties among plausibly-relevant candidates; it cannot promote
         topically distant ones).
      2. Re-sort and rerank as usual.
      3. **Fallback**: if not a single chunk of *category* survived into the
         result, the classifier's signal was uncorroborated by the pool —
         return the exact baseline ranking instead, so a wrong category can
         never score or order differently than the original pipeline.

    ``None``, ``""`` and ``"general"`` select the baseline behaviour directly.
    """
    if not pool:
        return []
    if not category or category == "general":
        return _rerank_chunks(pool, top_k)

    best_raw = max(c.get("score", 0.0) for c in pool)
    boost = _boost_amount(category)
    boosted: List[dict] = []
    for chunk in pool:
        score = chunk.get("score", 0.0)
        if (
            chunk.get("category") == category
            and (best_raw - score) <= _BOOST_BAND
        ):
            score = min(1.0, score + boost)
        boosted.append({**chunk, "score": score})
    boosted.sort(key=lambda c: c.get("score", 0.0), reverse=True)

    result = _rerank_chunks(boosted, top_k)
    if any(c.get("category") == category for c in result):
        return result
    # Uncorroborated signal → exact baseline.
    return _rerank_chunks(pool, top_k)


class ChromaRetriever:
    """Encapsulates embedding + Chroma similarity search."""

    def __init__(self) -> None:
        settings = get_settings()

        logger.info("Loading embedding model: %s", EMBEDDING_MODEL)
        self._encoder = SentenceTransformer(EMBEDDING_MODEL)

        logger.info("Opening Chroma at: %s", settings.chroma_path)
        self._client = chromadb.PersistentClient(
            path=settings.chroma_path,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "Chroma collection '%s' ready — %d chunks",
            COLLECTION_NAME,
            self._collection.count(),
        )

    # ── Connectivity ──────────────────────────────────────────────────────────

    def ping(self) -> bool:
        """Return True if the Chroma collection is accessible."""
        try:
            self._collection.count()
            return True
        except Exception:
            return False

    # ── Search ────────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int,
        category: Optional[str] = None,
    ) -> List[dict]:
        """
        Embed *query* and return the top-k most similar chunks.

        Parameters
        ----------
        category:
            Optional taxonomy value (e.g. ``"admissions_fees"``) from the
            intent classifier.  Matching chunks receive a relevance-banded
            score boost (see ``assemble_results``); ``None``/``"general"``
            keeps the original baseline behaviour exactly.

        Each result dict contains:
            text        – the chunk body
            source_url  – where the chunk was scraped from
            doc_type    – "html" or "pdf"
            title       – page/document title
            category    – taxonomy tag of the chunk ("" when untagged)
            score       – cosine similarity (0–1; higher = more similar)
        """
        if self._collection.count() == 0:
            logger.warning("Chroma collection is empty — run refresh_kb.py first")
            return []

        query_embedding = self._encoder.encode(query, normalize_embeddings=True).tolist()

        # Targeted document lookups: when the query names an individual, pull
        # their own resume/profile to the front; when it names a branch
        # syllabus ("third year syllabus for cse"), pull that document's
        # course-content chunks.  Without this, a name lives only in
        # aggregate tables (AISHE/NIRF) and a syllabus only behind shared
        # assessment boilerplate — both outrank the dedicated document, so
        # the model answers from — and links — the wrong source.
        targeted = find_person_chunks(self._collection, query)
        if not targeted:
            targeted = find_syllabus_chunks(self._collection, query)

        # Over-fetch, then assemble_results category-boosts near-relevant
        # matches and _rerank_chunks drops header/footer stubs and caps
        # per-source repeats so the LLM sees diverse, content-bearing chunks.
        pool = min(max(top_k * _POOL_MULTIPLIER, top_k), _POOL_MAX)
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(pool, self._collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        chunks: List[dict] = []
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for doc, meta, dist in zip(documents, metadatas, distances):
            # Chroma with cosine space returns distance (0 = identical, 2 = opposite).
            # Convert to similarity score in [0, 1].
            score = max(0.0, 1.0 - dist / 2.0)
            chunks.append(
                {
                    "text": doc,
                    "source_url": meta.get("source_url", ""),
                    "doc_type": meta.get("doc_type", "html"),
                    "title": meta.get("title", ""),
                    "category": meta.get("category", ""),
                    "score": score,
                }
            )

        if targeted:
            seen = {c["source_url"] for c in targeted}
            chunks = targeted + [c for c in chunks if c["source_url"] not in seen]
            logger.info(
                "Doc targeting: %d chunk(s) from %s",
                len(targeted),
                targeted[0]["source_url"],
            )

        return assemble_results(chunks, top_k, category)

    # ── Upsert (used by refresh_kb.py) ───────────────────────────────────────

    def upsert(
        self,
        ids: List[str],
        documents: List[str],
        embeddings: List[List[float]],
        metadatas: List[dict],
    ) -> None:
        """Upsert chunks into the collection (idempotent — re-runs are safe).

        Chroma's Rust backend caps a single upsert (max batch size ≈ 5,461 on
        this machine; limit varies with ONNX runtime threads), so slice into
        small batches instead of sending everything in one call.
        """
        batch = 4000
        for start in range(0, len(ids), batch):
            end = start + batch
            self._collection.upsert(
                ids=ids[start:end],
                documents=documents[start:end],
                embeddings=embeddings[start:end],
                metadatas=metadatas[start:end],
            )

    def get_encoder(self) -> SentenceTransformer:
        """Expose the encoder so refresh_kb.py can embed chunks."""
        return self._encoder
