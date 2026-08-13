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
from typing import List, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer

from app.config import get_settings

logger = logging.getLogger(__name__)

COLLECTION_NAME = "college_kb"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


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

    def search(self, query: str, top_k: int) -> List[dict]:
        """
        Embed *query* and return the top-k most similar chunks.

        Each result dict contains:
            text        – the chunk body
            source_url  – where the chunk was scraped from
            doc_type    – "html" or "pdf"
            title       – page/document title
            score       – cosine similarity (0–1; higher = more similar)
        """
        if self._collection.count() == 0:
            logger.warning("Chroma collection is empty — run refresh_kb.py first")
            return []

        query_embedding = self._encoder.encode(query, normalize_embeddings=True).tolist()

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self._collection.count()),
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
                    "score": score,
                }
            )

        return chunks

    # ── Upsert (used by refresh_kb.py) ───────────────────────────────────────

    def upsert(
        self,
        ids: List[str],
        documents: List[str],
        embeddings: List[List[float]],
        metadatas: List[dict],
    ) -> None:
        """Upsert chunks into the collection (idempotent — re-runs are safe)."""
        self._collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def get_encoder(self) -> SentenceTransformer:
        """Expose the encoder so refresh_kb.py can embed chunks."""
        return self._encoder
