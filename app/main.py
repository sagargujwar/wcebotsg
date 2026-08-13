"""
app/main.py
───────────
FastAPI application: routes, middleware, lifespan, rate limiting.

Endpoints:
  POST /chat  — main RAG query endpoint
  GET  /health — service health check
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.cache import RedisCache
from app.config import get_settings
from app.llm import LLMService
from app.models import ChatRequest, ChatResponse, ErrorResponse, HealthResponse
from app.retrieval import ChromaRetriever

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ── Application state (initialised in lifespan) ───────────────────────────────
_cache: Optional[RedisCache] = None
_retriever: Optional[ChromaRetriever] = None
_llm: Optional[LLMService] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise heavyweight singletons once at startup."""
    global _cache, _retriever, _llm

    logger.info("WCE Bot starting up …")
    _cache = RedisCache()
    _retriever = ChromaRetriever()
    _llm = LLMService()
    logger.info("WCE Bot ready.")

    yield

    logger.info("WCE Bot shutting down …")


# ── App factory ───────────────────────────────────────────────────────────────
settings = get_settings()

app = FastAPI(
    title="WCE Bot — College Information RAG API",
    description=(
        "Retrieval-Augmented Generation chatbot answering student questions "
        "from college documents (fee structure, syllabus, admissions, notices)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins_list(),
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


# ── Rate limiting (manual Redis-based, no extra dependency) ───────────────────
async def _check_rate_limit(request: Request) -> None:
    """Raise HTTP 429 if the client IP has exceeded the configured rate limit."""
    # Extract real IP (handles X-Forwarded-For for proxied deployments)
    forwarded_for = request.headers.get("X-Forwarded-For")
    client_ip = forwarded_for.split(",")[0].strip() if forwarded_for else (
        request.client.host if request.client else "unknown"
    )

    if _cache and not _cache.check_rate_limit(
        client_ip,
        settings.rate_limit_requests,
        settings.rate_limit_window_seconds,
    ):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=(
                f"Rate limit exceeded — maximum {settings.rate_limit_requests} "
                f"requests per {settings.rate_limit_window_seconds} seconds."
            ),
            headers={"Retry-After": str(settings.rate_limit_window_seconds)},
        )


# ── Routes ────────────────────────────────────────────────────────────────────

@app.post(
    "/chat",
    response_model=ChatResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid query"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
    summary="Ask WCE Bot a question about the college",
)
async def chat(request: Request, body: ChatRequest) -> ChatResponse:
    """
    Full RAG query-time flow:
      1. Validate input (handled by Pydantic model)
      2. Rate-limit check per client IP
      3. Redis cache lookup
      4. Chroma vector search
      5. Similarity threshold gate
      6. LLM call with grounding system prompt
      7. PDF source URL injection
      8. Cache + return
    """
    # Step 2 — Rate limit
    await _check_rate_limit(request)

    query = body.query

    # Step 3 — Cache lookup
    if _cache:
        cached = _cache.get(query)
        if cached:
            logger.info("Cache HIT for query: %.60s", query)
            return ChatResponse(**cached, cached=True)

    # Step 4 — Vector search
    chunks = []
    if _retriever:
        chunks = _retriever.search(query, settings.top_k)

    # Step 5 — Similarity threshold gate
    if not chunks or chunks[0]["score"] < settings.similarity_threshold:
        logger.info(
            "No relevant chunks (best score=%.3f) — returning fallback",
            chunks[0]["score"] if chunks else 0.0,
        )
        fallback_answer = (
            "I couldn't find that in the college's published documents — "
            f"you can check {settings.official_site_fallback_url} directly."
        )
        return ChatResponse(answer=fallback_answer, source_url=None, cached=False)

    # Step 6 — LLM call
    answer = _UNAVAILABLE_MSG
    if _llm:
        answer = _llm.generate(query, chunks)

    # Step 7 — Inject PDF source URL when the top chunk is a PDF
    top_chunk = chunks[0]
    source_url: Optional[str] = None
    if top_chunk.get("doc_type") == "pdf":
        source_url = top_chunk.get("source_url") or None
    else:
        source_url = top_chunk.get("source_url") or None

    # Step 8 — Cache response (only non-error answers)
    response_payload = {"answer": answer, "source_url": source_url}
    if _cache and answer != _UNAVAILABLE_MSG:
        _cache.set(query, response_payload)

    return ChatResponse(answer=answer, source_url=source_url, cached=False)


_UNAVAILABLE_MSG = (
    "I'm sorry, the WCE Bot service is temporarily unavailable. "
    "Please try again in a few moments."
)


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Service health check",
)
async def health() -> HealthResponse:
    """Return connectivity status for Chroma and Redis."""
    chroma_ok = _retriever.ping() if _retriever else False
    redis_ok = _cache.ping() if _cache else False
    overall = "ok" if (chroma_ok and redis_ok) else "degraded"
    return HealthResponse(status=overall, chroma_ok=chroma_ok, redis_ok=redis_ok)


# ── Custom exception handlers ─────────────────────────────────────────────────

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
        headers=getattr(exc, "headers", None) or {},
    )
