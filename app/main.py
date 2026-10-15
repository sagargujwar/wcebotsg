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
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from app.cache import RedisCache
from app.config import get_settings
from app.constants import NO_INFO_MESSAGE, UNAVAILABLE_MESSAGE
from app.llm import LLMService
from app.models import ChatRequest, ChatResponse, ErrorResponse, HealthResponse
from app.retrieval import ChromaRetriever
from app.social import SocialService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ── Application state (initialised in lifespan) ───────────────────────────────
_cache: Optional[RedisCache] = None
_retriever: Optional[ChromaRetriever] = None
_llm: Optional[LLMService] = None
_social: Optional[SocialService] = None

_FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise heavyweight singletons once at startup."""
    global _cache, _retriever, _llm, _social

    logger.info("WCE Bot starting up …")
    _cache = RedisCache()
    _retriever = ChromaRetriever()
    _llm = LLMService()
    _social = SocialService()
    logger.info(
        "WCE Bot ready. (YouTube feed: %s)",
        "enabled" if _social.enabled else "disabled — set YOUTUBE_CHANNEL_ID",
    )

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
# Exact origins come from ALLOWED_ORIGINS; the regex additionally allows any
# localhost/127.0.0.1 port (file:// pages send Origin "null", which is matched
# via allow_origins). Without this, serving the frontend from e.g. Live Server
# (http://localhost:5500) is blocked by the browser with a CORS error.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins_list(),
    allow_origin_regex=settings.allowed_origin_regex,
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

def _is_no_info(answer: str) -> bool:
    """
    True when *answer* is the standard "no information" refusal.

    The LLM is instructed to reply with NO_INFO_MESSAGE only, but it may wrap
    it in quotes or add whitespace — normalise before comparing so the source
    link is suppressed either way.
    """
    normalised = answer.strip().strip("\"'“”‘’").lstrip("-–—: ").strip()
    return normalised.startswith(NO_INFO_MESSAGE)


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
      4. Chroma vector search (+ optional live YouTube context)
      5. Similarity threshold gate
      6. LLM call with grounding system prompt
      7. Source URL injection — suppressed for "no information" replies
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
            if _is_no_info(cached.get("answer", "")):
                cached["source_url"] = None
            return ChatResponse(**cached, cached=True)

    # Step 4 — Vector search + live social context
    chunks = []
    if _retriever:
        chunks = _retriever.search(query, settings.top_k)

    if _social:
        social_chunks = _social.context_chunks(query)
        if social_chunks:
            logger.info("Adding %d live social chunk(s)", len(social_chunks))
            chunks = list(chunks) + social_chunks

    # Step 5 — Similarity threshold gate
    best_score = max((c.get("score", 0.0) for c in chunks), default=0.0)
    if not chunks or best_score < settings.similarity_threshold:
        logger.info(
            "No relevant chunks (best score=%.3f) — returning fallback",
            best_score,
        )
        # Out of scope / not in the KB: exact refusal, never a document link.
        return ChatResponse(answer=NO_INFO_MESSAGE, source_url=None, cached=False)

    # Step 6 — LLM call
    answer = UNAVAILABLE_MESSAGE
    if _llm:
        answer = _llm.generate(query, chunks)

    # Step 7 — Source URL; suppressed for refusals
    source_url: Optional[str] = None
    if not _is_no_info(answer):
        # Prefer the highest-scoring chunk that actually carries a URL (static
        # informational chunks such as "no official X account" have none).
        with_url = [c for c in chunks if c.get("source_url")]
        if with_url:
            top_chunk = max(with_url, key=lambda c: c.get("score", 0.0))
            source_url = top_chunk.get("source_url") or None

    # Step 8 — Cache response (only real answers — never errors or refusals,
    # so an improved KB/LLM is not masked by a stale "no info" for 24 h)
    response_payload = {"answer": answer, "source_url": source_url}
    if (
        _cache
        and answer != UNAVAILABLE_MESSAGE
        and not _is_no_info(answer)
    ):
        _cache.set(query, response_payload)

    return ChatResponse(answer=answer, source_url=source_url, cached=False)


# ── Embeddable widget assets ─────────────────────────────────────────────────

@app.get(
    "/widget",
    summary="Chat panel for the floating widget iframe",
    response_class=FileResponse,
)
async def widget_panel() -> FileResponse:
    """Serve the chat UI — loaded inside the widget's <iframe>."""
    return FileResponse(_FRONTEND_DIR / "index.html", media_type="text/html")


@app.get(
    "/widget.js",
    summary="Floating chat button embed loader",
    response_class=FileResponse,
)
async def widget_script() -> FileResponse:
    """
    Serve the one-line embed script. Host pages include it as:

        <script src="http://localhost:8000/widget.js" defer></script>
    """
    return FileResponse(
        _FRONTEND_DIR / "widget.js", media_type="application/javascript"
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
