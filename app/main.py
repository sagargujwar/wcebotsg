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
import re
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from app.cache import RedisCache
from app.categories import Category, expand_query, get_spec, threshold_for
from app.config import get_settings
from app.constants import NO_INFO_MESSAGE, UNAVAILABLE_MESSAGE
from app.intent import IntentResult, QueryClassifier
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
_intent: Optional[QueryClassifier] = None

# Baseline classification — used when no classifier is wired (tests, degraded
# startup).  Reproduces the original pipeline exactly: no category, no boost,
# global threshold, no rubric.
_BASELINE_INTENT = IntentResult(category=Category.GENERAL, confidence=0.0, method="none")

_FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise heavyweight singletons once at startup."""
    global _cache, _retriever, _llm, _social, _intent

    logger.info("WCE Bot starting up …")
    _cache = RedisCache()
    _retriever = ChromaRetriever()
    _llm = LLMService()
    _social = SocialService()
    # Reuse the retriever's already-loaded encoder for prototype scoring —
    # no second embedding model in memory.
    _intent = QueryClassifier(encoder=_retriever.get_encoder())
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


# ── Source attribution ───────────────────────────────────────────────────────
# "View Document" must open the document the answer was *grounded in*, not
# merely the highest-scoring chunk.  For person queries a different faculty
# member's resume routinely outranks the record the facts came from (same
# layout, shared co-authorship lines), which sent students to someone else's
# profile.  Preference order: LLM-cited URL → lexical grounding of the answer
# in each chunk → the predicted category's first chunk in list order (the
# list is sorted by retrieval score, adjusted by the context-meaning
# re-rank, so "first" is what the model read first).
_CONTENT_TOKEN_RE = re.compile(r"[a-z0-9]{3,}")
_URL_RE = re.compile(r"https?://[^\s)\]>\"']+")

# Generic words that appear in most answers and most chunks — useless for
# telling documents apart.
_STOPWORDS = frozenset(
    """
    the and for are was were with from this that have has had you your not
    but all any can who what when where how why about into over than then
    them they their there these those its his her our out one two per also
    please tell does done please information details please answer question
    walchand college engineering wce student students
    """.split()
)

# Below this many shared content tokens the grounding signal is noise.
_MIN_GROUNDING_SUPPORT = 3


def _strip_source_line(answer: str) -> str:
    """Drop the trailing "Source: …" citation the LLM is instructed to append."""
    return re.sub(r"\n\s*Source\s*:.*$", "", answer.strip(), flags=re.IGNORECASE)


def _content_tokens(text: str) -> set:
    """Lowercased word/number tokens (≥3 chars) minus generic stopwords."""
    return {
        t for t in _CONTENT_TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS
    }


def _cited_source_url(answer: str, chunks: list) -> Optional[str]:
    """Return the chunk URL the LLM explicitly cited, if it is a known chunk."""
    cited = {u.rstrip(".,;)") for u in _URL_RE.findall(answer)}
    for chunk in chunks:
        url = chunk.get("source_url") or ""
        if url and url in cited:
            return url
    return None


def _pick_source_url(answer: str, chunks: list, intent: IntentResult) -> Optional[str]:
    """
    Choose the document link for *answer* from the retrieved *chunks*.

    1. If the answer cites a URL that matches one of the chunks, trust it —
       the system prompt requires the LLM to name the document it used.
    2. Otherwise count how many of the answer's content tokens each chunk
       actually contains; the best-supported chunk is the one the facts came
       from (category and retrieval score break ties).
    3. When nothing is well supported (heavy paraphrase, no overlap), fall
       back to list order within the predicted category: the list arrives in
       the order the model read it — retrieval score, then the
       context-meaning re-rank — so the first chunk is the one the answer
       was most likely grounded in.
    """
    with_url = [c for c in chunks if c.get("source_url")]
    if not with_url:
        return None

    cited = _cited_source_url(answer, with_url)
    if cited:
        return cited

    body_tokens = _content_tokens(_strip_source_line(answer))
    if body_tokens:
        support = {
            id(c): len(body_tokens & _content_tokens(c.get("text", "")))
            for c in with_url
        }
        if max(support.values()) >= _MIN_GROUNDING_SUPPORT:
            def rank(chunk: dict):
                same_category = (
                    intent.is_specific
                    and chunk.get("category") == intent.category.value
                )
                return (support[id(chunk)], same_category, chunk.get("score", 0.0))

            return max(with_url, key=rank).get("source_url") or None

    # The list reaches the model in retrieval order (score, adjusted by the
    # context-meaning re-rank) with social chunks appended last — so its
    # first same-category entry is what the model read first.  Picking by
    # max(score) instead would re-derive a winner the re-rank deliberately
    # did not move: a promoted chunk's score is unchanged, so the link could
    # disagree with the ordering the answer was grounded in.
    same_category = (
        [c for c in with_url if c.get("category") == intent.category.value]
        if intent.is_specific
        else []
    )
    top_chunk = (same_category or with_url)[0]
    return top_chunk.get("source_url") or None


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
      4. Query rewriting (abbreviation expansion) + intent classification
         into the query taxonomy (app/categories.py, app/intent.py)
      5. Chroma vector search with the predicted category (+ live YouTube
         context); per-category similarity threshold gate
      6. LLM call with grounding system prompt + category rubric
      7. Source URL injection — suppressed for "no information" replies
      8. Cache + return
    """
    # Step 2 — Rate limit
    await _check_rate_limit(request)

    query = body.query

    # Step 3 — Cache lookup (payload carries the category it was answered with)
    if _cache:
        cached = _cache.get(query)
        if cached:
            logger.info("Cache HIT for query: %.60s", query)
            if _is_no_info(cached.get("answer", "")):
                cached["source_url"] = None
            return ChatResponse(**cached, cached=True)

    # Step 4 — Query rewriting + intent classification.
    # The *expanded* query feeds both the classifier and retrieval (full forms
    # of abbreviations sharpen both); what the ambiguous words *mean* in this
    # domain is applied inside retrieval as a re-rank of the chunks found
    # (app/retrieval.py: apply_sense_rerank), never as query text — so the
    # LLM, the cache key and the UI keep the student's original wording.
    search_query = expand_query(query)
    if search_query != query:
        logger.debug("Query expanded: %.80s → %.80s", query, search_query)

    intent = _BASELINE_INTENT
    if _intent:
        intent = _intent.classify(search_query)
    if intent.is_specific:
        logger.info(
            "Intent: %s (confidence %.2f, %s)",
            intent.category.value,
            intent.confidence,
            intent.method,
        )

    # Step 5 — Vector search + live social context
    chunks = []
    if _retriever:
        chunks = _retriever.search(
            search_query,
            settings.top_k,
            category=intent.category.value if intent.is_specific else None,
        )

    if _social:
        social_chunks = _social.context_chunks(query)
        if social_chunks:
            logger.info("Adding %d live social chunk(s)", len(social_chunks))
            chunks = list(chunks) + social_chunks

    # Similarity threshold gate — global default unless the category overrides.
    gate = threshold_for(intent.category, settings.similarity_threshold)
    best_score = max((c.get("score", 0.0) for c in chunks), default=0.0)
    if not chunks or best_score < gate:
        logger.info(
            "No relevant chunks (best score=%.3f, gate=%.3f) — returning fallback",
            best_score,
            gate,
        )
        # Out of scope / not in the KB: exact refusal, never a document link.
        return ChatResponse(answer=NO_INFO_MESSAGE, source_url=None, cached=False)

    # Step 6 — LLM call (grounded prompt + category rubric)
    answer = UNAVAILABLE_MESSAGE
    if _llm:
        answer = _llm.generate(query, chunks, rubric=get_spec(intent.category).rubric or None)

    # Step 7 — Source URL; suppressed for refusals.  The link must open the
    # document the answer is grounded in (see _pick_source_url) — the highest
    # scored chunk is often a different document with a similar layout.
    source_url: Optional[str] = None
    if not _is_no_info(answer) and answer != UNAVAILABLE_MESSAGE:
        source_url = _pick_source_url(answer, chunks, intent)

    # Step 8 — Cache response (only real answers — never errors or refusals,
    # so an improved KB/LLM is not masked by a stale "no info" for 24 h)
    category_out = intent.category.value if intent.is_specific else None
    if _is_no_info(answer) or answer == UNAVAILABLE_MESSAGE:
        category_out = None  # refusals/errors carry no category chip either
    response_payload = {
        "answer": answer,
        "source_url": source_url,
        "category": category_out,
    }
    if (
        _cache
        and answer != UNAVAILABLE_MESSAGE
        and not _is_no_info(answer)
    ):
        _cache.set(query, response_payload)

    return ChatResponse(
        answer=answer,
        source_url=source_url,
        cached=False,
        category=category_out,
    )


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
