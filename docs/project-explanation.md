# WCE Bot — Project Explanation

**WCE Bot** is a Retrieval-Augmented Generation (RAG) chatbot that answers student questions about Walchand College of Engineering (fees, syllabus, admissions, notices) strictly from the college's own published documents. It scrapes the college website, embeds the content into a local vector database, and answers questions through an LLM that is *grounded* on those documents — so it cites sources instead of hallucinating.

- **Backend:** FastAPI + ChromaDB + Redis + Gemini/Groq
- **Frontend:** single-file vanilla HTML/CSS/JS (`frontend/index.html`) + floating widget loader (`frontend/widget.js`)
- **Tests:** 117 pytest tests, all mocked (no API keys / Redis / real site needed)

---

## 1. High-level architecture

```
┌─────────────────────────── INGESTION (offline, run manually/cron) ───────────────────────────┐
│                                                                                              │
│  college website ──► CollegeScraper ──► TextChunker ──► SentenceTransformer ──► ChromaDB      │
│  (BFS crawl,            (HTML/PDF      (paragraph      (all-MiniLM-L6-v2,      (local file    │
│   robots.txt,            → clean text)   chunks +         local embedding)       store)       │
│   0.5s delay)                                         scripts/refresh_kb.py                   │
└──────────────────────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────── QUERY TIME (online, per request) ────────────────────────────────┐
│                                                                                              │
│  Browser (frontend/index.html)                                                              │
│      │ POST /chat {query}                                                                   │
│      ▼                                                                                      │
│  FastAPI  app/main.py  ──► 1. validate  ──► 2. rate limit (Redis, per IP)                   │
│                             ──► 3. cache lookup (Redis, 24h TTL)                            │
│                             ──► 4. vector search (Chroma, top_k=6) + live YouTube context    │
│                             ──► 5. similarity gate (< 0.35 → exact refusal, no link)        │
│                             ──► 6. LLM generate (Gemini → Groq fallback)                    │
│                             ──► 7. attach source_url (suppressed on refusal) ──► 8. cache    │
└──────────────────────────────────────────────────────────────────────────────────────────────┘
```

Two decoupled halves: the **knowledge base (KB)** is built offline by `scripts/refresh_kb.py`, and the **API** only ever *reads* it. The API never crawls.

---

## 2. Request lifecycle (`POST /chat`)

Implemented in `app/main.py:139` — the docstring enumerates the same 8 steps:

| # | Step | Module | Detail |
|---|---|---|---|
| 1 | Validate input | `app/models.py:16` | Pydantic strips, rejects empty and >500-char queries (422) |
| 2 | Rate limit | `app/main.py:93`, `app/cache.py:100` | Per client IP, `20 req / 60 s`, honors `X-Forwarded-For`; returns 429 + `Retry-After` |
| 3 | Cache lookup | `app/cache.py:75` | SHA-256 of the normalized query; hit returns with `cached: true` (a defensive re-clear of `source_url` still guards any legacy cached refusal) |
| 4 | Vector search | `app/retrieval.py:122`, `app/main.py:168` | Query embedded locally, Chroma returns a candidate pool (top_k × 10, max 100) that is reranked — stub chunks dropped, ≤2 chunks per source URL — down to top_k=6, with cosine distances converted to a 0–1 similarity score (`score = 1 − distance/2`); a **live YouTube RSS chunk** (score 1.0) is appended when the query looks video-related (`app/social.py`) |
| 5 | Similarity gate | `app/main.py:176` | Best score `< 0.35` → no LLM call; returns the exact `NO_INFO_MESSAGE` with `source_url: null` (`app/constants.py`) |
| 6 | LLM generation | `app/llm.py:128` | Grounded prompt built from the chunks; **Gemini (single attempt, 25 s hard timeout via ThreadPoolExecutor) → Groq (15 s) → graceful "service unavailable"**; never raises. The prompt forbids guessing — out-of-scope replies must be the same exact refusal |
| 7 | Source injection | `app/main.py:191` | Suppressed when the answer is a refusal (`_is_no_info`); otherwise the highest-scoring chunk with a non-empty `source_url` wins — the UI labels it **View Document** |
| 8 | Cache + return | `app/main.py:201` | **Only real answers are cached 24 h** — refusals and service-unavailable errors are never written, so an improved KB/LLM is not masked by a stale "no info" |

`GET /health` (`app/main.py:247`) reports `chroma_ok` + `redis_ok` → `ok` or `degraded`. Interactive docs at `GET /docs` (Swagger). Embeddable widget assets: `GET /widget` (iframe panel, `app/main.py:216`) and `GET /widget.js` (floating-button loader, `app/main.py:226`).

---

## 3. Knowledge-base pipeline

### Scraper — `scraper/scraper.py`
- **BFS crawl** from `BASE_URL` up to `MAX_CRAWL_DEPTH` hops (`scraper.py:184`): depth 0 = homepage only, 1 = homepage + its links, 2 = + links-of-links.
- **Politeness & safety:** `robots.txt` checked per URL (`_is_allowed`), custom User-Agent, `0.5 s` delay between fetches, 15 s timeout, 3 retries on 5xx/connection errors (4xx fails fast), exact same-domain restriction, fragments stripped.
- **HTML handling:** strips `script/style/nav/footer/header/aside/noscript/iframe/form`, prefers `<main>`/`<article>`/`<body>` text.
- **PDF handling:** downloads to `data/pdfs/`, extracts text with `pdfplumber`, never expands links from a PDF.
- Custom CA bundle (`certs/ca-bundle.pem`) is required — the target site omits its intermediate certificate.

### Chunker — `scraper/chunker.py`
- Splits on paragraph boundaries/headings, merges tiny fragments (<50 chars), hard cap 1500 chars (enforced at sentence/word boundaries), and prepends the last 1–2 sentences of the previous chunk (~10–15 % overlap, capped) so answers spanning boundaries survive.
- **Deterministic IDs:** `sha256(f"{source_url}:::{chunk_index}")` (`chunker.py:42`) → re-running the refresh **upserts** instead of duplicating.

### Embedding & storage — `app/retrieval.py`
- `SentenceTransformer("all-MiniLM-L6-v2")`, normalized embeddings, batch size 64.
- Chroma collection `college_kb` persisted at `CHROMA_PATH` (`./data/chroma_db`), cosine distance.
- Metadata per chunk: `source_url, doc_type, title, last_scraped_at, chunk_index`.
- **Query-time reranking** (`_rerank_chunks`): searches a candidate pool of top_k × 10 (max 100), then drops stub chunks (page-header/footer fragments), caps repeats at 2 chunks per source URL, and backfills by score so the caller always gets up to top_k results.
- **Upsert-only:** there is no delete/reset anywhere — the KB accumulates; lowering crawl depth later does *not* remove already-ingested chunks.
- Upserts are batched (4,000 rows per call) to stay under Chroma's max-batch limit.

---

## 4. Frontend — `frontend/index.html`

Single self-contained file (inline `<style>` + `<script>`, Inter font, no build step). Open directly as `file://` — the backend allows the `null` origin.

| Feature | Notes |
|---|---|
| Send / Enter-to-send, Shift+Enter newline | `sendMessage()`, `handleKeydown()` |
| **Response-time badge** | `performance.now()` before `fetch`, delta after `response.json()` → `⏱ Responded in X.XXs` (purely client-side; no backend change needed) |
| Cached badge | `⚡ Cached response` when the API returns `cached: true` |
| Source link | Quiet `📄 View Document` pill opening the scraped page/PDF — rendered **only** when `source_url` is non-null (never for refusals) |
| Suggestion cards | 4 quick questions; `sendChip()` strips the leading emoji from `textContent` |
| Skeleton loader | Shown while awaiting the reply (`#typing-indicator` / `#skeleton`) |
| Copy answer / Retry on failure | `copyBubble()`, `retrySend()` re-sends `lastQuery` |
| Clear chat | `clearChat()` resets to the welcome bubble |
| Char counter | `n/500`, amber >450 (matches backend `MAX_QUERY_LENGTH`) |
| XSS safety | User input via `textContent`; all interpolated values pass `escapeHtml`/`escapeAttr` |
| Error toast | 4-second toast for 400/429/network failures |

**CORS:** allowed origins come from `ALLOWED_ORIGINS` plus a regex allowing any `localhost`/`127.0.0.1` port (Live Server etc.) — `app/main.py` + `app/config.py`. Verified by `tests/test_cors.py`.

### Embeddable floating widget

- `frontend/widget.js` — one-line snippet for any host page:
  `<script src="http://localhost:8000/widget.js" defer></script>` → injects a
  floating action button; clicking it opens the chat panel **on the same
  screen** as an overlay (no navigation, no popup).
- The panel is an `<iframe>` loaded from `GET /widget`, so `/chat` is
  same-origin for the iframe and the host page's CSS can't break the UI.
- `scripts/demo_proxy.py` — local mirror of the real college site (`BASE_URL`)
  with the snippet injected (default port `9000`), for demoing the button on
  the actual college page without any site access.

---

## 5. Repository layout

```
wcebot/
├── app/                  # FastAPI application
│   ├── main.py           # routes, lifespan, CORS, rate-limit middleware, widget endpoints
│   ├── config.py         # Settings (pydantic-settings, reads .env)
│   ├── constants.py      # NO_INFO_MESSAGE — the single exact refusal reply
│   ├── models.py         # ChatRequest / ChatResponse / Health / Error schemas
│   ├── llm.py            # LLMService: grounding prompt, Gemini → Groq fallback
│   ├── social.py         # SocialService: live YouTube RSS context (keyless, cached)
│   ├── retrieval.py      # ChromaRetriever: search / upsert / encoder
│   └── cache.py          # RedisCache: answer cache + per-IP rate limiting
├── scraper/
│   ├── scraper.py        # CollegeScraper: BFS crawl, HTML/PDF processing, robots, social links
│   └── chunker.py        # TextChunker: paragraph chunks + overlap + stable IDs
├── scripts/
│   ├── refresh_kb.py     # crawl → chunk → embed → upsert (the ingestion CLI)
│   └── demo_proxy.py     # local college-site mirror with the widget injected
├── static_docs/
│   └── social_media.json # official social profiles → ingested as KB chunks
├── frontend/
│   ├── index.html        # the entire UI (standalone page + widget iframe panel)
│   └── widget.js         # floating-button embed loader
├── tests/                # 117 tests (chat API, CORS, retrieval, scraper, social) + fixtures
├── docs/                 # gemini-prompts.md, project-explanation.md
├── certs/                # CA bundle for the target site's incomplete chain
├── data/                 # chroma_db/ and pdfs/ (git-ignored, runtime)
├── .env.example          # documented template — copy to .env
└── README.md             # setup & operations guide
```

---

## 6. Configuration (`.env`)

| Variable | Default | Purpose |
|---|---|---|
| `GEMINI_API_KEY` / `GROQ_API_KEY` | empty | LLM providers (either one is enough; both = redundancy) |
| `REDIS_URL` | `redis://localhost:6379/0` | answer cache + rate limiting |
| `BASE_URL` | target college site | crawl seed |
| `MAX_CRAWL_DEPTH` | `3` in `.env` (`2` in config default) | link-hops to crawl |
| `MAX_PAGES` | `20000` | crawl safety cap (documents, 0 = unlimited) |
| `SSL_CA_BUNDLE` | `./certs/ca-bundle.pem` | required for the target site's TLS chain |
| `CHROMA_PATH` / `PDF_DOWNLOAD_PATH` | `./data/chroma_db`, `./data/pdfs` | storage |
| `SIMILARITY_THRESHOLD` | `0.35` | minimum cosine score before the LLM is called |
| `TOP_K` | `6` | chunks retrieved per query |
| `RATE_LIMIT_REQUESTS` / `_WINDOW_SECONDS` | `20 / 60` | per-IP chat limit |
| `CACHE_TTL_SECONDS` | `86400` | answer cache lifetime |
| `MAX_QUERY_LENGTH` | `500` | query validation |
| `ALLOWED_ORIGINS` | localhost list + `null` | CORS allow-list |
| `OFFICIAL_SITE_FALLBACK_URL` | — | link used in the "couldn't find it" answer |
| `YOUTUBE_CHANNEL_ID` | empty in config; `.env` sets `UCkWEgeBCY5U6sl7jS9YShwA` (Media WCE) | live YouTube RSS context; empty = disabled |
| `WIDGET_API_BASE` | `http://localhost:8000` | origin used to build the widget embed snippet |
| `DEMO_PROXY_PORT` | `9000` | port of the local college-site demo mirror |

Secrets live only in `.env` (git-ignored). `Settings` never prints key values — they are `SecretStr`.

---

## 7. Running it

```bash
python -m venv .venv && .venv\Scripts\activate       # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
copy .env.example .env                               # then fill in keys / BASE_URL

python scripts/refresh_kb.py      # build/refresh the knowledge base (crawl + embed)
uvicorn app.main:app --reload     # backend on :8000
# open frontend/index.html in a browser (file:// works — CORS allows Origin: null)
# optional: python scripts/demo_proxy.py → http://localhost:9000 (college site + widget)

pytest tests/ -v                  # 117 tests
```

Schedule KB refreshes with cron (Linux/macOS) or Task Scheduler (Windows) — see `README.md` §12.

---

## 8. Testing strategy

`pytest` runs **117 tests** with *everything external mocked* — no Redis, no Chroma, no LLM keys, no network:

- `tests/test_chat_api.py` (26) — happy path, validation (422s), similarity-threshold fallback, cache hit/miss, rate limiting (429), source URL injection, **exact refusal + `source_url: null`**, refusals never cached
- `tests/test_cors.py` (6) — localhost any-port allowed, `null` (file://) allowed, foreign origins blocked, preflight
- `tests/test_retrieval.py` (15) — Chroma + encoder behavior, stub filtering / per-source reranking
- `tests/test_scraper.py` (40) — HTML/PDF processing, depth-0/1 crawling, redirects, robots handling, retries, link extraction
- `tests/test_social.py` (30) — YouTube feed parsing, video-intent matching, social-link extraction, disabled-mode behavior (all offline)

Mock singletons are injected into `app.main`'s module globals (`_cache`, `_retriever`, `_llm`, `_social`) before each client is created.

---

## 9. Security & correctness notes

- `.env` is git-ignored; API keys are `SecretStr`.
- HTML from the site is treated as data: rendered text is stripped of scripts; user input is never interpolated as HTML.
- Per-IP rate limiting + query-length validation on every request.
- The crawler respects `robots.txt` and stays on the seeded domain.
- TLS verification is enforced via a custom CA bundle (not disabled).
- Error paths return structured `ErrorResponse` JSON; the LLM layer never raises out of `generate()`.

---

## 10. Known limitations / possible next steps

| Item | Status |
|---|---|
| Frontend professional redesign | ✅ applied (prompts retained in `docs/gemini-prompts.md`) |
| Response-time badge | ✅ implemented (client-side) |
| CORS any-localhost-port fix | ✅ implemented + tested |
| Exact out-of-scope refusal | ✅ implemented — `NO_INFO_MESSAGE` + `source_url: null`, double-checked in the frontend |
| Social media answers | ✅ implemented — static profile links in the KB + live YouTube RSS |
| Floating widget on any site | ✅ implemented — `GET /widget.js` snippet, `GET /widget` iframe, `scripts/demo_proxy.py` demo |
| `MAX_PAGES` cap | ✅ implemented — `MAX_PAGES` (`.env`, default 20000) stops the crawl with a warning; progress logged every 25 pages |
| PDF filename collisions | Filenames use the URL basename only — two `syllabus.pdf` overwrite each other |
| KB never pruned | Chunks from removed/depth-reduced pages stay in Chroma forever |
| Stale answers up to 24 h | Refreshing the KB does not invalidate the Redis cache |
| Depth / scale | `.env` uses `MAX_CRAWL_DEPTH=3`, `MAX_PAGES=20000`; full run = 847 docs → 25,181 chunks (~75 min typical, ~4 h on a slow day) |

---

*Companion docs: `README.md` (setup & operations) · `docs/gemini-prompts.md` (AI prompts used for the UI redesign)*
