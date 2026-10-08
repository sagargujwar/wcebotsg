# WCE Bot — Project Explanation

**WCE Bot** is a Retrieval-Augmented Generation (RAG) chatbot that answers student questions about Walchand College of Engineering (fees, syllabus, admissions, notices) strictly from the college's own published documents. It scrapes the college website, embeds the content into a local vector database, and answers questions through an LLM that is *grounded* on those documents — so it cites sources instead of hallucinating.

- **Backend:** FastAPI + ChromaDB + Redis + Gemini/Groq
- **Frontend:** single-file vanilla HTML/CSS/JS (`frontend/index.html`)
- **Tests:** 45 pytest tests, all mocked (no API keys / Redis / real site needed)

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
│                             ──► 4. vector search (Chroma, top_k=4)                          │
│                             ──► 5. similarity gate (score < 0.35 → fallback answer)         │
│                             ──► 6. LLM generate (Gemini → Groq fallback)                    │
│                             ──► 7. attach source_url  ──► 8. cache + return                 │
└──────────────────────────────────────────────────────────────────────────────────────────────┘
```

Two decoupled halves: the **knowledge base (KB)** is built offline by `scripts/refresh_kb.py`, and the **API** only ever *reads* it. The API never crawls.

---

## 2. Request lifecycle (`POST /chat`)

Implemented in `app/main.py:117` — the docstring enumerates the same 8 steps:

| # | Step | Module | Detail |
|---|---|---|---|
| 1 | Validate input | `app/models.py:15` | Pydantic strips, rejects empty and >500-char queries (422) |
| 2 | Rate limit | `app/main.py:78`, `app/cache.py:100` | Per client IP, `20 req / 60 s`, honors `X-Forwarded-For`; returns 429 + `Retry-After` |
| 3 | Cache lookup | `app/cache.py:75` | SHA-256 of the normalized query; hit returns immediately with `cached: true` |
| 4 | Vector search | `app/retrieval.py:63` | Query embedded locally, Chroma returns top_k=4 with cosine distances converted to a 0–1 similarity score (`score = 1 − distance/2`) |
| 5 | Similarity gate | `app/main.py:147` | Best score `< 0.35` → no LLM call, returns a "couldn't find that" fallback pointing at the official site |
| 6 | LLM generation | `app/llm.py:120` | Grounded prompt built from the chunks; **Gemini (2 attempts, hard timeout via ThreadPoolExecutor) → Groq → graceful "service unavailable"**; never raises |
| 7 | Source injection | `app/main.py:163` | Top chunk's `source_url` returned so the UI can show a "View source document" link |
| 8 | Cache + return | `app/main.py:171` | Successful answers cached 24 h; errors are not cached |

`GET /health` (`app/main.py:190`) reports `chroma_ok` + `redis_ok` → `ok` or `degraded`. Interactive docs at `GET /docs` (Swagger).

---

## 3. Knowledge-base pipeline

### Scraper — `scraper/scraper.py`
- **BFS crawl** from `BASE_URL` up to `MAX_CRAWL_DEPTH` hops (`scraper.py:70`): depth 0 = homepage only, 1 = homepage + its links, 2 = + links-of-links.
- **Politeness & safety:** `robots.txt` checked per URL (`_is_allowed`), custom User-Agent, `0.5 s` delay between fetches, 10 s timeout, exact same-domain restriction, fragments stripped.
- **HTML handling:** strips `script/style/nav/footer/header/aside/form`, prefers `<main>`/`<article>`/`<body>` text.
- **PDF handling:** downloads to `data/pdfs/`, extracts text with `pdfplumber`, never expands links from a PDF.
- Custom CA bundle (`certs/ca-bundle.pem`) is required — the target site omits its intermediate certificate.

### Chunker — `scraper/chunker.py`
- Splits on paragraph boundaries/headings, merges tiny fragments (<50 chars), soft cap 1500 chars, and prepends the last 1–2 sentences of the previous chunk (~10–15 % overlap) so answers spanning boundaries survive.
- **Deterministic IDs:** `sha256(f"{source_url}:::{chunk_index}")` (`chunker.py:34`) → re-running the refresh **upserts** instead of duplicating.

### Embedding & storage — `app/retrieval.py`
- `SentenceTransformer("all-MiniLM-L6-v2")`, normalized embeddings, batch size 64.
- Chroma collection `college_kb` persisted at `CHROMA_PATH` (`./data/chroma_db`), cosine distance.
- Metadata per chunk: `source_url, doc_type, title, last_scraped_at, chunk_index`.
- **Upsert-only:** there is no delete/reset anywhere — the KB accumulates; lowering crawl depth later does *not* remove already-ingested chunks.

---

## 4. Frontend — `frontend/index.html`

Single self-contained file (inline `<style>` + `<script>`, Inter font, no build step). Open directly as `file://` — the backend allows the `null` origin.

| Feature | Notes |
|---|---|
| Send / Enter-to-send, Shift+Enter newline | `sendMessage()`, `handleKeydown()` |
| **Response-time badge** | `performance.now()` before `fetch`, delta after `response.json()` → `⏱ Responded in X.XXs` (purely client-side; no backend change needed) |
| Cached badge | `⚡ Cached response` when the API returns `cached: true` |
| Source link | Opens the scraped page/PDF the answer came from |
| Suggestion cards | 4 quick questions; `sendChip()` strips the leading emoji from `textContent` |
| Skeleton loader | Shown while awaiting the reply (`#typing-indicator` / `#skeleton`) |
| Copy answer / Retry on failure | `copyBubble()`, `retrySend()` re-sends `lastQuery` |
| Clear chat | `clearChat()` resets to the welcome bubble |
| Char counter | `n/500`, amber >450 (matches backend `MAX_QUERY_LENGTH`) |
| XSS safety | User input via `textContent`; all interpolated values pass `escapeHtml`/`escapeAttr` |
| Error toast | 4-second toast for 400/429/network failures |

**CORS:** allowed origins come from `ALLOWED_ORIGINS` plus a regex allowing any `localhost`/`127.0.0.1` port (Live Server etc.) — `app/main.py` + `app/config.py`. Verified by `tests/test_cors.py`.

---

## 5. Repository layout

```
wcebot/
├── app/                  # FastAPI application
│   ├── main.py           # routes, lifespan, CORS, rate-limit middleware
│   ├── config.py         # Settings (pydantic-settings, reads .env)
│   ├── models.py         # ChatRequest / ChatResponse / Health / Error schemas
│   ├── llm.py            # LLMService: grounding prompt, Gemini → Groq fallback
│   ├── retrieval.py      # ChromaRetriever: search / upsert / encoder
│   └── cache.py          # RedisCache: answer cache + per-IP rate limiting
├── scraper/
│   ├── scraper.py        # CollegeScraper: BFS crawl, HTML/PDF processing, robots
│   └── chunker.py        # TextChunker: paragraph chunks + overlap + stable IDs
├── scripts/
│   └── refresh_kb.py     # crawl → chunk → embed → upsert (the ingestion CLI)
├── frontend/
│   └── index.html        # the entire UI
├── tests/                # 45 tests (chat API, CORS, retrieval, scraper) + fixtures
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
| `MAX_CRAWL_DEPTH` | `1` in `.env` (`2` in config default) | link-hops to crawl |
| `SSL_CA_BUNDLE` | `./certs/ca-bundle.pem` | required for the target site's TLS chain |
| `CHROMA_PATH` / `PDF_DOWNLOAD_PATH` | `./data/chroma_db`, `./data/pdfs` | storage |
| `SIMILARITY_THRESHOLD` | `0.35` | minimum cosine score before the LLM is called |
| `TOP_K` | `4` | chunks retrieved per query |
| `RATE_LIMIT_REQUESTS` / `_WINDOW_SECONDS` | `20 / 60` | per-IP chat limit |
| `CACHE_TTL_SECONDS` | `86400` | answer cache lifetime |
| `MAX_QUERY_LENGTH` | `500` | query validation |
| `ALLOWED_ORIGINS` | localhost list + `null` | CORS allow-list |
| `OFFICIAL_SITE_FALLBACK_URL` | — | link used in the "couldn't find it" answer |

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

pytest tests/ -v                  # 45 tests
```

Schedule KB refreshes with cron (Linux/macOS) or Task Scheduler (Windows) — see `README.md` §9.

---

## 8. Testing strategy

`pytest` runs **45 tests** with *everything external mocked* — no Redis, no Chroma, no LLM keys, no network:

- `tests/test_chat_api.py` — happy path, validation (422s), similarity-threshold fallback, cache hit/miss, rate limiting (429), source URL injection
- `tests/test_cors.py` — localhost any-port allowed, `null` (file://) allowed, foreign origins blocked, preflight
- `tests/test_retrieval.py` — Chroma + encoder behavior
- `tests/test_scraper.py` — HTML/PDF processing, depth-0/1 crawling, robots handling

Mock singletons are injected into `app.main`'s module globals (`_cache`, `_retriever`, `_llm`) before each client is created.

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
| Frontend professional redesign | Prompt prepared in `docs/gemini-prompts.md`; not yet applied |
| Response-time badge | ✅ implemented (client-side) |
| CORS any-localhost-port fix | ✅ implemented + tested |
| No `MAX_PAGES` cap in the crawler | A `MAX_CRAWL_DEPTH=2` run is bounded only by robots + dedup + depth |
| PDF filename collisions | Filenames use the URL basename only — two `syllabus.pdf` overwrite each other |
| KB never pruned | Chunks from removed/depth-reduced pages stay in Chroma forever |
| Stale answers up to 24 h | Refreshing the KB does not invalidate the Redis cache |
| Depth 1 vs 2 | Current `.env` uses `1` (homepage + its links ≈ 145 pages already ingested); `2` would add roughly 1,000–3,000 pages and run 30 min–hours |

---

*Companion docs: `README.md` (setup & operations) · `docs/gemini-prompts.md` (AI prompts used for the UI redesign)*
