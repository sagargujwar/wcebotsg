# WCE Bot — Project Explanation

**WCE Bot** is a Retrieval-Augmented Generation (RAG) chatbot that answers student questions about Walchand College of Engineering (fees, syllabus, admissions, notices) strictly from the college's own published documents. It scrapes the college website, embeds the content into a local vector database, and answers questions through an LLM that is *grounded* on those documents — so it cites sources instead of hallucinating.

- **Backend:** FastAPI + ChromaDB + Redis + Gemini/Groq
- **Frontend:** single-file vanilla HTML/CSS/JS (`frontend/index.html`) + floating widget loader (`frontend/widget.js`)
- **Query taxonomy:** every query is classified (academics, admissions, notices, …) and the category steers retrieval, gating, and the LLM rubric — measured by `scripts/eval_accuracy.py`
- **Tests:** 360 pytest tests, all mocked (no API keys / Redis / real site needed)

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
│                             ──► 4. rewrite query + classify into the taxonomy               │
│                                   (app/intent.py: rules + prototypes, 9 categories)         │
│                             ──► 5. vector search (top_k=6, category prior boosts            │
│                                   matching chunks) + per-category similarity gate           │
│                             ──► 6. LLM generate (Gemini → Groq fallback, category rubric)   │
│                             ──► 7. attach source_url (suppressed on refusal) ──► 8. cache    │
└──────────────────────────────────────────────────────────────────────────────────────────────┘
```

Two decoupled halves: the **knowledge base (KB)** is built offline by `scripts/refresh_kb.py`, and the **API** only ever *reads* it. The API never crawls.

---

## 2. Request lifecycle (`POST /chat`)

Implemented in `app/main.py:150` — the docstring enumerates the same 8 steps:

| # | Step | Module | Detail |
|---|---|---|---|
| 1 | Validate input | `app/models.py:15` | Pydantic strips, rejects empty and >500-char queries (422) |
| 2 | Rate limit | `app/main.py:104`, `app/cache.py:100` | Per client IP, `20 req / 60 s`, honors `X-Forwarded-For`; returns 429 + `Retry-After` |
| 3 | Cache lookup | `app/cache.py:75` | SHA-256 of the normalized query; hit returns with `cached: true` (a defensive re-clear of `source_url` still guards any legacy cached refusal); the payload carries the `category` it was answered with |
| 4 | Rewrite + classify | `app/main.py:178`, `app/categories.py:584`, `app/intent.py:145` | Abbreviation expansion (e.g. `ntc` → `notice`) for the embedding query only; the hybrid classifier predicts a taxonomy category (`rules 0.45 + seed-query prototypes 0.55`, `MIN_SCORE=0.34`, `MIN_MARGIN=0.05`) or falls back to `general` |
| 5 | Vector search + gate | `app/retrieval.py:181`, `app/main.py:197` | Chroma candidate pool (top_k × 10, max 100) reranked to top_k=6 (`score = 1 − distance/2`); when the query is classified, same-category chunks get a small relevance-banded boost (`assemble_results`, `app/retrieval.py:102`); a **live YouTube RSS chunk** (score 1.0) is appended when the query looks video-related (`app/social.py`); best score below the **per-category gate** (`threshold_for`, default 0.35) → exact refusal, no LLM call (`app/main.py:213`) |
| 6 | LLM generation | `app/llm.py:142`, `app/main.py:224` | Grounded prompt + the category's **rubric** (`app/llm.py:74`); **Gemini (single attempt, 25 s hard timeout via ThreadPoolExecutor) → Groq (15 s) → graceful "service unavailable"**; never raises. The prompt forbids guessing — out-of-scope replies must be the same exact refusal |
| 7 | Source injection | `app/main.py:229` | Suppressed when the answer is a refusal (`_is_no_info`); otherwise the highest-scoring chunk with a non-empty `source_url` wins — preferring chunks of the predicted category — and the UI labels it **View Document** |
| 8 | Cache + return | `app/main.py:246` | **Only real answers are cached 24 h** — refusals and service-unavailable errors are never written, so an improved KB/LLM is not masked by a stale "no info" |

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

## 4. Query taxonomy & category-aware retrieval

A plain RAG pipeline treats every query the same. This project adds a **query
taxonomy**: each question is first classified into one of nine categories, and
the predicted category then steers retrieval, the similarity gate, the LLM
prompt, the source shown to the user, and the UI chip. Everything lives in two
modules — `app/categories.py` (the taxonomy) and `app/intent.py` (the
classifier) — with `scripts/eval_accuracy.py` measuring the effect.

### Categories (`app/categories.py`)

Ordered by project scope — 1–6 are what students actually ask a college bot:

| # | Category | Examples |
|---|---|---|
| 1 | `admissions_fees` | fees, caps, CET/DTE, seats, rounds, scholarships |
| 2 | `academics` | syllabus, departments, subjects, exams, timetable |
| 3 | `placements_careers` | placements, recruiters, internships, packages |
| 4 | `people_admin` | director/HODs, committees, contacts, recruitment |
| 5 | `infra_campus` | labs, library, hostels, wifi, transport, canteen |
| 6 | `notices_events` | notices, announcements, workshops, deadlines |
| 7 | `research` | papers, patents, PhD, MoUs, funded projects |
| 8 | `general_info` | vision/mission, NIRF/NAAC, address, history |
| — | `general` | low-confidence fallback → **exactly the baseline pipeline** |

(`out_of_scope` exists as a label but is never predicted — the refusal path
belongs to the similarity gate and the grounded LLM, not the classifier.)

Each `CategorySpec` carries: **keywords + regex patterns**, **seed queries**
(→ embedding prototypes), **URL prefixes** (→ KB tagging), a **threshold
override**, a **boost**, and an **answer rubric** for the prompt.

### Classification (`app/intent.py`)

Hybrid of two cheap signals, fused and gated:

- **Rules** — keyword hit = 1.0, richer regex hit = 1.5 (e.g. `₹`, phone
  numbers, `\bannounce\w*\b`), summed per category and normalized to 0–1.
- **Prototypes** — the seed queries are batch-encoded once at startup into a
  per-category mean vector (centroid); cosine similarity of the query gives a
  0–1 prototype score.
- **Fused:** `0.45 × rules + 0.55 × prototypes`. The winner must clear
  `MIN_SCORE = 0.34` *and* beat the runner-up by `MIN_MARGIN = 0.05`,
  otherwise the query falls back to `general` (no boost, global threshold, no
  rubric). The gates were calibrated against the eval set.

`expand_query()` rewrites abbreviations (`ntc` → `notice`, `hod` → `head of
department`) for the **embedding/classification text only** — the LLM and the
user still see the original wording.

### Context meaning of ambiguous words — `CONTEXT_SENSES`

A dictionary word is only useful once we write down what it means *here*.
`CONTEXT_SENSES` (`app/categories.py`) curates the genuinely ambiguous
college-domain words and, for each, the phrases the document that *is* the
thing carries:

```python
"syllabus": ["course contents", "course name", "course code", "credits"],
"notice":   ["notice board", "circular", "announcement", "notification"],
```

At the end of `search()`, `apply_sense_rerank()` (`app/retrieval.py`) uses
those meanings on the **chunks already retrieved**: one inside the relevance
band of the best result that carries ≥ 2 defining phrases moves up. So when a
student asks for the *syllabus*, the document built from course contents and
credits is what the LLM reads first and cites — not the annual report that
merely repeats the word a dozen times.

Two properties are deliberate, and both are pinned by tests:

- **Order only.** Scores are untouched (the similarity gate reads the *best*
  score, never the first chunk, so out-of-scope refusals cannot move) and no
  chunk enters or leaves the list — answer coverage (hit@6) is preserved by
  construction; only MRR can improve.
- **The meaning never becomes query text.** Appending the gloss to the query
  so the embedding sees it was implemented and measured first: the extra
  generic vocabulary pulls the embedding toward the big reports
  (hit@6 50 → 47–49, MRR 0.536 → 0.492), so it was rejected in favour of
  re-ranking the found chunks.

### KB tagging

Chunks are tagged at ingestion: `url_to_category(url, title)` picks the
**longest matching URL prefix**, else a unique title-lexicon hit, else
`general`. Existing databases are migrated metadata-only (no re-embedding):

```bash
python scripts/tag_categories.py --force   # after any URL-rule change
```

Current distribution over 25,181 chunks: `general` 10,778 · `academics` 8,828 ·
`general_info` 4,627 · `people_admin` 642 · `admissions_fees` 128 ·
`placements_careers` 73 · `research` 49 · `infra_campus` 41 ·
`notices_events` 15 (most KB content is PDFs under `/wp-content/`, whose
filenames map via the title lexicon).

### How the category is used at query time

1. **Retrieval** — `search(query, category=…)` → `assemble_results`
   (`app/retrieval.py:102`) adds a *relevance-banded* boost to chunks whose
   `category` matches, relative to the pool's best score (so it re-orders
   near-tied candidates, never injects junk). If no matching chunk would
   survive the top-k, a corroboration fallback keeps the plain baseline rerank.
   Then `apply_sense_rerank()` re-orders the finished list by what the query's
   ambiguous words *mean* here (`CONTEXT_SENSES`) — again order only: scores
   and membership are left as they are.
2. **Gate** — `threshold_for()` (`app/categories.py:470`) may lower/raise the
   global 0.35 gate per category (e.g. `notices_events` = 0.30).
3. **Prompt** — the spec's rubric is injected as a `{rubric}` block
   (`app/llm.py:74`): category-specific guardrails and evidence expectations.
4. **Source & cache** — chunks of the predicted category are preferred when
   picking the cited `source_url` (`app/main.py:237`), and the response/cache
   payload carries `category`.
5. **UI** — the frontend renders a 🏷 chip with the category label (never on
   refusals) next to the cached/time badges.

### Measured effect (`scripts/eval_accuracy.py`)

86 labelled queries (`tests/eval/queries.jsonl`: 76 answerable across 8
categories ≥ 5 each, 10 out-of-scope). No LLM calls — embedding + Chroma only:

| metric | baseline | category-aware |
|---|---|---|
| classification accuracy | — | **100 % (86/86)** |
| hit@6 (76 answerable) | 53.9 % (41/76) | **65.8 % (50/76)** |
| MRR | 0.369 | **0.570** |
| out-of-scope queries routed to a category | — | **0/10** (all declined → no prior applied) |

9 queries gained a relevant top-6 hit, **0 lost**. Removing the
context-meaning re-rank drops MRR to 0.369 / 0.536 with hit@6 unchanged
(41/76 and 50/76) — the meaning layer moves the right document *up*, it never
adds or removes one. Full breakdown per category and per query:
`docs/eval-report.md` (regenerate with
`python scripts/eval_accuracy.py --out docs/eval-report.md`).

---

## 5. Frontend — `frontend/index.html`

Single self-contained file (inline `<style>` + `<script>`, Inter font, no build step). Open directly as `file://` — the backend allows the `null` origin.

| Feature | Notes |
|---|---|
| Send / Enter-to-send, Shift+Enter newline | `sendMessage()`, `handleKeydown()` |
| **Response-time badge** | `performance.now()` before `fetch`, delta after `response.json()` → `⏱ Responded in X.XXs` (purely client-side; no backend change needed) |
| Cached badge | `⚡ Cached response` when the API returns `cached: true` |
| Category chip | `🏷 <label>` for the taxonomy category the query was routed to (from `ChatResponse.category`; never shown on refusals) |
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

## 6. Repository layout

```
wcebot/
├── app/                  # FastAPI application
│   ├── main.py           # routes, lifespan, CORS, rate-limit middleware, widget endpoints
│   ├── config.py         # Settings (pydantic-settings, reads .env)
│   ├── constants.py      # NO_INFO_MESSAGE — the single exact refusal reply
│   ├── models.py         # ChatRequest / ChatResponse / Health / Error schemas
│   ├── categories.py     # query taxonomy: specs, lexicons, URL rules, rubrics
│   │                     #   + context meanings of ambiguous words (CONTEXT_SENSES)
│   ├── intent.py         # QueryClassifier: rules + prototypes, confidence gates
│   ├── llm.py            # LLMService: grounding prompt + rubric, Gemini → Groq fallback
│   ├── social.py         # SocialService: live YouTube RSS context (keyless, cached)
│   ├── retrieval.py      # ChromaRetriever: search / upsert / encoder + category boost
│   │                     #   + context-meaning re-rank (apply_sense_rerank)
│   └── cache.py          # RedisCache: answer cache + per-IP rate limiting
├── scraper/
│   ├── scraper.py        # CollegeScraper: BFS crawl, HTML/PDF processing, robots, social links
│   └── chunker.py        # TextChunker: paragraph chunks + overlap + stable IDs
├── scripts/
│   ├── refresh_kb.py     # crawl → chunk → embed → upsert (the ingestion CLI)
│   ├── tag_categories.py # backfill chunk categories in Chroma (metadata-only)
│   ├── eval_accuracy.py  # baseline vs category-aware evaluation → docs/eval-report.md
│   └── demo_proxy.py     # local college-site mirror with the widget injected
├── static_docs/
│   └── social_media.json # official social profiles → ingested as KB chunks
├── frontend/
│   ├── index.html        # the entire UI (standalone page + widget iframe panel)
│   └── widget.js         # floating-button embed loader
├── tests/                # 360 tests (chat API, taxonomy, sense layer, people,
│   │                     #   syllabus targeting, eval set, CORS, retrieval,
│   │                     #   scraper, social) + fixtures
│   └── eval/queries.jsonl  # 86 labelled queries for scripts/eval_accuracy.py
├── docs/                 # eval-report.md (generated), project-explanation.md, gemini-prompts.md
├── certs/                # CA bundle for the target site's incomplete chain
├── data/                 # chroma_db/ and pdfs/ (git-ignored, runtime)
├── .env.example          # documented template — copy to .env
└── README.md             # setup & operations guide
```

---

## 7. Configuration (`.env`)

| Variable | Default | Purpose |
|---|---|---|
| `GEMINI_API_KEY` / `GROQ_API_KEY` | empty | LLM providers (either one is enough; both = redundancy) |
| `REDIS_URL` | `redis://localhost:6379/0` | answer cache + rate limiting |
| `BASE_URL` | target college site | crawl seed |
| `MAX_CRAWL_DEPTH` | `3` in `.env` (`2` in config default) | link-hops to crawl |
| `MAX_PAGES` | `20000` | crawl safety cap (documents, 0 = unlimited) |
| `SSL_CA_BUNDLE` | `./certs/ca-bundle.pem` | required for the target site's TLS chain |
| `CHROMA_PATH` / `PDF_DOWNLOAD_PATH` | `./data/chroma_db`, `./data/pdfs` | storage |
| `SIMILARITY_THRESHOLD` | `0.35` | minimum cosine score before the LLM is called (per-category overrides in `app/categories.py`) |
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

## 8. Running it

```bash
python -m venv .venv && .venv\Scripts\activate       # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
copy .env.example .env                               # then fill in keys / BASE_URL

python scripts/refresh_kb.py      # build/refresh the knowledge base (crawl + embed)
uvicorn app.main:app --reload     # backend on :8000
# open frontend/index.html in a browser (file:// works — CORS allows Origin: null)
# optional: python scripts/demo_proxy.py → http://localhost:9000 (college site + widget)

pytest tests/ -v                                    # 360 tests
python scripts/eval_accuracy.py --out docs/eval-report.md   # taxonomy evaluation
```

Schedule KB refreshes with cron (Linux/macOS) or Task Scheduler (Windows) — see `README.md` §12.

---

## 9. Testing strategy

`pytest` runs **360 tests** with *everything external mocked* — no Redis, no Chroma, no LLM keys, no network:

- `tests/test_chat_api.py` (42) — happy path, validation (422s), similarity-threshold fallback, cache hit/miss, rate limiting (429), source URL injection, **exact refusal + `source_url: null`**, refusals never cached, **category-aware behaviour** (response category, per-category gate, rubric passing, source preference, cached category)
- `tests/test_intent.py` (35) — taxonomy specs, lexicon/URL rules, abbreviation expansion, prototype round-trips, confidence gates
- `tests/test_senses.py` (79) — context-meaning layer: trigger detection, phrase matching, and `apply_sense_rerank` (promotes the document that *is* the thing; scores, membership and targeted chunks untouched)
- `tests/test_eval_set.py` (6) — schema validation of `tests/eval/queries.jsonl` (size, categories, ids, sources)
- `tests/test_retrieval.py` (35) — Chroma + encoder behavior, stub filtering / per-source reranking, `assemble_results` category boost, person/syllabus document targeting
- `tests/test_people.py` (47) — name extraction and person-document targeting
- `tests/test_syllabus.py` (40) — branch/year/level parsing and syllabus-document targeting
- `tests/test_cors.py` (6) — localhost any-port allowed, `null` (file://) allowed, foreign origins blocked, preflight
- `tests/test_scraper.py` (40) — HTML/PDF processing, depth-0/1 crawling, redirects, robots handling, retries, link extraction
- `tests/test_social.py` (30) — YouTube feed parsing, video-intent matching, social-link extraction, disabled-mode behavior (all offline)

The end-to-end taxonomy quality (classification + retrieval) is measured
separately by `scripts/eval_accuracy.py` — see §4 and `docs/eval-report.md`.

Mock singletons are injected into `app.main`'s module globals (`_cache`, `_retriever`, `_llm`, `_social`, `_intent`) before each client is created.

---

## 10. Security & correctness notes

- `.env` is git-ignored; API keys are `SecretStr`.
- HTML from the site is treated as data: rendered text is stripped of scripts; user input is never interpolated as HTML.
- Per-IP rate limiting + query-length validation on every request.
- The crawler respects `robots.txt` and stays on the seeded domain.
- TLS verification is enforced via a custom CA bundle (not disabled).
- Error paths return structured `ErrorResponse` JSON; the LLM layer never raises out of `generate()`.

---

## 11. Known limitations / possible next steps

| Item | Status |
|---|---|
| Frontend professional redesign | ✅ applied (prompts retained in `docs/gemini-prompts.md`) |
| Response-time badge | ✅ implemented (client-side) |
| CORS any-localhost-port fix | ✅ implemented + tested |
| Exact out-of-scope refusal | ✅ implemented — `NO_INFO_MESSAGE` + `source_url: null`, double-checked in the frontend |
| Social media answers | ✅ implemented — static profile links in the KB + live YouTube RSS |
| Floating widget on any site | ✅ implemented — `GET /widget.js` snippet, `GET /widget` iframe, `scripts/demo_proxy.py` demo |
| Query taxonomy + category-aware retrieval | ✅ implemented — 9 categories, hybrid classifier (100 % on the eval set), relevance-banded boost, per-category gate, prompt rubric, 🏷 chip; hit@6 **53.9 % → 65.8 %**, MRR **0.369 → 0.570** (`docs/eval-report.md`) |
| Context meaning of ambiguous words | ✅ implemented — `CONTEXT_SENSES` meanings applied as a re-rank of the found chunks (`apply_sense_rerank`): MRR **0.536 → 0.570** on the category-aware path, hit@6 / gate / refusals unchanged. Query-text glossing was measured and rejected (it cost hits) |
| `MAX_PAGES` cap | ✅ implemented — `MAX_PAGES` (`.env`, default 20000) stops the crawl with a warning; progress logged every 25 pages |
| Eval set size | 86 hand-labelled queries — good enough to catch regressions, not a benchmark; extend `tests/eval/queries.jsonl` to grow it |
| PDF filename collisions | Filenames use the URL basename only — two `syllabus.pdf` overwrite each other |
| KB never pruned | Chunks from removed/depth-reduced pages stay in Chroma forever |
| Stale answers up to 24 h | Refreshing the KB does not invalidate the Redis cache |
| Depth / scale | `.env` uses `MAX_CRAWL_DEPTH=3`, `MAX_PAGES=20000`; full run = 847 docs → 25,181 chunks (~75 min typical, ~4 h on a slow day) |

---

*Companion docs: `README.md` (setup & operations) · `docs/eval-report.md` (generated evaluation results) · `docs/gemini-prompts.md` (AI prompts used for the UI redesign)*
