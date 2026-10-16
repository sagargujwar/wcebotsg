# 🎓 WCE Bot — College Information RAG Chatbot

A production-quality **Retrieval-Augmented Generation (RAG)** chatbot that answers student questions about Walchand College of Engineering (or any college) using content scraped from the official website and its PDFs.

Answers come exclusively from retrieved college documents — the bot never guesses or hallucinates. Source links are returned alongside answers.

---

## Architecture

```
Browser Widget (frontend/index.html)
         │  POST /chat {"query": "..."}
         ▼
   FastAPI /chat  (app/main.py)
         │
         ├─ Validate input (Pydantic, max 500 chars)
         ├─ Rate-limit check (Redis counter per IP → 429 if exceeded)
         ├─ Redis cache lookup (normalised query hash)
         │       └─ HIT  → return {answer, source_url, cached: true}
         │
    ▼  MISS
    Query rewrite (abbreviation expansion) + category classification
    (app/intent.py — hybrid keyword rules + seed-query prototypes)
    ChromaDB search  (all-MiniLM-L6-v2, local, no API)
    Retrieve top-6 chunks — when a category is predicted, same-category
    chunks get a small relevance-banded boost (plain ranking otherwise)
           │
           ├─ + live YouTube feed context when the question is about videos
           │
           ├─ Similarity < per-category threshold (default 0.35)
           │   → "I don't have information about this." (no LLM call)
           │
           ▼  Above threshold
    LLM  (Gemini gemini-3.8-flash 25s → Groq openai/gpt-oss-120b → graceful fail)
    Grounded system prompt: answer ONLY from context, cite sources;
    + category rubric (guardrails for that category);
    out-of-scope / missing info → "I don't have information about this."
           │
           ▼
     Source URL injected — suppressed when the answer is the no-info refusal
     Write to Redis cache (TTL 24 h; refusals are never cached)
          │
          ▼
   Return {answer, source_url, category, cached: false}
```

---

## Project Structure

```
college-rag-chatbot/
├── app/
│   ├── main.py          # FastAPI app, routes, rate limiting, widget endpoints
│   ├── config.py        # pydantic-settings (reads .env)
│   ├── constants.py     # the exact "I don't have information about this." reply
│   ├── categories.py    # query taxonomy: 9 categories, lexicons, URL rules, rubrics
│   ├── intent.py        # hybrid query classifier (rules + seed prototypes)
│   ├── retrieval.py     # ChromaDB search + SentenceTransformer + category boost
│   ├── llm.py           # Gemini + Groq calls, fallback, system prompt + rubric
│   ├── social.py        # live YouTube RSS context (keyless, cached)
│   ├── cache.py         # Redis wrapper + rate-limit counters
│   └── models.py        # Pydantic request/response schemas
├── scraper/
│   ├── scraper.py       # BFS HTML crawler + PDF downloader + social links
│   └── chunker.py       # Paragraph-boundary chunker with overlap
├── scripts/
│   ├── refresh_kb.py    # End-to-end pipeline (crawl → chunk → embed → upsert)
│   ├── tag_categories.py  # Backfill chunk categories in Chroma (metadata-only)
│   ├── eval_accuracy.py # Baseline vs category-aware retrieval evaluation
│   └── demo_proxy.py    # Local college-site mirror with the widget injected
├── static_docs/
│   └── social_media.json  # Official social profiles → ingested into the KB
├── frontend/
│   ├── index.html       # Chat UI (standalone page + widget iframe panel)
│   └── widget.js        # Floating-button embed loader for host pages
├── docs/
│   ├── project-explanation.md  # Architecture & design deep-dive
│   ├── eval-report.md          # Generated evaluation results (see §11)
│   └── gemini-prompts.md       # Prompts used for the UI redesign
├── tests/
│   ├── fixtures/
│   │   ├── index.html         # Mock college homepage
│   │   ├── syllabus.html      # Mock syllabus page
│   │   ├── fee_structure.pdf  # Generated test PDF (run generate_fixtures.py)
│   │   └── generate_fixtures.py
│   ├── eval/queries.jsonl     # 79 labelled queries for the eval harness
│   ├── conftest.py
│   ├── test_scraper.py
│   ├── test_retrieval.py
│   ├── test_intent.py
│   ├── test_eval_set.py
│   ├── test_social.py
│   ├── test_cors.py
│   └── test_chat_api.py
├── certs/
│   ├── ca-bundle.pem       # Custom CA bundle: certifi + Sectigo intermediate
│   └── sectigo-dv-r36.pem  # The intermediate alone (see step 5 TLS note)
├── data/
│   ├── chroma_db/       # ChromaDB persistent storage (git-ignored)
│   └── pdfs/            # Downloaded PDFs (git-ignored)
├── .env.example
├── requirements.txt
└── README.md
```

---

## Prerequisites

- **Python 3.10+**
- **Redis** (local installation — see below)
- Free **Gemini** and/or **Groq** API keys

---

## 1 · Install Redis Locally

Redis must be running before you start the backend.

### Windows (recommended: Redis via WSL or the official Windows port)

**Option A — WSL2 (recommended):**
```bash
# Inside WSL2 terminal
sudo apt update && sudo apt install redis-server -y
sudo service redis-server start
redis-cli ping   # should reply PONG
```

**Option B — Native Windows (Redis 7 for Windows):**

1. Download `Redis-7.4.9-Windows-x64-msys2-with-Service.zip` (or newer) from
   https://github.com/redis-windows/redis-windows/releases
2. Extract it somewhere stable, e.g. `C:\Users\<you>\Desktop\Redis` — don't move or
   delete that folder later; the installed service points at it.
3. In an **Admin PowerShell** opened inside that folder:
   ```powershell
   New-Item -ItemType Directory -Path C:\Redis\data -Force   # required — redis-server crashes at startup without it
   .\RedisService.exe install --port 6379 --dir C:\Redis\data
   Start-Service Redis
   redis-cli ping   # should reply PONG
   ```

> Alternative: the **5.0.14.1 MSI** from https://github.com/tporadowski/redis/releases
> installs a service with one click — but that project is unmaintained since 2022 and is
> **Redis 5, not Redis 7**, despite older docs saying so.

### macOS
```bash
brew install redis
brew services start redis
redis-cli ping   # should reply PONG
```

### Linux (Debian/Ubuntu)
```bash
sudo apt install redis-server -y
sudo systemctl enable --now redis-server
redis-cli ping
```

---

## 2 · Clone & Set Up Python Environment

```bash
git clone <repo-url> college-rag-chatbot
cd college-rag-chatbot

# Create and activate virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

---

## 3 · Configure Environment Variables

```bash
cp .env.example .env
```

Edit `.env` and fill in:

| Variable | Where to get it |
|---|---|
| `GEMINI_API_KEY` | https://aistudio.google.com/apikey (free) |
| `GROQ_API_KEY` | https://console.groq.com/keys (free) |
| `BASE_URL` | Your college website URL |
| `REDIS_URL` | `redis://localhost:6379/0` (default) |
| `SSL_CA_BUNDLE` | `./certs/ca-bundle.pem` (default) — custom CA bundle; needed when the target site serves its certificate without the intermediate |
| `YOUTUBE_CHANNEL_ID` | Pre-filled with WCE's official channel `UCkWEgeBCY5U6sl7jS9YShwA` (Media WCE) — enables live YouTube-feed answers; set empty to disable |
| `WIDGET_API_BASE` | `http://localhost:8000` — origin used in the widget embed snippet |
| `DEMO_PROXY_PORT` | `9000` — port for the local college-site demo mirror |

**Getting free API keys:**
- **Gemini**: Go to https://aistudio.google.com/apikey → Sign in with Google → Create API key. Free tier includes generous daily limits.
- **Groq**: Go to https://console.groq.com → Sign up → API Keys → Create. Free tier available.

At least one key is needed for live queries. Tests run without any keys.

---

## 4 · Generate Test PDF Fixture

```bash
python tests/fixtures/generate_fixtures.py
```

This creates `tests/fixtures/fee_structure.pdf`. Run once.

---

## 5 · Populate the Knowledge Base

### Using the mock fixtures (offline, for testing)

```bash
# Point BASE_URL at a local fixture server, or use the test fixtures directly
# For a real demo, point at your college URL (see step 6)
python scripts/refresh_kb.py
```

### Against a real college website

1. Set `BASE_URL=https://your-college-website.edu` in `.env`
2. Optionally set `MAX_CRAWL_DEPTH=2` (start with 1 for a quick test)
3. Run:
   ```bash
   python scripts/refresh_kb.py
   ```
   This crawls the site, downloads PDFs, chunks text, embeds locally, and upserts into ChromaDB. It is idempotent — safe to re-run at any time.

> **TLS note:** if the crawl fails with `SSLCertVerificationError ... certificate verify
> failed` while the same URL opens fine in your browser, the server isn't sending its
> intermediate certificate (common with Sectigo chains). `SSL_CA_BUNDLE=./certs/ca-bundle.pem`
> — certifi's roots plus the intermediate from `http://crt.sectigo.com/SectigoPublicServerAuthenticationCADVR36.crt` —
> fixes it. Rebuild the bundle the same way if the CA ever rotates its intermediate.

---

## 6 · Start the Backend

```bash
uvicorn app.main:app --reload
```

The API is now available at `http://localhost:8000`.

Useful endpoints:
- `POST /chat` — main query endpoint
- `GET /health` — service status
- `GET /widget` — chat panel HTML (loaded inside the floating widget's iframe)
- `GET /widget.js` — floating-button embed loader
- `GET /docs` — interactive Swagger UI

---

## 7 · Open the Frontend

Open `frontend/index.html` directly in your browser. No web server needed.

The widget POSTs to `http://localhost:8000` by default. To change the backend URL, edit the `API_BASE` constant at the top of the `<script>` block in `index.html`.

> A `file://` page sends `Origin: null`, so `ALLOWED_ORIGINS` must include `null`
> (the shipped default does). If the widget shows "Could not reach WCE Bot" while
> `GET /health` works, add `null` to `ALLOWED_ORIGINS` in `.env` and **restart** the
> backend — `.env` is only read at startup, and `uvicorn --reload` watches `.py` files only.

---

## 8 · Floating Chat Widget (embed on any site)

Add the chatbot to any page with **one line**:

```html
<script src="http://localhost:8000/widget.js" defer></script>
```

The script injects a floating action button (bottom-right). Clicking it opens
the chat panel **on the same screen** as an overlay — no navigation, no popup.
The panel is an `<iframe>` served from `GET /widget`, so `/chat` is
same-origin for the iframe and the host page's CSS can't break the UI.

- Backend must be running (`uvicorn app.main:app`).
- Backend on a different host/port:
  `<script src="https://cdn.example.com/widget.js" data-api="http://localhost:8000" defer></script>`
- A page served over **https** needs the backend on https too (browsers block
  active mixed content).

### Demo it on the college site (no site access needed)

`BASE_URL` (e.g. `https://walchandsangli.ac.in/`) is a live site we don't
control, so a local proxy mirrors it and injects the snippet:

```bash
python scripts/demo_proxy.py     # mirrors BASE_URL and injects widget.js
# open http://localhost:9000
```

You see the real college page **with the floating WCE Bot button**; clicking
it opens the chat overlay without leaving the page. Change the port with
`DEMO_PROXY_PORT` in `.env`.

---

## 9 · Social Media Answers

- **Static profiles** — `static_docs/social_media.json` lists the official
  accounts (Instagram, Facebook, LinkedIn, YouTube). `refresh_kb.py` ingests
  **one chunk per platform** with `source_url` set to the profile URL, so
  "What is WCE's Instagram?" answers with the official link. The crawler also
  captures social links found on crawled pages and adds any platform missing
  from the JSON. Re-run `python scripts/refresh_kb.py` after changing the file.
- **Live YouTube feed** — `.env` ships with `YOUTUBE_CHANNEL_ID` set to WCE's
  official channel (**Media WCE**, `UCkWEgeBCY5U6sl7jS9YShwA`). Questions like
  "latest WCE videos" then include the 5 newest videos from YouTube's **keyless
  RSS feed** (8 s timeout, cached 15 minutes; verified live — recent uploads
  with titles, dates, and watch links). To use another channel, open it →
  View Page Source → search `channelId`, or take the `youtube.com/channel/UC…`
  URL. Set the variable empty to disable — static links still work.
- Instagram/Facebook/X have no keyless feeds (their APIs require registered
  apps or paid tiers), so their links are static KB content instead.

---

## 10 · Out-of-Scope Questions

Anything out of scope or missing from the sources gets the exact reply:

> **I don't have information about this.**

…with **no "View Document" link** — the backend returns `source_url: null`
for every refusal (threshold-gate fallback and LLM refusal alike), and the
frontend double-checks before rendering the link. Refusals are also **never
written to the Redis cache**, so a later KB refresh is never masked by a
stale "no info" answer.

The query classifier deliberately **never predicts an out-of-scope category**
(it declines the route and falls back to `general`, no category prior is
applied) — producing the refusal stays with the similarity gate and the
grounded LLM prompt, exactly as above. Verified on the eval set: 10/10
out-of-scope queries declined (see §11).

---

## 11 · Run Tests

```bash
pytest tests/ -v
```

177 tests — all passing, without any real API keys or a running Redis instance.
The `test_pdf_text_extracted` and `test_missing_content_type_sniffs_pdf` tests are
automatically skipped if the PDF fixture hasn't been generated.

To run a specific test file:
```bash
pytest tests/test_chat_api.py -v
pytest tests/test_intent.py -v      # query-taxonomy classifier
pytest tests/test_retrieval.py -v
pytest tests/test_scraper.py -v
pytest tests/test_social.py -v
pytest tests/test_cors.py -v
```

### Evaluate retrieval accuracy (query taxonomy)

The bot classifies every query into a category (academics, admissions, notices, …)
and uses it to steer retrieval. A labelled set of 79 queries
(`tests/eval/queries.jsonl`) measures both halves — no LLM calls, embedding +
Chroma only:

```bash
python scripts/eval_accuracy.py --out docs/eval-report.md
```

Latest results (see `docs/eval-report.md`):

| metric | baseline | category-aware |
|---|---|---|
| classification accuracy | — | **100% (79/79)** |
| hit@6 (69 answerable queries) | 53.6% (37/69) | **62.3% (43/69)** |
| MRR | 0.345 | **0.523** |
| out-of-scope queries routed to a category | — | 0/10 (all correctly declined) |

6 queries gained a relevant top-6 hit; **0 lost**.

After changing any URL→category rule, retag the existing KB (metadata only, no
re-embedding):

```bash
python scripts/tag_categories.py --force
```

---

## 12 · Schedule Automatic KB Refresh (Cron)

The knowledge base should be refreshed periodically as the college website updates.

### Linux / macOS (cron)

```bash
crontab -e
```

Add a line to refresh every day at 2 AM:
```
0 2 * * * cd /path/to/college-rag-chatbot && /path/to/.venv/bin/python scripts/refresh_kb.py >> logs/refresh.log 2>&1
```

### Windows (Task Scheduler)

1. Open Task Scheduler → Create Basic Task
2. **Trigger**: Daily at 02:00
3. **Action**: Start a program
   - Program: `C:\path\to\.venv\Scripts\python.exe`
   - Arguments: `scripts/refresh_kb.py`
   - Start in: `C:\path\to\college-rag-chatbot`

After the first run, check **Last Run Result** in Task Scheduler: `0x0` = success.
Task output isn't logged by default — to see errors, run the same command manually.
If the console shows a stale "Running" row after the job finished, press F5 to refresh
the view; don't cancel it.

---

## 13 · Pointing at a Real College Site

1. Set `BASE_URL` in `.env` to the college homepage (e.g. `https://walchandsangli.ac.in`)
2. Adjust `MAX_CRAWL_DEPTH` (1–3 recommended; higher = more pages but slower)
3. Run `python scripts/refresh_kb.py`
4. Start the backend — it will now answer from real college content

**Tip**: Run with `MAX_CRAWL_DEPTH=1` first to see what's scraped, then increase.

---

## Security Notes

- API keys are read only from `.env` / environment — never hardcoded
- `.env` is git-ignored; `.env.example` (no secrets) is committed
- CORS restricted to `ALLOWED_ORIGINS` in `.env` (includes `null` so the local `file://` widget works)
- All user input is length-checked and sanitised before prompt interpolation
- No `eval`, no shell injection

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `redis.exceptions.ConnectionError` | Start Redis first (`Start-Service Redis` on Windows, `redis-server` or `brew services start redis` elsewhere) |
| `GEMINI_API_KEY not set` warning | Fill in `.env` — tests still pass without it |
| `/chat` always returns "couldn't find" | Run `python scripts/refresh_kb.py` to populate ChromaDB first |
| `fee_structure.pdf` skip message | Run `python tests/fixtures/generate_fixtures.py` |
| Slow first request | SentenceTransformer model is loaded once on startup — subsequent requests are fast |
| `SSLCertVerificationError ... certificate verify failed` (crawl gets 0 documents) | Site omits its intermediate cert — ensure `SSL_CA_BUNDLE=./certs/ca-bundle.pem` is set and the file exists (step 5) |
| `404 ... model not found` / `model ... no longer available` | Providers rename models — update names in `app/llm.py` (currently `gemini-3.8-flash` and `openai/gpt-oss-120b`) |
| Widget: "Could not reach WCE Bot" but `/health` is ok | `ALLOWED_ORIGINS` must include `null` for `file://` pages; restart the backend after editing `.env` |
| Floating button doesn't appear on the host page | Backend must be running; check the browser console — an https page blocks an http `widget.js` (mixed content) |
| `demo_proxy.py` returns "Upstream unavailable" | `BASE_URL` unreachable — check network / `.env` |
| `WinError 10013` starting uvicorn | Port 8000 already in use — find and stop it: `Get-NetTCPConnection -LocalPort 8000` |
| Scheduled task `Last Run Result: 0x1` | Refresh failed silently — run `python scripts/refresh_kb.py` manually to see the error |
| `python -m venv` → "Unable to copy venvlauncher.exe" | A process is using `.venv` (e.g. a running uvicorn) — stop it first, then retry |
| First chat answer takes 4–30 s | Gemini latency from your network varies; Groq fallback covers timeouts, and answers are cached in Redis for 24 h |
