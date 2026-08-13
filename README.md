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
   ChromaDB search  (all-MiniLM-L6-v2, local, no API)
   Retrieve top-4 most similar chunks
         │
         ├─ Similarity < 0.35 → "couldn't find that" (no LLM call)
         │
         ▼  Above threshold
   LLM  (Gemini gemini-2.5-flash → retry → Groq llama-3.3-70b → graceful fail)
   Grounded system prompt: answer ONLY from context, cite sources
         │
         ▼
   Inject PDF source_url if top chunk is a PDF
   Write to Redis cache (TTL 24 h)
         │
         ▼
   Return {answer, source_url, cached: false}
```

---

## Project Structure

```
college-rag-chatbot/
├── app/
│   ├── main.py          # FastAPI app, routes, rate limiting
│   ├── config.py        # pydantic-settings (reads .env)
│   ├── retrieval.py     # ChromaDB search + SentenceTransformer
│   ├── llm.py           # Gemini + Groq calls, fallback, system prompt
│   ├── cache.py         # Redis wrapper + rate-limit counters
│   └── models.py        # Pydantic request/response schemas
├── scraper/
│   ├── scraper.py       # BFS HTML crawler + PDF downloader
│   └── chunker.py       # Paragraph-boundary chunker with overlap
├── scripts/
│   └── refresh_kb.py    # End-to-end pipeline (crawl → chunk → embed → upsert)
├── frontend/
│   └── index.html       # Embeddable chat widget (plain HTML/CSS/JS)
├── tests/
│   ├── fixtures/
│   │   ├── index.html         # Mock college homepage
│   │   ├── syllabus.html      # Mock syllabus page
│   │   ├── fee_structure.pdf  # Generated test PDF (run generate_fixtures.py)
│   │   └── generate_fixtures.py
│   ├── conftest.py
│   ├── test_scraper.py
│   ├── test_retrieval.py
│   └── test_chat_api.py
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
Download and install from https://github.com/tporadowski/redis/releases
Then run:
```powershell
redis-server
```

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

---

## 6 · Start the Backend

```bash
uvicorn app.main:app --reload
```

The API is now available at `http://localhost:8000`.

Useful endpoints:
- `POST /chat` — main query endpoint
- `GET /health` — service status
- `GET /docs` — interactive Swagger UI

---

## 7 · Open the Frontend

Open `frontend/index.html` directly in your browser. No web server needed.

The widget POSTs to `http://localhost:8000` by default. To change the backend URL, edit the `API_BASE` constant at the top of the `<script>` block in `index.html`.

---

## 8 · Run Tests

```bash
pytest tests/ -v
```

All tests pass without any real API keys or a running Redis instance.
The `test_pdf_extracted` test is automatically skipped if the PDF fixture hasn't been generated.

To run a specific test file:
```bash
pytest tests/test_chat_api.py -v
pytest tests/test_scraper.py -v
pytest tests/test_retrieval.py -v
```

---

## 9 · Schedule Automatic KB Refresh (Cron)

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

---

## 10 · Pointing at a Real College Site

1. Set `BASE_URL` in `.env` to the college homepage (e.g. `https://walchandsangli.ac.in`)
2. Adjust `MAX_CRAWL_DEPTH` (1–3 recommended; higher = more pages but slower)
3. Run `python scripts/refresh_kb.py`
4. Start the backend — it will now answer from real college content

**Tip**: Run with `MAX_CRAWL_DEPTH=1` first to see what's scraped, then increase.

---

## Security Notes

- API keys are read only from `.env` / environment — never hardcoded
- `.env` is git-ignored; `.env.example` (no secrets) is committed
- CORS restricted to `ALLOWED_ORIGINS` in `.env`
- All user input is length-checked and sanitised before prompt interpolation
- No `eval`, no shell injection

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `redis.exceptions.ConnectionError` | Start Redis first (`redis-server` or `brew services start redis`) |
| `GEMINI_API_KEY not set` warning | Fill in `.env` — tests still pass without it |
| `/chat` always returns "couldn't find" | Run `python scripts/refresh_kb.py` to populate ChromaDB first |
| `fee_structure.pdf` skip message | Run `python tests/fixtures/generate_fixtures.py` |
| Slow first request | SentenceTransformer model is loaded once on startup — subsequent requests are fast |
