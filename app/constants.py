"""
app/constants.py
────────────────
Shared application-wide string constants.

NO_INFO_MESSAGE is the single, exact reply for:
  - queries whose retrieval score is below the similarity threshold, and
  - questions the LLM cannot answer from the retrieved context
    (including out-of-scope questions that have nothing to do with WCE).

It must match the phrase hard-coded in the system prompt (app/llm.py) and the
guard in the frontend (frontend/index.html) — change all three together.
"""
from __future__ import annotations

NO_INFO_MESSAGE = "I don't have information about this."

UNAVAILABLE_MESSAGE = (
    "I'm sorry, the WCE Bot service is temporarily unavailable. "
    "Please try again in a few moments."
)
