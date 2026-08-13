"""
app/llm.py
──────────
LLM service with Gemini primary → Groq fallback logic.

Fallback behaviour (as specified):
  1. Call Gemini gemini-2.5-flash, timeout 10s.
  2. On any exception: retry once (same provider).
  3. On second failure: try Groq llama-3.3-70b-versatile.
  4. On both failing: return a graceful unavailability string — never raise.

The system prompt is authored in full here (not delegated to config) because
it is a core part of the application's grounding logic.
"""
from __future__ import annotations

import logging
import textwrap
from typing import List, Optional

from app.config import get_settings

logger = logging.getLogger(__name__)

# ── System prompt ─────────────────────────────────────────────────────────────
# Written out in full as required. {context} is interpolated at call time.
_SYSTEM_PROMPT_TEMPLATE = textwrap.dedent(
    """\
    You are WCE Bot, a helpful college information assistant for Walchand College of Engineering \
    (or the college whose documents are provided below).
    Your job is to answer student questions about the college using ONLY the information \
    provided in the context below.

    Rules you must follow without exception:
    1. Answer ONLY from the provided context. Do not use any prior knowledge, do not infer,
       do not guess, and do not hallucinate facts.
    2. If the context does not contain enough information to fully answer the question, say exactly:
       "I don't have that information in the college's published documents."
       Do not attempt to fill gaps with general knowledge.
    3. Always cite which source document you used by mentioning its title or URL at the end
       of your answer (e.g., "Source: <title or URL>").
    4. Be concise and student-friendly. Prefer bullet points for structured information such as
       fee tables, syllabus lists, exam schedules, and admission criteria.
    5. Never make up fees, dates, deadlines, contact details, or admission criteria —
       even if you believe you know the answer from training data.
    6. If the student asks for a document or a direct link, refer them to the source_url
       in the context rather than constructing a URL yourself.

    Context (retrieved from college documents):
    ─────────────────────────────────────────────
    {context}
    ─────────────────────────────────────────────
    """
)

_UNAVAILABLE = (
    "I'm sorry, the WCE Bot service is temporarily unavailable. "
    "Please try again in a few moments."
)


def _build_prompt(context_chunks: List[dict]) -> str:
    """Combine retrieved chunks into the context string for the system prompt."""
    parts = []
    for i, chunk in enumerate(context_chunks, 1):
        parts.append(
            f"[{i}] Title: {chunk.get('title', 'Unknown')}\n"
            f"    Source: {chunk.get('source_url', '')}\n"
            f"    Content: {chunk.get('text', '')}"
        )
    context = "\n\n".join(parts)
    return _SYSTEM_PROMPT_TEMPLATE.format(context=context)


class LLMService:
    """Orchestrates Gemini → Groq fallback calls."""

    def __init__(self) -> None:
        self._settings = get_settings()
        self._gemini_client = None
        self._groq_client = None
        self._init_clients()

    def _init_clients(self) -> None:
        """Lazily initialise LLM clients — missing keys are warned, not fatal."""
        settings = self._settings

        gemini_key = settings.gemini_api_key.get_secret_value()
        if gemini_key:
            try:
                import google.generativeai as genai  # type: ignore

                genai.configure(api_key=gemini_key)
                self._gemini_client = genai.GenerativeModel("gemini-2.5-flash")
                logger.info("Gemini client initialised")
            except Exception as exc:
                logger.warning("Gemini client init failed: %s", exc)
        else:
            logger.warning("GEMINI_API_KEY not set — Gemini disabled")

        groq_key = settings.groq_api_key.get_secret_value()
        if groq_key:
            try:
                from groq import Groq  # type: ignore

                self._groq_client = Groq(api_key=groq_key)
                logger.info("Groq client initialised")
            except Exception as exc:
                logger.warning("Groq client init failed: %s", exc)
        else:
            logger.warning("GROQ_API_KEY not set — Groq disabled")

    # ── Public API ────────────────────────────────────────────────────────────

    def generate(self, query: str, context_chunks: List[dict]) -> str:
        """
        Generate an answer grounded in *context_chunks*.

        Never raises — returns a graceful string on total failure.
        """
        system_prompt = _build_prompt(context_chunks)
        full_prompt = f"{system_prompt}\n\nStudent question: {query}"

        # 1 — Try Gemini (up to 2 attempts)
        answer = self._try_gemini(full_prompt)
        if answer is not None:
            return answer

        # 2 — Fallback to Groq
        answer = self._try_groq(full_prompt)
        if answer is not None:
            return answer

        # 3 — Both failed
        logger.error("Both Gemini and Groq failed for query: %.80s", query)
        return _UNAVAILABLE

    # ── Private helpers ───────────────────────────────────────────────────────

    def _try_gemini(self, prompt: str) -> Optional[str]:
        if self._gemini_client is None:
            return None
        for attempt in range(2):
            try:
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(
                        self._gemini_client.generate_content, prompt
                    )
                    result = future.result(timeout=10)
                text = result.text.strip()
                if text:
                    logger.info("Gemini answered (attempt %d)", attempt + 1)
                    return text
            except Exception as exc:
                logger.warning("Gemini attempt %d failed: %s", attempt + 1, exc)
        return None

    def _try_groq(self, prompt: str) -> Optional[str]:
        if self._groq_client is None:
            return None
        try:
            response = self._groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                timeout=15,
            )
            text = response.choices[0].message.content.strip()
            if text:
                logger.info("Groq answered as fallback")
                return text
        except Exception as exc:
            logger.warning("Groq fallback failed: %s", exc)
        return None
