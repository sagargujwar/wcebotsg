"""
app/llm.py
──────────
LLM service with Gemini primary → Groq fallback logic.

Fallback behaviour (as specified):
  1. Call Gemini gemini-3.8-flash, timeout 25s (single attempt — responses from
     this network routinely take 8–15s, so a tight budget with retries never helps).
  2. On failure or timeout: try Groq openai/gpt-oss-120b (timeout 15s).
  3. On both failing: return a graceful unavailability string — never raise.

The system prompt is authored in full here (not delegated to config) because
it is a core part of the application's grounding logic.
"""
from __future__ import annotations

import concurrent.futures
import logging
import textwrap
from typing import List, Optional

from app.config import get_settings
from app.constants import NO_INFO_MESSAGE, UNAVAILABLE_MESSAGE

logger = logging.getLogger(__name__)

# Real-world Gemini latency from this network is 8–15s even for trivial prompts;
# the previous 10s cap (with 2 blocking retries) cut responses off mid-flight.
_GEMINI_TIMEOUT_SECONDS = 25

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
    2. The context may include the college's official social-media links (Instagram, Facebook,
       LinkedIn, YouTube) and, when available, a list of recent YouTube videos. You may answer
       questions about those social-media accounts using that context, including the links.
    3. Some retrieved chunks may be irrelevant to the question — ignore them and answer
       using the relevant ones. Only when NO chunk contains information relevant to the
       question, reply with EXACTLY this sentence and nothing else: "{no_info_message}"
       Do not add a Source line, extra words, or an explanation when you reply this way.
    4. If the question is not about this college (out of scope — for example general knowledge,
       other institutions, opinions, jokes, or unrelated tasks), reply with EXACTLY this
       sentence and nothing else: "{no_info_message}"
    5. When you do answer a question from the context, cite which source document you used by
       mentioning its title or URL at the end of your answer (e.g., "Source: <title or URL>").
       Rules 3 and 4 override this rule — refusal replies carry no Source line.
    6. Be concise and student-friendly. Prefer bullet points for structured information such as
       fee tables, syllabus lists, exam schedules, and admission criteria.
    7. Never make up fees, dates, deadlines, contact details, admission criteria, or social
       accounts — even if you believe you know the answer from training data.
    8. If the student asks for a document or a direct link, refer them to the source_url
       in the context rather than constructing a URL yourself.

    {rubric}
    Context (retrieved from college documents):
    ─────────────────────────────────────────────
    {context}
    ─────────────────────────────────────────────
    """
)

_UNAVAILABLE = UNAVAILABLE_MESSAGE


def _build_prompt(context_chunks: List[dict], rubric: Optional[str] = None) -> str:
    """Combine retrieved chunks into the context string for the system prompt.

    *rubric* is the category-specific guidance from ``app/categories.py``
    (answer shape + guardrails), interpolated as a normal format value —
    only the template itself is parsed for placeholders.
    """
    parts = []
    for i, chunk in enumerate(context_chunks, 1):
        parts.append(
            f"[{i}] Title: {chunk.get('title', 'Unknown')}\n"
            f"    Source: {chunk.get('source_url', '')}\n"
            f"    Content: {chunk.get('text', '')}"
        )
    context = "\n\n".join(parts)
    rubric_block = ""
    if rubric and rubric.strip():
        rubric_block = (
            "Additional guidance for this category of question:\n"
            f"{rubric.strip()}\n\n"
        )
    return _SYSTEM_PROMPT_TEMPLATE.format(
        context=context,
        no_info_message=NO_INFO_MESSAGE,
        rubric=rubric_block,
    )


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
                self._gemini_client = genai.GenerativeModel("gemini-3.8-flash")
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

    def generate(
        self,
        query: str,
        context_chunks: List[dict],
        rubric: Optional[str] = None,
    ) -> str:
        """
        Generate an answer grounded in *context_chunks*.

        *rubric* is optional category-specific guidance appended to the
        grounded system prompt (see app/categories.py).

        Never raises — returns a graceful string on total failure.
        """
        system_prompt = _build_prompt(context_chunks, rubric=rubric)
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
        # Manual executor (no `with`): on timeout we must NOT wait for the
        # stranded call to finish, or the "timeout" costs the full call duration.
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        try:
            future = executor.submit(self._gemini_client.generate_content, prompt)
            result = future.result(timeout=_GEMINI_TIMEOUT_SECONDS)
            text = result.text.strip()
            if text:
                logger.info("Gemini answered")
                return text
            logger.warning("Gemini returned an empty response")
        except Exception as exc:
            # TimeoutError has an empty str() — fall back to the type name so
            # the log line isn't blank.
            logger.warning(
                "Gemini failed (timeout=%ds): %s",
                _GEMINI_TIMEOUT_SECONDS,
                str(exc) or type(exc).__name__,
            )
        finally:
            executor.shutdown(wait=False, cancel_futures=True)
        return None

    def _try_groq(self, prompt: str) -> Optional[str]:
        if self._groq_client is None:
            return None
        try:
            response = self._groq_client.chat.completions.create(
                model="openai/gpt-oss-120b",
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
