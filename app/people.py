"""
app/people.py
────────────
Person-aware retrieval for "who is <name>" style queries.

Why this exists
---------------
A student asking about a specific faculty member should be pointed at that
person's own document — their faculty profile or CV — not at an aggregate
report (AISHE / NIRF / annual report) that merely lists their name in a
table.  Aggregate tables match the *name* but carry no biography, and the
dedicated profile pages often rank too low for the embedding model to
surface them at all (a profile's name lives on a different chunk from its
publication list, so no single chunk contains the whole query).

Approach
--------
Rather than nudging scores (which cannot promote a doc that never enters the
candidate pool), we *target* the lookup: extract the rare name terms from the
query, then find chunks that live in a dedicated person document and contain
those terms.  Only chunks from documents that look like a resume/CV/profile
are considered, so a name shared with an unrelated doc cannot win, and a
generic query ("who is the principal") yields no name terms and never fires.
"""
from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# One-time index of every dedicated person document in the KB (built lazily on
# the first person query, then reused).  53 docs × ~8 chunks is small enough to
# hold in memory, and it sidesteps the fact that a profile often ranks outside
# the embedding top-k for its own owner's name.
_PERSON_DOCS_CACHE: Optional[Dict[str, List[dict]]] = None

# ── Query → name terms ───────────────────────────────────────────────────────
# Words that are about a *role* rather than a person.  Stripping them stops
# "who is the head of computer engineering department" from yielding
# ("head", "computer", …), which would match dozens of unrelated CVs.
_NAME_STOP = frozenset(
    """
    the a an and or is are was were of to in on at for by with from who whom
    whose what when where why how which that this these those it its i me my
    we our you your he she they them his her their do does did done have has
    had can could would should will shall may might must tell give show list
    get know about faculty professor staff member someone person name head
    hod principal director department dean coordinator chairman
    walchand college university institute department engineering computer
    information technology science electronic electrical mechanical civil
    resume curriculum vitae declaration attachment
   """.split()
)

# Name-ish tokens: alphabetic, ≥4 chars.  Filters "d" in "siddharaj d pujari"
# (too short to be a surname signal) while keeping "anant", "umbarkar".
_NAME_TOKEN_RE = re.compile(r"[a-zA-Z]{4,}")

# How many name terms a query must yield before we treat it as a person query.
_MIN_NAME_TERMS = 1

# ── Document → "is this one person's own document?" ──────────────────────────
# Matches the *filename*, not the whole URL, so "/faculty/" listing pages are
# not mistaken for an individual profile.  Order matters little; all alternatives
# denote a single person's resume/CV/profile.
_PERSON_DOC_RE = re.compile(
    r"""
    (?:^|[-_. ])                 # filename separator
    (?:
        resume                   # Mr.-Atul-Patane-Resume.pdf
      | cv                       # CV-SDR.pdf, CV_NSB.pdf
      | faculty[ _-]?profile     # Faculty_Profile_Anant-J-Umbarkar.pdf
      | profile                  # Profile_NLG.pdf, SDP1.pdf is caught below
    )
    (?:$|[-_. ])
    |
    ^sdp\d+\.pdf$                # opaque initials export (SDP1.pdf)
    """,
    re.IGNORECASE | re.VERBOSE,
)

# A person doc must also look like one in its text — stops a syllabus whose
# filename happens to contain "profile".
_PERSON_TEXT_RE = re.compile(
    r"(date of (?:birth|joining)|designation|qualification|email[- ]?id"
    r"|total experience|areas of (?:specialization|expertise)"
    r"|previous experience|about faculty|objective)",
    re.IGNORECASE,
)

# The document's own subject, as declared on the bio page.  A CV routinely
# *mentions* other people (co-authors, students, guides), so matching a
# queried name anywhere in the file is not enough — the name must be the
# person the document is about.  Handles the shapes seen in this KB:
#   "Profile of Mr. Pujari Siddharaj Dattatray"   (SDP1.pdf)
#   "Name Atul Siddheshwar Patane"                (name, no colon)
#   "Name\n: Dr. RashmiJadhav"                    (colon on next line)
# "Name of Board/Institution" table headers are excluded.
_SUBJECT_RE = re.compile(
    r"(?:profile\s+of\b|curriculum\s+vita(?:e|\s+of)\b|name\b(?!s?\s+of)\s*:?)"
    r"\s*([^\n]+)",
    re.IGNORECASE,
)

# "RashmiJadhav" (PDF extraction dropped the space) → "Rashmi Jadhav".
_CAMEL_RE = re.compile(r"(?<=[a-z])(?=[A-Z])")

# "Mr. Pujari Siddharaj Dattatray" / "Dr. Atul S. Patane" → content words.
_TITLE_RE = re.compile(r"\b(?:mr|mrs|ms|dr|prof|professor|er|shri|smt)\b\.?", re.IGNORECASE)

# Words that mark the *document*, never the person (filename terms only).
_DOC_WORDS = frozenset(
    "resume cv profile faculty curriculum vitae biodata declaration attachment".split()
)

# Cap on how many targeted chunks we inject.
_MAX_PERSON_CHUNKS = 3


def _filename(url: str) -> str:
    return (url or "").rsplit("/", 1)[-1]


def _is_person_doc(url: str, text: str = "") -> bool:
    """True when *url* is a dedicated single-person resume/CV/profile file."""
    name = _filename(url)
    if not name:
        return False
    if not _PERSON_DOC_RE.search(name):
        return False
    # Text guard is skipped when we have no text (caller may pass none).
    return True if not text else bool(_PERSON_TEXT_RE.search(text))


def name_terms(query: str) -> List[str]:
    """
    Rare, name-like terms from *query*, lowercased, in first-seen order.

    Returns [] for role queries ("who is the principal"), so the person path
    simply never fires and baseline retrieval is untouched.
    """
    seen: List[str] = []
    for raw in _NAME_TOKEN_RE.findall(query or ""):
        term = raw.lower()
        if term in _NAME_STOP or term in seen:
            continue
        seen.append(term)
    return seen if len(seen) >= _MIN_NAME_TERMS else []


def _name_tokens(line: str) -> List[str]:
    """Content name words of *line*: titles/stopwords stripped, camelCase
    split ("RashmiJadhav" → rashmi, jadhav), lowercased.  The token regex
    already requires ≥4 letters, so initials never survive."""
    line = _TITLE_RE.sub(" ", line)
    line = _CAMEL_RE.sub(" ", line)
    out: List[str] = []
    for raw in _NAME_TOKEN_RE.findall(line):
        term = raw.lower()
        if term in _NAME_STOP or term in _DOC_WORDS or term in out:
            continue
        out.append(term)
    return out


def _subject_terms(url: str, text: str) -> List[str]:
    """
    Name words the document claims to be *about*.

    Two sources, in order: the filename ("Faculty_Profile_Anant-J-Umbarkar.pdf"
    → anant, umbarkar) and the declared subject lines ("Profile of Mr. Pujari
    Siddharaj Dattatray" → pujari, siddharaj, dattatray).  Union, deduped.
    """
    stem = re.sub(r"\.[A-Za-z]+$", "", _filename(url))
    terms = _name_tokens(stem.replace("-", " ").replace("_", " "))
    for match in _SUBJECT_RE.finditer(text or ""):
        for term in _name_tokens(match.group(1)):
            if term not in terms:
                terms.append(term)
    return terms


def _covers(subject: List[str], term: str) -> bool:
    """True when a subject word accounts for *term* — exact, or prefix in
    either direction, so query "anant" covers subject "anantkumar" and vice
    versa (PDF text and filenames abbreviate names inconsistently)."""
    return any(term == s or term.startswith(s) or s.startswith(term) for s in subject)


# ── Chroma lookup ────────────────────────────────────────────────────────────

def _person_doc_index(collection) -> Dict[str, List[dict]]:
    """
    All dedicated person documents in the KB, grouped by URL (cached).

    Built once per process from a full collection scan rather than a per-query
    embedding probe, because a person's own profile routinely ranks *below*
    the top-k for their own name (their name sits on the bio page while the
    query's other words sit in the publication list, so no single chunk is a
    good embedding match).  Scanning also lets us match a name split across
    chunks, which a per-chunk lexical filter cannot do.
    """
    global _PERSON_DOCS_CACHE
    if _PERSON_DOCS_CACHE is not None:
        return _PERSON_DOCS_CACHE

    grouped: Dict[str, List[dict]] = {}
    try:
        raw = collection.get(include=["documents", "metadatas"])
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Person-doc index build failed (%s)", exc)
        return {}

    for text, meta in zip(raw.get("documents") or [], raw.get("metadatas") or []):
        meta = meta or {}
        url = meta.get("source_url", "")
        if not url or not _is_person_doc(url):
            continue
        grouped.setdefault(url, []).append(
            {
                "text": text,
                "source_url": url,
                "doc_type": meta.get("doc_type", "pdf"),
                "title": meta.get("title", ""),
                "category": meta.get("category", ""),
                # Exact lexical hit on the person's own file.
                "score": 1.0,
            }
        )

    _PERSON_DOCS_CACHE = grouped
    logger.info("Person-doc index: %d dedicated resume/profile documents", len(grouped))
    return grouped


def reset_person_doc_cache() -> None:
    """Drop the cached index (tests, and after a KB refresh)."""
    global _PERSON_DOCS_CACHE
    _PERSON_DOCS_CACHE = None


def find_person_chunks(
    collection,
    query: str,
    max_chunks: int = _MAX_PERSON_CHUNKS,
) -> List[dict]:
    """
    Return up to *max_chunks* chunks from the named person's own document.

    *collection* is a Chroma collection (injected so this module stays free of
    embedding/network concerns and is trivially unit-testable with a stub).

    Returns chunk dicts shaped like the retriever's output (text, source_url,
    doc_type, title, category, score) with score set to 1.0 — these are exact
    lexical hits on the person's own file, which outrank any embedding
    similarity for this query shape.  Empty list when the query names nobody
    or no matching profile exists.
    """
    terms = name_terms(query)
    if not terms:
        return []

    grouped = _person_doc_index(collection)

    # A CV is paginated, so the queried name often lands on one page while
    # other terms land on another.  Match per *document* over its combined text.
    #
    # The decisive test is subject identity, not mere mention: a CV routinely
    # names other people (co-authors, guides, students), so "siddharaj pujari"
    # occurs inside Atul Patane's resume as a co-author.  We only accept a
    # document whose own declared subject ("Profile of Mr. Pujari Siddharaj
    # Dattatray", "Name : Mr. Atul Siddheshwar Patane") covers the query.
    hits: List[tuple] = []  # (rank, url, chunks) — rank 0 when a bio chunk leads
    for _url, chunks in grouped.items():
        combined = "\n".join(c["text"] for c in chunks)
        subject = _subject_terms(_url, combined)
        # Require every query term to be the *subject* of this document.
        # Matching a name that merely appears in the file lets one faculty
        # member's CV win for a colleague's query (co-author lists), which is
        # exactly the bug this check exists to prevent.
        if not all(_covers(subject, t) for t in terms):
            continue

        mentioned = [c for c in chunks if any(t in c["text"].lower() for t in terms)]
        if not mentioned:
            continue  # name only in the filename; text never discusses them
        # Lead with chunks that read like a biography, but accept a
        # marker-less CV (some resumes say "Work Experience" rather than
        # "Designation") instead of dropping the person entirely.
        bio = [c for c in mentioned if _PERSON_TEXT_RE.search(c["text"])]
        if bio:
            lead = sorted(bio, key=lambda c: len(c["text"]), reverse=True)
        else:
            lead = mentioned  # document order: header chunk first
        rest = [c for c in chunks if c not in lead]
        hits.append((0 if bio else 1, _url, lead + rest))

    if not hits:
        return []

    out: List[dict] = []
    for _rank, _url, chunks in sorted(hits, key=lambda h: (h[0], h[1])):
        out.extend(chunks)
        if len(out) >= max_chunks:
            break
    return out[:max_chunks]
