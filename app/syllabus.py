"""
app/syllabus.py
───────────────
Syllabus-aware retrieval for "<branch> syllabus <year>" style queries.

Why this exists
---------------
A student asking "current third year syllabus for cse" expects the CSE
syllabus PDF, but the embedding model cannot surface it: every syllabus PDF
in the KB shares near-identical assessment boilerplate ("MSE, ISE and ESE"),
so the pool for a syllabus query floods with those fragments plus academic-
calendar dates, while the actual course-content chunks of the target document
rank #105–190 of 25k — far outside the candidate pool.  The gate then passes
on boilerplate similarity and the LLM correctly refuses for lack of real
content.  (Same structural failure person-docs had; see app/people.py.)

Approach
--------
The branch, year-of-study and academic year a student wants are encoded in
the *document filename* ("5.-TY-CSE-Syllabus-25-26-CC.pdf" → cse + third
year + 25-26), not discriminatively in any single chunk's text.  So we
target the lookup: parse those constraints from the query, match them
against a cached index of syllabus documents, and inject chunks from the
single best-matching document.  Queries without a named branch, or without
syllabus intent, never fire — baseline retrieval is untouched.
"""
from __future__ import annotations

import logging
import re
from typing import Dict, FrozenSet, List, Optional, Tuple

logger = logging.getLogger(__name__)

# One-time index of every syllabus document in the KB (built lazily on the
# first syllabus query, then reused).  Filename-derived metadata only — the
# match is on document identity, not chunk text.
_SYLLABUS_DOCS_CACHE: Optional[Dict[str, dict]] = None

# Cap on how many targeted chunks we inject.
_MAX_SYLLABUS_CHUNKS = 3

# ── Query → "is this a syllabus request?" ────────────────────────────────────
_SYLLABUS_INTENT_RE = re.compile(
    r"\b(?:syllabus|curricul(?:a|um)\b|course[\s-]?contents?\b|subjects?\b"
    r"|credits?[\s-]?structure\b|class[\s-]?schedule\b)",
    re.IGNORECASE,
)

# ── Branch aliases (query side → document-side token set) ────────────────────
# Each entry: (query pattern, document filename tokens that satisfy it).
_DEPT_ALIASES: List[Tuple[re.Pattern, Tuple[str, ...]]] = [
    (
        re.compile(
            r"\b(?:cse|cs|computer(?:[\s-]*(?:science|engineering))?"
            r"(?:[\s-]*and[\s-]*engineering)?)\b",
            re.IGNORECASE,
        ),
        ("cse",),
    ),
    # Bare "it" only counts next to syllabus/branch wording — the English
    # pronoun ("is it possible to get the syllabus") must not target IT.
    (
        re.compile(
            r"\b(?:information[\s-]*technology"
            r"|(?:for|of|in|by)\s+it\b"
            r"|\bit\s+(?:syllabus|curricul(?:a|um)|course[\s-]?contents?|subjects?"
            r"|branch|department))\b",
            re.IGNORECASE,
        ),
        ("it",),
    ),
    (
        re.compile(
            r"\b(?:aiml|a\.?i\.?[\s-]*m\.?l\.?|artificial[\s-]*intelligence"
            r"(?:[\s-]*and[\s-]*machine[\s-]*learning)?|machine[\s-]*learning)\b",
            re.IGNORECASE,
        ),
        ("aiml",),
    ),
    (re.compile(r"\b(?:mech|mechanical)\b", re.IGNORECASE), ("mech",)),
    (
        re.compile(
            r"\b(?:eln|ece|entc|electronics?|electronic"
            r"|electronics[\s-]*and[\s-]*communication)\b",
            re.IGNORECASE,
        ),
        ("eln", "elect", "electronics"),
    ),
    (re.compile(r"\belectrical\b", re.IGNORECASE), ("electrical",)),
    (re.compile(r"\bcivil\b", re.IGNORECASE), ("civil",)),
    (
        re.compile(
            r"\b(?:ra|robotics(?:[\s-]*and[\s-]*automation)?|automation)\b",
            re.IGNORECASE,
        ),
        ("ra", "robotics"),
    ),
    (re.compile(r"\bvlsi\b", re.IGNORECASE), ("vlsi",)),
    (re.compile(r"\bdata[\s-]*science\b", re.IGNORECASE), ("datascience",)),
    (re.compile(r"\bmba\b", re.IGNORECASE), ("mba",)),
]

# Document-side department detection (on the normalised filename + title).
_DEPT_DOC_RES: Dict[str, re.Pattern] = {
    "cse": re.compile(r"\bcse\b", re.IGNORECASE),
    "it": re.compile(r"\bit\b", re.IGNORECASE),
    "aiml": re.compile(r"\baiml\b", re.IGNORECASE),
    "mech": re.compile(r"\b(?:mech|mechanical)\b", re.IGNORECASE),
    "eln": re.compile(r"\b(?:eln|electronics?|elect)\b", re.IGNORECASE),
    "electrical": re.compile(r"\belectrical\b", re.IGNORECASE),
    "civil": re.compile(r"\bcivil\b", re.IGNORECASE),
    "ra": re.compile(r"\b(?:ra|robotics)\b", re.IGNORECASE),
    "vlsi": re.compile(r"\bvlsi\b", re.IGNORECASE),
    "datascience": re.compile(r"\bdata[\s-]*science\b", re.IGNORECASE),
    "mba": re.compile(r"\bmba\b", re.IGNORECASE),
}

# ── Year of study ────────────────────────────────────────────────────────────
# Query and document sides share these patterns (applied to normalised text).
_YOS_RES: List[Tuple[re.Pattern, int]] = [
    (re.compile(r"\b(?:final|fourth|4th)\b", re.IGNORECASE), 4),
    (re.compile(r"\b(?:fy|f\.?\s?y\.?|first|1st)\b", re.IGNORECASE), 1),
    (re.compile(r"\b(?:sy|s\.?\s?y\.?|second|2nd)\b", re.IGNORECASE), 2),
    (re.compile(r"\b(?:ty|t\.?\s?y\.?|third|3rd)\b", re.IGNORECASE), 3),
]

# ── Academic year ────────────────────────────────────────────────────────────
# Applied to the *raw* filename/title (dashes intact); normalisation turns
# separators into spaces, so both separators and spaces are accepted here.
# "19 to 23" (a span, not an AY) and bare dates like "18 7 24" never match.
_AY_LONG_RE = re.compile(
    r"(?<!\d)((?:19|20)\d{2})[\s_-]+((?:19|20)\d{2}|\d{2})(?!\d)"
)
_AY_SHORT_RE = re.compile(r"(?<!\d)(1[5-9]|2[0-9])[\s_-]+(1[5-9]|2[0-9])(?!\d)")

# Degree level (query + document): "mtech"/"mba" are opt-in filters; every
# other syllabus doc is treated as B.Tech-level course content.
_MTECH_RE = re.compile(r"\bm\.?\s?tech\b", re.IGNORECASE)
_MBA_RE = re.compile(r"\bmba\b", re.IGNORECASE)

# ── Document predicate ───────────────────────────────────────────────────────
# Applied to the *normalised* filename + title (separators → spaces), so
# "Course_Contents" and "Course-Contents" both match.  Primary: an explicit
# syllabus keyword.  Secondary: a known branch token plus a year-of-study
# marker — catches terse exports like "5.-FY-CSE-26_27.pdf".  A doc must also
# carry a branch token: branch-less mega-docs (Course_Contents-19-to-23,
# MDM, Credit_System) are exactly the boilerplate a targeted query must
# never land on, and no query can name them anyway.
_SYLLABUS_DOC_RE = re.compile(
    r"(?:^|\s)(?:syllabus|curricul(?:a|um)\b|course\s+contents?"
    r"|course\s+content|credit\s+structure|cc)(?=\s|$)",
    re.IGNORECASE,
)
_YOS_MARKER_RE = re.compile(
    r"\b(?:fy|sy|ty|f\.?\s?y|s\.?\s?y|t\.?\s?y|final|first|second|third|cc)\b",
    re.IGNORECASE,
)
_SYLLABUS_WORD_RE = re.compile(r"syllabus", re.IGNORECASE)

# Chunks that lead the injected set: they carry the programme/course header
# the LLM needs to name actual subjects.
_COURSE_HEADER_RE = re.compile(
    r"course\s*(?:name|code|title)|programme\b|class,?\s*semester"
    r"|semester\s+(?:first|second|third|final)",
    re.IGNORECASE,
)
_STUB_MAX_CHARS = 150


def _filename(url: str) -> str:
    return (url or "").rsplit("/", 1)[-1]


def _normalise(url: str, title: str) -> str:
    """Filename + title, lowercased, separators → spaces (AY parsing happens
    on the raw string before this, so '25-26' and '25 26' both work)."""
    stem = re.sub(r"\.[A-Za-z]+$", "", _filename(url))
    return re.sub(r"[-_.]+", " ", f"{stem} {title or ''}").lower()


def _parse_ay(text: str) -> Optional[Tuple[int, int]]:
    """First academic-year pair in *text*, normalised to 2-digit (26, 27)."""
    m = _AY_LONG_RE.search(text)
    if m:
        return (int(m.group(1)) % 100, int(m.group(2)) % 100)
    m = _AY_SHORT_RE.search(text)
    if m:
        return (int(m.group(1)), int(m.group(2)))
    return None


def _parse_yos(norm: str) -> Optional[int]:
    """Year of study from the *first* marker occurring in the text.

    First-occurrence matters: "TY-AIML-Syllabus-with-codes-final" names T.Y.
    up front and only uses "final" as a filename adjective — pattern-list
    order would misread it as final year.
    """
    best: Optional[Tuple[int, int]] = None
    for rx, value in _YOS_RES:
        m = rx.search(norm)
        if m and (best is None or m.start() < best[0]):
            best = (m.start(), value)
    return best[1] if best else None


def _doc_depts(norm: str) -> FrozenSet[str]:
    return frozenset(key for key, rx in _DEPT_DOC_RES.items() if rx.search(norm))


def _doc_level(raw: str) -> str:
    if _MTECH_RE.search(raw):
        return "mtech"
    if _MBA_RE.search(raw):
        return "mba"
    return "btech"


def is_syllabus_doc(url: str, title: str = "") -> bool:
    """True when *url* is a branch-tagged syllabus or course-content file."""
    if not _filename(url):
        return False
    norm = _normalise(url, title)
    # Without a branch token the doc cannot satisfy any query constraint —
    # and the branch-less mega-docs are precisely the boilerplate to avoid.
    if not _doc_depts(norm):
        return False
    if _SYLLABUS_DOC_RE.search(norm):
        return True
    # Terse exports ("5.-FY-CSE-26_27.pdf"): branch + a year-of-study marker.
    return bool(_YOS_MARKER_RE.search(norm))


# ── Query → targeting constraints ────────────────────────────────────────────

def parse_syllabus_query(query: str) -> Optional[dict]:
    """
    Constraints a syllabus query imposes, or None when the query is not a
    syllabus request or names no branch (branch-less questions like "what
    subjects are taught in B.Tech first year?" are ambiguous across departments
    and stay with baseline retrieval).
    """
    q = query or ""
    if not _SYLLABUS_INTENT_RE.search(q):
        return None
    dept_tokens: Optional[Tuple[str, ...]] = None
    for rx, tokens in _DEPT_ALIASES:
        if rx.search(q):
            dept_tokens = tokens
            break
    if not dept_tokens:
        return None
    if _MTECH_RE.search(q):
        level = "mtech"
    elif _MBA_RE.search(q):
        level = "mba"
    else:
        level = "btech"
    return {
        "dept": dept_tokens,
        "yos": _parse_yos(q),
        "ay": _parse_ay(q),
        "level": level,
    }


# ── Chroma lookup ────────────────────────────────────────────────────────────

def _syllabus_doc_index(collection) -> Dict[str, dict]:
    """
    All syllabus documents in the KB, keyed by URL (cached).

    Built once per process from a full collection scan: the target document's
    chunks routinely rank below the candidate pool for the query that names it
    (see module docstring), and the matching signals — branch, year of study,
    academic year — live in the filename, which only a scan can see.
    """
    global _SYLLABUS_DOCS_CACHE
    if _SYLLABUS_DOCS_CACHE is not None:
        return _SYLLABUS_DOCS_CACHE

    grouped: Dict[str, dict] = {}
    try:
        raw = collection.get(include=["documents", "metadatas"])
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Syllabus-doc index build failed (%s)", exc)
        return {}

    for text, meta in zip(raw.get("documents") or [], raw.get("metadatas") or []):
        meta = meta or {}
        url = meta.get("source_url", "")
        title = meta.get("title", "")
        if not url or not is_syllabus_doc(url, title):
            continue
        entry = grouped.get(url)
        if entry is None:
            raw_name = f"{_filename(url)} {title}"
            entry = grouped[url] = {
                "url": url,
                "chunks": [],
                "depts": _doc_depts(_normalise(url, title)),
                "yos": _parse_yos(_normalise(url, title)),
                "ay": _parse_ay(raw_name),
                "level": _doc_level(raw_name),
                "named": bool(_SYLLABUS_WORD_RE.search(_filename(url))),
            }
        entry["chunks"].append(
            {
                "text": text,
                "source_url": url,
                "doc_type": meta.get("doc_type", "pdf"),
                "title": title,
                "category": meta.get("category", ""),
                "chunk_index": meta.get("chunk_index", 0),
            }
        )

    _SYLLABUS_DOCS_CACHE = grouped
    logger.info("Syllabus-doc index: %d course-content documents", len(grouped))
    return grouped


def reset_syllabus_index_cache() -> None:
    """Drop the cached index (tests, and after a KB refresh)."""
    global _SYLLABUS_DOCS_CACHE
    _SYLLABUS_DOCS_CACHE = None


def _level_ok(doc: dict, want: str) -> bool:
    if want in ("mtech", "mba"):
        return doc["level"] == want
    return doc["level"] == "btech"


def _select_doc(docs: List[dict], cons: dict) -> Optional[dict]:
    """
    Best document for the query constraints, via progressively relaxed
    filters: (branch + year + AY) → (branch + year) → (branch + AY) →
    (branch).  Year of study outranks academic year — a student in T.Y. wants
    the T.Y. syllabus even if the published AY differs from the one they
    typed.  Ties prefer files literally named "…Syllabus…" over terse
    course-content exports, then newest AY, then URL, so the choice is
    deterministic.
    """
    def has_dept(d: dict) -> bool:
        return bool(cons["dept"][0] in d["depts"] and _level_ok(d, cons["level"]))

    base = [d for d in docs if has_dept(d)]
    tiers: List[List[dict]] = []
    if cons["yos"] is not None and cons["ay"] is not None:
        tiers.append(
            [d for d in base if d["yos"] == cons["yos"] and d["ay"] == cons["ay"]]
        )
        tiers.append([d for d in base if d["yos"] == cons["yos"]])
        tiers.append([d for d in base if d["ay"] == cons["ay"]])
    elif cons["yos"] is not None:
        tiers.append([d for d in base if d["yos"] == cons["yos"]])
    elif cons["ay"] is not None:
        tiers.append([d for d in base if d["ay"] == cons["ay"]])
    tiers.append(base)

    for tier in tiers:
        if tier:
            def rank(d: dict):
                ay = d["ay"] or (-1, -1)
                return (
                    0 if (cons["ay"] and d["ay"] == cons["ay"]) else 1,
                    0 if (cons["yos"] and d["yos"] == cons["yos"]) else 1,
                    0 if d.get("named") else 1,
                    -ay[0],
                    -ay[1],
                    d["url"],
                )
            return sorted(tier, key=rank)[0]
    return None


def find_syllabus_chunks(
    collection,
    query: str,
    max_chunks: int = _MAX_SYLLABUS_CHUNKS,
) -> List[dict]:
    """
    Return up to *max_chunks* chunks from the syllabus document that matches
    the query's branch / year-of-study / academic-year constraints.

    *collection* is a Chroma collection (injected so this module stays free of
    embedding/network concerns and is trivially unit-testable with a stub).

    Returns chunk dicts shaped like the retriever's output (text, source_url,
    doc_type, title, category, score) with score set to 1.0 — these are exact
    lexical hits on the document the query names, which outrank any embedding
    similarity for this query shape.  Empty list when the query is not a
    syllabus request, names no branch, or no branch syllabus exists in the KB.
    """
    cons = parse_syllabus_query(query)
    if not cons:
        return []

    index = _syllabus_doc_index(collection)
    if not index:
        return []

    best = _select_doc(list(index.values()), cons)
    if best is None:
        return []

    doc_chunks = sorted(
        best["chunks"], key=lambda c: c.get("chunk_index", 0)
    )
    lead = [
        c for c in doc_chunks
        if len(c["text"]) >= _STUB_MAX_CHARS and _COURSE_HEADER_RE.search(c["text"])
    ]
    body = [
        c for c in doc_chunks
        if len(c["text"]) >= _STUB_MAX_CHARS and c not in lead
    ]
    ordered = lead + body + [c for c in doc_chunks if c not in lead and c not in body]

    out: List[dict] = []
    for c in ordered[:max_chunks]:
        out.append(
            {
                "text": c["text"],
                "source_url": c["source_url"],
                "doc_type": c["doc_type"],
                "title": c["title"],
                "category": c["category"],
                "score": 1.0,
            }
        )
    logger.info(
        "Syllabus-doc targeting: %d chunk(s) from %s",
        len(out),
        best["url"].rsplit("/", 1)[-1],
    )
    return out
