"""
app/categories.py
─────────────────
The query taxonomy — single source of truth for category-aware retrieval.

Every incoming query is assigned to one of a small set of student-facing
categories (admissions & fees, academics, placements, people/administration,
infrastructure, notices, research, general info).  Each category carries:

  • a weighted keyword/regex lexicon   — deterministic rule scoring at query time
  • seed queries                       — prototypes for embedding-based scoring
  • URL prefixes                       — deterministic tagging of KB chunks
  • a similarity-threshold override    — the retrieval gate, tuned per category
  • a retrieval boost                  — score bonus for matching-category chunks
  • an answer rubric                   — prompt addendum with category guardrails

`Category.GENERAL` is the low-confidence fallback: it disables every
category-specific behaviour, i.e. it reproduces the original baseline pipeline.
`Category.OUT_OF_SCOPE` is never predicted by the classifier — it exists so the
taxonomy documents the refusal path (handled by the similarity gate and the
grounded prompt).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse


class Category(str, Enum):
    """Query/KB categories. Values are stored verbatim in Chroma metadata."""

    ADMISSIONS_FEES = "admissions_fees"
    ACADEMICS = "academics"
    PLACEMENTS_CAREERS = "placements_careers"
    PEOPLE_ADMIN = "people_admin"
    INFRA_CAMPUS = "infra_campus"
    NOTICES_EVENTS = "notices_events"
    RESEARCH = "research"
    GENERAL_INFO = "general_info"
    GENERAL = "general"            # low-confidence fallback → baseline behaviour
    OUT_OF_SCOPE = "out_of_scope"  # refusal path (never classifier-predicted)


# Categories the classifier may actually predict (GENERAL is the fallback).
PREDICTABLE_CATEGORIES: Tuple[Category, ...] = (
    Category.ADMISSIONS_FEES,
    Category.ACADEMICS,
    Category.PLACEMENTS_CAREERS,
    Category.PEOPLE_ADMIN,
    Category.INFRA_CAMPUS,
    Category.NOTICES_EVENTS,
    Category.RESEARCH,
    Category.GENERAL_INFO,
)

UI_LABELS: Dict[Category, str] = {
    Category.ADMISSIONS_FEES: "Admissions & Fees",
    Category.ACADEMICS: "Academics",
    Category.PLACEMENTS_CAREERS: "Placements & Careers",
    Category.PEOPLE_ADMIN: "People & Administration",
    Category.INFRA_CAMPUS: "Infrastructure & Campus",
    Category.NOTICES_EVENTS: "Notices & Events",
    Category.RESEARCH: "Research",
    Category.GENERAL_INFO: "About WCE",
    Category.GENERAL: "",
    Category.OUT_OF_SCOPE: "",
}


@dataclass(frozen=True)
class CategorySpec:
    """Everything the pipeline needs to know about one category."""

    id: Category
    keywords: Tuple[str, ...] = ()       # word/phrase hits, weight 1.0 each
    patterns: Tuple[str, ...] = ()       # regex hits, weight 1.5 each
    seed_queries: Tuple[str, ...] = ()   # embedding prototypes
    url_prefixes: Tuple[str, ...] = ()   # deterministic KB-chunk tagging
    threshold: Optional[float] = None    # None → use global SIMILARITY_THRESHOLD
    boost: float = 0.06                  # score bonus for matching-category chunks
    rubric: str = ""                     # prompt addendum for this category


# ── The taxonomy ──────────────────────────────────────────────────────────────
# Priority from the project-scope point of view: 1–6 are what students actually
# ask a college bot about; 7–8 are secondary.

CATEGORY_SPECS: Dict[Category, CategorySpec] = {
    Category.ADMISSIONS_FEES: CategorySpec(
        id=Category.ADMISSIONS_FEES,
        keywords=(
            "fee", "fees", "fee structure", "tuition", "admission", "admissions",
            "admit", "apply", "cap", "cet", "jee", "cutoff", "cut-off",
            "eligibility", "eligible", "counselling", "counseling", "scholarship",
            "documents required", "provisional", "merit", "quota", "nri", "pio",
            "oci", "ciwg", "intake", "round", "seat", "dte", "fra",
            "second year",
        ),
        patterns=(
            r"₹\s*[\d,]+",
            r"\brs\.?\s*[\d,]+",
            r"\bcap\s*round\b",
            r"\bmht\s*[- ]?\s*cet\b",
            r"\b\d{2,3}\s*(?:%|percentile)\b",
        ),
        seed_queries=(
            "What is the fee structure for B.E. first year?",
            "Which documents are required for CAP admission?",
            "What is the MHT-CET cutoff for Computer Engineering?",
            "How do I apply for direct second year B.Tech?",
            "Is hostel fee separate from tuition fee?",
            "What are the eligibility criteria for M.Tech admission?",
            "Tell me the admission process through CAP round",
            "Are NRI students eligible for admission?",
            "What scholarship options are available for SC/ST students?",
            "How much is the tuition fee for Open category?",
            "When is the last date to pay the fees?",
            "Give me the fee structure document",
        ),
        url_prefixes=(
            "/admissions/",
            "/mtechadmissions2026/",
            "/master-of-business-administration",
            "/bachelor-of-design",
            "/integrated-b-tech-m-tech",
            "/diploma/",
            "/post-graduate/",
            "/btech-dse-working-professionals",
            "/certificate-courses",
            "/campus-life/student-scholarships",
        ),
        threshold=None,
        boost=0.06,
        rubric=(
            "Answer with exact figures, dates and eligibility criteria ONLY as "
            "printed in the context. Prefer bullets or a compact table for fee "
            "structures and document lists; never round, extend or extrapolate "
            "an amount or a date."
        ),
    ),
    Category.ACADEMICS: CategorySpec(
        id=Category.ACADEMICS,
        keywords=(
            "syllabus", "semester", "subject", "subjects", "course", "courses",
            "curriculum", "regulation", "regulations", "credit", "credits",
            "exam", "exams", "examination", "timetable", "time table", "calendar",
            "lecture", "lectures", "class", "classes", "department", "academic",
            "academics", "attendance", "result", "results", "theory", "practical",
            "programme", "programmes", "program", "degree", "diploma",
            "minor", "scheme",
        ),
        patterns=(
            r"\bsem(?:ester)?[- ]?\s*\d\b",
            r"\byear[- ]?\d\b",
            r"\bb\.?tech\b",
            r"\bb\.?e\.?\b",
            r"\bm\.?tech\b",
        ),
        seed_queries=(
            "What is the syllabus of Computer Engineering semester 5?",
            "List the subjects in second year mechanical engineering",
            "What are the academic regulations for the credit system?",
            "When is the examination schedule released?",
            "How many credits does a semester carry?",
            "Which courses does the IT department offer?",
            "Tell me the academic calendar for this year",
            "What is the attendance requirement per subject?",
            "Which semester covers Data Structures and Algorithms?",
            "Is there a polytechnic diploma programme?",
            "What are the course outcomes for this subject?",
            "Tell me about the multi-disciplinary minor programmes",
        ),
        url_prefixes=(
            "/academics-programmes/",
            "/department-of-",
            "/overview-of-programmes",
            "/polytechnic-academics/",
            "/nba-dcs/",
            "/circle/",
        ),
        threshold=None,
        boost=0.06,
        rubric=(
            "State programme, subject and semester facts precisely as given. "
            "List subjects, credits or regulations as bullets, and name the "
            "regulation document when the context identifies one."
        ),
    ),
    Category.PLACEMENTS_CAREERS: CategorySpec(
        id=Category.PLACEMENTS_CAREERS,
        keywords=(
            "placement", "placements", "placed", "package", "recruiter",
            "recruiters", "internship", "internships", "career", "careers",
            "cdc", "job", "jobs", "higher education", "training", "entrepreneur",
            "entrepreneurship", "incubation", "alumni", "campus drive", "drive",
            "selection", "company", "companies", "industry",
        ),
        patterns=(
            r"\b(?:lpa|lakh per annum|ctc)\b",
            r"\bplacement[s]?\s*(?:stat|report|data)",
        ),
        seed_queries=(
            "What are the placement statistics for last year?",
            "Which companies visit the campus for recruitment?",
            "Does WCE provide internships?",
            "What is the average placement package?",
            "How do I register with the Career Development Cell?",
            "What higher education options do students go for?",
            "Are there training programmes for placements?",
            "Tell me about the entrepreneurship cell",
            "Do alumni networks help with placements?",
            "Which students got placed in the latest campus drive?",
            "Are there internships in the summer?",
            "Which recruiters visited recently?",
        ),
        url_prefixes=("/student-careers/",),
        threshold=0.33,
        boost=0.06,
        rubric=(
            "Report placement, package and internship information only when the "
            "context states it, and include the year or batch it refers to. "
            "Never invent a recruiter name or a package figure."
        ),
    ),
    Category.PEOPLE_ADMIN: CategorySpec(
        id=Category.PEOPLE_ADMIN,
        keywords=(
            "director", "principal", "hod", "head", "head of department", "dean",
            "professor", "faculty", "staff", "committee", "council", "contact",
            "phone", "email", "office", "address", "recruitment", "vacancy",
            "manager", "trust", "society", "organogram", "deputy", "coordinator",
            "chairman", "governing", "administrative", "designation", "cell",
        ),
        patterns=(
            r"\b[\w.+-]+@[\w-]+\.[\w.]+",
            # Grouped phone numbers only (… 5-5, 3-4-4 …) — a bare digit run
            # like a date "2026-3-10" must NOT read as a phone number.
            r"\+?\d{1,3}[-\s]\d{4,5}[-\s]\d{4,5}|\+\d{10,13}\b",
        ),
        seed_queries=(
            "Who is the Director of WCE?",
            "What is the contact number of the principal's office?",
            "List the heads of departments",
            "Who is the Head of the Computer Engineering department?",
            "How many members are on the Academic Council?",
            "What is the email address for admission queries?",
            "Is there a faculty recruitment notification?",
            "Who manages the college trust?",
            "Tell me about the college committees",
            "What is the designation of the Deputy Director?",
            "Where is the college office located?",
            "Who is the IQAC coordinator?",
        ),
        url_prefixes=(
            "/about/",
            "/communication/",   # contact directory — people, not "about"
            "/notice_board/service-rules",
            "/notice_board/apply-for-faculty",
            "/notice_board/finance-budget",
        ),
        threshold=0.31,
        boost=0.06,
        rubric=(
            "Give names together with their exact designation, and include "
            "contact details only when they appear in the context. Do not guess "
            "a role, a phone number or an email address."
        ),
    ),
    Category.INFRA_CAMPUS: CategorySpec(
        id=Category.INFRA_CAMPUS,
        keywords=(
            "hostel", "hostels", "library", "librarian", "laboratory",
            "laboratories", "lab", "labs", "wifi", "wi-fi", "internet",
            "campus", "building", "buildings", "auditorium", "sports", "gym",
            "canteen", "playground", "reading room", "infrastructure",
            "facility", "facilities", "club", "clubs", "hall", "mess",
            "computing", "network", "stationery", "books", "journals",
        ),
        patterns=(
            r"\btilak\s*hall\b",
            r"\bcomputer\s*centre\b|\bcomputer\s*center\b",
        ),
        seed_queries=(
            "How many students can stay in the hostel?",
            "Is Wi-Fi available on campus?",
            "What facilities does the library provide?",
            "Tell me about the sports facilities",
            "Are there labs in the computer department?",
            "What are the reading room timings?",
            "Is there a canteen on campus?",
            "Describe the academic buildings",
            "What clubs can students join?",
            "Is there a bus facility for students?",
            "How big is the college campus?",
            "What books and journals are available in the library?",
        ),
        url_prefixes=("/campus-life/",),
        threshold=None,
        boost=0.06,
        rubric=(
            "Describe facilities exactly as documented (hostel, library, labs, "
            "connectivity, sports). Mention location, capacity or charges only "
            "when the context provides them."
        ),
    ),
    Category.NOTICES_EVENTS: CategorySpec(
        id=Category.NOTICES_EVENTS,
        keywords=(
            "notice", "notices", "announcement", "announcements", "event",
            "events", "workshop", "fdp", "seminar", "webinar", "circular",
            "circulars", "update", "updates", "news", "last date", "deadline",
            "happening", "notification", "schedule", "upcoming",
        ),
        patterns=(
            r"\blast\s*date\b",
            r"\bfrom\s+\d{1,2}[-–]\d{1,2}\s+\w+\b",
            r"\bannounce\w*\b",
            r"\bany\s+notic\w*\b",
        ),
        seed_queries=(
            "What are the latest notices?",
            "Any new announcements from the college?",
            "When is the last date for the current notice?",
            "List recent events and workshops",
            "Is there any new notice about admissions?",
            "Which FDPs are being conducted?",
            "Show me the notice board updates",
            "Any news about the college recently?",
            "What is the latest circular for students?",
            "Are there any upcoming seminars?",
            "What is happening on campus this week?",
            "Tell me about the recent notification",
        ),
        url_prefixes=(
            "/notices/",
            "/notice_board/",
            "/news-archive/",
            "/blogs/",
        ),
        threshold=0.30,
        boost=0.06,
        rubric=(
            "Prefer the most recent notice in the context and give its date and "
            "title verbatim. When several notices apply, list them newest first."
        ),
    ),
    Category.RESEARCH: CategorySpec(
        id=Category.RESEARCH,
        keywords=(
            "research", "phd", "ph.d", "publication", "publications", "paper",
            "papers", "patent", "patents", "citation",
            "mou", "mous", "sponsored", "consultancy", "funded", "project",
            "projects", "scopus", "index", "scientometric",
        ),
        patterns=(
            r"\bph\.?\s*d\b",
            r"\br&d\b",
            r"\b(?:h.index|h-index)\b",
            r"\bpublish\w*\b",
        ),
        seed_queries=(
            "What research projects are funded by DST?",
            "How many patents has the college filed?",
            "Tell me about the R&D cell",
            "Which journals do faculty publish in?",
            "Is there a PhD programme at WCE?",
            "List the MoUs with industries",
            "What sponsored research projects are ongoing?",
            "How do I apply for PhD admission?",
            "Tell me about the research centres and labs",
            "Are student publications encouraged?",
            "What grants did the department receive?",
            "Which faculty are listed in the AD scientific index?",
        ),
        url_prefixes=(
            "/research-consultancy/",
            "/admissions/phd/",
        ),
        threshold=0.33,
        boost=0.06,
        rubric=(
            "Summarise research output and facilities only from the context; "
            "quote project, patent or MoU titles exactly as written."
        ),
    ),
    Category.GENERAL_INFO: CategorySpec(
        id=Category.GENERAL_INFO,
        keywords=(
            # NB: "about" is deliberately absent — it is a function word
            # ("tell me about X") that routed queries away from their real
            # category in the eval set.
            "vision", "mission", "history", "nirf", "naac", "nba",
            "accreditation", "accredited", "ranking", "rank", "gallery",
            "social", "instagram", "facebook", "linkedin", "youtube", "website",
            "established", "glance", "rti", "disclosure", "disclosures",
            "approval", "approvals", "ugc", "aicte", "annual report",
            "video", "videos", "trust",
        ),
        patterns=(
            r"\bat\s*a\s*glance\b",
            r"\bestablished\s+in\b",
        ),
        seed_queries=(
            "What is the vision and mission of WCE?",
            "When was the college established?",
            "What is WCE's NIRF ranking?",
            "Is the college NAAC accredited?",
            "What is the official Instagram handle of WCE?",
            "Give me the college address",
            "Tell me about Walchand Trust",
            "What approvals does the college have from UGC?",
            "Where can I see the annual report?",
            "Tell me about WCE at a glance",
            "Latest videos from the college YouTube channel",
            "What does the college stand for?",
        ),
        url_prefixes=(
            # Specific /about/ pages are institutional facts, not people queries.
            "/about/wce-at-glance/",
            "/about/vision-mission/",
            "/about/nirf-nba-and-naac/",
            "/about/mandatory-disclosures/",
            "/about/annual-reports/",
            "/about/iqac-reports/",
            "/about/academic-audits/",
            "/about/nirf-submissions/",
            "/about/rti/",
            "/about/ugc-aicte-and-shivaji-university/",
            "/about/walchand-trust/",
            "/about/mte-society/",
            "/online-platforms/",
            "/360vr/",
        ),
        threshold=0.33,
        boost=0.06,
        rubric=(
            "Answer institutional facts (vision, accreditation, approvals, "
            "social links) from the context and cite the page title."
        ),
    ),
    # ── Non-content categories ────────────────────────────────────────────────
    Category.GENERAL: CategorySpec(id=Category.GENERAL),
    Category.OUT_OF_SCOPE: CategorySpec(id=Category.OUT_OF_SCOPE),
}


# ── URL → category (deterministic KB-chunk tagging) ───────────────────────────
# Longest prefix wins, so "/admissions/phd/" beats "/admissions/" and
# "/campus-life/student-scholarships" beats "/campus-life/".
_URL_RULES: Tuple[Tuple[str, Category], ...] = tuple(
    sorted(
        (
            (prefix, spec.id)
            for spec in CATEGORY_SPECS.values()
            for prefix in spec.url_prefixes
        ),
        key=lambda rule: -len(rule[0]),
    )
)


def get_spec(category: Category) -> CategorySpec:
    """Return the spec for *category* (unknown values fall back to GENERAL)."""
    return CATEGORY_SPECS.get(category, CATEGORY_SPECS[Category.GENERAL])


def threshold_for(category: Category, default: float) -> float:
    """Per-category similarity gate; falls back to the global threshold."""
    override = get_spec(category).threshold
    return default if override is None else override


def boost_for(category: Category) -> float:
    """Score bonus applied to chunks tagged with *category* (0 for GENERAL)."""
    if category in (Category.GENERAL, Category.OUT_OF_SCOPE, None):
        return 0.0
    return get_spec(category).boost


def label_for(category: Category) -> str:
    """Human-readable label (empty string for GENERAL / OUT_OF_SCOPE)."""
    return UI_LABELS.get(category, "")


def _keyword_regex(keyword: str) -> "re.Pattern[str]":
    """Word-boundary regex for a keyword; plural 's' matches singular too."""
    body = re.escape(keyword)
    if keyword.isalpha():
        return re.compile(rf"(?<!\w){body}s?(?!\w)", re.IGNORECASE)
    return re.compile(rf"(?<!\w){body}(?!\w)", re.IGNORECASE)


_KEYWORD_RES: Dict[Category, Tuple["re.Pattern[str]", ...]] = {
    spec.id: tuple(_keyword_regex(k) for k in spec.keywords)
    for spec in CATEGORY_SPECS.values()
    if spec.keywords
}
_PATTERN_RES: Dict[Category, Tuple["re.Pattern[str]", ...]] = {
    spec.id: tuple(re.compile(p, re.IGNORECASE) for p in spec.patterns)
    for spec in CATEGORY_SPECS.values()
    if spec.patterns
}

# Weights: a keyword match = 1.0, a (richer) regex match = 1.5.
_PATTERN_WEIGHT = 1.5
# raw >= _HEURISTIC_MIN points and a unique best → confident enough to tag a
# chunk whose URL matched no prefix.
_HEURISTIC_MIN = 1.0


def rule_hits(text: str) -> Dict[Category, float]:
    """Raw (un-normalised) rule hits of *text* against every category lexicon."""
    scores: Dict[Category, float] = {}
    for category, patterns in _KEYWORD_RES.items():
        hits = sum(1.0 for rx in patterns if rx.search(text))
        hits += _PATTERN_WEIGHT * sum(
            1.0 for rx in _PATTERN_RES.get(category, ()) if rx.search(text)
        )
        if hits:
            scores[category] = hits
    return scores


def rule_scores(text: str) -> Dict[Category, float]:
    """Lexicon scores normalised to [0, 1] (2 raw points = 1.0)."""
    return {c: min(1.0, raw / 2.0) for c, raw in rule_hits(text).items()}


def url_to_category(url: str, title: str = "") -> Category:
    """
    Deterministically tag a knowledge-base chunk.

    1. Longest matching URL prefix → its category.
    2. Otherwise: lexicon heuristic over the title + URL path (PDFs live under
       `/wp-content/uploads/…` with opaque names, so their titles carry the
       signal — e.g. "Fee Structure", "Recruitment Advertisement").
    3. Otherwise GENERAL (no category-specific behaviour at query time).
    """
    path = urlparse(url).path.lower() if "://" in url or url.startswith("/") else url.lower()
    for prefix, category in _URL_RULES:
        if path.startswith(prefix):
            return category

    probe = f"{title} {path}"
    hits = rule_hits(probe)
    if hits:
        best = max(hits.values())
        ordered = sorted(hits.values(), reverse=True)
        if best >= _HEURISTIC_MIN and (len(ordered) == 1 or ordered[0] > ordered[1]):
            return max(hits, key=lambda c: hits[c])
    return Category.GENERAL


# ── Query rewriting (abbreviation expansion) ──────────────────────────────────
# Deterministic, zero-cost rewrite used ONLY for the embedding query — the
# student's original wording is what the LLM and the UI still see.
ABBREVIATIONS: Dict[str, str] = {
    "hod": "head of department",
    "cdc": "career development cell",
    "cap": "centralised admission process",
    "cet": "common entrance test",
    "dte": "directorate of technical education",
    "nirf": "national institutional ranking framework",
    "naac": "national assessment and accreditation council",
    "nba": "national board of accreditation",
    "iqac": "internal quality assurance cell",
    "rti": "right to information",
    "mou": "memorandum of understanding",
    "fdp": "faculty development programme",
    "ug": "under graduate",
    "pg": "post graduate",
    "rd": "research and development",
    "cse": "computer science and engineering",
    "aiml": "artificial intelligence and machine learning",
    "mech": "mechanical engineering",
    "entc": "electronics and telecommunication",
    "ty": "third year",
    "sy": "second year",
    "fy": "first year",
}
_ABBREV_RES: Dict[str, "re.Pattern[str]"] = {
    abbr: re.compile(rf"(?<!\w){re.escape(abbr)}(?!\w)", re.IGNORECASE)
    for abbr in ABBREVIATIONS
    if ABBREVIATIONS[abbr]
}


def expand_query(query: str, max_expansions: int = 3) -> str:
    """
    Append full forms of abbreviations found in *query* so the embedding model
    matches pages that spell them out ("HOD" → also "head of department").

    Deterministic and bounded: at most *max_expansions* expansions, deduped.
    """
    expansions: List[str] = []
    seen: set = set()
    for abbr, full in ABBREVIATIONS.items():
        if not full or full in seen:
            continue
        rx = _ABBREV_RES.get(abbr)
        if rx and rx.search(query):
            expansions.append(full)
            seen.add(full)
            if len(expansions) >= max_expansions:
                break
    if not expansions:
        return query
    return f"{query} ({'; '.join(expansions)})"
