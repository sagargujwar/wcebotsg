#!/usr/bin/env python
"""
Evaluate the query-taxonomy pipeline against the baseline, offline.

Two things are measured on tests/eval/queries.jsonl (labelled with the
expected category, an acceptable source page, and answer/refuse intent):

  1. CLASSIFICATION — hybrid classifier accuracy (overall + per category,
     with the misclassified queries listed).
  2. RETRIEVAL — two modes over the same live Chroma knowledge base:
       baseline : no category prior, global threshold 0.35
       aware    : category boost + per-category threshold
     Both modes run production's query rewrite (abbreviation expansion) and
     its context-meaning re-rank of the found chunks (inside search()), so the
     two rows differ only in category-awareness; compare two runs to measure
     a rewrite change.
     reporting coverage (passed the gate), hit@6 (an expected source page in
     the top-6), MRR, answered∧hit (the LLM would receive relevant context),
     and refusal correctness on out-of-scope queries.

No LLM calls, no Redis, no network — embedding + Chroma only.

Usage (from the repo root):
    python scripts/eval_accuracy.py                # print report
    python scripts/eval_accuracy.py --out docs/eval-report.md
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)  # settings use relative paths (./data/chroma_db)

from app.categories import (  # noqa: E402
    PREDICTABLE_CATEGORIES,
    Category,
    expand_query,
    threshold_for,
)
from app.config import get_settings  # noqa: E402
from app.intent import QueryClassifier  # noqa: E402
from app.retrieval import ChromaRetriever  # noqa: E402

logging.basicConfig(level=logging.WARNING)

# Windows consoles default to cp1252 — report glyphs (✓, ∧, …) must not crash it.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DEFAULT_JSONL = os.path.join("tests", "eval", "queries.jsonl")


# ── Data loading ──────────────────────────────────────────────────────────────

def load_queries(path: str) -> List[dict]:
    items = []
    with open(path, encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_no}: invalid JSON — {exc}") from exc
    return items


# ── Evaluation ────────────────────────────────────────────────────────────────

def classify_all(items: List[dict], classifier: QueryClassifier) -> List[dict]:
    rows = []
    for item in items:
        intent = classifier.classify(expand_query(item["query"]))
        rows.append({**item, "predicted": intent.category.value, "confidence": intent.confidence})
    return rows


def evaluate_retrieval(
    items: List[dict],
    retriever: ChromaRetriever,
    top_k: int,
    mode: str,
) -> List[dict]:
    """Run every query through one pipeline mode and score the outcome."""
    settings = get_settings()
    rows = []
    for item in items:
        # Mirror app/main.py step 4: the classifier and retrieval both see the
        # abbreviation-expanded query; the context-meaning re-rank happens
        # inside retriever.search() itself, so nothing extra to pass here.
        query = expand_query(item["query"])
        category: Optional[str] = None
        if mode == "aware" and item["predicted"] != Category.GENERAL.value:
            category = item["predicted"]

        chunks = retriever.search(query, top_k, category=category)
        best = max((c.get("score", 0.0) for c in chunks), default=0.0)

        if mode == "baseline":
            gate = settings.similarity_threshold
        else:
            gate = threshold_for(Category(item["predicted"]), settings.similarity_threshold)
        answered = bool(chunks) and best >= gate

        urls = [c.get("source_url", "") for c in chunks]
        rank: Optional[int] = None
        for i, url in enumerate(urls, 1):
            if any(sub.lower() in url.lower() for sub in item.get("sources", [])):
                rank = i
                break

        rows.append({**item, "mode": mode, "answered": answered, "best": best, "gate": gate, "rank": rank})
    return rows


def _fmt_pct(n: int, d: int) -> str:
    return f"{100.0 * n / d:5.1f}%" if d else "  n/a"


def build_report(items: List[dict], classified: List[dict], results: Dict[str, List[dict]]) -> str:
    lines: List[str] = []
    total = len(items)
    answer_items = [i for i in items if i["type"] == "answer"]
    refuse_items = [i for i in items if i["type"] == "refuse"]

    # ── Classification ────────────────────────────────────────────────────────
    correct = sum(1 for r in classified if r["predicted"] == r["category"])
    lines.append("# Query-taxonomy evaluation report")
    lines.append("")
    lines.append(f"Eval set: `{DEFAULT_JSONL}` — {total} queries "
                 f"({len(answer_items)} answerable, {len(refuse_items)} out-of-scope).")
    lines.append("")
    lines.append("## 1. Classification (hybrid rules + prototypes)")
    lines.append("")
    lines.append(f"**Accuracy: {_fmt_pct(correct, total).strip()} ({correct}/{total})**")
    lines.append("")

    per_cat: Dict[str, List[bool]] = defaultdict(list)
    for r in classified:
        per_cat[r["category"]].append(r["predicted"] == r["category"])
    lines.append("| expected category | n | accuracy |")
        # header separator appended below in loop order — build rows first
    rows = []
    for spec in list(PREDICTABLE_CATEGORIES) + [Category.GENERAL]:
        key = spec.value
        vals = per_cat.get(key, [])
        if vals:
            rows.append(f"| {key} | {len(vals)} | {_fmt_pct(sum(vals), len(vals)).strip()} |")
    lines.append("|---|---|---|")
    lines.extend(rows)
    lines.append("")

    mis = [r for r in classified if r["predicted"] != r["category"]]
    if mis:
        lines.append("Misclassified queries:")
        lines.append("")
        for r in mis:
            lines.append(f"- `{r['id']}` expected **{r['category']}** → got **{r['predicted']}** "
                         f"(conf {r['confidence']:.2f}): {r['query']!r}")
        lines.append("")
    else:
        lines.append("No misclassifications.")
        lines.append("")

    # ── Retrieval ─────────────────────────────────────────────────────────────
    lines.append("## 2. Retrieval — baseline vs category-aware")
    lines.append("")
    lines.append("| mode | answered | hit@6 | MRR | answered ∧ hit |")
    lines.append("|---|---|---|---|---|")
    for mode in ("baseline", "aware"):
        rows_m = [r for r in results[mode] if r["type"] == "answer"]
        n = len(rows_m)
        answered = sum(1 for r in rows_m if r["answered"])
        hits = sum(1 for r in rows_m if r["rank"] is not None)
        mrr = sum(1.0 / r["rank"] for r in rows_m if r["rank"]) / n if n else 0.0
        both = sum(1 for r in rows_m if r["answered"] and r["rank"] is not None)
        lines.append(
            f"| {mode} | {_fmt_pct(answered, n)} ({answered}/{n}) "
            f"| {_fmt_pct(hits, n)} ({hits}/{n}) "
            f"| {mrr:.3f} "
            f"| {_fmt_pct(both, n)} ({both}/{n}) |"
        )
    lines.append("")

    # ── Refusals ──────────────────────────────────────────────────────────────
    # The exact refusal text is produced by the grounded LLM (prompt rule 4)
    # and, at retrieval level, by the similarity gate.  Offline we can verify
    # that out-of-scope queries receive NO category prior (they must classify
    # as general → no boost, global threshold) and that the aware path leaves
    # their gate outcome identical to baseline.
    lines.append("## 3. Out-of-scope queries (no category prior applied)")
    lines.append("")
    routed = sum(1 for r in classified if r["type"] == "refuse"
                 and r["predicted"] != Category.GENERAL.value)
    lines.append(f"- Classifier declined to route: {len(refuse_items) - routed}/{len(refuse_items)}")
    if routed:
        routed_ids = ", ".join(
            r["id"] for r in classified
            if r["type"] == "refuse" and r["predicted"] != Category.GENERAL.value
        )
        lines.append(f"  - routed to a category anyway: {routed_ids}")
    lines.append("- The final \"I don't have information about this.\" reply comes from the")
    lines.append("  grounded LLM prompt (rule 4); this harness makes no LLM calls.")
    lines.append("")
    lines.append("| mode | cleared similarity gate (would reach the LLM) |")
    lines.append("|---|---|")
    for mode in ("baseline", "aware"):
        rows_m = [r for r in results[mode] if r["type"] == "refuse"]
        cleared = sum(1 for r in rows_m if r["answered"])
        lines.append(f"| {mode} | {cleared}/{len(rows_m)} |")
    lines.append("")

    # ── Deltas ────────────────────────────────────────────────────────────────
    b = {r["id"]: r for r in results["baseline"] if r["type"] == "answer"}
    a = {r["id"]: r for r in results["aware"] if r["type"] == "answer"}
    gained = [i for i in b if a[i]["rank"] is not None and b[i]["rank"] is None]
    lost = [i for i in b if b[i]["rank"] is not None and a[i]["rank"] is None]
    lines.append("## 4. Per-query effect of the category-aware path")
    lines.append("")
    lines.append(f"- Queries that **gained** a relevant top-6 hit: {len(gained)}"
                 + (f" — {', '.join(gained)}" if gained else ""))
    lines.append(f"- Queries that **lost** a relevant top-6 hit: {len(lost)}"
                 + (f" — {', '.join(lost)}" if lost else ""))
    lines.append("")
    lines.append("---")
    lines.append("*Generated by `scripts/eval_accuracy.py` (no LLM calls; "
                 "embedding + Chroma only).*")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline vs category-aware accuracy report")
    parser.add_argument("--jsonl", default=DEFAULT_JSONL, help="Labelled eval set (JSONL).")
    parser.add_argument("--out", default="", help="Also write the Markdown report to this path.")
    args = parser.parse_args()

    items = load_queries(args.jsonl)
    print(f"Loaded {len(items)} labelled queries from {args.jsonl}")

    retriever = ChromaRetriever()
    classifier = QueryClassifier(encoder=retriever.get_encoder())
    top_k = get_settings().top_k

    classified = classify_all(items, classifier)
    correct = sum(1 for r in classified if r["predicted"] == r["category"])
    print(f"Classification accuracy: {correct}/{len(items)} ({100.0 * correct / len(items):.1f}%)")

    results: Dict[str, List[dict]] = {}
    for mode in ("baseline", "aware"):
        results[mode] = evaluate_retrieval(classified, retriever, top_k, mode)
        rows = [r for r in results[mode] if r["type"] == "answer"]
        hits = sum(1 for r in rows if r["rank"] is not None)
        both = sum(1 for r in rows if r["answered"] and r["rank"] is not None)
        print(f"[{mode:8s}] answerable: hit@6 {hits}/{len(rows)}, "
              f"answered∧hit {both}/{len(rows)}")

    report = build_report(items, classified, results)
    print()
    print(report)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(report)
        print(f"Report written to {args.out}")


if __name__ == "__main__":
    main()
