"""
tests/test_eval_set.py
──────────────────────
Schema/consistency checks for the labelled evaluation set
(tests/eval/queries.jsonl).  Keeps the eval data honest without needing the
live Chroma DB (the retrieval half runs via scripts/eval_accuracy.py).
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.categories import PREDICTABLE_CATEGORIES, Category  # noqa: E402

EVAL_PATH = os.path.join(os.path.dirname(__file__), "eval", "queries.jsonl")


def _load():
    with open(EVAL_PATH, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


@pytest.fixture(scope="module")
def items():
    return _load()


class TestEvalSet:
    def test_size(self, items):
        assert len(items) >= 60, "eval set should cover all categories"

    def test_ids_unique(self, items):
        ids = [i["id"] for i in items]
        assert len(ids) == len(set(ids))

    def test_known_types_and_categories(self, items):
        predictable = {c.value for c in PREDICTABLE_CATEGORIES} | {Category.GENERAL.value}
        for item in items:
            assert item["type"] in ("answer", "refuse")
            assert item["category"] in predictable, item
            assert item["query"].strip()

    def test_answer_items_have_expected_sources(self, items):
        for item in items:
            if item["type"] == "answer":
                assert item["sources"], f"{item['id']} must list acceptable sources"
                assert item["category"] != Category.GENERAL.value
            else:
                assert item["category"] == Category.GENERAL.value
                assert item["sources"] == []

    def test_every_core_category_covered(self, items):
        counts = Counter(i["category"] for i in items if i["type"] == "answer")
        for category in PREDICTABLE_CATEGORIES:
            assert counts[category.value] >= 5, (
                f"category {category.value} has only {counts[category.value]} queries"
            )

    def test_refusals_present(self, items):
        assert sum(1 for i in items if i["type"] == "refuse") >= 8
