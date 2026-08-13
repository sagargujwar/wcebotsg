"""
tests/conftest.py
─────────────────
Shared pytest fixtures used across the test suite.
"""
from __future__ import annotations

import os
import sys

import pytest

# Ensure the project root is on sys.path when tests are run from any directory.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")


@pytest.fixture
def fixture_dir() -> str:
    return FIXTURES_DIR


@pytest.fixture
def index_html_path(fixture_dir) -> str:
    return os.path.join(fixture_dir, "index.html")


@pytest.fixture
def syllabus_html_path(fixture_dir) -> str:
    return os.path.join(fixture_dir, "syllabus.html")


@pytest.fixture
def fee_pdf_path(fixture_dir) -> str:
    return os.path.join(fixture_dir, "fee_structure.pdf")


@pytest.fixture
def index_html_content(index_html_path) -> str:
    with open(index_html_path, encoding="utf-8") as f:
        return f.read()


@pytest.fixture
def syllabus_html_content(syllabus_html_path) -> str:
    with open(syllabus_html_path, encoding="utf-8") as f:
        return f.read()
