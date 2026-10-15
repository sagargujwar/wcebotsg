"""
app/models.py
─────────────
Pydantic request/response schemas for all API endpoints.
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field, field_validator

from app.config import get_settings


class ChatRequest(BaseModel):
    query: str = Field(..., description="The student's question")

    @field_validator("query")
    @classmethod
    def _validate_query(cls, v: str) -> str:
        v = v.strip()
        settings = get_settings()
        if not v:
            raise ValueError("query must not be empty")
        if len(v) > settings.max_query_length:
            raise ValueError(
                f"query exceeds maximum length of {settings.max_query_length} characters"
            )
        return v


class ChatResponse(BaseModel):
    answer: str
    source_url: Optional[str] = None
    cached: bool = False
    # Taxonomy category the query was routed to ("admissions_fees", …).
    # None when the classifier fell back to the baseline/general path.
    category: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    chroma_ok: bool
    redis_ok: bool


class ErrorResponse(BaseModel):
    detail: str
