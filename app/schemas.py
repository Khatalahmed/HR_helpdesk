from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, description="Employee question")
    top_k: Optional[int] = Field(None, ge=1, le=10, description="Optional retrieval override")
    include_sources: bool = Field(True, description="Include source chunks in response")


class SourceChunk(BaseModel):
    index: int
    filename: str
    heading: str
    chunk_id: str
    score: float
    preview: str


class AskResponse(BaseModel):
    answer: str
    route: Optional[str]
    timings: dict[str, float]
    sources: list[SourceChunk]


class RetrieveDebugRequest(BaseModel):
    question: str = Field(..., min_length=3)
    top_k: Optional[int] = Field(None, ge=1, le=10)


class RetrieveDebugResponse(BaseModel):
    route: Optional[str]
    sources: list[SourceChunk]


class FeedbackRequest(BaseModel):
    question: str = Field(..., min_length=3)
    answer: str = Field(..., min_length=1)
    rating: int = Field(..., ge=1, le=5, description="1=bad, 5=great")
    notes: str = Field("", description="Optional feedback")


class HealthResponse(BaseModel):
    status: str
    resources_loaded: bool
    generation_model: str
    embedding_model: str
    collection_name: str

