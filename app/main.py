from __future__ import annotations

import json
import logging
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, Security
from fastapi.security import APIKeyHeader

from app.config import get_settings
from app.logging_config import setup_logging
from app.rag.service import RAGService
from app.schemas import (
    AskRequest,
    AskResponse,
    FeedbackRequest,
    HealthResponse,
    RetrieveDebugRequest,
    RetrieveDebugResponse,
)
from app.security import InMemoryRateLimiter, enforce_api_key, enforce_rate_limit


settings = get_settings()
setup_logging(settings)
rag_service = RAGService(settings=settings)
rate_limiter = InMemoryRateLimiter(settings.rate_limit_per_minute)
http_logger = logging.getLogger("snailcloud.http")
http_logger.setLevel(logging.INFO)

app = FastAPI(
    title="SnailCloud HR Helpdesk API",
    version="1.0.0",
    description="Backend API for SnailCloud HR policy assistant (RAG + Gemini + pgvector).",
    docs_url=None if settings.is_production else "/docs",
    redoc_url=None if settings.is_production else "/redoc",
    openapi_url=None if settings.is_production else "/openapi.json",
)

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def _protect(request: Request, provided_api_key: str | None) -> None:
    enforce_api_key(provided_api_key, settings)
    enforce_rate_limit(request, rate_limiter, provided_api_key)


def _log_http_request(
    request: Request,
    request_id: str,
    status_code: int,
    duration_ms: float,
    error: str | None = None,
) -> None:
    payload = {
        "event": "http_request",
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "request_id": request_id,
        "method": request.method,
        "path": request.url.path,
        "status_code": status_code,
        "duration_ms": round(duration_ms, 2),
        "client_ip": request.client.host if request.client else "unknown",
    }
    if error:
        payload["error"] = error

    http_logger.info(json.dumps(payload, separators=(",", ":")))


@app.middleware("http")
async def request_observability_middleware(request: Request, call_next):
    request_id = request.headers.get("x-request-id", "").strip() or uuid.uuid4().hex
    request.state.request_id = request_id

    started = time.perf_counter()
    response = None
    error_message = None
    try:
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response
    except Exception as exc:
        error_message = str(exc)
        raise
    finally:
        elapsed_ms = (time.perf_counter() - started) * 1000
        status_code = response.status_code if response is not None else 500
        _log_http_request(
            request=request,
            request_id=request_id,
            status_code=status_code,
            duration_ms=elapsed_ms,
            error=error_message,
        )


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        resources_loaded=rag_service.is_ready,
        generation_model=settings.generation_model,
        embedding_model=settings.embedding_model,
        collection_name=settings.collection_name,
    )


@app.post("/ask", response_model=AskResponse)
def ask(
    payload: AskRequest,
    request: Request,
    api_key: str | None = Security(api_key_header),
) -> AskResponse:
    _protect(request, api_key)
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        result = rag_service.ask(
            question=question,
            include_sources=payload.include_sources,
            top_k_override=payload.top_k,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"RAG processing failed: {exc}") from exc

    return AskResponse(**result)


@app.post("/retrieve-debug", response_model=RetrieveDebugResponse)
def retrieve_debug(
    payload: RetrieveDebugRequest,
    request: Request,
    api_key: str | None = Security(api_key_header),
) -> RetrieveDebugResponse:
    _protect(request, api_key)
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        result = rag_service.retrieve_debug(question=question, top_k_override=payload.top_k)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {exc}") from exc

    return RetrieveDebugResponse(route=result["route"], sources=result["sources"])


@app.post("/feedback")
def feedback(
    payload: FeedbackRequest,
    request: Request,
    api_key: str | None = Security(api_key_header),
) -> dict:
    """
    Beginner-friendly feedback sink.
    In production, move this to a DB table or logging service.
    """
    _protect(request, api_key)
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    feedback_file = log_dir / "feedback.log"

    timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    line = (
        f"{timestamp} | rating={payload.rating} | "
        f"question={payload.question!r} | notes={payload.notes!r}\n"
    )
    with open(feedback_file, "a", encoding="utf-8") as file:
        file.write(line)

    return {"status": "saved"}
