from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient


def _load_main_module(
    monkeypatch: pytest.MonkeyPatch,
    app_api_key: str | None = None,
    rate_limit_per_minute: int | None = None,
    app_env: str = "development",
    log_dir: str | None = None,
):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    if app_api_key is None:
        app_api_key = ""
    if rate_limit_per_minute is None:
        rate_limit_per_minute = 1000
    monkeypatch.setenv("APP_API_KEY", app_api_key)
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", str(rate_limit_per_minute))
    monkeypatch.setenv("APP_ENV", app_env)
    monkeypatch.setenv("LOG_DIR", log_dir or "logs")
    monkeypatch.setenv("APP_LOG_FILE", "api.log")
    monkeypatch.setenv("HTTP_LOG_FILE", "http.jsonl")
    monkeypatch.setenv("LOG_MAX_BYTES", "5242880")
    monkeypatch.setenv("LOG_BACKUP_COUNT", "3")
    monkeypatch.setenv("LOG_LEVEL", "INFO")
    module_name = "app.main"
    if module_name in sys.modules:
        return importlib.reload(sys.modules[module_name])
    return importlib.import_module(module_name)


@pytest.fixture()
def main_module(monkeypatch: pytest.MonkeyPatch):
    return _load_main_module(monkeypatch)


@pytest.fixture()
def client(main_module):
    return TestClient(main_module.app)


def test_health_endpoint_returns_contract(client: TestClient):
    response = client.get("/health")
    payload = response.json()

    assert response.status_code == 200
    assert payload["status"] == "ok"
    assert "resources_loaded" in payload
    assert "generation_model" in payload
    assert "embedding_model" in payload
    assert "collection_name" in payload
    assert response.headers.get("x-request-id")


def test_request_id_is_echoed_when_provided(client: TestClient):
    response = client.get("/health", headers={"X-Request-ID": "rid-123"})
    assert response.status_code == 200
    assert response.headers.get("x-request-id") == "rid-123"


def test_health_emits_structured_log(client: TestClient, caplog: pytest.LogCaptureFixture):
    caplog.set_level("INFO", logger="snailcloud.http")
    response = client.get("/health", headers={"X-Request-ID": "rid-log-1"})
    assert response.status_code == 200

    records = [record for record in caplog.records if record.name == "snailcloud.http"]
    assert records
    log_payload = json.loads(records[-1].message)
    assert log_payload["event"] == "http_request"
    assert log_payload["request_id"] == "rid-log-1"
    assert log_payload["method"] == "GET"
    assert log_payload["path"] == "/health"
    assert log_payload["status_code"] == 200
    assert isinstance(log_payload["duration_ms"], (int, float))


def test_health_writes_structured_log_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    log_dir = tmp_path / "logs"
    main_module = _load_main_module(monkeypatch, log_dir=str(log_dir))
    client = TestClient(main_module.app)

    response = client.get("/health", headers={"X-Request-ID": "rid-file-1"})
    assert response.status_code == 200

    http_log_file = log_dir / "http.jsonl"
    assert http_log_file.exists()
    lines = [line.strip() for line in http_log_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert lines
    payload = json.loads(lines[-1])
    assert payload["event"] == "http_request"
    assert payload["request_id"] == "rid-file-1"
    assert payload["path"] == "/health"


def test_docs_are_hidden_in_production(monkeypatch: pytest.MonkeyPatch):
    main_module = _load_main_module(monkeypatch, app_env="production")
    client = TestClient(main_module.app)

    docs_response = client.get("/docs")
    openapi_response = client.get("/openapi.json")

    assert docs_response.status_code == 404
    assert openapi_response.status_code == 404


def test_ask_endpoint_success(main_module, client: TestClient, monkeypatch: pytest.MonkeyPatch):
    mock_response: dict[str, Any] = {
        "answer": "You get 18 annual leaves per year.",
        "route": "leave",
        "timings": {"retrieval": 0.101, "generation": 0.322, "total": 0.423},
        "sources": [
            {
                "index": 1,
                "filename": "Leave_Policy",
                "heading": "Annual Leave",
                "chunk_id": "leave-1",
                "score": 0.52,
                "preview": "Employees are eligible for annual leave as per policy.",
            }
        ],
    }

    def _mock_ask(question: str, include_sources: bool, top_k_override: int | None):
        assert question == "How many annual leaves do I get?"
        assert include_sources is True
        assert top_k_override == 4
        return mock_response

    monkeypatch.setattr(main_module.rag_service, "ask", _mock_ask)

    response = client.post(
        "/ask",
        json={
            "question": "  How many annual leaves do I get?  ",
            "include_sources": True,
            "top_k": 4,
        },
    )

    assert response.status_code == 200
    assert response.json() == mock_response


def test_ask_endpoint_empty_question_returns_400(client: TestClient):
    response = client.post("/ask", json={"question": "   ", "include_sources": True})

    assert response.status_code == 400
    assert response.json()["detail"] == "Question cannot be empty."


def test_ask_endpoint_validation_error(client: TestClient):
    response = client.post("/ask", json={"question": "hi", "include_sources": True})

    assert response.status_code == 422


def test_ask_endpoint_service_error_returns_500(
    main_module,
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
):
    def _mock_ask(question: str, include_sources: bool, top_k_override: int | None):
        raise RuntimeError("simulated service failure")

    monkeypatch.setattr(main_module.rag_service, "ask", _mock_ask)
    response = client.post("/ask", json={"question": "What is salary credit date?"})

    assert response.status_code == 500
    assert "RAG processing failed: simulated service failure" in response.json()["detail"]


def test_retrieve_debug_endpoint_success(
    main_module,
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
):
    mock_response: dict[str, Any] = {
        "route": "payroll_discrepancy",
        "sources": [
            {
                "index": 1,
                "filename": "Payroll_and_Salary_Processing_Policy",
                "heading": "Payroll Discrepancy",
                "chunk_id": "payroll-3",
                "score": 0.41,
                "preview": "Employees should report discrepancies within 3 working days.",
            }
        ],
        "timings": {"retrieval": 0.074},
    }

    def _mock_retrieve_debug(question: str, top_k_override: int | None):
        assert question == "How do I raise payroll discrepancy?"
        assert top_k_override == 3
        return mock_response

    monkeypatch.setattr(main_module.rag_service, "retrieve_debug", _mock_retrieve_debug)

    response = client.post(
        "/retrieve-debug",
        json={"question": "  How do I raise payroll discrepancy?  ", "top_k": 3},
    )

    assert response.status_code == 200
    assert response.json() == {
        "route": mock_response["route"],
        "sources": mock_response["sources"],
    }


def test_retrieve_debug_endpoint_empty_question_returns_400(client: TestClient):
    response = client.post("/retrieve-debug", json={"question": "   "})

    assert response.status_code == 400
    assert response.json()["detail"] == "Question cannot be empty."


def test_feedback_endpoint_writes_log(client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)

    response = client.post(
        "/feedback",
        json={
            "question": "How do I apply for leave?",
            "answer": "Use the HRMS portal and submit manager approval.",
            "rating": 4,
            "notes": "Helpful answer.",
        },
    )

    assert response.status_code == 200
    assert response.json() == {"status": "saved"}

    feedback_log = tmp_path / "logs" / "feedback.log"
    assert feedback_log.exists()
    content = feedback_log.read_text(encoding="utf-8")
    assert "rating=4" in content
    assert "How do I apply for leave?" in content


def test_ask_requires_api_key_when_configured(monkeypatch: pytest.MonkeyPatch):
    main_module = _load_main_module(monkeypatch, app_api_key="secret-123", rate_limit_per_minute=1000)
    client = TestClient(main_module.app)

    mock_response: dict[str, Any] = {
        "answer": "Salary is credited on the last working day.",
        "route": None,
        "timings": {"retrieval": 0.1, "generation": 0.2, "total": 0.3},
        "sources": [],
    }

    monkeypatch.setattr(main_module.rag_service, "ask", lambda **kwargs: mock_response)

    missing_header_response = client.post("/ask", json={"question": "What is the salary credit date?"})
    assert missing_header_response.status_code == 401
    assert missing_header_response.json()["detail"] == "Invalid or missing API key."

    wrong_header_response = client.post(
        "/ask",
        json={"question": "What is the salary credit date?"},
        headers={"X-API-Key": "wrong"},
    )
    assert wrong_header_response.status_code == 401

    ok_response = client.post(
        "/ask",
        json={"question": "What is the salary credit date?"},
        headers={"X-API-Key": "secret-123"},
    )
    assert ok_response.status_code == 200
    assert ok_response.json()["answer"] == mock_response["answer"]


def test_ask_rate_limit_returns_429(monkeypatch: pytest.MonkeyPatch):
    main_module = _load_main_module(monkeypatch, app_api_key="ratelimit-key", rate_limit_per_minute=2)
    client = TestClient(main_module.app)

    mock_response: dict[str, Any] = {
        "answer": "Mock",
        "route": None,
        "timings": {"retrieval": 0.1, "generation": 0.2, "total": 0.3},
        "sources": [],
    }
    monkeypatch.setattr(main_module.rag_service, "ask", lambda **kwargs: mock_response)

    headers = {"X-API-Key": "ratelimit-key"}
    payload = {"question": "What is the salary credit date?"}

    response_1 = client.post("/ask", json=payload, headers=headers)
    response_2 = client.post("/ask", json=payload, headers=headers)
    response_3 = client.post("/ask", json=payload, headers=headers)

    assert response_1.status_code == 200
    assert response_2.status_code == 200
    assert response_3.status_code == 429
    assert response_3.json()["detail"] == "Rate limit exceeded. Try again later."
