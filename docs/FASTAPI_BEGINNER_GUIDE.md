# FastAPI Beginner Guide

This guide helps you run the backend API for your HR Helpdesk project.

## 1) Install dependencies

From project root:

```powershell
venv\Scripts\pip.exe install -r requirements.txt
```

## 2) Start the API server

Option A:

```powershell
venv\Scripts\python.exe run_api.py
```

Option B:

```powershell
venv\Scripts\uvicorn.exe app.main:app --reload
```

Production-style start (single command):

```powershell
venv\Scripts\python.exe run_api_production.py
```

Windows note:
- Keep `API_WORKERS=1` for local Windows runs.
- On Linux production server, increase workers (for example `2` or `4`).

## 3) Open API docs

Browser URL:

- `http://127.0.0.1:8000/docs`

If `APP_API_KEY` is enabled:

1. Click **Authorize** in Swagger UI.
2. Enter your key value for `APIKeyHeader`.
3. Click **Authorize** and close the dialog.
4. Now `POST` endpoints can be tested directly from `/docs`.

If `APP_ENV=production`, docs are hidden:

- `/docs` -> disabled
- `/openapi.json` -> disabled

## 4) Main endpoints

- `GET /health`
- `POST /ask`
- `POST /retrieve-debug`
- `POST /feedback`

If `APP_API_KEY` is set in `.env`, all `POST` endpoints require request header:

- `X-API-Key: <your_key>`

## 5) Request Tracing (X-Request-ID)

The API now adds `X-Request-ID` on every response.

- If you send header `X-Request-ID`, API echoes the same value.
- If you do not send it, API generates one automatically.

Use this ID to trace a request in backend logs.

Persistent log files (with rotation) are written to:

- `logs/api.log` (general app logs)
- `logs/http.jsonl` (structured request logs)

## 6) Example `/ask` request body

```json
{
  "question": "What is the salary credit date?",
  "include_sources": true
}
```

Example with API key in PowerShell:

```powershell
curl -X POST "http://127.0.0.1:8000/ask" `
  -H "Content-Type: application/json" `
  -H "X-API-Key: your_key_here" `
  -d "{\"question\":\"What is the salary credit date?\",\"include_sources\":true,\"top_k\":4}"
```

## 7) File overview

- `app/main.py` -> API endpoints
- `app/config.py` -> environment settings
- `app/schemas.py` -> request/response models
- `app/rag/router.py` -> query routing and scoring helpers
- `app/rag/service.py` -> retrieval + generation service
- `app/rag/prompt.py` -> prompt template

## 8) Run production checks locally (before push)

```powershell
venv\Scripts\python.exe -m ruff check app tests test_api.py test_models.py scripts\check_quality_gate.py
venv\Scripts\python.exe -m mypy app tests test_api.py test_models.py scripts\check_quality_gate.py --ignore-missing-imports --follow-imports=skip
venv\Scripts\python.exe -m pyright -p .
venv\Scripts\python.exe -m pytest tests -q
venv\Scripts\python.exe scripts\check_quality_gate.py --summary rag_evaluation_summary.json
```

These are the same checks now configured in GitHub Actions CI.

## 9) Docker Deployment

For production-style local deployment, follow:

- `docs/DOCKER_DEPLOYMENT_BEGINNER.md`
