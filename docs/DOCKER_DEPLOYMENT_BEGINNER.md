# Docker Deployment Beginner Guide

This guide helps you run the project in Docker for production-style setup.

## 1) Prerequisites

- Docker Desktop installed and running
- `GOOGLE_API_KEY` available

## 2) Prepare env file

Create `.env` in project root (or copy from `.env.example`), then set:

```env
GOOGLE_API_KEY=your_real_google_api_key
APP_API_KEY=choose_a_secret_key
RATE_LIMIT_PER_MINUTE=60
```

You can keep the DB values from `.env.example` as-is.  
`docker-compose.yml` overrides DB host/port for containers automatically.

## 3) Start services

From project root:

```powershell
docker compose up -d --build
```

This starts:
- `pgvector-db` on host port `5433`
- `api` on host port `8000`

## 4) Check status

```powershell
docker compose ps
docker compose logs -f api
```

Open:
- API docs: `http://127.0.0.1:8000/docs`
- Health: `http://127.0.0.1:8000/health`

Logs inside API container:

```powershell
docker compose exec api ls logs
docker compose exec api tail -n 30 logs/http.jsonl
```

## 5) One-time data ingestion (if DB is empty)

Run these in order:

```powershell
docker compose exec api python 01_load_documents.py
docker compose exec api python 02_chunk_documents.py
docker compose exec api python 03_embed_and_store.py
```

Then test (`X-API-Key` header is required when `APP_API_KEY` is set):

```powershell
curl -X POST "http://127.0.0.1:8000/ask" `
  -H "Content-Type: application/json" `
  -H "X-API-Key: choose_a_secret_key" `
  -d "{\"question\":\"What is the salary credit date?\",\"include_sources\":true,\"top_k\":4}"
```

## 6) Stop services

```powershell
docker compose down
```

To remove DB data also:

```powershell
docker compose down -v
```
