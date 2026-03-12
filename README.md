# HR Helpdesk RAG

AI-powered HR Helpdesk using FastAPI + Gemini + PostgreSQL pgvector, with a Streamlit UI and RAGAS evaluation.

## Architecture

- `docs/*.md`: source HR policy documents.
- `02_chunk_documents.py`: heading-aware chunking and `chunks.pkl` output.
- `03_embed_and_store.py`: embeddings + pgvector indexing.
- `app/main.py`: FastAPI API (`/health`, `/ask`, `/retrieve-debug`, `/feedback`).
- `app/rag/service.py`: shared retrieval + generation logic (single source of truth).
- `05_streamlit_app.py`: thin client UI calling FastAPI.
- `06_rag_evaluation.py`: governance and quality evaluation via RAGAS.

## Quick Start

1. Create `.env` from `.env.example` and set `GOOGLE_API_KEY`.
2. Start PostgreSQL with pgvector.
3. Build index:
   - `python 02_chunk_documents.py`
   - `python 03_embed_and_store.py`
4. Run API:
   - Dev: `python run_api.py`
   - Prod-style: `python run_api_production.py`
5. Run UI:
   - `streamlit run 05_streamlit_app.py`

## Security Notes

- If `APP_API_KEY` is set, all POST endpoints require `X-API-Key`.
- Rate limiter backends:
  - `RATE_LIMIT_BACKEND=memory` (single process)
  - `RATE_LIMIT_BACKEND=sqlite` (shared across local workers)
  - `RATE_LIMIT_BACKEND=redis` (distributed across containers/hosts via Redis)
- Feedback logs are privacy-safe metadata only (hash + length), not raw user text.

## Testing

- Run tests: `python -m pytest -q`
- Utility scripts (not tests):
  - `python scripts/check_google_api.py`
  - `python scripts/check_google_models.py`

## Evaluation

- Run: `python 06_rag_evaluation.py`
- Outputs:
  - `rag_evaluation_report.csv`
  - `rag_evaluation_summary.json`
