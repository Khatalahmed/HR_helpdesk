"""
Production entrypoint for server deployment.

Run:
    python run_api_production.py
"""

from __future__ import annotations

import os

import uvicorn

from app.config import get_settings


if __name__ == "__main__":
    settings = get_settings()
    workers = settings.api_workers
    # Windows local servers are more stable with a single worker process.
    if os.name == "nt" and workers > 1:
        print(
            f"[run_api_production] API_WORKERS={workers} on Windows can fail; "
            "forcing API_WORKERS=1."
        )
        workers = 1

    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=False,
        workers=workers,
        log_config=None,
        access_log=False,
    )
