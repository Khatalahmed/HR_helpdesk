from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class Settings:
    google_api_key: str
    db_host: str = "localhost"
    db_port: str = "5433"
    db_name: str = "hr_helpdesk"
    db_user: str = "postgres"
    db_password: str = "postgres"
    collection_name: str = "hr_helpdesk"
    generation_model: str = "gemini-2.5-flash"
    embedding_model: str = "models/gemini-embedding-001"
    app_api_key: str = ""
    rate_limit_per_minute: int = 60
    app_env: str = "development"
    log_level: str = "INFO"
    log_dir: str = "logs"
    app_log_file: str = "api.log"
    http_log_file: str = "http.jsonl"
    log_max_bytes: int = 5_242_880
    log_backup_count: int = 5
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 1

    @property
    def is_production(self) -> bool:
        return self.app_env.strip().lower() == "production"

    @property
    def connection_string(self) -> str:
        return (
            f"postgresql+psycopg2://{self.db_user}:{self.db_password}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
        )


def get_settings() -> Settings:
    api_key = os.getenv("GOOGLE_API_KEY", "").strip()
    if not api_key:
        raise ValueError("GOOGLE_API_KEY is not set. Add it to your .env file.")

    return Settings(
        google_api_key=api_key,
        db_host=os.getenv("DB_HOST", "localhost"),
        db_port=os.getenv("DB_PORT", "5433"),
        db_name=os.getenv("DB_NAME", "hr_helpdesk"),
        db_user=os.getenv("DB_USER", "postgres"),
        db_password=os.getenv("DB_PASSWORD", "postgres"),
        collection_name=os.getenv("COLLECTION_NAME", "hr_helpdesk"),
        generation_model=os.getenv("GENERATION_MODEL", "gemini-2.5-flash"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-001"),
        app_api_key=os.getenv("APP_API_KEY", "").strip(),
        rate_limit_per_minute=max(1, int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))),
        app_env=os.getenv("APP_ENV", "development").strip() or "development",
        log_level=os.getenv("LOG_LEVEL", "INFO").strip().upper() or "INFO",
        log_dir=os.getenv("LOG_DIR", "logs").strip() or "logs",
        app_log_file=os.getenv("APP_LOG_FILE", "api.log").strip() or "api.log",
        http_log_file=os.getenv("HTTP_LOG_FILE", "http.jsonl").strip() or "http.jsonl",
        log_max_bytes=max(10_000, int(os.getenv("LOG_MAX_BYTES", "5242880"))),
        log_backup_count=max(1, int(os.getenv("LOG_BACKUP_COUNT", "5"))),
        api_host=os.getenv("API_HOST", "0.0.0.0").strip() or "0.0.0.0",
        api_port=max(1, int(os.getenv("API_PORT", "8000"))),
        api_workers=max(1, int(os.getenv("API_WORKERS", "1"))),
    )
