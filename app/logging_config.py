from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from app.config import Settings


def _has_file_handler(logger: logging.Logger, file_path: Path) -> bool:
    normalized_target = str(file_path.resolve())
    for handler in logger.handlers:
        if isinstance(handler, RotatingFileHandler) and hasattr(handler, "baseFilename"):
            if str(Path(handler.baseFilename).resolve()) == normalized_target:
                return True
    return False


def _has_console_handler(logger: logging.Logger) -> bool:
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler) and not isinstance(handler, RotatingFileHandler):
            return True
    return False


def _safe_level(name: str) -> int:
    return getattr(logging, name.upper(), logging.INFO)


def setup_logging(settings: Settings) -> None:
    log_dir = Path(settings.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    level = _safe_level(settings.log_level)
    app_log_path = log_dir / settings.app_log_file
    http_log_path = log_dir / settings.http_log_file

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    if not _has_console_handler(root_logger):
        console = logging.StreamHandler()
        console.setLevel(level)
        console.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
        root_logger.addHandler(console)

    if not _has_file_handler(root_logger, app_log_path):
        app_handler = RotatingFileHandler(
            app_log_path,
            maxBytes=settings.log_max_bytes,
            backupCount=settings.log_backup_count,
            encoding="utf-8",
        )
        app_handler.setLevel(level)
        app_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
        root_logger.addHandler(app_handler)

    http_logger = logging.getLogger("snailcloud.http")
    http_logger.setLevel(logging.INFO)

    if not _has_file_handler(http_logger, http_log_path):
        http_handler = RotatingFileHandler(
            http_log_path,
            maxBytes=settings.log_max_bytes,
            backupCount=settings.log_backup_count,
            encoding="utf-8",
        )
        http_handler.setLevel(logging.INFO)
        http_handler.setFormatter(logging.Formatter("%(message)s"))
        http_logger.addHandler(http_handler)
