import logging
import os


def _log_level_from_env() -> int:
    raw = os.getenv("ACTIONLAB_LOG_LEVEL", "INFO").strip().upper()
    return getattr(logging, raw, logging.INFO)


def get_logger(name: str) -> logging.Logger:
    """
    Production-safe logger for FastAPI + Uvicorn.

    - Prevents duplicate log lines
    - Does not override uvicorn root config
    - Attaches handler only once
    - Allows ACTIONLAB_LOG_LEVEL override (e.g. DEBUG/INFO/WARNING/ERROR)
    """

    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False

    logger.setLevel(_log_level_from_env())
    return logger
