import logging


def get_logger(name: str) -> logging.Logger:
    """
    Production-safe logger for FastAPI + Uvicorn.

    - Prevents duplicate log lines
    - Does not override uvicorn root config
    - Attaches handler only once
    """

    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s"
        )
        handler.setFormatter(formatter)

        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

        logger.propagate = False

    return logger

