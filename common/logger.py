import logging


def get_logger(name: str):
    """
    Simple, process-safe logger.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s"
        )

    return logger
