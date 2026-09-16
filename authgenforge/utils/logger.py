from __future__ import annotations

from authgenforge import *


def get_logger(name="forge", log_dir="logs"):
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    # File logger
    fh = logging.FileHandler(
        os.path.join(log_dir, "logs.log"),
        encoding="utf-8",
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    # Console logger
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)

    # Configure package logger
    pkg_logger = logging.getLogger("authgenforge")
    pkg_logger.setLevel(logging.DEBUG)
    pkg_logger.propagate = False

    if not pkg_logger.handlers:
        pkg_logger.addHandler(fh)
        pkg_logger.addHandler(ch)

    logger.info("Logger initialized: %s", name)

    return logger