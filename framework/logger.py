"""
framework/logger.py
===================
Lightweight structured logger for the framework.
Writes to both stdout (coloured) and a rotating log file.
"""

from __future__ import annotations
import json
import logging
import os
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler


# ANSI colours (disabled automatically on non-TTY outputs)
_COLOURS = {
    "DEBUG":    "\033[36m",   # cyan
    "INFO":     "\033[32m",   # green
    "WARNING":  "\033[33m",   # yellow
    "ERROR":    "\033[31m",   # red
    "CRITICAL": "\033[35m",   # magenta
    "RESET":    "\033[0m",
}

_USE_COLOUR = sys.stdout.isatty()


class _ColouredFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        msg = super().format(record)
        if _USE_COLOUR:
            colour = _COLOURS.get(record.levelname, "")
            reset  = _COLOURS["RESET"]
            return f"{colour}{msg}{reset}"
        return msg


def get_logger(
    name: str = "ai_trainer",
    log_dir: str = "logs",
    level: int = logging.DEBUG,
) -> logging.Logger:
    """
    Build and return a configured logger.

    - StreamHandler  : coloured output to stdout
    - RotatingFileHandler : JSON-formatted lines in logs/<name>.log
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured

    logger.setLevel(level)
    os.makedirs(log_dir, exist_ok=True)

    # ── Console handler ───────────────────────────────────────────────────────
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(
        _ColouredFormatter(
            fmt="%(asctime)s  %(levelname)-8s  %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    logger.addHandler(ch)

    # ── File handler (JSON lines) ─────────────────────────────────────────────
    log_path = os.path.join(log_dir, f"{name}.log")
    fh = RotatingFileHandler(log_path, maxBytes=1_000_000, backupCount=3)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(message)s"))   # raw JSON line

    class _JsonFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            line = {
                "ts":      datetime.utcnow().isoformat(),
                "level":   record.levelname,
                "logger":  record.name,
                "message": record.getMessage(),
            }
            record.msg = json.dumps(line)
            record.args = ()
            return True

    fh.addFilter(_JsonFilter())
    logger.addHandler(fh)

    return logger
