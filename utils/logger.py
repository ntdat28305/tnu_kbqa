"""
utils/logger.py
Logging helper dùng chung toàn project.

Dùng:
    from utils.logger import get_logger
    logger = get_logger(__name__, log_file="logs/ten_module.log")

    logger.info("Thông báo bình thường")
    logger.warning("Cảnh báo")
    logger.error("Lỗi")
    logger.debug("Debug chi tiết (chỉ ghi vào file)")
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path


# ── Màu sắc cho console (Windows + Unix) ──────────────────────

COLORS = {
    "DEBUG":    "\033[36m",   # Cyan
    "INFO":     "\033[32m",   # Green
    "WARNING":  "\033[33m",   # Yellow
    "ERROR":    "\033[31m",   # Red
    "CRITICAL": "\033[35m",   # Magenta
    "RESET":    "\033[0m",
}


class ColorFormatter(logging.Formatter):
    """Formatter thêm màu cho console output."""

    FMT = "%(asctime)s %(levelname)-8s %(name)s — %(message)s"

    def format(self, record: logging.LogRecord) -> str:
        color = COLORS.get(record.levelname, COLORS["RESET"])
        reset = COLORS["RESET"]
        formatter = logging.Formatter(
            f"{color}{self.FMT}{reset}",
            datefmt="%H:%M:%S",
        )
        return formatter.format(record)


class PlainFormatter(logging.Formatter):
    """Formatter không màu cho file output."""

    FMT = "%(asctime)s [%(levelname)s] %(name)s — %(message)s"

    def __init__(self):
        super().__init__(fmt=self.FMT, datefmt="%Y-%m-%d %H:%M:%S")


# ── Cache logger để tránh duplicate handlers ──────────────────

_loggers: dict[str, logging.Logger] = {}


def get_logger(
    name:     str,
    log_file: str | None = None,
    level:    int = logging.DEBUG,
) -> logging.Logger:
    """
    Trả về logger với:
      - Console handler : INFO trở lên, có màu
      - File handler    : DEBUG trở lên, không màu (nếu log_file được truyền)

    Args:
        name:     Tên logger, thường dùng __name__
        log_file: Đường dẫn file log, ví dụ "logs/loader.log"
        level:    Log level tối thiểu (mặc định DEBUG)

    Returns:
        logging.Logger đã cấu hình sẵn
    """
    if name in _loggers:
        return _loggers[name]

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False   # Không bubble up lên root logger

    # ── Console handler ──
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(ColorFormatter())
    logger.addHandler(console_handler)

    # ── File handler ──
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(PlainFormatter())
        logger.addHandler(file_handler)

    _loggers[name] = logger
    return logger


# ── Shortcut: tạo logger cho từng module ──────────────────────

def get_pipeline_logger(module_name: str) -> logging.Logger:
    """
    Shortcut tự động đặt tên file log theo module.

    Ví dụ:
        get_pipeline_logger("loader")
        → logger tên "pipeline.loader", log ra "logs/loader.log"
    """
    return get_logger(
        name=f"pipeline.{module_name}",
        log_file=f"logs/{module_name}.log",
    )