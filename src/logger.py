import os
import sys
import logging
from logging.handlers import RotatingFileHandler
from typing import Optional


class HsBatLogger:
    _instance: Optional["HsBatLogger"] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, log_dir: str = "logs", log_level: str = "DEBUG"):
        if hasattr(self, "_initialized") and self._initialized:
            return
        self._initialized = True
        os.makedirs(log_dir, exist_ok=True)
        self.logger = logging.getLogger("HsBat")
        self.logger.setLevel(getattr(logging, log_level.upper(), logging.DEBUG))

        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        file_handler = RotatingFileHandler(
            os.path.join(log_dir, "hsbat.log"),
            maxBytes=10 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

    def get_logger(self, name: Optional[str] = None) -> logging.Logger:
        if name:
            return logging.getLogger(f"HsBat.{name}")
        return self.logger
