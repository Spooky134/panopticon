import logging
import logging.config
import sys
from typing import Dict, Any


def setup_logging(level: str = "INFO") -> None:
    logging_config: Dict[str, Any] = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
            "detailed": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
        },
        "handlers": {
            "default": {
                "level": level,
                "formatter": "default",
                "class": "logging.StreamHandler",
                "stream": sys.stdout,
            },
            "file": {
                "level": "INFO",
                "formatter": "detailed",
                "class": "logging.handlers.RotatingFileHandler",
                "filename": "app.log",
                "maxBytes": 10485760,  # 10MB
                "backupCount": 3,
            },
        },
        "loggers": {
            "app": {
                "handlers": ["default", "file"],
                "level": level,
                "propagate": False,
            },
            "uvicorn": {
                "handlers": ["default"],
                "level": "INFO",
                "propagate": False,
            },
        },
        "root": {
            "handlers": ["default"],
            "level": "WARNING",
        }
    }

    logging.config.dictConfig(logging_config)


