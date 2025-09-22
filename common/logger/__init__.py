"""Logging utilities for the sentiment pipeline."""

import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
from pathlib import Path
from colorlog import ColoredFormatter
from pythonjsonlogger import jsonlogger
# from src.logger.global_logging import LOG_SESSION_TIME

LOG_DIR = "logs"
MAX_LOG_SIZE = 5 * 1024 * 1024  # 5 MB
BACKUP_COUNT = 3

def configure_logger(logger_name:str, level:str = "INFO", 
                     to_console: bool = True, 
                     to_file: bool = True, 
                     log_file_name: str = "service.log")-> logging.Logger:
    """
    Configure a logger with optional console and rotating file handlers to output structured JSON logs.

    :param logger_name: Name of the logger (e.g., __name__)
    :param level: Logging level ('DEBUG', 'INFO', etc.)
    :param to_console: Enable console logging
    :param to_file: Enable file logging
    :param log_file_name: Custom log file name (defaults to timestamp)
    :return: Configured Logger instance
    """
    # Determine project root (2 levels up: common/logger -> common -> project root)
    base_dir = Path(__file__).resolve().parents[2]
    log_dir_path = base_dir / LOG_DIR 
    log_dir_path.mkdir(parents=True, exist_ok=True)
    
    # Create or retrieve the logger
    logger = logging.getLogger(name=logger_name)
    logger.handlers[:] = []
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Set logging level
    log_level = getattr(logging, level.upper(),logging.INFO)
    logger.setLevel(level=log_level)

    # Define Colored Log Formatter for better console readability
    console_formatter = ColoredFormatter(    
        "%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        log_colors={
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "bold_red"
        })
    
    # Setup Console Handler
    if to_console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    # Define a standard JSON log format
    # This format is excellent for services like CloudWatch
    json_formatter = jsonlogger.JsonFormatter(
        '%(asctime)s %(name)s %(levelname)s %(message)s'
    )
    # Setup Rotating File Handler for JSON logs
    log_file_path = log_dir_path / log_file_name
    file_handler = RotatingFileHandler(
        filename=str(log_file_path),
        encoding="utf-8",
        maxBytes=MAX_LOG_SIZE,
        backupCount=BACKUP_COUNT
    )
    file_handler.setFormatter(json_formatter)
    logger.addHandler(file_handler)
    
    # Prevent log propagation to root logger
    logger.propagate = False

    return logger