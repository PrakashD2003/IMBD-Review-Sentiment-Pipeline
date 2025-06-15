import logging
import os
from logging.handlers import RotatingFileHandler
from datetime import datetime
from pathlib import Path
from colorlog import ColoredFormatter
# from src.logger.global_logging import LOG_SESSION_TIME

LOG_DIR = "logs"
MAX_LOG_SIZE = 5 * 1024 * 1024  # 5 MB
BACKUP_COUNT = 3

def configure_logger(logger_name:str, level:str = "INFO", 
                     to_console: bool = True, 
                     to_file: bool = True, 
                     log_file_name: str = None)-> logging.Logger:
    """
    Configure a logger with optional console and rotating file handlers.

    :param logger_name: Name of the logger (e.g., __name__)
    :param level: Logging level ('DEBUG', 'INFO', etc.)
    :param to_console: Enable console logging
    :param to_file: Enable file logging
    :param log_file_name: Custom log file name (defaults to timestamp)
    :return: Configured Logger instance
    """
    # Determine project root (2 levels up: src/logger -> src -> project root)
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

    # Define Colored Log Formatter
    formatter = ColoredFormatter(    
        "%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        log_colors={
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "bold_red"
        })
    
    # Setup Console Handeler
    if to_console:
        console_handeler = logging.StreamHandler()
        console_handeler.setLevel(log_level)
        console_handeler.setFormatter(formatter)
        logger.addHandler(console_handeler)
    
    # Setup Rotating File Handeler
    if to_file:
        if log_file_name is None:
            log_file_name = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
        log_file_path = log_dir_path / f"{log_file_name}.log"
        file_handeler = RotatingFileHandler(
            filename= str(log_file_path),
            encoding= "utf-8",
            maxBytes= MAX_LOG_SIZE,
            backupCount= BACKUP_COUNT
        )
        file_handeler.setLevel(log_level)
        file_handeler.setFormatter(formatter)
        logger.addHandler(file_handeler)
    
    # Prevent log propagation to root logger
    logger.propagate = False

    return logger