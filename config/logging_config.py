import logging
from logging.handlers import TimedRotatingFileHandler
import sys
from pathlib import Path

# --- CONFIGURATION ---
LOG_DIRECTORY = "logs"  # Storing all logs in a dedicated 'logs' folder
LOG_BASENAME = "app"    # No .log extension, suffix will handle that
WHEN_TO_ROTATE = "midnight"  # Rotate daily at midnight
ROTATION_INTERVAL = 1
BACKUP_COUNT = 30  # Keep 30 old log files

def setup_logging():
    """
    Configures logging to output to both the console and a daily rotating file.
    """
    # Create the log directory if it doesn't exist
    log_dir = Path(LOG_DIRECTORY)
    log_dir.mkdir(exist_ok=True)
    log_file_path = log_dir / LOG_BASENAME

    # Get the root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Create a standard formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Prevent duplicate handlers if this function is called multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    # --- 1. CONSOLE HANDLER ---
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # --- 2. TIMED ROTATING FILE HANDLER ---
    # This handler creates a new log file each day with suffix like app-2025-08-17.log
    file_handler = TimedRotatingFileHandler(
        filename=log_file_path,
        when=WHEN_TO_ROTATE,
        interval=ROTATION_INTERVAL,
        backupCount=BACKUP_COUNT,
        encoding='utf-8',
        utc=False  # set True if you want UTC instead of local time
    )
    file_handler.suffix = "%Y-%m-%d.log"
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logging.info("Logging configured successfully to console and daily rotating files.")