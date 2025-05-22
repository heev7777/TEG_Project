# app/logging_config.py
import logging
from pathlib import Path

def configure_logger():
    logger = logging.getLogger("app.mcp_server")
    logger.setLevel(logging.INFO)

    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    LOG_FILE = PROJECT_ROOT / "mcp_server.log"

    # Prevent duplicate handlers on reload
    if not logger.handlers:
        # File handler
        fh = logging.FileHandler(LOG_FILE, mode='a')
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

        logger.addHandler(fh)
        logger.addHandler(ch)

    logger.info("Logger configured. Logging to file: %s", LOG_FILE)
    return logger
