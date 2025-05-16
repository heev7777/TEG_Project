import logging
import sys
from typing import Optional

def setup_logger(name: str, level: Optional[int] = None) -> logging.Logger:
    """
    Set up a logger with the specified name and level.
    
    Args:
        name: Name of the logger
        level: Logging level (defaults to INFO)
        
    Returns:
        logging.Logger: Configured logger instance
    """
    if level is None:
        level = logging.INFO
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Create console handler if no handlers exist
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(handler)
    
    return logger 