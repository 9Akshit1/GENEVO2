"""
Logging configuration for the simulation framework.
"""

import logging
import os
from datetime import datetime
from typing import Optional


def setup_logging(log_dir: str = "logs", 
                 log_level: str = "INFO",
                 log_to_file: bool = True) -> logging.Logger:
    """
    Configure logging for the simulation framework.
    
    Args:
        log_dir: Directory for log files
        log_level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
        log_to_file: Whether to write logs to file
    
    Returns:
        Configured logger instance
    """
    # Create logs directory
    if log_to_file and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Create logger
    logger = logging.getLogger('bone_biosensor')
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
    )
    simple_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Console handler (INFO and above)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    logger.addHandler(console_handler)
    
    # File handler (all levels)
    if log_to_file:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"simulation_{timestamp}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)
        
        logger.info(f"Logging to file: {log_file}")
    
    return logger