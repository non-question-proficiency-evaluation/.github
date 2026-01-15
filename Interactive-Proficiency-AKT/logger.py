"""
Logging utility module for AKT training and evaluation.

This module provides a centralized logging system that can be used throughout
the codebase for consistent log formatting and output management.
"""
import logging
import os
from datetime import datetime
from typing import Optional


def setup_logger(name: str = "akt", log_file: Optional[str] = None, 
                 level: int = logging.INFO) -> logging.Logger:
    """
    Set up and configure a logger with console and optional file handlers.
    
    Creates a logger with formatted output that includes timestamps, logger name,
    log level, and messages. Supports both console output and file logging.
    
    Args:
        name (str, optional): Name identifier for the logger. Defaults to "akt".
        log_file (str, optional): Path to log file. If provided, logs will also
            be written to this file. If None, only console logging is enabled.
            Defaults to None.
        level (int, optional): Logging level. Defaults to logging.INFO.
            Options: logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR.
    
    Returns:
        logging.Logger: Configured logger instance ready for use.
    
    Note:
        - If log_file is provided, the directory will be created if it doesn't exist.
        - Log format: '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        - Console handler always outputs to stdout.
        - File handler appends to existing log files.
    
    Example:
        >>> logger = setup_logger("akt_training")
        >>> logger.info("Starting training...")
        >>> 
        >>> # With file logging
        >>> logger = setup_logger("akt_training", log_file="logs/training.log")
        >>> logger.info("Training epoch 1")
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid adding duplicate handlers if logger already exists
    if logger.handlers:
        return logger
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if log_file is provided)
    if log_file:
        # Create log directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, mode='a')  # Append mode
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_log_file_path(base_dir: str = "logs", experiment_name: str = "experiment") -> str:
    """
    Generate a log file path with timestamp.
    
    Creates a log file path that includes the experiment name and timestamp,
    making it easy to track different training runs.
    
    Args:
        base_dir (str, optional): Base directory for log files. Defaults to "logs".
        experiment_name (str, optional): Name identifier for the experiment.
            Defaults to "experiment".
    
    Returns:
        str: Full path to the log file.
            Format: <base_dir>/<experiment_name>_<timestamp>.log
    
    Example:
        >>> log_path = get_log_file_path("logs", "akt_assist2009")
        >>> # Returns: "logs/akt_assist2009_2024-01-15_14-30-45.log"
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = f"{experiment_name}_{timestamp}.log"
    return os.path.join(base_dir, log_filename)
