"""
Logging setup utilities for LLM efficiency benchmarking.
Provides consistent logging configuration across the benchmarking system.
"""

import os
import logging
import sys
from pathlib import Path
from datetime import datetime


def setup_logging(log_dir: str = "./logs", log_level: str = "INFO", 
                 log_to_file: bool = True, log_to_console: bool = True) -> logging.Logger:
    """
    Set up comprehensive logging for the benchmarking system.
    
    Args:
        log_dir: Directory to save log files
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_to_file: Whether to log to file
        log_to_console: Whether to log to console
        
    Returns:
        Configured logger instance
    """
    
    # Create log directory
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear any existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    simple_formatter = logging.Formatter(
        '%(levelname)s: %(message)s'
    )
    
    # File handler
    if log_to_file:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"efficiency_benchmark_{timestamp}.log"
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(detailed_formatter)
        root_logger.addHandler(file_handler)
    
    # Console handler
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level.upper()))
        console_handler.setFormatter(simple_formatter)
        root_logger.addHandler(console_handler)
    
    return root_logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the specified name."""
    return logging.getLogger(name)


def log_system_info(logger: logging.Logger = None):
    """Log basic system information."""
    if logger is None:
        logger = logging.getLogger(__name__)
    
    import platform
    import torch
    
    logger.info("=== SYSTEM INFORMATION ===")
    logger.info(f"Platform: {platform.platform()}")
    logger.info(f"Python: {platform.python_version()}")
    logger.info(f"PyTorch: {torch.__version__}")
    logger.info(f"CUDA Available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        logger.info(f"CUDA Version: {torch.version.cuda}")
        logger.info(f"GPU Count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    logger.info("=== START BENCHMARKING ===")


def log_benchmark_start(logger: logging.Logger, model_type: str, config_path: str):
    """Log the start of a benchmark run."""
    logger.info(f"Starting benchmark for model: {model_type}")
    logger.info(f"Config file: {config_path}")
    logger.info("-" * 50)


def log_benchmark_complete(logger: logging.Logger, model_type: str, output_dir: str):
    """Log the completion of a benchmark run."""
    logger.info("-" * 50)
    logger.info(f"Benchmark completed for model: {model_type}")
    logger.info(f"Results saved to: {output_dir}")


def log_error(logger: logging.Logger, error: Exception, context: str = ""):
    """Log an error with context."""
    if context:
        logger.error(f"Error in {context}: {str(error)}")
    else:
        logger.error(f"Error: {str(error)}")
    
    # Log traceback at debug level
    import traceback
    logger.debug(f"Traceback: {traceback.format_exc()}")


# Configure default logging if this module is imported
_default_logger = None

def get_default_logger() -> logging.Logger:
    """Get the default logger for the benchmarking system."""
    global _default_logger
    if _default_logger is None:
        _default_logger = setup_logging()
    return _default_logger