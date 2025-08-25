#!/usr/bin/env python
"""
Centralized logging configuration for CustomKB.

This module provides standardized logging setup with consistent formats,
levels, and handlers across all modules.
"""

import os
import sys
import logging
from typing import Optional, Dict, Any
from pathlib import Path

# Standard log format for all modules
DEFAULT_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
DETAILED_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(funcName)s() - %(message)s'
SIMPLE_FORMAT = '%(levelname)s: %(message)s'

# Performance metrics format for structured logging
METRICS_FORMAT = '%(asctime)s - METRICS - %(message)s'

# Color codes for terminal output
COLORS = {
  'DEBUG': '\033[36m',    # Cyan
  'INFO': '\033[32m',     # Green
  'WARNING': '\033[33m',  # Yellow
  'ERROR': '\033[31m',    # Red
  'CRITICAL': '\033[35m', # Magenta
  'RESET': '\033[0m'      # Reset
}

class ColoredFormatter(logging.Formatter):
  """Custom formatter that adds colors to log output."""
  
  def format(self, record):
    if sys.stdout.isatty():
      levelname = record.levelname
      if levelname in COLORS:
        record.levelname = f"{COLORS[levelname]}{levelname}{COLORS['RESET']}"
    return super().format(record)


class ContextFilter(logging.Filter):
  """Add contextual information to log records."""
  
  def __init__(self, context: Dict[str, Any] = None):
    super().__init__()
    self.context = context or {}
  
  def filter(self, record):
    # Add context to the record
    for key, value in self.context.items():
      setattr(record, key, value)
    
    # Add memory usage if available
    try:
      import psutil
      process = psutil.Process()
      record.memory_mb = process.memory_info().rss / 1024 / 1024
    except (ImportError, AttributeError):
      record.memory_mb = 0
    
    return True


def get_log_level(env_var: str = 'LOG_LEVEL', default: str = 'INFO') -> int:
  """
  Get log level from environment variable.
  
  Args:
      env_var: Environment variable name
      default: Default level if not set
      
  Returns:
      Logging level constant
  """
  level_name = os.getenv(env_var, default).upper()
  level_map = {
    'DEBUG': logging.DEBUG,
    'INFO': logging.INFO,
    'WARNING': logging.WARNING,
    'ERROR': logging.ERROR,
    'CRITICAL': logging.CRITICAL
  }
  return level_map.get(level_name, logging.INFO)


def setup_file_handler(
  logger: logging.Logger,
  log_file: str,
  level: int = logging.INFO,
  format_string: str = DEFAULT_FORMAT
) -> logging.FileHandler:
  """
  Add file handler to logger.
  
  Args:
      logger: Logger instance
      log_file: Path to log file
      level: Log level
      format_string: Format string
      
  Returns:
      FileHandler instance
  """
  # Ensure log directory exists
  log_path = Path(log_file)
  log_path.parent.mkdir(parents=True, exist_ok=True)
  
  # Create handler
  handler = logging.FileHandler(log_file)
  handler.setLevel(level)
  handler.setFormatter(logging.Formatter(format_string))
  
  logger.addHandler(handler)
  return handler


def setup_console_handler(
  logger: logging.Logger,
  level: int = logging.INFO,
  colored: bool = True,
  format_string: str = SIMPLE_FORMAT
) -> logging.StreamHandler:
  """
  Add console handler to logger.
  
  Args:
      logger: Logger instance
      level: Log level
      colored: Use colored output
      format_string: Format string
      
  Returns:
      StreamHandler instance
  """
  handler = logging.StreamHandler()
  handler.setLevel(level)
  
  if colored and sys.stdout.isatty():
    handler.setFormatter(ColoredFormatter(format_string))
  else:
    handler.setFormatter(logging.Formatter(format_string))
  
  logger.addHandler(handler)
  return handler


def configure_root_logger(
  level: int = None,
  log_file: Optional[str] = None,
  console: bool = True,
  colored: bool = True
) -> logging.Logger:
  """
  Configure the root logger with standard settings.
  
  Args:
      level: Log level (uses env var if not specified)
      log_file: Optional log file path
      console: Enable console output
      colored: Use colored console output
      
  Returns:
      Configured root logger
  """
  # Get root logger
  root_logger = logging.getLogger()
  
  # Clear existing handlers
  root_logger.handlers.clear()
  
  # Set level
  if level is None:
    level = get_log_level()
  root_logger.setLevel(level)
  
  # Add console handler
  if console:
    setup_console_handler(
      root_logger,
      level=level,
      colored=colored,
      format_string=SIMPLE_FORMAT if level > logging.DEBUG else DEFAULT_FORMAT
    )
  
  # Add file handler
  if log_file:
    setup_file_handler(
      root_logger,
      log_file,
      level=logging.DEBUG,  # File gets all debug messages
      format_string=DETAILED_FORMAT
    )
  
  return root_logger


def get_logger(
  name: str,
  level: Optional[int] = None,
  context: Optional[Dict[str, Any]] = None
) -> logging.Logger:
  """
  Get a configured logger for a module.
  
  This is the primary function modules should use to get their logger.
  
  Args:
      name: Logger name (usually __name__)
      level: Optional log level override
      context: Optional context to add to all log records
      
  Returns:
      Configured logger instance
  """
  logger = logging.getLogger(name)
  
  # Set level if specified
  if level is not None:
    logger.setLevel(level)
  
  # Add context filter if provided
  if context:
    logger.addFilter(ContextFilter(context))
  
  return logger


def log_performance_metrics(
  logger: logging.Logger,
  operation: str,
  duration: float,
  **kwargs
) -> None:
  """
  Log performance metrics in a structured format.
  
  Args:
      logger: Logger instance
      operation: Operation name
      duration: Duration in seconds
      **kwargs: Additional metrics
  """
  metrics = {
    'operation': operation,
    'duration_ms': round(duration * 1000, 2),
    **kwargs
  }
  
  # Format as key=value pairs
  metrics_str = ' '.join([f'{k}={v}' for k, v in metrics.items()])
  logger.info(f"PERF: {metrics_str}")


def configure_module_loggers():
  """
  Configure log levels for specific modules.
  
  This can be used to reduce noise from verbose third-party libraries.
  """
  # Reduce noise from verbose libraries
  logging.getLogger('urllib3').setLevel(logging.WARNING)
  logging.getLogger('requests').setLevel(logging.WARNING)
  logging.getLogger('httpx').setLevel(logging.WARNING)
  logging.getLogger('openai').setLevel(logging.WARNING)
  logging.getLogger('anthropic').setLevel(logging.WARNING)
  logging.getLogger('faiss').setLevel(logging.WARNING)
  logging.getLogger('sentence_transformers').setLevel(logging.WARNING)
  logging.getLogger('transformers').setLevel(logging.WARNING)
  logging.getLogger('torch').setLevel(logging.WARNING)
  logging.getLogger('matplotlib').setLevel(logging.WARNING)
  logging.getLogger('PIL').setLevel(logging.WARNING)
  logging.getLogger('spacy').setLevel(logging.WARNING)
  logging.getLogger('nltk').setLevel(logging.WARNING)


# Compatibility functions - delegate to logging_utils for now
def setup_logging(verbose: bool, debug: bool = False, 
                 log_file: Optional[str] = None,
                 log_to_file: bool = True,
                 config_file: Optional[str] = None,
                 kb_directory: Optional[str] = None,
                 kb_name: Optional[str] = None) -> Optional[logging.Logger]:
  """Set up logging configuration - compatibility wrapper."""
  from utils.logging_utils import setup_logging as _setup_logging
  return _setup_logging(verbose, debug, log_file, log_to_file, config_file, kb_directory, kb_name)

def dashes(offset: int = 0, char: str = '-') -> str:
  """Generate dashes for terminal output."""
  from utils.logging_utils import dashes as _dashes
  return _dashes(offset, char)

def elapsed_time(start_time: int, end_time: Optional[int] = None) -> str:
  """Calculate elapsed time."""
  from utils.logging_utils import elapsed_time as _elapsed_time
  return _elapsed_time(start_time, end_time)

def time_to_finish(start_time: int, records_processed: int, total_records: int) -> str:
  """Estimate time to completion."""
  from utils.logging_utils import time_to_finish as _time_to_finish
  return _time_to_finish(start_time, records_processed, total_records)

def get_kb_info_from_config(config_file: str) -> tuple:
  """Get KB info from config file."""
  from utils.logging_utils import get_kb_info_from_config as _get_kb_info
  return _get_kb_info(config_file)

def log_file_operation(logger, operation_type: str, filepath: str, **kwargs):
  """Log file operations."""
  from utils.logging_utils import log_file_operation as _log_file_operation
  return _log_file_operation(logger, operation_type, filepath, **kwargs)

def log_operation_error(logger, operation: str, error: Exception, **context):
  """Log operation errors."""
  from utils.logging_utils import log_operation_error as _log_operation_error
  return _log_operation_error(logger, operation, error, **context)

class OperationLogger:
  """Operation logging context manager."""
  def __init__(self, logger, operation_type: str, **kwargs):
    from utils.logging_utils import OperationLogger as _OperationLogger
    self._wrapped = _OperationLogger(logger, operation_type, **kwargs)
  
  def __enter__(self):
    return self._wrapped.__enter__()
  
  def __exit__(self, exc_type, exc_val, exc_tb):
    return self._wrapped.__exit__(exc_type, exc_val, exc_tb)

# Initialize on import if running as main program
if __name__ != '__main__':
  # Configure module loggers to reduce noise
  configure_module_loggers()

#fin