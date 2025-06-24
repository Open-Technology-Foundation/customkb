#!/usr/bin/env python
"""
Logging utilities for CustomKB.
Provides consistent logging with color support, time tracking, and progress reporting.
"""

import logging
import logging.handlers
import colorlog
import shutil
import time
import os
import psutil
import configparser
import json
from typing import Optional, Dict, Any, Union, Tuple

# Note: Global logger removed - use get_logger(__name__) in each module

def get_kb_info_from_config(config_file: str) -> Tuple[str, str]:
  """
  Extract knowledge base directory and name from config file path.
  
  Args:
      config_file: Path to config file (e.g., "/path/to/mycompany.cfg")
      
  Returns:
      (kb_directory, kb_name) tuple (e.g., ("/path/to/", "mycompany"))
  """
  from utils.text_utils import split_filepath
  
  if not config_file:
    raise ValueError("Config file path cannot be empty")
  
  directory, basename, extension, _ = split_filepath(config_file)
  
  # Remove .cfg extension if present
  if basename.endswith('.cfg'):
    basename = basename[:-4]
  
  return directory, basename

def get_log_file_path(config_log_file: str, kb_directory: str, kb_name: str) -> str:
  """
  Resolve log file path based on configuration.
  
  Args:
      config_log_file: Log file setting from config ('auto', relative, or absolute path)
      kb_directory: Knowledge base directory
      kb_name: Knowledge base name
      
  Returns:
      Fully resolved log file path
  """
  if config_log_file == 'auto':
    # Auto-generate: {kb_directory}/logs/{kb_name}.log
    log_dir = os.path.join(kb_directory, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    return os.path.join(log_dir, f'{kb_name}.log')
  elif os.path.isabs(config_log_file):
    # Absolute path: use as-is
    log_dir = os.path.dirname(config_log_file)
    if log_dir:
      os.makedirs(log_dir, exist_ok=True)
    return config_log_file
  else:
    # Relative path: resolve against KB directory
    log_path = os.path.join(kb_directory, config_log_file)
    log_dir = os.path.dirname(log_path)
    if log_dir:
      os.makedirs(log_dir, exist_ok=True)
    return log_path

def load_logging_config(config_file: Optional[str] = None, 
                       kb_directory: Optional[str] = None) -> Dict[str, Any]:
  """
  Load logging configuration from .cfg file or environment variables.
  
  Args:
      config_file: Path to configuration file (optional).
      kb_directory: Knowledge base directory for relative path resolution (optional).
      
  Returns:
      Dictionary containing logging configuration.
  """
  # Default logging configuration
  defaults = {
    'log_level': 'INFO',
    'log_file': 'auto',
    'max_log_size': 10485760,    # 10MB
    'backup_count': 5,
    'console_colors': True,
    'file_logging': True,
    'json_format': False,
    'module_levels': {}          # Per-module log levels
  }
  
  # Load from config file if provided
  if config_file and os.path.exists(config_file):
    try:
      config = configparser.ConfigParser()
      config.read(config_file)
      
      if 'LOGGING' in config:
        logging_section = config['LOGGING']
        
        # Update defaults with config file values
        for key in defaults:
          if key == 'module_levels':
            # Handle module-specific levels
            module_levels = {}
            try:
              for option in logging_section:
                if option.startswith('level_'):
                  module_name = option[6:]  # Remove 'level_' prefix
                  module_levels[module_name] = logging_section[option]
              if module_levels:
                defaults[key] = module_levels
            except Exception:
              pass  # Skip module levels if parsing fails
          elif key in logging_section:
            try:
              if isinstance(defaults[key], bool):
                defaults[key] = logging_section.getboolean(key)
              elif isinstance(defaults[key], int):
                defaults[key] = logging_section.getint(key)
              else:
                defaults[key] = logging_section.get(key)
            except ValueError:
              pass  # Keep default value if conversion fails
    except Exception as e:
      # If config loading fails, use defaults
      logging.warning(f"Failed to load logging config from {config_file}: {e}")
  
  # Override with environment variables
  env_overrides = {
    'LOGGING_LEVEL': 'log_level',
    'LOGGING_FILE': 'log_file', 
    'LOGGING_MAX_SIZE': 'max_log_size',
    'LOGGING_BACKUP_COUNT': 'backup_count',
    'LOGGING_CONSOLE_COLORS': 'console_colors',
    'LOGGING_FILE_ENABLED': 'file_logging',
    'LOGGING_JSON_FORMAT': 'json_format'
  }
  
  for env_var, config_key in env_overrides.items():
    env_value = os.getenv(env_var)
    if env_value is not None:
      try:
        if isinstance(defaults[config_key], bool):
          defaults[config_key] = env_value.lower() in ('true', '1', 'yes', 'on')
        elif isinstance(defaults[config_key], int):
          defaults[config_key] = int(env_value)
        else:
          defaults[config_key] = env_value
      except ValueError:
        pass  # Keep default if conversion fails
  
  return defaults

def get_json_formatter() -> logging.Formatter:
  """
  Get JSON formatter for structured logging.
  
  Returns:
      JSON logging formatter.
  """
  class JSONFormatter(logging.Formatter):
    def format(self, record):
      log_entry = {
        'timestamp': self.formatTime(record, self.datefmt),
        'level': record.levelname,
        'logger': record.name,
        'message': record.getMessage(),
        'module': record.module,
        'function': record.funcName,
        'line': record.lineno
      }
      
      # Add extra fields if present
      if hasattr(record, 'operation'):
        log_entry['operation'] = record.operation
      if hasattr(record, 'file_path'):
        log_entry['file_path'] = record.file_path
      if hasattr(record, 'model'):
        log_entry['model'] = record.model
      if hasattr(record, 'duration'):
        log_entry['duration'] = record.duration
      if hasattr(record, 'memory_usage'):
        log_entry['memory_usage'] = record.memory_usage
      
      # Add any other extra fields
      for key, value in record.__dict__.items():
        if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 
                       'filename', 'module', 'lineno', 'funcName', 'created', 
                       'msecs', 'relativeCreated', 'thread', 'threadName',
                       'processName', 'process', 'message', 'exc_info', 
                       'exc_text', 'stack_info'] and not key.startswith('_'):
          log_entry[key] = value
      
      return json.dumps(log_entry)
  
  return JSONFormatter()

def configure_module_logging(module_levels: Dict[str, str]):
  """
  Configure per-module logging levels.
  
  Args:
      module_levels: Dictionary mapping module names to log levels.
  """
  for module_name, level_str in module_levels.items():
    try:
      level = getattr(logging, level_str.upper())
      logger = logging.getLogger(module_name)
      logger.setLevel(level)
    except AttributeError:
      logging.warning(f"Invalid log level '{level_str}' for module '{module_name}'")

def get_logger(name: str) -> logging.Logger:
  """
  Get a module-specific logger with proper namespacing.
  
  This function returns a properly isolated logger for each module,
  allowing for module-specific log filtering and configuration.

  Args:
      name: The name of the logger (typically __name__).

  Returns:
      A module-specific logger instance.
  """
  return logging.getLogger(name)

def get_root_logger() -> logging.Logger:
  """
  Get the root logger for configuration purposes.
  
  Returns:
      The root logger instance.
  """
  return logging.getLogger()

def setup_file_logging(log_file: str, max_bytes: int = 10485760, 
                      backup_count: int = 5) -> logging.Handler:
  """
  Setup rotating file logging handler.
  
  Args:
      log_file: Path to the log file.
      max_bytes: Maximum size of each log file (default: 10MB).
      backup_count: Number of backup files to keep (default: 5).
      
  Returns:
      Configured rotating file handler.
  """
  # Ensure log directory exists
  log_dir = os.path.dirname(log_file)
  if log_dir:
    os.makedirs(log_dir, exist_ok=True)
  
  file_handler = logging.handlers.RotatingFileHandler(
    log_file, 
    maxBytes=max_bytes, 
    backupCount=backup_count,
    encoding='utf-8'
  )
  
  file_formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
  )
  file_handler.setFormatter(file_formatter)
  
  return file_handler

def get_console_formatter(verbose: bool) -> colorlog.ColoredFormatter:
  """
  Get console formatter with appropriate detail level.
  
  Args:
      verbose: Whether to include detailed formatting.
      
  Returns:
      Configured colored formatter.
  """
  if verbose:
    logformat = "%(log_color)s%(module)s:%(levelname)s: %(message)s"
  else:
    logformat = "%(log_color)s%(module)s:%(levelname)s: %(message)s"
  
  return colorlog.ColoredFormatter(
    logformat,
    datefmt=None,
    reset=True,
    log_colors={
      'DEBUG': 'cyan',
      'INFO': 'green',
      'WARNING': 'yellow',
      'ERROR': 'red',
      'CRITICAL': 'red,bg_white'
    },
    secondary_log_colors={},
    style='%'
  )

def setup_logging(verbose: bool, debug: bool = False, 
                 log_file: Optional[str] = None,
                 log_to_file: bool = True,
                 config_file: Optional[str] = None,
                 kb_directory: Optional[str] = None,
                 kb_name: Optional[str] = None) -> Optional[logging.Logger]:
  """
  Set up enhanced logging configuration with console and optional file output.

  Args:
      verbose: Whether to enable verbose logging (INFO level).
      debug: Whether to enable debug logging (DEBUG level).
      log_file: Optional path to log file. If None, uses config or auto-generates.
      log_to_file: Whether to enable file logging (default: True).
      config_file: Optional path to configuration file for logging settings.
      kb_directory: Knowledge base directory for per-KB logging.
      kb_name: Knowledge base name for per-KB logging.

  Returns:
      The configured root logger instance, or None for commands that don't need logging.
  """
  
  # Return None if this is a help/version command (no KB context provided)
  if not kb_directory or not kb_name:
    return None
    
  # Load logging configuration with KB context
  logging_config = load_logging_config(config_file, kb_directory)
  
  root_logger = logging.getLogger()

  # Remove existing handlers to avoid duplicates
  for handler in root_logger.handlers[:]:
    root_logger.removeHandler(handler)

  # Determine log level (command line overrides config)
  if debug:
    log_level = logging.DEBUG
  elif verbose:
    log_level = logging.INFO
  else:
    # Quiet mode: only show warnings and errors
    log_level = logging.WARNING
  
  root_logger.setLevel(log_level)

  # Console handler with color formatting
  console_handler = colorlog.StreamHandler()
  
  # Use colors based on config
  if logging_config.get('console_colors', True):
    console_formatter = get_console_formatter(verbose or debug)
  else:
    # Plain text formatter without colors
    console_formatter = logging.Formatter(
      '%(name)s:%(levelname)s: %(message)s'
    )
  
  console_handler.setFormatter(console_formatter)
  root_logger.addHandler(console_handler)

  # File handler with rotation
  file_logging_enabled = log_to_file and logging_config.get('file_logging', True)
  
  if file_logging_enabled:
    # Determine KB-specific log file path
    if not log_file:
      config_log_file = logging_config.get('log_file', 'auto')
      log_file = get_log_file_path(config_log_file, kb_directory, kb_name)
    
    try:
      max_bytes = logging_config.get('max_log_size', 10485760)
      backup_count = logging_config.get('backup_count', 5)
      
      file_handler = setup_file_logging(log_file, max_bytes, backup_count)
      
      # Use JSON formatter if configured
      if logging_config.get('json_format', False):
        json_formatter = get_json_formatter()
        file_handler.setFormatter(json_formatter)
      
      root_logger.addHandler(file_handler)
      if verbose or debug:
        format_type = "JSON" if logging_config.get('json_format', False) else "text"
        root_logger.info(f"Logging to file: {log_file} (format: {format_type})")
    except Exception as e:
      # Fallback: log warning but continue without file logging
      root_logger.warning(f"Failed to setup file logging to {log_file}: {e}")

  # Configure per-module logging levels
  module_levels = logging_config.get('module_levels', {})
  if module_levels:
    configure_module_logging(module_levels)
    if verbose or debug:
      root_logger.debug(f"Applied module-specific log levels: {module_levels}")

  return root_logger

def dashes(offset: int = 0, char: str = '-') -> str:
  """
  Generate a string of dashes to match the terminal width.

  Args:
      offset: Number of characters to subtract from the width.
      char: The character to use for the line.

  Returns:
      A string of dashes.
  """
  width, _ = shutil.get_terminal_size()
  return (char[0] * (width - offset))

def elapsed_time(start_time: int, end_time: Optional[int] = None) -> str:
  """
  Calculate and format the elapsed time.

  Args:
      start_time: The start time in seconds.
      end_time: The end time in seconds. If None, uses the current time.

  Returns:
      A formatted string representing the elapsed time (e.g., "01h 23m 45s").
  """
  if end_time is None:
    end_time = int(time.time())

  el = int(end_time - start_time)
  result = ''

  n = el // 86400
  if n:
    result += f"{n}d "
    el -= n * 86400

  n = el // 3600
  if n:
    result += f"{n:02}h "
    el -= n * 3600

  n = el // 60
  if n:
    result += f"{n:02}m "
    el -= n * 60

  result += f"{el:02}s"

  return result

def time_to_finish(start_time: int, records_processed: int, total_records: int) -> str:
  """
  Estimate the time remaining to finish processing based on current rate.

  Args:
      start_time: The start time in seconds.
      records_processed: The number of records processed so far.
      total_records: The total number of records to process.

  Returns:
      A formatted string representing the estimated time to finish.
  """
  current_time = int(time.time())
  elapsed = current_time - start_time

  if records_processed == 0 or elapsed == 0:
    return ""

  records_per_second = records_processed / elapsed
  remaining_records = total_records - records_processed
  estimated_seconds = remaining_records / records_per_second

  return f"{elapsed_time(0, int(estimated_seconds))}"

def log_operation_error(logger: logging.Logger, operation: str, error: Exception, **context):
  """
  Log errors with consistent formatting and context.
  
  Provides structured error logging with operation name, error details,
  and additional context. Automatically masks sensitive data in context values.
  
  Args:
      logger: The logger instance to use.
      operation: Name of the operation that failed.
      error: The exception that occurred.
      **context: Additional context as key-value pairs.
  """
  from utils.security_utils import mask_sensitive_data
  
  # Build context string with sensitive data masking
  context_items = []
  for key, value in context.items():
    safe_value = str(value)
    # Mask values that might contain sensitive information
    if any(sensitive in key.lower() for sensitive in ['key', 'token', 'password', 'secret']):
      safe_value = mask_sensitive_data(safe_value)
    context_items.append(f"{key}={safe_value}")
  
  context_str = " | ".join(context_items) if context_items else ""
  error_msg = f"Operation '{operation}' failed: {error}"
  
  if context_str:
    logger.error(f"{error_msg} | Context: {context_str}")
  else:
    logger.error(error_msg)

def log_performance_metrics(logger: logging.Logger, operation: str, duration: float, **metrics):
  """
  Log performance data with structured metrics.
  
  Records performance information for operations including timing,
  throughput, and custom metrics.
  
  Args:
      logger: The logger instance to use.
      operation: Name of the operation measured.
      duration: Duration in seconds.
      **metrics: Additional performance metrics.
  """
  metric_items = [f"duration={duration:.2f}s"]
  
  # Add standard metrics with formatting
  for key, value in metrics.items():
    if key in ['memory_usage', 'memory_delta']:
      # Format memory values in MB
      metric_items.append(f"{key}={value/1024/1024:.1f}MB")
    elif key in ['throughput', 'rate']:
      # Format rates
      metric_items.append(f"{key}={value:.2f}/s")
    elif key in ['size', 'count', 'total']:
      # Format counts
      metric_items.append(f"{key}={value:,}")
    else:
      # Generic formatting
      metric_items.append(f"{key}={value}")
  
  metrics_str = " | ".join(metric_items)
  logger.info(f"Performance [{operation}]: {metrics_str}")

def log_file_operation(logger: logging.Logger, operation: str, file_path: str, **kwargs):
  """
  Log file operations with structured metadata.
  
  Args:
      logger: The logger instance to use.
      operation: Type of file operation (read, write, process, etc.).
      file_path: Path to the file being operated on.
      **kwargs: Additional context like file_size, duration, etc.
  """
  try:
    # Safely gather file metadata
    file_exists = os.path.exists(file_path)
    file_size = os.path.getsize(file_path) if file_exists else 0
    file_ext = os.path.splitext(file_path)[1]
    file_name = os.path.basename(file_path)
    
    # Build structured context for JSON logging
    extra_fields = {
      'operation': operation,
      'file_path': file_path,
      'file_name': file_name,
      'file_extension': file_ext,
      'file_exists': file_exists,
      **kwargs
    }
    
    if file_exists:
      extra_fields['file_size'] = file_size
    
    # Format message for text logging
    context_items = []
    for key, value in extra_fields.items():
      if key == 'file_path':
        continue  # Don't include full path in text format
      elif key == 'file_size':
        context_items.append(f"{key}={value:,}bytes")
      else:
        context_items.append(f"{key}={value}")
    
    context_str = " | ".join(context_items)
    
    # Log with both text format and structured data
    logger.info(f"File {operation}: {file_name} | {context_str}", extra=extra_fields)
    
  except Exception as e:
    # Fallback to simple logging if metadata gathering fails
    logger.warning(f"Failed to log file operation metadata: {e}")
    logger.info(f"File {operation}: {os.path.basename(file_path)}", 
               extra={'operation': operation, 'file_path': file_path})

def log_model_operation(logger: logging.Logger, operation: str, model: str, **kwargs):
  """
  Log AI model operations with metadata.
  
  Args:
      logger: The logger instance to use.
      operation: Type of model operation (embedding, query, etc.).
      model: Name of the AI model being used.
      **kwargs: Additional context like token_count, batch_size, etc.
  """
  # Build structured context for JSON logging
  extra_fields = {
    'operation': operation,
    'model': model,
    'timestamp': int(time.time()),
    **kwargs
  }
  
  # Format context for text logging with appropriate units
  context_items = []
  for key, value in extra_fields.items():
    if key in ['token_count', 'max_tokens']:
      context_items.append(f"{key}={value:,}tokens")
    elif key == 'temperature':
      context_items.append(f"{key}={value:.2f}")
    elif key == 'timestamp':
      continue  # Skip timestamp in display
    else:
      context_items.append(f"{key}={value}")
  
  context_str = " | ".join(context_items)
  
  # Log with both text format and structured data
  logger.info(f"Model {operation}: {model} | {context_str}", extra=extra_fields)

class OperationLogger:
  """
  Context manager for logging operations with timing and metrics.
  
  Automatically logs operation start, measures duration, tracks memory usage,
  and logs completion or failure with detailed context.
  """
  
  def __init__(self, logger: logging.Logger, operation: str, **initial_context):
    """
    Initialize operation logger.
    
    Args:
        logger: The logger instance to use.
        operation: Name of the operation being logged.
        **initial_context: Initial context for the operation.
    """
    self.logger = logger
    self.operation = operation
    self.context = initial_context.copy()
    self.start_time = None
    self.start_memory = None
  
  def __enter__(self):
    """Start the operation and begin timing/monitoring."""
    self.start_time = time.time()
    
    # Track memory usage if psutil is available
    try:
      process = psutil.Process()
      self.start_memory = process.memory_info().rss
    except (ImportError, psutil.NoSuchProcess):
      self.start_memory = None
    
    # Log operation start
    if self.context:
      context_str = " | ".join(f"{k}={v}" for k, v in self.context.items())
      self.logger.debug(f"Starting {self.operation} | {context_str}")
    else:
      self.logger.debug(f"Starting {self.operation}")
    
    return self
  
  def __exit__(self, exc_type, exc_val, exc_tb):
    """Complete the operation and log results."""
    duration = time.time() - self.start_time if self.start_time else 0
    
    # Calculate memory usage delta
    memory_delta = None
    current_memory = None
    if self.start_memory:
      try:
        process = psutil.Process()
        current_memory = process.memory_info().rss
        memory_delta = current_memory - self.start_memory
      except (ImportError, psutil.NoSuchProcess):
        pass
    
    # Prepare metrics
    metrics = {'duration': duration, **self.context}
    if current_memory:
      metrics['memory_usage'] = current_memory
    if memory_delta:
      metrics['memory_delta'] = memory_delta
    
    if exc_type:
      # Operation failed
      log_operation_error(
        self.logger, self.operation, exc_val, 
        duration=duration, **self.context
      )
    else:
      # Operation succeeded
      log_performance_metrics(
        self.logger, self.operation, duration, **self.context
      )
  
  def add_context(self, **kwargs):
    """Add additional context during operation."""
    self.context.update(kwargs)
  
  def checkpoint(self, message: str, **metrics):
    """Log a checkpoint during the operation."""
    if self.start_time:
      elapsed = time.time() - self.start_time
      metric_str = " | ".join(f"{k}={v}" for k, v in metrics.items()) if metrics else ""
      checkpoint_msg = f"{self.operation} checkpoint: {message} (elapsed: {elapsed:.2f}s)"
      if metric_str:
        checkpoint_msg += f" | {metric_str}"
      self.logger.debug(checkpoint_msg)

#fin
