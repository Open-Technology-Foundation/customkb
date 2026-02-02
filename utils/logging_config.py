#!/usr/bin/env python
"""Centralized logging for CustomKB: setup, time tracking, operation logging."""
import configparser
import logging
import logging.handlers
import os
import shutil
import time
from typing import Any

import colorlog


def get_kb_info_from_config(config_file: str) -> tuple[str, str]:
  """Extract (kb_directory, kb_name) from config file path."""
  from utils.text_utils import split_filepath
  if not config_file:
    raise ValueError("Config file path cannot be empty")
  directory, basename, _, _ = split_filepath(config_file)
  return directory, basename[:-4] if basename.endswith('.cfg') else basename

def get_log_file_path(cfg_val: str, kb_dir: str, kb_name: str) -> str:
  """Resolve log file path from config ('auto', relative, or absolute)."""
  if cfg_val == 'auto':
    d = os.path.join(kb_dir, 'logs')
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f'{kb_name}.log')
  path = cfg_val if os.path.isabs(cfg_val) else os.path.join(kb_dir, cfg_val)
  d = os.path.dirname(path)
  if d:
    os.makedirs(d, exist_ok=True)
  return path

def load_logging_config(config_file: str | None = None,
                        kb_directory: str | None = None) -> dict[str, Any]:
  """Load logging config from .cfg [LOGGING] section and env vars."""
  d: dict[str, Any] = {'log_level': 'INFO', 'log_file': 'auto', 'max_log_size': 10485760,
    'backup_count': 5, 'console_colors': True, 'file_logging': True,
    'json_format': False, 'module_levels': {}}
  if config_file and os.path.exists(config_file):
    try:
      cp = configparser.ConfigParser()
      cp.read(config_file)
      if 'LOGGING' in cp:
        s = cp['LOGGING']
        for key in d:
          if key == 'module_levels':
            lvls = {o[6:]: s[o] for o in s if o.startswith('level_')}
            if lvls:
              d[key] = lvls
          elif key in s:
            try:
              if isinstance(d[key], bool):
                d[key] = s.getboolean(key)
              elif isinstance(d[key], int):
                d[key] = s.getint(key)
              else:
                d[key] = s.get(key)
            except ValueError:
              pass
    except (FileNotFoundError, OSError, ValueError, configparser.Error) as e:
      logging.warning(f"Failed to load logging config from {config_file}: {e}")
  for ev, ck in {'LOGGING_LEVEL': 'log_level', 'LOGGING_FILE': 'log_file',
      'LOGGING_MAX_SIZE': 'max_log_size', 'LOGGING_BACKUP_COUNT': 'backup_count',
      'LOGGING_CONSOLE_COLORS': 'console_colors', 'LOGGING_FILE_ENABLED': 'file_logging',
      'LOGGING_JSON_FORMAT': 'json_format'}.items():
    v = os.getenv(ev)
    if v is not None:
      try:
        if isinstance(d[ck], bool):
          d[ck] = v.lower() in ('true', '1', 'yes', 'on')
        elif isinstance(d[ck], int):
          d[ck] = int(v)
        else:
          d[ck] = v
      except ValueError:
        pass
  return d

def get_logger(name: str) -> logging.Logger:
  """Get a module-specific logger (typically called with __name__)."""
  return logging.getLogger(name)

def setup_logging(verbose: bool, debug: bool = False, log_file: str | None = None,
                  log_to_file: bool = True, config_file: str | None = None,
                  kb_directory: str | None = None, kb_name: str | None = None) -> logging.Logger | None:
  """Set up logging with console and optional rotating file output."""
  if not kb_directory or not kb_name:
    return None
  cfg = load_logging_config(config_file, kb_directory)
  root = logging.getLogger()
  for h in root.handlers[:]:
    root.removeHandler(h)
  level = logging.DEBUG if debug else logging.INFO if verbose else logging.WARNING
  root.setLevel(level)
  con = colorlog.StreamHandler()
  if cfg.get('console_colors', True):
    con.setFormatter(colorlog.ColoredFormatter("%(log_color)s%(module)s:%(levelname)s: %(message)s",
      reset=True, log_colors={'DEBUG':'cyan','INFO':'green','WARNING':'yellow',
                               'ERROR':'red','CRITICAL':'red,bg_white'}))
  else:
    con.setFormatter(logging.Formatter('%(name)s:%(levelname)s: %(message)s'))
  root.addHandler(con)
  if log_to_file and cfg.get('file_logging', True):
    if not log_file:
      log_file = get_log_file_path(cfg.get('log_file', 'auto'), kb_directory, kb_name)
    try:
      fh = logging.handlers.RotatingFileHandler(log_file,
        maxBytes=cfg.get('max_log_size', 10485760), backupCount=cfg.get('backup_count', 5), encoding='utf-8')
      fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
      root.addHandler(fh)
      if verbose or debug:
        root.info(f"Logging to file: {log_file}")
    except (OSError, PermissionError) as e:
      root.warning(f"Failed to setup file logging to {log_file}: {e}")
  for mod, ls in cfg.get('module_levels', {}).items():
    try:
      logging.getLogger(mod).setLevel(getattr(logging, ls.upper()))
    except AttributeError:
      logging.warning(f"Invalid log level '{ls}' for module '{mod}'")
  return root

def dashes(offset: int = 0, char: str = '-') -> str:
  """Generate a line of characters matching terminal width."""
  width, _ = shutil.get_terminal_size()
  return char[0] * (width - offset)

def elapsed_time(start_time: int, end_time: int | None = None) -> str:
  """Format elapsed time as '1d 02h 03m 04s'."""
  if end_time is None:
    end_time = int(time.time())
  el = int(end_time - start_time)
  result = ''
  for unit, secs in [('d', 86400), ('h', 3600), ('m', 60)]:
    n = el // secs
    if n:
      result += f"{n:02}{unit} " if unit != 'd' else f"{n}{unit} "
      el -= n * secs
  return result + f"{el:02}s"

def time_to_finish(start_time: int, records_processed: int, total_records: int) -> str:
  """Estimate remaining time based on processing rate."""
  elapsed = int(time.time()) - start_time
  if records_processed == 0 or elapsed == 0:
    return ""
  return elapsed_time(0, int((total_records - records_processed) / (records_processed / elapsed)))

def log_operation_error(logger: logging.Logger, operation: str, error: Exception, **context):
  """Log operation errors with context, masking sensitive values."""
  from utils.security_utils import mask_sensitive_data
  sensitive = ('key', 'token', 'password', 'secret')
  items = [f"{k}={mask_sensitive_data(str(v)) if any(s in k.lower() for s in sensitive) else v}"
           for k, v in context.items()]
  msg = f"Operation '{operation}' failed: {error}"
  logger.error(f"{msg} | Context: {' | '.join(items)}" if items else msg)

def log_performance_metrics(logger: logging.Logger, operation: str, duration: float, **metrics):
  """Log performance metrics with formatting for memory/rate/count values."""
  items = [f"duration={duration:.2f}s"]
  for k, v in metrics.items():
    if k in ('memory_usage', 'memory_delta'):
      items.append(f"{k}={v/1024/1024:.1f}MB")
    elif k in ('throughput', 'rate'):
      items.append(f"{k}={v:.2f}/s")
    elif k in ('size', 'count', 'total'):
      items.append(f"{k}={v:,}")
    else:
      items.append(f"{k}={v}")
  logger.info(f"Performance [{operation}]: {' | '.join(items)}")

def log_file_operation(logger: logging.Logger, operation: str, file_path: str, **kwargs):
  """Log file operations with metadata."""
  try:
    exists, name = os.path.exists(file_path), os.path.basename(file_path)
    extra = {'operation': operation, 'file_path': file_path, 'file_name': name,
             'file_extension': os.path.splitext(file_path)[1], 'file_exists': exists, **kwargs}
    if exists:
      extra['file_size'] = os.path.getsize(file_path)
    parts = [f"{k}={v:,}bytes" if k == 'file_size' else f"{k}={v}"
             for k, v in extra.items() if k != 'file_path']
    logger.info(f"File {operation}: {name} | {' | '.join(parts)}", extra=extra)
  except (FileNotFoundError, OSError, PermissionError) as e:
    logger.warning(f"Failed to log file operation metadata: {e}")
    logger.info(f"File {operation}: {os.path.basename(file_path)}",
                extra={'operation': operation, 'file_path': file_path})

def log_model_operation(logger: logging.Logger, operation: str, model: str, **kwargs):
  """Log AI model operations with metadata."""
  extra = {'operation': operation, 'model': model, **kwargs}
  def fmt(k, v):
    return (f"{k}={v:,}tokens" if k in ('token_count', 'max_tokens')
      else f"{k}={v:.2f}" if k == 'temperature' else f"{k}={v}")
  parts = [fmt(k, v) for k, v in extra.items()]
  logger.info(f"Model {operation}: {model} | {' | '.join(parts)}", extra=extra)

class OperationLogger:
  """Context manager for logging operations with timing."""
  def __init__(self, logger: logging.Logger, operation: str, **initial_context):
    self.logger, self.operation = logger, operation
    self.context = initial_context.copy()
    self.start_time: float | None = None
  def __enter__(self):
    self.start_time = time.time()
    ctx = " | ".join(f"{k}={v}" for k, v in self.context.items())
    self.logger.debug(f"Starting {self.operation} | {ctx}" if ctx else f"Starting {self.operation}")
    return self
  def __exit__(self, exc_type, exc_val, exc_tb):
    dur = time.time() - self.start_time if self.start_time else 0
    if exc_type:
      log_operation_error(self.logger, self.operation, exc_val, duration=dur, **self.context)
    else:
      log_performance_metrics(self.logger, self.operation, dur, **self.context)
  def add_context(self, **kwargs):
    """Add context during operation."""
    self.context.update(kwargs)
  def checkpoint(self, message: str, **metrics):
    """Log a checkpoint during the operation."""
    if self.start_time:
      el = time.time() - self.start_time
      parts = " | ".join(f"{k}={v}" for k, v in metrics.items()) if metrics else ""
      msg = f"{self.operation} checkpoint: {message} (elapsed: {el:.2f}s)"
      self.logger.debug(f"{msg} | {parts}" if parts else msg)

# Silence verbose third-party loggers on import
for _lib in ('urllib3', 'requests', 'httpx', 'openai', 'anthropic', 'faiss',
             'sentence_transformers', 'transformers', 'torch', 'matplotlib',
             'PIL', 'spacy', 'nltk'):
  logging.getLogger(_lib).setLevel(logging.WARNING)

#fin
