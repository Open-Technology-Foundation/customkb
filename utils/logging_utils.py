#!/usr/bin/env python
"""
Logging utilities for CustomKB.
Provides consistent logging with color support, time tracking, and progress reporting.
"""

import logging
import colorlog
import shutil
import time
from typing import Optional

# Global logger
logger = None

def get_logger(name: str) -> logging.Logger:
  """
  Get a logger with the specified name.

  Args:
      name: The name of the logger.

  Returns:
      A configured logger instance.
  """
  global logger
  if logger is None:
    logger = logging.getLogger(name)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(name)s: %(message)s'))
    logger.addHandler(handler)
    logger.setLevel(logging.WARNING)
  return logger

def setup_logging(verbose: bool, debug: bool = False) -> logging.Logger:
  """
  Set up logging configuration with color formatting.

  Args:
      verbose: Whether to enable verbose logging (INFO level).
      debug: Whether to enable debug logging (DEBUG level).

  Returns:
      A configured logger instance.
  """
  global logger
  logger = logging.getLogger()

  # Remove existing handlers
  for handler in logger.handlers[:]:
    logger.removeHandler(handler)

  handler = colorlog.StreamHandler()

  if verbose or debug:
    if debug:
      logger.setLevel(logging.DEBUG)
    else:
      logger.setLevel(logging.INFO)
    logformat = f"%(log_color)s%(module)s:%(levelname)s: %(message)s"
  else:
    logger.setLevel(logging.ERROR)
    logformat = f"%(log_color)s%(module)s:%(levelname)s: %(message)s"

  formatter = colorlog.ColoredFormatter(
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

  handler.setFormatter(formatter)
  logger.addHandler(handler)

  return logger

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

#fin
