#!/usr/bin/env python
"""
Context managers for CustomKB.

This module provides reusable context managers for common operations
like database connections, file handling, and resource management.
"""

import os
import sqlite3
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Tuple, Any, Generator, Union

from utils.logging_config import get_logger
from utils.exceptions import (
  DatabaseError,
  ConnectionError as CustomConnectionError,
  FileSystemError,
  PermissionError as CustomPermissionError,
  ResourceError,
  TemporaryError,
)

logger = get_logger(__name__)


@contextmanager
def database_connection(
  db_path: str,
  timeout: float = 30.0,
  check_same_thread: bool = False,
  read_only: bool = False
) -> Generator[Tuple[sqlite3.Connection, sqlite3.Cursor], None, None]:
  """
  Context manager for SQLite database connections.
  
  Ensures proper connection handling and cleanup.
  
  Args:
      db_path: Path to the database file
      timeout: Connection timeout in seconds
      check_same_thread: Whether to check same thread (default False for multi-threading)
      read_only: Open database in read-only mode
      
  Yields:
      Tuple of (connection, cursor)
      
  Raises:
      ConnectionError: If connection fails
      DatabaseError: If database operation fails
  """
  conn = None
  cursor = None
  
  try:
    # Validate database path
    db_file = Path(db_path)
    if not db_file.exists():
      raise CustomConnectionError(f"Database not found: {db_path}")
    
    # Open connection with appropriate mode
    if read_only:
      uri = f"file:{db_path}?mode=ro"
      conn = sqlite3.connect(uri, uri=True, timeout=timeout, check_same_thread=check_same_thread)
    else:
      conn = sqlite3.connect(db_path, timeout=timeout, check_same_thread=check_same_thread)
    
    # Set pragmas for performance
    cursor = conn.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA synchronous=NORMAL")
    cursor.execute("PRAGMA cache_size=-64000")  # 64MB cache
    cursor.execute("PRAGMA temp_store=MEMORY")
    
    logger.debug(f"Database connection established: {db_path}")
    
    yield conn, cursor
    
    # Commit any pending transactions
    if not read_only:
      conn.commit()
    
  except sqlite3.Error as e:
    logger.error(f"Database error: {e}")
    if conn:
      conn.rollback()
    raise DatabaseError(f"Database operation failed: {e}") from e
  
  except Exception as e:
    logger.error(f"Unexpected error in database connection: {e}")
    raise CustomConnectionError(f"Failed to connect to database: {e}") from e
  
  finally:
    # Clean up resources
    if cursor:
      cursor.close()
    if conn:
      conn.close()
    logger.debug(f"Database connection closed: {db_path}")


@contextmanager
def atomic_write(
  filepath: Union[str, Path],
  mode: str = 'w',
  encoding: str = 'utf-8',
  create_dirs: bool = True
) -> Generator[Any, None, None]:
  """
  Context manager for atomic file writes.
  
  Writes to a temporary file and moves it to the target location
  only if the write succeeds, preventing partial writes.
  
  Args:
      filepath: Target file path
      mode: File open mode ('w' or 'wb')
      encoding: Text encoding (ignored for binary mode)
      create_dirs: Create parent directories if they don't exist
      
  Yields:
      File handle for writing
      
  Raises:
      FileSystemError: If file operation fails
      PermissionError: If lacking write permissions
  """
  filepath = Path(filepath)
  temp_file = None
  
  try:
    # Create parent directories if needed
    if create_dirs:
      filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Create temporary file in same directory (for atomic rename)
    temp_fd, temp_path = tempfile.mkstemp(
      dir=filepath.parent,
      prefix=f'.{filepath.name}.',
      suffix='.tmp'
    )
    
    # Open temp file with appropriate mode
    if 'b' in mode:
      temp_file = os.fdopen(temp_fd, mode)
    else:
      temp_file = os.fdopen(temp_fd, mode, encoding=encoding)
    
    logger.debug(f"Writing to temporary file: {temp_path}")
    
    yield temp_file
    
    # Close and move to final location
    temp_file.close()
    temp_file = None
    
    # Atomic rename (on same filesystem)
    Path(temp_path).replace(filepath)
    logger.debug(f"Atomic write completed: {filepath}")
    
  except PermissionError as e:
    logger.error(f"Permission denied writing to {filepath}: {e}")
    raise CustomPermissionError(str(filepath), "write") from e
  
  except Exception as e:
    logger.error(f"Failed to write file {filepath}: {e}")
    # Clean up temp file on error
    if temp_file:
      temp_file.close()
    if temp_path and Path(temp_path).exists():
      Path(temp_path).unlink()
    raise FileSystemError(f"Failed to write file: {e}") from e
  
  finally:
    # Ensure temp file is closed
    if temp_file:
      temp_file.close()


@contextmanager
def timed_operation(
  operation_name: str,
  timeout: Optional[float] = None,
  logger_instance: Optional[Any] = None
) -> Generator[dict, None, None]:
  """
  Context manager for timing operations.
  
  Tracks operation duration and logs performance metrics.
  
  Args:
      operation_name: Name of the operation for logging
      timeout: Optional timeout in seconds
      logger_instance: Logger to use (defaults to module logger)
      
  Yields:
      Dictionary to store operation results/metrics
      
  Raises:
      TimeoutError: If operation exceeds timeout
  """
  if logger_instance is None:
    logger_instance = logger
  
  start_time = time.time()
  metrics = {'operation': operation_name, 'start_time': start_time}
  
  logger_instance.debug(f"Starting operation: {operation_name}")
  
  try:
    yield metrics
    
    # Calculate duration
    duration = time.time() - start_time
    metrics['duration'] = duration
    metrics['success'] = True
    
    # Check timeout
    if timeout and duration > timeout:
      raise TimeoutError(f"Operation '{operation_name}' exceeded timeout: {duration:.2f}s > {timeout}s")
    
    logger_instance.info(f"Operation '{operation_name}' completed in {duration:.2f}s")
    
  except Exception as e:
    duration = time.time() - start_time
    metrics['duration'] = duration
    metrics['success'] = False
    metrics['error'] = str(e)
    
    logger_instance.error(f"Operation '{operation_name}' failed after {duration:.2f}s: {e}")
    raise
  
  finally:
    # Log final metrics
    if logger_instance.isEnabledFor(logger.DEBUG):
      logger_instance.debug(f"Metrics for '{operation_name}': {metrics}")


@contextmanager
def resource_limit(
  max_memory_mb: Optional[int] = None,
  max_cpu_percent: Optional[float] = None
) -> Generator[None, None, None]:
  """
  Context manager for resource limiting.
  
  Monitors resource usage and raises errors if limits are exceeded.
  
  Args:
      max_memory_mb: Maximum memory usage in MB
      max_cpu_percent: Maximum CPU usage percentage
      
  Yields:
      None
      
  Raises:
      ResourceError: If resource limits are exceeded
  """
  try:
    import psutil
    process = psutil.Process()
    
    # Get initial resource usage
    initial_memory = process.memory_info().rss / 1024 / 1024
    initial_cpu = process.cpu_percent(interval=0.1)
    
    logger.debug(f"Resource limits: memory={max_memory_mb}MB, cpu={max_cpu_percent}%")
    
    yield
    
    # Check final resource usage
    final_memory = process.memory_info().rss / 1024 / 1024
    final_cpu = process.cpu_percent(interval=0.1)
    
    memory_used = final_memory - initial_memory
    
    # Check limits
    if max_memory_mb and final_memory > max_memory_mb:
      raise ResourceError(f"Memory limit exceeded: {final_memory:.1f}MB > {max_memory_mb}MB")
    
    if max_cpu_percent and final_cpu > max_cpu_percent:
      raise ResourceError(f"CPU limit exceeded: {final_cpu:.1f}% > {max_cpu_percent}%")
    
    logger.debug(f"Resource usage: memory_delta={memory_used:.1f}MB, cpu={final_cpu:.1f}%")
    
  except ImportError:
    logger.warning("psutil not available, resource limiting disabled")
    yield
  except Exception as e:
    logger.error(f"Resource monitoring error: {e}")
    raise


@contextmanager
def retry_on_error(
  max_retries: int = 3,
  delay: float = 1.0,
  backoff: float = 2.0,
  exceptions: tuple = (TemporaryError, TimeoutError, ConnectionError)
) -> Generator[dict, None, None]:
  """
  Context manager for automatic retry logic.
  
  Retries operations that fail with specific exceptions.
  
  Args:
      max_retries: Maximum number of retry attempts
      delay: Initial delay between retries in seconds
      backoff: Backoff multiplier for delay
      exceptions: Tuple of exceptions to retry on
      
  Yields:
      Dictionary with retry metadata
      
  Raises:
      The last exception if all retries fail
  """
  retry_info = {
    'attempt': 0,
    'max_retries': max_retries,
    'delays': []
  }
  
  last_exception = None
  current_delay = delay
  
  for attempt in range(max_retries + 1):
    retry_info['attempt'] = attempt + 1
    
    try:
      if attempt > 0:
        logger.info(f"Retry attempt {attempt}/{max_retries} after {current_delay:.1f}s delay")
        time.sleep(current_delay)
        retry_info['delays'].append(current_delay)
        current_delay *= backoff
      
      yield retry_info
      
      # Success - exit
      return
      
    except exceptions as e:
      last_exception = e
      if attempt < max_retries:
        logger.warning(f"Retryable error (attempt {attempt + 1}/{max_retries + 1}): {e}")
        continue
      else:
        logger.error(f"All retry attempts failed: {e}")
        raise
    
    except Exception as e:
      # Non-retryable error
      logger.error(f"Non-retryable error: {e}")
      raise
  
  # Should not reach here, but just in case
  if last_exception:
    raise last_exception


@contextmanager
def batch_processor(
  items: list,
  batch_size: int = 100,
  progress_callback: Optional[callable] = None
) -> Generator[list, None, None]:
  """
  Context manager for batch processing.
  
  Processes items in batches with optional progress tracking.
  
  Args:
      items: List of items to process
      batch_size: Number of items per batch
      progress_callback: Optional callback for progress updates
      
  Yields:
      Current batch of items
  """
  total_items = len(items)
  processed = 0
  
  logger.info(f"Starting batch processing: {total_items} items in batches of {batch_size}")
  
  try:
    for i in range(0, total_items, batch_size):
      batch = items[i:i + batch_size]
      batch_num = (i // batch_size) + 1
      total_batches = (total_items + batch_size - 1) // batch_size
      
      logger.debug(f"Processing batch {batch_num}/{total_batches} ({len(batch)} items)")
      
      yield batch
      
      processed += len(batch)
      
      # Call progress callback if provided
      if progress_callback:
        progress_callback(processed, total_items)
      
      # Log progress at intervals
      if processed % (batch_size * 10) == 0 or processed == total_items:
        progress_pct = (processed / total_items) * 100
        logger.info(f"Progress: {processed}/{total_items} ({progress_pct:.1f}%)")
    
    logger.info(f"Batch processing completed: {processed} items processed")
    
  except Exception as e:
    logger.error(f"Batch processing failed at item {processed}: {e}")
    raise


@contextmanager
def safe_import(module_name: str, package: Optional[str] = None) -> Generator[Any, None, None]:
  """
  Context manager for safe module imports.
  
  Handles import errors gracefully and provides fallbacks.
  
  Args:
      module_name: Name of module to import
      package: Optional package name for relative imports
      
  Yields:
      Imported module or None if import fails
  """
  import importlib
  
  module = None
  
  try:
    if package:
      module = importlib.import_module(module_name, package)
    else:
      module = importlib.import_module(module_name)
    
    logger.debug(f"Successfully imported module: {module_name}")
    yield module
    
  except ImportError as e:
    logger.warning(f"Failed to import module '{module_name}': {e}")
    yield None
  
  except Exception as e:
    logger.error(f"Unexpected error importing '{module_name}': {e}")
    yield None


# Example usage functions
def example_database_usage():
  """Example of using database_connection context manager."""
  with database_connection('/path/to/db.sqlite') as (conn, cursor):
    cursor.execute("SELECT COUNT(*) FROM table")
    count = cursor.fetchone()[0]
    print(f"Row count: {count}")


def example_atomic_write_usage():
  """Example of using atomic_write context manager."""
  with atomic_write('/path/to/file.txt') as f:
    f.write("This write is atomic!\n")
    f.write("Either all of this is written, or none of it is.\n")


def example_retry_usage():
  """Example of using retry_on_error context manager."""
  with retry_on_error(max_retries=3, delay=1.0) as retry_info:
    print(f"Attempt {retry_info['attempt']}")
    # Potentially failing operation here
    if retry_info['attempt'] < 3:
      raise TemporaryError("Simulated temporary failure")


#fin