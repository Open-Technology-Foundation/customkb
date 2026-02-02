"""
Resource management and monitoring for CustomKB.

This module provides functionality to monitor and limit system resource usage
to prevent crashes and ensure stable operation.
"""

import resource
import signal
import sys
import threading
import time
from collections.abc import Callable
from contextlib import contextmanager

import psutil

from utils.logging_config import get_logger

logger = get_logger(__name__)


class ResourceMonitor:
  """
  Monitor and control system resource usage.

  Features:
  - Memory usage monitoring and limits
  - Process count limits
  - File descriptor limits
  - Automatic cleanup when approaching limits
  """

  def __init__(self, max_memory_gb: float = None, max_memory_percent: float = 80.0):
    """
    Initialize resource monitor.

    Args:
        max_memory_gb: Maximum memory usage in GB (None = no limit)
        max_memory_percent: Maximum percentage of system memory to use
    """
    self.max_memory_gb = max_memory_gb
    self.max_memory_percent = max_memory_percent
    self.process = psutil.Process()
    self._monitoring = False
    self._monitor_thread = None
    self._callbacks = []

    # Track initial state
    self.initial_memory = self.get_memory_info()

  def get_memory_info(self) -> dict[str, float]:
    """Get current memory usage information."""
    mem_info = self.process.memory_info()
    system_mem = psutil.virtual_memory()

    return {
      'rss_mb': mem_info.rss / 1024 / 1024,
      'vms_mb': mem_info.vms / 1024 / 1024,
      'percent': self.process.memory_percent(),
      'system_percent': system_mem.percent,
      'system_available_mb': system_mem.available / 1024 / 1024,
      'system_total_mb': system_mem.total / 1024 / 1024
    }

  def check_memory_limits(self) -> tuple[bool, str]:
    """
    Check if memory usage is within limits.

    Returns:
        Tuple of (is_within_limits, message)
    """
    info = self.get_memory_info()

    # Check absolute memory limit
    if self.max_memory_gb and info['rss_mb'] > self.max_memory_gb * 1024:
      return False, f"Memory limit exceeded: {info['rss_mb']:.1f}MB > {self.max_memory_gb * 1024}MB"

    # Check percentage limit
    if info['system_percent'] > self.max_memory_percent:
      return False, f"System memory usage too high: {info['system_percent']:.1f}% > {self.max_memory_percent}%"

    # Check if system is running low on memory
    if info['system_available_mb'] < 500:  # Less than 500MB available
      return False, f"System memory critically low: {info['system_available_mb']:.1f}MB available"

    return True, f"Memory OK: {info['rss_mb']:.1f}MB ({info['percent']:.1f}%)"

  def register_cleanup_callback(self, callback: Callable[[], None]):
    """Register a callback to be called when approaching limits."""
    self._callbacks.append(callback)

  def _trigger_cleanup(self):
    """Trigger all registered cleanup callbacks."""
    logger.warning("Triggering resource cleanup callbacks")
    for callback in self._callbacks:
      try:
        callback()
      except Exception as e:
        logger.error(f"Cleanup callback failed: {e}")

  def start_monitoring(self, interval: float = 5.0, warning_threshold: float = 0.9):
    """
    Start background monitoring thread.

    Args:
        interval: Check interval in seconds
        warning_threshold: Trigger cleanup at this fraction of limit
    """
    if self._monitoring:
      return

    self._monitoring = True

    def monitor_loop():
      while self._monitoring:
        try:
          info = self.get_memory_info()

          # Check if approaching limits
          if self.max_memory_gb:
            usage_fraction = info['rss_mb'] / (self.max_memory_gb * 1024)
            if usage_fraction > warning_threshold:
              logger.warning(f"Approaching memory limit: {usage_fraction:.1%}")
              self._trigger_cleanup()

          if info['system_percent'] > self.max_memory_percent * warning_threshold:
            logger.warning(f"System memory usage high: {info['system_percent']:.1f}%")
            self._trigger_cleanup()

        except Exception as e:
          logger.error(f"Error in resource monitor: {e}")

        time.sleep(interval)

    self._monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
    self._monitor_thread.start()
    logger.info("Resource monitoring started")

  def stop_monitoring(self):
    """Stop background monitoring."""
    self._monitoring = False
    if self._monitor_thread:
      self._monitor_thread.join(timeout=5)
    logger.info("Resource monitoring stopped")


@contextmanager
def memory_limit(limit_mb: int):
  """
  Context manager to set memory limits for a block of code.

  Note: This uses resource.setrlimit which may not work on all systems.

  Args:
      limit_mb: Memory limit in megabytes

  Example:
      >>> with memory_limit(500):  # 500MB limit
      ...     process_large_data()
  """
  if sys.platform == 'win32':
    logger.warning("Memory limits not supported on Windows")
    yield
    return

  # Get current limit
  soft, hard = resource.getrlimit(resource.RLIMIT_AS)

  try:
    # Set new limit
    new_limit = limit_mb * 1024 * 1024  # Convert to bytes
    resource.setrlimit(resource.RLIMIT_AS, (new_limit, hard))
    logger.debug(f"Set memory limit to {limit_mb}MB")
    yield
  finally:
    # Restore original limit
    resource.setrlimit(resource.RLIMIT_AS, (soft, hard))


@contextmanager
def resource_limited(memory_mb: int | None = None,
                    cpu_seconds: int | None = None,
                    file_descriptors: int | None = None):
  """
  Context manager for multiple resource limits.

  Args:
      memory_mb: Memory limit in MB
      cpu_seconds: CPU time limit in seconds
      file_descriptors: Maximum number of open files

  Example:
      >>> with resource_limited(memory_mb=1000, cpu_seconds=60):
      ...     run_intensive_task()
  """
  if sys.platform == 'win32':
    logger.warning("Resource limits not fully supported on Windows")
    yield
    return

  # Store original limits
  original_limits = {}

  try:
    # Set memory limit
    if memory_mb:
      original_limits[resource.RLIMIT_AS] = resource.getrlimit(resource.RLIMIT_AS)
      resource.setrlimit(resource.RLIMIT_AS,
                        (memory_mb * 1024 * 1024, original_limits[resource.RLIMIT_AS][1]))

    # Set CPU time limit
    if cpu_seconds:
      original_limits[resource.RLIMIT_CPU] = resource.getrlimit(resource.RLIMIT_CPU)
      resource.setrlimit(resource.RLIMIT_CPU,
                        (cpu_seconds, original_limits[resource.RLIMIT_CPU][1]))

    # Set file descriptor limit
    if file_descriptors:
      original_limits[resource.RLIMIT_NOFILE] = resource.getrlimit(resource.RLIMIT_NOFILE)
      resource.setrlimit(resource.RLIMIT_NOFILE,
                        (file_descriptors, original_limits[resource.RLIMIT_NOFILE][1]))

    yield

  finally:
    # Restore original limits
    for limit_type, (soft, hard) in original_limits.items():
      try:
        resource.setrlimit(limit_type, (soft, hard))
      except (ValueError, OSError) as e:
        logger.error(f"Failed to restore limit {limit_type}: {e}")


class ResourceGuard:
  """
  Guard against resource exhaustion with automatic cleanup.
  """

  def __init__(self, memory_limit_gb: float = None):
    """
    Initialize resource guard.

    Args:
        memory_limit_gb: Memory limit in GB (None = 80% of system memory)
    """
    if memory_limit_gb is None:
      # Default to 80% of system memory
      total_gb = psutil.virtual_memory().total / 1024 / 1024 / 1024
      memory_limit_gb = total_gb * 0.8

    self.monitor = ResourceMonitor(max_memory_gb=memory_limit_gb)
    self._cleanup_handlers = []

  def register_cleanup(self, name: str, handler: Callable[[], None]):
    """Register a cleanup handler."""
    self._cleanup_handlers.append((name, handler))
    self.monitor.register_cleanup_callback(handler)

  def check_resources(self) -> bool:
    """
    Check if resources are OK.

    Returns:
        True if resources are within limits
    """
    ok, msg = self.monitor.check_memory_limits()
    if not ok:
      logger.warning(f"Resource check failed: {msg}")
    return ok

  def force_cleanup(self):
    """Force all cleanup handlers to run."""
    logger.info("Forcing resource cleanup")
    for name, handler in self._cleanup_handlers:
      try:
        logger.debug(f"Running cleanup: {name}")
        handler()
      except Exception as e:
        logger.error(f"Cleanup handler '{name}' failed: {e}")

  @contextmanager
  def guarded_operation(self, operation_name: str):
    """
    Context manager for operations that need resource guarding.

    Args:
        operation_name: Name of the operation for logging

    Example:
        >>> guard = ResourceGuard(memory_limit_gb=4)
        >>> with guard.guarded_operation("large_data_processing"):
        ...     process_data()
    """
    logger.debug(f"Starting guarded operation: {operation_name}")

    # Check resources before starting
    if not self.check_resources():
      self.force_cleanup()
      if not self.check_resources():
        raise MemoryError(f"Insufficient resources for {operation_name}")

    # Start monitoring
    self.monitor.start_monitoring(interval=2.0)

    try:
      yield
    finally:
      # Stop monitoring
      self.monitor.stop_monitoring()

      # Log final resource usage
      info = self.monitor.get_memory_info()
      delta_mb = info['rss_mb'] - self.monitor.initial_memory['rss_mb']
      logger.info(f"Operation '{operation_name}' completed. "
                  f"Memory: {info['rss_mb']:.1f}MB (+{delta_mb:.1f}MB)")


# Global resource guard instance (lazy initialization)
_global_guard: ResourceGuard | None = None


def get_resource_guard(memory_limit_gb: float = None) -> ResourceGuard:
  """Get or create the global resource guard."""
  global _global_guard
  if _global_guard is None:
    _global_guard = ResourceGuard(memory_limit_gb)
  return _global_guard


def cleanup_caches():
  """Cleanup function that can be registered with resource guard."""
  import gc

  # Force garbage collection
  gc.collect()

  # Try to clear various caches
  try:
    # Clear embedding cache
    from embedding.embed_manager import cache_manager
    if hasattr(cache_manager, '_memory_cache'):
      cache_manager._memory_cache.clear()
      cache_manager._memory_cache_keys.clear()
      logger.info("Cleared embedding cache")
  except (AttributeError, ImportError) as e:
    logger.error(f"Failed to clear embedding cache: {e}")

  # Clear any pyplot figures
  try:
    import matplotlib.pyplot as plt
    plt.close('all')
  except ImportError:
    # matplotlib not installed, skip cleanup
    pass
  except RuntimeError as e:
    logger.debug(f"Could not close matplotlib figures: {e}")

  logger.info("Cache cleanup completed")


def setup_signal_handlers():
  """
  Set up signal handlers for graceful shutdown on resource limits.
  """
  def handle_memory_limit(signum, frame):
    logger.error("Memory limit reached! Initiating emergency cleanup")
    cleanup_caches()
    # Try to save any critical state here
    sys.exit(1)

  if hasattr(signal, 'SIGXFSZ'):
    signal.signal(signal.SIGXFSZ, handle_memory_limit)


# Initialize signal handlers on module load
setup_signal_handlers()

#fin
