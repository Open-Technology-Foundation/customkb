#!/usr/bin/env python
"""
Embedding cache management for CustomKB.

This module handles in-memory and disk-based caching of embeddings
with thread-safe operations and LRU eviction.
"""

import atexit
import contextlib
import hashlib
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from utils.logging_config import get_logger

logger = get_logger(__name__)

# Embedding cache directory (created lazily when cache files are written)
CACHE_DIR = os.path.join(os.getenv('VECTORDBS', '/var/lib/vectordbs'), '.embedding_cache')

# Model dimension mappings
MODEL_DIMENSIONS = {
  'text-embedding-ada-002': 1536,
  'text-embedding-3-small': 1536,
  'text-embedding-3-large': 3072,
  'gemini-embedding-001': None,  # Variable dimensions
  'bge-m3': 1024,
  'all-minilm-l6-v2': 384,
}

def get_expected_dimensions(model: str) -> int | None:
  """Get expected dimensions for a model.

  Args:
      model: Model name

  Returns:
      Expected dimensions or None if variable/unknown
  """
  return MODEL_DIMENSIONS.get(model)


class CacheThreadManager:
  """Manages thread pool and cache operations for embedding storage."""

  def __init__(self, max_workers: int = 4):
    self._executor = None
    self._max_workers = max_workers
    self._lock = threading.RLock()
    self._memory_cache = {}
    self._memory_cache_keys = []
    self._memory_cache_size = 10000  # Default size
    self._max_memory_mb = 500  # Default 500MB limit for cache
    self._embedding_size_bytes = {}  # Track size of each embedding

    # Performance monitoring
    self._metrics = {
      'cache_hits': 0,
      'cache_misses': 0,
      'cache_adds': 0,
      'cache_evictions': 0,
      'thread_pool_tasks': 0,
      'memory_usage_mb': 0.0
    }

  def _ensure_executor(self):
    """Ensure thread pool executor is initialized."""
    if self._executor is None:
      self._executor = ThreadPoolExecutor(max_workers=self._max_workers)
      atexit.register(self._cleanup)

  def _cleanup(self):
    """Clean up thread pool executor."""
    if self._executor is not None:
      self._executor.shutdown(wait=True)
      self._executor = None

  def submit_cache_task(self, func, *args, **kwargs):
    """Submit a cache task to the thread pool."""
    with self._lock:
      self._ensure_executor()
      self._metrics['thread_pool_tasks'] += 1
      return self._executor.submit(func, *args, **kwargs)

  def get_from_memory_cache(self, cache_key: str) -> list[float] | None:
    """Thread-safe retrieval from memory cache."""
    with self._lock:
      if cache_key in self._memory_cache:
        # Move to end for LRU
        self._memory_cache_keys.remove(cache_key)
        self._memory_cache_keys.append(cache_key)
        self._metrics['cache_hits'] += 1
        return self._memory_cache[cache_key]
      else:
        self._metrics['cache_misses'] += 1
    return None

  def add_to_memory_cache(self, cache_key: str, embedding: list[float], kb=None):
    """Thread-safe addition to memory cache with LRU eviction based on memory usage."""
    with self._lock:
      # Configure cache size from KB config if available
      cache_size = getattr(kb, 'memory_cache_size', self._memory_cache_size) if kb else self._memory_cache_size
      memory_limit_mb = getattr(kb, 'cache_memory_limit_mb', self._max_memory_mb) if kb else self._max_memory_mb

      # Calculate embedding size (4 bytes per float)
      embedding_size = len(embedding) * 4

      # Calculate current memory usage
      current_memory = sum(self._embedding_size_bytes.values()) / (1024 * 1024)

      # Evict if over memory limit or cache size limit
      while ((current_memory + embedding_size / (1024 * 1024)) > memory_limit_mb or
             len(self._memory_cache) >= cache_size) and self._memory_cache_keys:
        # Remove oldest (LRU)
        evict_key = self._memory_cache_keys.pop(0)
        evicted_size = self._embedding_size_bytes.pop(evict_key, 0)
        current_memory -= evicted_size / (1024 * 1024)
        del self._memory_cache[evict_key]
        self._metrics['cache_evictions'] += 1

      # Add new embedding
      if cache_key not in self._memory_cache:
        self._memory_cache[cache_key] = embedding
        self._memory_cache_keys.append(cache_key)
        self._embedding_size_bytes[cache_key] = embedding_size
        self._metrics['cache_adds'] += 1
        self._metrics['memory_usage_mb'] = (current_memory + embedding_size / (1024 * 1024))

  def configure(self, max_workers: int = None, memory_cache_size: int = None,
                max_memory_mb: float = None):
    """Configure cache settings."""
    with self._lock:
      if max_workers is not None:
        self._max_workers = max_workers
        # Recreate executor with new worker count
        if self._executor is not None:
          self._cleanup()
      if memory_cache_size is not None:
        self._memory_cache_size = memory_cache_size
      if max_memory_mb is not None:
        self._max_memory_mb = max_memory_mb

  def get_metrics(self) -> dict[str, Any]:
    """Get cache performance metrics."""
    with self._lock:
      total_requests = self._metrics['cache_hits'] + self._metrics['cache_misses']
      hit_ratio = self._metrics['cache_hits'] / total_requests if total_requests > 0 else 0

      return {
        'cache_hits': self._metrics['cache_hits'],
        'cache_misses': self._metrics['cache_misses'],
        'cache_hit_ratio': hit_ratio,
        'cache_size': len(self._memory_cache),
        'max_cache_size': self._memory_cache_size,
        'cache_evictions': self._metrics['cache_evictions'],
        'cache_adds': self._metrics['cache_adds'],
        'memory_usage_mb': self._metrics['memory_usage_mb'],
        'memory_limit_mb': self._max_memory_mb,
        'thread_pool_tasks': self._metrics['thread_pool_tasks']
      }

  def reset_metrics(self):
    """Reset all performance metrics to zero."""
    with self._lock:
      self._metrics = {
        'cache_hits': 0,
        'cache_misses': 0,
        'cache_adds': 0,
        'cache_evictions': 0,
        'thread_pool_tasks': 0,
        'memory_usage_mb': 0.0
      }

  def clear_cache(self):
    """Clear all cached embeddings from memory."""
    with self._lock:
      self._memory_cache.clear()
      self._memory_cache_keys.clear()
      self._embedding_size_bytes.clear()
      self._metrics['memory_usage_mb'] = 0.0


# Global cache manager instance
cache_manager = CacheThreadManager()


def configure_cache_manager(kb: Any) -> None:
  """Configure the global cache manager from KnowledgeBase settings."""
  cache_thread_pool_size = getattr(kb, 'cache_thread_pool_size', 4)
  memory_cache_size = getattr(kb, 'memory_cache_size', 10000)
  cache_memory_limit_mb = getattr(kb, 'cache_memory_limit_mb', 500)

  cache_manager.configure(
    max_workers=cache_thread_pool_size,
    memory_cache_size=memory_cache_size,
    max_memory_mb=cache_memory_limit_mb
  )

  logger.debug(f"Cache manager configured: threads={cache_thread_pool_size}, "
              f"cache_size={memory_cache_size}, memory_limit={cache_memory_limit_mb}MB")


def get_cache_key(text: str, model: str) -> str:
  """
  Generate a cache key for the embedding.

  Args:
      text: The text to be embedded
      model: The embedding model name

  Returns:
      Cache key in format: {model}_{sha256_hash}
  """
  # Create a unique hash based on text
  text_hash = hashlib.sha256(text.encode()).hexdigest()
  # Return format that includes model name for cache organization
  return f"{model}_{text_hash}"


def get_cache_file_path(cache_key: str) -> str:
  """
  Get the file path for a cache key.

  Args:
      cache_key: The cache key

  Returns:
      Full path to cache file
  """
  # Use first 2 chars for directory to avoid too many files in one dir
  subdir = os.path.join(CACHE_DIR, cache_key[:2])
  os.makedirs(subdir, exist_ok=True)
  return os.path.join(subdir, f"{cache_key}.json")


def get_cached_embedding(text: str, model: str) -> list[float] | None:
  """
  Check if embedding exists in cache (memory first, then disk).

  Args:
      text: The text that was embedded
      model: The embedding model name

  Returns:
      Cached embedding if found, None otherwise
  """
  cache_key = get_cache_key(text, model)

  # Check memory cache first
  embedding = cache_manager.get_from_memory_cache(cache_key)
  if embedding is not None:
    return embedding

  # Check disk cache
  cache_file = get_cache_file_path(cache_key)
  if os.path.exists(cache_file):
    try:
      with open(cache_file) as f:
        cache_data = json.load(f)

      # Validate cache data
      if cache_data.get('model') == model and cache_data.get('text_hash') == cache_key:
        embedding = cache_data['embedding']

        # Validate embedding dimensions based on model
        expected_dims = get_expected_dimensions(model)
        if expected_dims and len(embedding) != expected_dims:
          logger.warning(f"Cached embedding has wrong dimensions: got {len(embedding)}, expected {expected_dims}. Removing cache file: {cache_file}")
          with contextlib.suppress(OSError):
            os.remove(cache_file)
          return None

        # Add to memory cache for faster future access
        cache_manager.add_to_memory_cache(cache_key, embedding)
        return embedding
    except (OSError, json.JSONDecodeError, KeyError) as e:
      logger.warning(f"Cache file corrupted or invalid: {cache_file}: {e}")
      # Remove corrupted cache file
      with contextlib.suppress(OSError):
        os.remove(cache_file)

  return None


def add_to_memory_cache(cache_key: str, embedding: list[float], kb=None) -> None:
  """
  Add embedding to memory cache.

  Args:
      cache_key: The cache key
      embedding: The embedding vector
      kb: Optional KnowledgeBase for configuration
  """
  cache_manager.add_to_memory_cache(cache_key, embedding, kb)


def save_embedding_to_cache(text: str, model: str, embedding: list[float], kb=None) -> None:
  """
  Save embedding to both memory and disk cache.

  Args:
      text: The original text
      model: The embedding model name
      embedding: The embedding vector
      kb: Optional KnowledgeBase for configuration
  """
  cache_key = get_cache_key(text, model)

  # Add to memory cache
  cache_manager.add_to_memory_cache(cache_key, embedding, kb)

  # Save to disk cache asynchronously
  def _save_to_disk():
    cache_file = get_cache_file_path(cache_key)
    cache_data = {
      'model': model,
      'text_hash': cache_key,
      'embedding': embedding,
      'text_preview': text[:100] if len(text) > 100 else text  # Store preview for debugging
    }

    try:
      with open(cache_file, 'w') as f:
        json.dump(cache_data, f)
    except OSError as e:
      logger.warning(f"Failed to save embedding to disk cache: {e}")

  # Submit disk save task to thread pool
  cache_manager.submit_cache_task(_save_to_disk)


def clear_cache(memory_only: bool = False) -> dict[str, int]:
  """
  Clear the embedding cache.

  Args:
      memory_only: If True, only clear memory cache, not disk cache

  Returns:
      Dictionary with counts of cleared items
  """
  stats = {'memory_cleared': 0, 'disk_cleared': 0}

  # Clear memory cache
  with cache_manager._lock:
    stats['memory_cleared'] = len(cache_manager._memory_cache)
    cache_manager._memory_cache.clear()
    cache_manager._memory_cache_keys.clear()
    cache_manager._embedding_size_bytes.clear()
    cache_manager._metrics['memory_usage_mb'] = 0.0

  # Clear disk cache if requested
  if not memory_only:
    try:
      for root, _dirs, files in os.walk(CACHE_DIR):
        for file in files:
          if file.endswith('.json'):
            try:
              os.remove(os.path.join(root, file))
              stats['disk_cleared'] += 1
            except OSError:
              pass
    except OSError as e:
      logger.warning(f"Error clearing disk cache: {e}")

  logger.info(f"Cache cleared: {stats['memory_cleared']} from memory, "
             f"{stats['disk_cleared']} from disk")

  return stats


def get_cache_stats() -> dict[str, Any]:
  """
  Get cache statistics.

  Returns:
      Dictionary with cache statistics
  """
  metrics = cache_manager.get_metrics()

  # Count disk cache files
  disk_files = 0
  disk_size_mb = 0
  try:
    for root, _dirs, files in os.walk(CACHE_DIR):
      for file in files:
        if file.endswith('.json'):
          disk_files += 1
          file_path = os.path.join(root, file)
          disk_size_mb += os.path.getsize(file_path) / (1024 * 1024)
  except OSError:
    pass

  return {
    **metrics,
    'disk_files': disk_files,
    'disk_size_mb': disk_size_mb,
    'cache_dir': CACHE_DIR
  }


#fin
