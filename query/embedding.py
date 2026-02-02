#!/usr/bin/env python
"""
Query embedding generation and caching for CustomKB.

This module handles embedding generation for queries, caching,
and integration with various embedding providers.
"""

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from utils.exceptions import EmbeddingError
from utils.logging_config import get_logger
from utils.text_utils import clean_text

from .enhancement import enhance_query

logger = get_logger(__name__)

# Query embedding cache directory (created lazily when cache files are written)
QUERY_CACHE_DIR = os.path.join(os.getenv('VECTORDBS', '/var/lib/vectordbs'), '.query_embedding_cache')


def get_cache_key(query_text: str, model: str) -> str:
  """
  Generate a cache key for query embeddings.

  Args:
      query_text: The query text
      model: The embedding model name

  Returns:
      Cache key in format: {model}_{sha256_hash}
  """
  # Create a unique hash based on text
  text_hash = hashlib.sha256(query_text.encode()).hexdigest()
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
  subdir = os.path.join(QUERY_CACHE_DIR, cache_key[:2])
  os.makedirs(subdir, exist_ok=True)
  return os.path.join(subdir, f"{cache_key}.json")


def get_cached_query_embedding(query_text: str, model: str, kb=None) -> list[float] | None:
  """
  Retrieve cached query embedding.

  Args:
      query_text: The query text
      model: The embedding model name
      kb: Optional KnowledgeBase for configuration

  Returns:
      Cached embedding vector or None
  """
  try:
    cache_key = get_cache_key(query_text, model)
    cache_file = get_cache_file_path(cache_key)

    if os.path.exists(cache_file):
      # Check cache TTL (convert days to seconds)
      cache_ttl = getattr(kb, 'query_cache_ttl_days', 7) * 86400 if kb else 604800  # 7 days default in seconds

      file_age = time.time() - os.path.getmtime(cache_file)
      if file_age < cache_ttl:
        with open(cache_file) as f:
          cache_data = json.load(f)

        # Validate cache data
        if (cache_data.get('model') == model and
            cache_data.get('query_hash') == cache_key):
          logger.debug("Using cached query embedding")
          return cache_data['embedding']
      else:
        # Remove expired cache
        os.remove(cache_file)
        logger.debug("Removed expired query embedding cache")

  except (OSError, json.JSONDecodeError, KeyError, ValueError) as e:
    logger.debug(f"Query embedding cache retrieval failed: {e}")

  return None


def save_query_embedding_to_cache(query_text: str, model: str, embedding: list[float]) -> None:
  """
  Save query embedding to cache.

  Args:
      query_text: The original query text
      model: The embedding model name
      embedding: The embedding vector
  """
  try:
    cache_key = get_cache_key(query_text, model)
    cache_file = get_cache_file_path(cache_key)

    cache_data = {
      'model': model,
      'query_hash': cache_key,
      'embedding': embedding,
      'query_preview': query_text[:100] if len(query_text) > 100 else query_text,
      'timestamp': time.time()
    }

    with open(cache_file, 'w') as f:
      json.dump(cache_data, f)

    logger.debug("Query embedding saved to cache")

  except (OSError, json.JSONDecodeError) as e:
    logger.debug(f"Query embedding cache save failed: {e}")


async def generate_query_embedding(query_text: str, model: str, kb=None) -> np.ndarray:
  """
  Generate embedding for a query using the specified model.

  Args:
      query_text: Query text to embed
      model: Embedding model name
      kb: Optional KnowledgeBase for configuration

  Returns:
      Query embedding as numpy array
  """
  try:
    # Generate embedding using LiteLLM unified provider
    import embedding.litellm_provider as litellm_embed

    embeddings = await litellm_embed.get_embeddings([query_text], model)

    if not embeddings or not embeddings[0]:
      raise EmbeddingError("Failed to generate embedding for query")

    embedding = embeddings[0]

    # Convert to numpy array
    embedding_array = np.array(embedding, dtype=np.float32)

    logger.debug(f"Generated query embedding: {len(embedding)} dimensions")
    return embedding_array

  except (ValueError, ImportError, RuntimeError, OSError) as e:
    logger.error(f"Query embedding generation failed: {e}")
    raise EmbeddingError(f"Failed to generate query embedding: {e}") from e


async def get_query_embedding(query_text: str, model: str, kb: Any | None = None) -> np.ndarray:
  """
  Get embedding for a query, using cache if available.

  This is the main function for getting query embeddings with caching and enhancement.

  Args:
      query_text: The query text
      model: The model to use for embedding
      kb: KnowledgeBase instance for configuration

  Returns:
      Numpy array containing the embedding vector
  """
  try:
    # Clean and enhance the query
    clean_query = clean_text(query_text)

    # Apply query enhancement if enabled
    enhanced_query = enhance_query(clean_query, kb)

    # Log enhancement if there was a change
    if enhanced_query != clean_query and enhanced_query != query_text:
      logger.info(f"Query enhanced: '{clean_query}' -> '{enhanced_query}'")

    # Use enhanced query for caching and embedding generation
    query_for_embedding = enhanced_query

    # Check cache first
    cached_embedding = get_cached_query_embedding(query_for_embedding, model, kb)

    if cached_embedding:
      logger.debug("Using cached query embedding")
      return np.array(cached_embedding, dtype=np.float32)

    # Generate new embedding
    logger.debug(f"Generating new query embedding with model: {model}")
    embedding_array = await generate_query_embedding(query_for_embedding, model, kb)

    # Cache the embedding
    save_query_embedding_to_cache(query_for_embedding, model, embedding_array.tolist())

    return embedding_array

  except (ValueError, RuntimeError, OSError) as e:
    logger.error(f"Failed to get query embedding: {e}")
    raise EmbeddingError(f"Query embedding failed: {e}") from e


def clear_query_cache(model: str = None, older_than_hours: int = None) -> int:
  """
  Clear query embedding cache.

  Args:
      model: Optional model name to filter by
      older_than_hours: Optional age threshold in hours

  Returns:
      Number of files removed
  """
  removed_count = 0
  current_time = time.time()

  try:
    for cache_file in Path(QUERY_CACHE_DIR).rglob("*.json"):
      should_remove = False

      # Check age filter
      if older_than_hours:
        file_age_hours = (current_time - cache_file.stat().st_mtime) / 3600
        if file_age_hours < older_than_hours:
          continue

      # Check model filter
      if model:
        try:
          with open(cache_file) as f:
            cache_data = json.load(f)
          if cache_data.get('model') != model:
            continue
        except (json.JSONDecodeError, OSError):
          # If we can't read the file, mark for removal
          should_remove = True

      if should_remove or not model:
        try:
          cache_file.unlink()
          removed_count += 1
        except OSError:
          pass

    logger.info(f"Cleared {removed_count} query cache files")

  except (OSError, FileNotFoundError) as e:
    logger.error(f"Failed to clear query cache: {e}")

  return removed_count


def get_query_cache_stats() -> dict:
  """
  Get query embedding cache statistics.

  Returns:
      Dictionary with cache statistics
  """
  stats = {
    'cache_dir': QUERY_CACHE_DIR,
    'total_files': 0,
    'total_size_bytes': 0,
    'models': {},
    'oldest_cache': None,
    'newest_cache': None
  }

  try:
    cache_files = list(Path(QUERY_CACHE_DIR).rglob("*.json"))
    stats['total_files'] = len(cache_files)

    if cache_files:
      total_size = 0
      model_counts = {}
      file_times = []

      for cache_file in cache_files:
        try:
          file_size = cache_file.stat().st_size
          file_time = cache_file.stat().st_mtime
          total_size += file_size
          file_times.append(file_time)

          # Get model info
          with open(cache_file) as f:
            cache_data = json.load(f)

          model = cache_data.get('model', 'unknown')
          model_counts[model] = model_counts.get(model, 0) + 1

        except (json.JSONDecodeError, OSError):
          continue

      stats['total_size_bytes'] = total_size
      stats['models'] = model_counts

      if file_times:
        stats['oldest_cache'] = min(file_times)
        stats['newest_cache'] = max(file_times)

  except (OSError, FileNotFoundError) as e:
    logger.error(f"Failed to get query cache stats: {e}")

  return stats


def validate_embedding_dimensions(embedding: np.ndarray, expected_dims: int) -> bool:
  """
  Validate that embedding has expected dimensions.

  Args:
      embedding: Embedding vector
      expected_dims: Expected number of dimensions

  Returns:
      True if valid, False otherwise
  """
  if embedding is None:
    return False

  if not isinstance(embedding, np.ndarray):
    try:
      embedding = np.array(embedding)
    except (ValueError, TypeError):
      return False

  if embedding.shape != (expected_dims,) and embedding.shape != (1, expected_dims):
    logger.warning(f"Embedding dimension mismatch: got {embedding.shape}, expected ({expected_dims},)")
    return False

  # Check for NaN or infinite values
  if np.any(np.isnan(embedding)) or np.any(np.isinf(embedding)):
    logger.warning("Embedding contains NaN or infinite values")
    return False

  return True


#fin
