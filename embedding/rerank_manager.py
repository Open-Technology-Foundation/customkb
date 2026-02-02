"""
Reranking manager for improving search result relevance using cross-encoder models.

This module provides functionality to rerank search results using cross-encoder models
from sentence-transformers, implementing caching and batch processing for efficiency.
"""

import asyncio
import hashlib
import json
import os
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from utils.logging_config import get_logger

logger = get_logger(__name__)

# Global cross-encoder model instance (lazy loading)
_reranking_model = None
_model_lock = asyncio.Lock()

# Two-tier cache: memory (fast) and disk (persistent)
_memory_cache: OrderedDict[str, float] = OrderedDict()
_memory_cache_size = 1000  # Default, will be updated from config

# Cache directory (created lazily when cache files are written)
CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.cache', 'reranking')


def get_reranking_cache_key(query: str, document: str) -> str:
  """
  Generate a cache key for a query-document pair.

  Args:
      query: The query text.
      document: The document text.

  Returns:
      A cache key string.
  """
  # Combine query and document for unique key
  combined = f"{query}|||{document}"
  return hashlib.md5(combined.encode('utf-8')).hexdigest()


def get_cached_score(query: str, document: str) -> float | None:
  """
  Retrieve a cached reranking score if available.

  Args:
      query: The query text.
      document: The document text.

  Returns:
      The cached score or None if not found.
  """
  cache_key = get_reranking_cache_key(query, document)

  # Check memory cache first
  if cache_key in _memory_cache:
    # Move to end (LRU)
    _memory_cache.move_to_end(cache_key)
    return _memory_cache[cache_key]

  # Check disk cache (JSON format)
  cache_file = os.path.join(CACHE_DIR, f"{cache_key}.json")
  if os.path.exists(cache_file):
    try:
      with open(cache_file) as f:
        data = json.load(f)
        score = data.get('score') if isinstance(data, dict) else data
      # Promote to memory cache
      _memory_cache[cache_key] = score
      _enforce_memory_cache_size()
      return score
    except (json.JSONDecodeError, OSError, KeyError, ValueError) as e:
      logger.warning(f"Failed to load cached score from JSON: {e}")

  # Legacy pickle format is no longer supported (removed for security)
  # If old pickle cache files exist, they will be ignored and naturally age out
  old_cache_file = os.path.join(CACHE_DIR, f"{cache_key}.pkl")
  if os.path.exists(old_cache_file):
    logger.debug(f"Ignoring legacy pickle cache file: {cache_key}.pkl")
    logger.debug("Legacy pickle caches are no longer supported for security reasons.")
    logger.debug("The cache will be regenerated automatically.")

  return None


def cache_score(query: str, document: str, score: float):
  """
  Cache a reranking score.

  Args:
      query: The query text.
      document: The document text.
      score: The reranking score.
  """
  cache_key = get_reranking_cache_key(query, document)

  # Add to memory cache
  _memory_cache[cache_key] = score
  _enforce_memory_cache_size()

  # Save to disk cache (JSON format)
  cache_file = os.path.join(CACHE_DIR, f"{cache_key}.json")
  try:
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(cache_file, 'w') as f:
      json.dump({'score': score, 'version': '1.0'}, f)
  except (OSError, PermissionError) as e:
    logger.warning(f"Failed to save score to disk cache: {e}")


def _enforce_memory_cache_size():
  """Enforce the memory cache size limit using LRU eviction."""
  while len(_memory_cache) > _memory_cache_size:
    _memory_cache.popitem(last=False)  # Remove oldest


async def load_reranking_model(model_name: str, device: str = 'cpu'):
  """
  Load the cross-encoder model for reranking.

  Args:
      model_name: The model name/path.
      device: The device to use ('cpu' or 'cuda').

  Returns:
      The loaded model.
  """
  global _reranking_model

  async with _model_lock:
    if _reranking_model is None:
      try:
        from sentence_transformers import CrossEncoder

        logger.info(f"Loading reranking model: {model_name} on {device}")
        _reranking_model = CrossEncoder(model_name, device=device)
        logger.info("Reranking model loaded successfully")
      except (ImportError, RuntimeError, OSError, FileNotFoundError) as e:
        logger.error(f"Failed to load reranking model: {e}")
        raise

  return _reranking_model


def batch_predict(model, pairs: list[tuple[str, str]], batch_size: int) -> list[float]:
  """
  Predict scores for query-document pairs in batches.

  Args:
      model: The cross-encoder model.
      pairs: List of (query, document) tuples.
      batch_size: Batch size for prediction.

  Returns:
      List of scores.
  """
  scores = []

  for i in range(0, len(pairs), batch_size):
    batch = pairs[i:i + batch_size]
    try:
      batch_scores = model.predict(batch)
      scores.extend(batch_scores)
    except (RuntimeError, ValueError, TypeError, IndexError) as e:
      logger.error(f"Error in batch prediction: {e}")
      # Return zeros for failed batch
      scores.extend([0.0] * len(batch))

  return scores


async def rerank_documents(
    kb: Any,
    query: str,
    documents: list[tuple[int, str, float]],
    top_k: int | None = None
) -> list[tuple[int, str, float]]:
  """
  Rerank documents using a cross-encoder model.

  Args:
      kb: KnowledgeBase instance for configuration.
      query: The query text.
      documents: List of (doc_id, doc_text, original_score) tuples.
      top_k: Number of top documents to rerank (default: from config).

  Returns:
      Reranked list of (doc_id, doc_text, reranking_score) tuples.
  """
  if not documents:
    return documents

  # Update cache size from config
  global _memory_cache_size
  _memory_cache_size = getattr(kb, 'reranking_cache_size', 1000)

  # Get configuration
  model_name = getattr(kb, 'reranking_model', 'cross-encoder/ms-marco-MiniLM-L-6-v2')
  device = getattr(kb, 'reranking_device', 'cpu')
  batch_size = getattr(kb, 'reranking_batch_size', 32)
  if top_k is None:
    top_k = getattr(kb, 'reranking_top_k', 20)

  # Limit documents to rerank
  documents_to_rerank = documents[:top_k]
  remaining_documents = documents[top_k:]

  # Load model
  model = await load_reranking_model(model_name, device)

  # Prepare pairs and check cache
  pairs_to_score = []
  cached_scores = []
  doc_indices = []

  for i, (_doc_id, doc_text, _orig_score) in enumerate(documents_to_rerank):
    cached_score = get_cached_score(query, doc_text)
    if cached_score is not None:
      cached_scores.append((i, cached_score))
    else:
      pairs_to_score.append((query, doc_text))
      doc_indices.append(i)

  # Score uncached pairs
  new_scores = []
  if pairs_to_score:
    logger.info(f"Reranking {len(pairs_to_score)} documents (batch size: {batch_size})")

    # Run prediction in thread pool to avoid blocking
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor(max_workers=1) as executor:
      new_scores = await loop.run_in_executor(
        executor, batch_predict, model, pairs_to_score, batch_size
      )

    # Cache new scores
    for (query_text, doc_text), score in zip(pairs_to_score, new_scores, strict=False):
      cache_score(query_text, doc_text, float(score))

  # Combine cached and new scores
  all_scores = []
  new_score_idx = 0

  for i, (doc_id, doc_text, _orig_score) in enumerate(documents_to_rerank):
    if any(idx == i for idx, _ in cached_scores):
      # Use cached score
      score = next(s for idx, s in cached_scores if idx == i)
    else:
      # Use new score
      score = new_scores[new_score_idx]
      new_score_idx += 1

    all_scores.append((doc_id, doc_text, float(score)))

  # Sort by reranking score (descending)
  reranked = sorted(all_scores, key=lambda x: x[2], reverse=True)

  # Append remaining documents (not reranked)
  reranked.extend(remaining_documents)

  logger.info(f"Reranking complete. Top score: {reranked[0][2]:.4f} (was position {documents.index(next(d for d in documents if d[0] == reranked[0][0])) + 1})")

  return reranked


async def rerank_search_results(
    kb: Any,
    query: str,
    search_results: list[tuple[int, float]]
) -> list[tuple[int, float]]:
  """
  Rerank search results by fetching document texts and applying cross-encoder.

  Args:
      kb: KnowledgeBase instance.
      query: The query text.
      search_results: List of (doc_id, distance) tuples from initial search.

  Returns:
      Reranked list of (doc_id, distance) tuples.
  """
  if not search_results or not getattr(kb, 'enable_reranking', False):
    return search_results

  top_k = min(getattr(kb, 'reranking_top_k', 20), len(search_results))

  # Fetch document texts for top-k results
  documents_to_rerank = []
  remaining_results = []

  for i, (doc_id, distance) in enumerate(search_results):
    if i < top_k:
      # Fetch document text
      kb.sql_cursor.execute(
        "SELECT originaltext FROM docs WHERE id = ?",
        (doc_id,)
      )
      result = kb.sql_cursor.fetchone()
      if result:
        doc_text = result[0]
        documents_to_rerank.append((doc_id, doc_text, distance))
      else:
        logger.warning(f"Could not fetch text for doc_id {doc_id}")
        remaining_results.append((doc_id, distance))
    else:
      remaining_results.append((doc_id, distance))

  if not documents_to_rerank:
    return search_results

  # Rerank documents
  reranked_docs = await rerank_documents(kb, query, documents_to_rerank, top_k)

  # Convert back to search result format
  # Note: We use the reranking score as a proxy for distance (lower is better)
  # So we invert the score: distance = 1 / (score + epsilon)
  reranked_results = []
  for doc_id, _, score in reranked_docs:
    # Convert score to distance-like metric
    distance = 1.0 / (score + 1e-6) - 1.0
    reranked_results.append((doc_id, distance))

  # Append remaining results
  reranked_results.extend(remaining_results)

  return reranked_results


def clear_reranking_cache():
  """Clear all reranking caches."""
  global _memory_cache
  _memory_cache.clear()

  # Clear disk cache (both old .pkl and new .json formats)
  for file in os.listdir(CACHE_DIR):
    if file.endswith('.pkl') or file.endswith('.json'):
      try:
        os.remove(os.path.join(CACHE_DIR, file))
      except (FileNotFoundError, PermissionError, OSError) as e:
        logger.warning(f"Failed to remove cache file {file}: {e}")

  logger.info("Reranking cache cleared")


#fin
