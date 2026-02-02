#!/usr/bin/env python
"""
Batch processing utilities for embeddings.

This module handles batch size optimization, checkpoint management,
and parallel batch processing for efficient embedding generation.
"""

import asyncio
import json
import os
from typing import Any

import numpy as np

# Load FAISS with proper GPU initialization
from utils.faiss_loader import get_faiss

faiss, FAISS_GPU_AVAILABLE = get_faiss()

from utils.exceptions import BatchError, ProcessingError
from utils.logging_config import get_logger

logger = get_logger(__name__)


def calculate_optimal_batch_size(chunks: list[str], model: str, max_batch_size: int, kb=None) -> int:
  """
  Calculate optimal batch size based on text length and model constraints.

  Args:
      chunks: List of text chunks
      model: Embedding model name
      max_batch_size: Maximum allowed batch size
      kb: Optional KnowledgeBase for configuration

  Returns:
      Optimal batch size
  """
  if not chunks:
    return 1

  # Model-specific token limits (approximate)
  model_limits = {
    'text-embedding-ada-002': 8191,
    'text-embedding-3-small': 8191,
    'text-embedding-3-large': 8191,
    'gemini-embedding-001': 30000,  # Gemini supports longer context
  }

  # Get token limit for model
  token_limit = model_limits.get(model, 8191)

  # Estimate average tokens per chunk (rough estimate: 1 token ≈ 4 chars)
  avg_chunk_tokens = sum(len(chunk) for chunk in chunks[:100]) // min(100, len(chunks)) // 4

  # Calculate batch size based on token limits
  # Leave some buffer for API overhead
  safe_token_limit = int(token_limit * 0.9)

  if avg_chunk_tokens == 0:
    return min(max_batch_size, len(chunks))

  # Calculate how many chunks can fit in one batch
  optimal_batch = min(
    max_batch_size,
    safe_token_limit // avg_chunk_tokens,
    len(chunks)
  )

  # Ensure at least 1
  optimal_batch = max(1, optimal_batch)

  # Dynamic adjustment based on KB config
  if kb:
    # Check if we should use conservative batching
    if getattr(kb, 'conservative_batching', False):
      optimal_batch = max(1, optimal_batch // 2)

    # Apply any batch size override
    if hasattr(kb, 'force_batch_size'):
      optimal_batch = min(kb.force_batch_size, optimal_batch)

  logger.debug(f"Calculated optimal batch size: {optimal_batch} for {len(chunks)} chunks")
  return optimal_batch


def save_checkpoint(kb: Any, index: faiss.Index, doc_ids: list[int],
                   processed_count: int, checkpoint_file: str) -> None:
  """
  Save processing checkpoint.

  Args:
      kb: KnowledgeBase instance
      index: FAISS index
      doc_ids: List of document IDs
      processed_count: Number of processed documents
      checkpoint_file: Path to checkpoint file
  """
  try:
    checkpoint_data = {
      'processed_count': processed_count,
      'doc_ids': doc_ids[:processed_count],
      'index_size': index.ntotal,
      'knowledge_base': kb.knowledge_base_db,
      'model': kb.vector_model
    }

    # Save checkpoint data
    with open(checkpoint_file + '.json', 'w') as f:
      json.dump(checkpoint_data, f)

    # Save FAISS index
    faiss.write_index(index, checkpoint_file + '.faiss')

    logger.debug(f"Checkpoint saved: {processed_count} documents processed")

  except (OSError, json.JSONDecodeError, RuntimeError) as e:
    logger.error(f"Failed to save checkpoint: {e}")
    raise ProcessingError(f"Checkpoint save failed: {e}") from e


def load_checkpoint(checkpoint_file: str) -> tuple[faiss.Index | None, dict | None]:
  """
  Load processing checkpoint.

  Args:
      checkpoint_file: Path to checkpoint file

  Returns:
      Tuple of (FAISS index, checkpoint data) or (None, None) if not found
  """
  try:
    # Check if checkpoint exists
    if not os.path.exists(checkpoint_file + '.json'):
      return None, None

    # Load checkpoint data
    with open(checkpoint_file + '.json') as f:
      checkpoint_data = json.load(f)

    # Load FAISS index
    if os.path.exists(checkpoint_file + '.faiss'):
      index = faiss.read_index(checkpoint_file + '.faiss')
    else:
      logger.warning("Checkpoint index file not found")
      return None, None

    logger.info(f"Checkpoint loaded: {checkpoint_data['processed_count']} documents already processed")
    return index, checkpoint_data

  except (FileNotFoundError, json.JSONDecodeError, OSError, RuntimeError) as e:
    logger.warning(f"Failed to load checkpoint: {e}")
    return None, None


def remove_checkpoint(checkpoint_file: str) -> None:
  """
  Remove checkpoint files.

  Args:
      checkpoint_file: Base path to checkpoint files
  """
  for ext in ['.json', '.faiss']:
    file_path = checkpoint_file + ext
    if os.path.exists(file_path):
      try:
        os.remove(file_path)
        logger.debug(f"Removed checkpoint file: {file_path}")
      except OSError as e:
        logger.warning(f"Failed to remove checkpoint file {file_path}: {e}")


async def process_batch_with_retry(process_func, batch: list[str], max_retries: int = 3,
                                  retry_delay: float = 1.0) -> list[list[float]]:
  """
  Process a batch with retry logic.

  Args:
      process_func: Async function to process the batch
      batch: List of texts to process
      max_retries: Maximum number of retries
      retry_delay: Delay between retries in seconds

  Returns:
      List of embeddings

  Raises:
      BatchError: If all retries fail
  """
  last_error = None

  for attempt in range(max_retries):
    try:
      return await process_func(batch)
    except (ConnectionError, TimeoutError, OSError, ValueError) as e:
      last_error = e
      if attempt < max_retries - 1:
        logger.warning(f"Batch processing failed (attempt {attempt + 1}/{max_retries}): {e}")
        await asyncio.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
      else:
        logger.error(f"Batch processing failed after {max_retries} attempts: {e}")

  raise BatchError("batch_unknown", f"Batch processing failed after {max_retries} retries: {last_error}") from last_error


async def process_batches_parallel(process_func, chunks: list[str], batch_size: int,
                                  max_concurrent: int = 3) -> list[list[float]]:
  """
  Process batches in parallel with concurrency control.

  Args:
      process_func: Async function to process each batch
      chunks: List of texts to process
      batch_size: Size of each batch
      max_concurrent: Maximum concurrent batches

  Returns:
      List of embeddings in order
  """
  embeddings = []
  semaphore = asyncio.Semaphore(max_concurrent)

  async def process_with_semaphore(batch: list[str], batch_idx: int) -> tuple[int, list[list[float]]]:
    async with semaphore:
      result = await process_func(batch)
      return batch_idx, result

  # Create tasks for all batches
  tasks = []
  for i in range(0, len(chunks), batch_size):
    batch = chunks[i:i + batch_size]
    batch_idx = i // batch_size
    tasks.append(process_with_semaphore(batch, batch_idx))

  # Process all batches
  results = await asyncio.gather(*tasks)

  # Sort results by batch index and flatten
  results.sort(key=lambda x: x[0])
  for _, batch_embeddings in results:
    embeddings.extend(batch_embeddings)

  return embeddings


def validate_batch_consistency(embeddings: list[list[float]], expected_count: int,
                              expected_dims: int) -> bool:
  """
  Validate batch processing results.

  Args:
      embeddings: List of embedding vectors
      expected_count: Expected number of embeddings
      expected_dims: Expected dimensions per embedding

  Returns:
      True if valid, raises exception otherwise

  Raises:
      BatchError: If validation fails
  """
  if len(embeddings) != expected_count:
    raise BatchError("validation", f"Embedding count mismatch: got {len(embeddings)}, expected {expected_count}")

  for i, emb in enumerate(embeddings):
    if len(emb) != expected_dims:
      raise BatchError(f"embedding_{i}", f"Embedding {i} has wrong dimensions: got {len(emb)}, expected {expected_dims}")

    # Check for NaN or infinite values
    emb_array = np.array(emb)
    if np.any(np.isnan(emb_array)) or np.any(np.isinf(emb_array)):
      raise BatchError(f"embedding_{i}", f"Embedding {i} contains NaN or infinite values")

  return True


class BatchProcessor:
  """Manages batch processing with progress tracking."""

  def __init__(self, kb: Any, total_items: int):
    """
    Initialize batch processor.

    Args:
        kb: KnowledgeBase instance
        total_items: Total number of items to process
    """
    self.kb = kb
    self.total_items = total_items
    self.processed_items = 0
    self.failed_items = 0
    self.start_time = None
    self.batch_times = []

  def start(self):
    """Start processing timer."""
    import time
    self.start_time = time.time()

  def update(self, batch_size: int, success: bool = True):
    """
    Update progress after processing a batch.

    Args:
        batch_size: Size of processed batch
        success: Whether batch was successful
    """
    import time

    if success:
      self.processed_items += batch_size
    else:
      self.failed_items += batch_size

    # Track batch processing time
    if self.start_time:
      current_time = time.time()
      elapsed = current_time - self.start_time
      self.batch_times.append(elapsed)

      # Estimate time remaining
      if self.processed_items > 0:
        avg_time_per_item = elapsed / self.processed_items
        remaining_items = self.total_items - self.processed_items
        eta_seconds = avg_time_per_item * remaining_items

        # Log progress
        progress_pct = (self.processed_items / self.total_items) * 100
        logger.info(f"Progress: {self.processed_items}/{self.total_items} ({progress_pct:.1f}%), "
                   f"ETA: {self._format_time(eta_seconds)}")

  def _format_time(self, seconds: float) -> str:
    """Format seconds into human-readable time."""
    if seconds < 60:
      return f"{seconds:.0f}s"
    elif seconds < 3600:
      return f"{seconds/60:.1f}m"
    else:
      return f"{seconds/3600:.1f}h"

  def get_summary(self) -> dict[str, Any]:
    """
    Get processing summary.

    Returns:
        Dictionary with processing statistics
    """
    import time

    elapsed = time.time() - self.start_time if self.start_time else 0

    return {
      'total_items': self.total_items,
      'processed_items': self.processed_items,
      'failed_items': self.failed_items,
      'success_rate': self.processed_items / self.total_items if self.total_items > 0 else 0,
      'total_time_seconds': elapsed,
      'avg_time_per_item': elapsed / self.processed_items if self.processed_items > 0 else 0,
      'items_per_second': self.processed_items / elapsed if elapsed > 0 else 0
    }


#fin
