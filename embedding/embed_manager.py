#!/usr/bin/env python
"""
Improved Embedding management for CustomKB.
Handles the generation and storage of vector embeddings with:
- Checkpoint updating after each batch
- Forced delay between API calls
- More robust error handling
"""

import os
import sys
import time
import numpy as np
import faiss
import argparse
import asyncio
import hashlib
import json
import threading
import atexit
from typing import List, Tuple, Optional, Dict, Any, Set
from concurrent.futures import ThreadPoolExecutor

from utils.logging_utils import setup_logging, get_logger, time_to_finish
from config.config_manager import KnowledgeBase, get_fq_cfg_filename
from database.db_manager import connect_to_database, close_database

# Import OpenAI client with validation
from openai import OpenAI, AsyncOpenAI
from utils.security_utils import validate_api_key, safe_log_error

def load_and_validate_openai_key():
  """Load and validate OpenAI API key securely."""
  openai_key = os.getenv('OPENAI_API_KEY')
  if not openai_key:
    raise EnvironmentError("OPENAI_API_KEY environment variable not set.")
  
  if not validate_api_key(openai_key, 'sk-', 40):
    raise ValueError("Invalid OpenAI API key format")
  
  return openai_key

try:
  OPENAI_API_KEY = load_and_validate_openai_key()
  openai_client = OpenAI(api_key=OPENAI_API_KEY)
  async_openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
except (EnvironmentError, ValueError) as e:
  # Don't use safe_log_error during module initialization
  # as logging may not be set up yet
  print(f"ERROR: OpenAI API key validation failed: {e}", file=sys.stderr)
  raise

logger = get_logger(__name__)

# Embedding cache directory
CACHE_DIR = os.path.join(os.getenv('VECTORDBS', '/var/lib/vectordbs'), '.embedding_cache')
os.makedirs(CACHE_DIR, exist_ok=True)

# Configuration constants - these are now loaded from KnowledgeBase config

# Thread-safe cache manager
class CacheThreadManager:
  """Manages thread pool and cache operations for embedding storage."""
  
  def __init__(self, max_workers: int = 4):
    self._executor = None
    self._max_workers = max_workers
    self._lock = threading.RLock()
    self._memory_cache = {}
    self._memory_cache_keys = []
    self._memory_cache_size = 10000  # Default size
    
    # Performance monitoring
    self._metrics = {
      'cache_hits': 0,
      'cache_misses': 0,
      'cache_adds': 0,
      'cache_evictions': 0,
      'thread_pool_tasks': 0
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
  
  def get_from_memory_cache(self, cache_key: str) -> Optional[List[float]]:
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
  
  def add_to_memory_cache(self, cache_key: str, embedding: List[float], kb=None):
    """Thread-safe addition to memory cache with LRU eviction."""
    with self._lock:
      # Configure cache size from KB config if available
      cache_size = getattr(kb, 'memory_cache_size', self._memory_cache_size) if kb else self._memory_cache_size
      
      # Remove oldest entries if cache is full
      evictions = 0
      while len(self._memory_cache_keys) >= cache_size:
        oldest_key = self._memory_cache_keys.pop(0)
        if oldest_key in self._memory_cache:
          del self._memory_cache[oldest_key]
          evictions += 1
      
      # Track metrics
      self._metrics['cache_adds'] += 1
      self._metrics['cache_evictions'] += evictions
      
      # Add to memory cache
      self._memory_cache[cache_key] = embedding
      self._memory_cache_keys.append(cache_key)
  
  def configure(self, max_workers: int = None, memory_cache_size: int = None):
    """Configure the cache manager."""
    with self._lock:
      if max_workers is not None and max_workers != self._max_workers:
        # Recreate executor with new size
        if self._executor is not None:
          self._executor.shutdown(wait=True)
          self._executor = None
        self._max_workers = max_workers
      
      if memory_cache_size is not None:
        self._memory_cache_size = memory_cache_size
  
  def get_metrics(self) -> Dict[str, Any]:
    """Get performance metrics for the cache manager."""
    with self._lock:
      metrics = self._metrics.copy()
      # Calculate derived metrics
      total_requests = metrics['cache_hits'] + metrics['cache_misses']
      metrics['cache_hit_ratio'] = metrics['cache_hits'] / total_requests if total_requests > 0 else 0.0
      metrics['cache_size'] = len(self._memory_cache)
      metrics['max_cache_size'] = self._memory_cache_size
      metrics['thread_pool_size'] = self._max_workers
      return metrics
  
  def reset_metrics(self):
    """Reset performance metrics (for testing or periodic monitoring)."""
    with self._lock:
      self._metrics = {
        'cache_hits': 0,
        'cache_misses': 0,
        'cache_adds': 0,
        'cache_evictions': 0,
        'thread_pool_tasks': 0
      }

# Global cache manager instance
cache_manager = CacheThreadManager()

# Thread-safe backward compatibility proxies
class ThreadSafeCacheProxy:
  """Thread-safe proxy for backward compatibility with direct cache access."""
  
  def __init__(self, cache_manager: CacheThreadManager):
    self._cache_manager = cache_manager
  
  def __contains__(self, key):
    with self._cache_manager._lock:
      return key in self._cache_manager._memory_cache
  
  def __getitem__(self, key):
    with self._cache_manager._lock:
      return self._cache_manager._memory_cache[key]
  
  def __setitem__(self, key, value):
    # Deprecated - use cache_manager.add_to_memory_cache instead
    import warnings
    warnings.warn("Direct cache assignment is deprecated. Use cache_manager.add_to_memory_cache()", 
                  DeprecationWarning, stacklevel=2)
    self._cache_manager.add_to_memory_cache(key, value)
  
  def get(self, key, default=None):
    result = self._cache_manager.get_from_memory_cache(key)
    return result if result is not None else default
  
  def keys(self):
    with self._cache_manager._lock:
      return list(self._cache_manager._memory_cache.keys())
  
  def values(self):
    with self._cache_manager._lock:
      return list(self._cache_manager._memory_cache.values())
  
  def items(self):
    with self._cache_manager._lock:
      return list(self._cache_manager._memory_cache.items())
  
  def __len__(self):
    with self._cache_manager._lock:
      return len(self._cache_manager._memory_cache)

class ThreadSafeCacheKeysProxy:
  """Thread-safe proxy for backward compatibility with direct cache keys access."""
  
  def __init__(self, cache_manager: CacheThreadManager):
    self._cache_manager = cache_manager
  
  def __len__(self):
    with self._cache_manager._lock:
      return len(self._cache_manager._memory_cache_keys)
  
  def __contains__(self, key):
    with self._cache_manager._lock:
      return key in self._cache_manager._memory_cache_keys
  
  def __iter__(self):
    with self._cache_manager._lock:
      return iter(self._cache_manager._memory_cache_keys.copy())
  
  def __getitem__(self, index):
    with self._cache_manager._lock:
      return self._cache_manager._memory_cache_keys[index]
  
  def append(self, key):
    # Deprecated - use cache_manager.add_to_memory_cache instead
    import warnings
    warnings.warn("Direct cache keys manipulation is deprecated. Use cache_manager.add_to_memory_cache()", 
                  DeprecationWarning, stacklevel=2)
  
  def remove(self, key):
    # Deprecated - managed automatically by cache_manager
    import warnings
    warnings.warn("Direct cache keys manipulation is deprecated. Cache keys are managed automatically.", 
                  DeprecationWarning, stacklevel=2)

# Backward compatibility - these use thread-safe proxies
embedding_memory_cache = ThreadSafeCacheProxy(cache_manager)
embedding_memory_cache_keys = ThreadSafeCacheKeysProxy(cache_manager)

def configure_cache_manager(kb: 'KnowledgeBase') -> None:
  """
  Configure the global cache manager from KnowledgeBase settings.
  
  Args:
      kb: KnowledgeBase instance with configuration.
  """
  cache_thread_pool_size = getattr(kb, 'cache_thread_pool_size', 4)
  memory_cache_size = getattr(kb, 'memory_cache_size', 10000)
  
  cache_manager.configure(
    max_workers=cache_thread_pool_size,
    memory_cache_size=memory_cache_size
  )

def get_cache_key(text: str, model: str) -> str:
  """
  Generate a cache key for an embedding.
  
  Args:
      text: The text to embed.
      model: The model used for embedding.
      
  Returns:
      A cache key string.
  """
  text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
  return f"{model}_{text_hash}"

def get_cached_embedding(text: str, model: str) -> Optional[List[float]]:
  """
  Retrieve a cached embedding if it exists, checking memory first then disk.
  
  Args:
      text: The text to embed.
      model: The model used for embedding.
      
  Returns:
      The cached embedding or None if not found.
  """
  cache_key = get_cache_key(text, model)
  
  # First check in-memory cache (faster, thread-safe)
  cached_embedding = cache_manager.get_from_memory_cache(cache_key)
  if cached_embedding is not None:
    return cached_embedding
  
  # Then check disk cache
  cache_file = os.path.join(CACHE_DIR, f"{cache_key}.json")
  
  if os.path.exists(cache_file):
    try:
      with open(cache_file, 'r') as f:
        embedding = json.load(f)
        # Store in memory cache for future use
        add_to_memory_cache(cache_key, embedding)
        return embedding
    except (json.JSONDecodeError, IOError):
      return None
  
  return None

def add_to_memory_cache(cache_key: str, embedding: List[float], kb=None) -> None:
  """
  Add an embedding to the in-memory cache with LRU eviction.
  
  Args:
      cache_key: The cache key.
      embedding: The embedding vector.
      kb: KnowledgeBase instance for configuration (optional).
  """
  cache_manager.add_to_memory_cache(cache_key, embedding, kb)

def save_embedding_to_cache(text: str, model: str, embedding: List[float], kb=None) -> None:
  """
  Save an embedding to both memory and disk cache.
  
  Args:
      text: The text that was embedded.
      model: The model used for embedding.
      embedding: The embedding vector.
      kb: KnowledgeBase instance for configuration (optional).
  """
  cache_key = get_cache_key(text, model)
  
  # Save to memory cache for immediate access (thread-safe)
  add_to_memory_cache(cache_key, embedding, kb)
  
  # Save to disk asynchronously using shared thread pool
  cache_file = os.path.join(CACHE_DIR, f"{cache_key}.json")
  
  def save_to_disk():
    try:
      with open(cache_file, 'w') as f:
        json.dump(embedding, f)
    except IOError as e:
      logger.warning(f"Failed to cache embedding: {e}")
  
  # Use shared thread pool - no resource leak
  cache_manager.submit_cache_task(save_to_disk)

def get_optimal_faiss_index(dimensions: int, dataset_size: int, kb=None) -> faiss.Index:
  """
  Create an optimal FAISS index based on dataset size.
  
  Args:
      dimensions: The dimensions of the embedding vectors.
      dataset_size: The expected size of the dataset.
      kb: KnowledgeBase instance for configuration (optional).
      
  Returns:
      A FAISS index optimized for the dataset.
  """
  # Get configurable thresholds
  high_dim_threshold = getattr(kb, 'high_dimension_threshold', 1536) if kb else 1536
  small_dataset_threshold = getattr(kb, 'small_dataset_threshold', 1000) if kb else 1000
  medium_dataset_threshold = getattr(kb, 'medium_dataset_threshold', 100000) if kb else 100000
  ivf_centroid_multiplier = getattr(kb, 'ivf_centroid_multiplier', 4) if kb else 4
  max_centroids = getattr(kb, 'max_centroids', 256) if kb else 256
  
  # For high-dimensional vectors, use a flat index 
  # which doesn't require training and works with any dimensionality
  if dimensions > high_dim_threshold:
    logger.info(f"Using IndexFlatIP due to high dimensionality: {dimensions}")
    index = faiss.IndexFlatIP(dimensions)
    return faiss.IndexIDMap(index)
    
  # For smaller datasets, use exact search
  if dataset_size < small_dataset_threshold:
    index = faiss.IndexFlatIP(dimensions)
  elif dataset_size < medium_dataset_threshold:
    # For medium datasets, use IVF with configurable centroids
    n_centroids = min(int(ivf_centroid_multiplier * (dataset_size ** 0.5)), max_centroids)  # Limit number of centroids
    quantizer = faiss.IndexFlatIP(dimensions)
    index = faiss.IndexIVFFlat(quantizer, dimensions, n_centroids, faiss.METRIC_INNER_PRODUCT)
    index.train_mode = True  # Enable training mode initially
  else:
    # For large datasets, use IVF with PQ for compression
    large_max_centroids = max_centroids * 2  # Allow more centroids for large datasets
    n_centroids = min(int(ivf_centroid_multiplier * (dataset_size ** 0.5)), large_max_centroids)  # Limit number of centroids
    quantizer = faiss.IndexFlatIP(dimensions)
    # Use 8-bit quantization with 16 subquantizers
    n_subquantizers = min(16, dimensions // 64)  # Ensure subquantizers fit dimensions
    index = faiss.IndexIVFPQ(quantizer, dimensions, n_centroids, n_subquantizers, 8, faiss.METRIC_INNER_PRODUCT)
    index.train_mode = True  # Enable training mode initially
  
  return faiss.IndexIDMap(index)

def calculate_optimal_batch_size(chunks: List[str], model: str, max_batch_size: int, kb=None) -> int:
  """
  Calculate the optimal batch size based on token limits.
  
  Args:
      chunks: The text chunks to embed.
      model: The embedding model.
      max_batch_size: The maximum batch size.
      kb: KnowledgeBase instance for configuration (optional).
      
  Returns:
      An optimal batch size.
  """
  # Get configurable parameters
  sample_size_config = getattr(kb, 'token_estimation_sample_size', 10) if kb else 10
  token_multiplier = getattr(kb, 'token_estimation_multiplier', 1.3) if kb else 1.3
  
  # More efficient token estimation
  # Sample a few chunks rather than processing all of them
  sample_size = min(sample_size_config, len(chunks))
  sample_chunks = chunks[:sample_size]
  avg_tokens = sum(len(chunk.split()) * token_multiplier for chunk in sample_chunks) / sample_size
  
  # Token limits per model
  model_limits = {
    "text-embedding-3-small": 8191,
    "text-embedding-3-large": 8191,
    "text-embedding-ada-002": 8191
  }
  
  token_limit = model_limits.get(model, 8191)
  
  # Calculate max chunks per batch
  max_chunks = min(max_batch_size, int(token_limit / avg_tokens))
  
  # Ensure at least one chunk per batch
  return max(1, max_chunks)

async def process_embedding_batch_async(kb: KnowledgeBase, chunks: List[str]) -> List[List[float]]:
  """
  Process a batch of text chunks to generate embeddings asynchronously.
  
  Args:
      kb: The KnowledgeBase instance.
      chunks: List of text chunks to embed.
      
  Returns:
      List of embedding vectors.
  """
  max_tries = getattr(kb, 'api_max_retries', 20)
  tries = 0
  cached_embeddings: List[Optional[List[float]]] = [get_cached_embedding(chunk, kb.vector_model) for chunk in chunks]
  uncached_indices = [i for i, emb in enumerate(cached_embeddings) if emb is None]
  
  if not uncached_indices:
    # All embeddings found in cache
    return [emb for emb in cached_embeddings if emb is not None]
  
  uncached_chunks = [chunks[i] for i in uncached_indices]
  
  # Split into smaller sub-batches to improve reliability
  max_batch_size_config = getattr(kb, 'embedding_batch_size', 100)
  sub_batch_size = min(max_batch_size_config, len(uncached_chunks))
  sub_batches = [uncached_chunks[i:i+sub_batch_size] for i in range(0, len(uncached_chunks), sub_batch_size)]
  sub_indices = [uncached_indices[i:i+sub_batch_size] for i in range(0, len(uncached_indices), sub_batch_size)]
  
  all_embeddings = []
  all_indices = []
  
  for sub_batch, sub_idx in zip(sub_batches, sub_indices):
    tries = 0
    while True:
      try:
        # Add forced delay to avoid rate limiting
        api_delay = getattr(kb, 'api_call_delay_seconds', 0.05)
        await asyncio.sleep(api_delay)
        
        response = await async_openai_client.embeddings.create(
          input=sub_batch, 
          model=kb.vector_model
        )
        
        new_embeddings = [data.embedding for data in response.data]
        
        # Cache the new embeddings
        for i, chunk_idx, emb in zip(range(len(sub_batch)), sub_idx, new_embeddings):
          save_embedding_to_cache(chunks[chunk_idx], kb.vector_model, emb, kb)
          all_embeddings.append((chunk_idx, emb))
        
        all_indices.extend(sub_idx)
        break
      except Exception as e:
        from utils.security_utils import safe_log_error
        safe_log_error(f"Embedding API error: {e}")
        safe_log_error(f"Retry attempt {tries} for model {kb.vector_model}")
        tries += 1
        if tries > max_tries:
          safe_log_error(f"Max retries reached for sub-batch. Skipping batch.")
          safe_log_error(f"Failed after {tries} attempts with model {kb.vector_model}")
          # Skip this batch instead of exiting the program
          break
        # Exponential backoff with jitter
        backoff_exponent = getattr(kb, 'backoff_exponent', 2)
        backoff_jitter = getattr(kb, 'backoff_jitter', 0.1)
        backoff = (tries ** backoff_exponent) + (backoff_jitter * np.random.random())
        logger.info(f"Rate limit hit. Backing off for {backoff:.2f} seconds")
        await asyncio.sleep(backoff)
  
  # Merge cached and new embeddings
  result = cached_embeddings.copy()
  for idx, emb in all_embeddings:
    result[idx] = emb
      
  return [emb for emb in result if emb is not None]

async def get_embeddings_for_batch(kb: KnowledgeBase, chunks: List[str]) -> List[List[float]]:
  """
  Get embeddings for a batch of text chunks without updating the index.
  Used for training the FAISS index.
  
  Args:
      kb: The KnowledgeBase instance.
      chunks: List of text chunks.
      
  Returns:
      List of embeddings as lists of floats.
  """
  try:
    embeddings_list = []
    cache_hits = 0
    
    # Check cache first
    for text in chunks:
      cached_embedding = get_cached_embedding(text, kb.vector_model)
      if cached_embedding:
        embeddings_list.append(cached_embedding)
        cache_hits += 1
      else:
        # Add placeholder for uncached
        embeddings_list.append(None)
    
    # Get indices of uncached texts
    uncached_indices = [i for i, emb in enumerate(embeddings_list) if emb is None]
    uncached_texts = [chunks[i] for i in uncached_indices]
    
    if uncached_texts:
      # Get embeddings from API
      response = openai_client.embeddings.create(
        input=uncached_texts,
        model=kb.vector_model
      )
      
      # Fill in the embeddings and cache them
      for idx, (text_idx, embedding_data) in enumerate(zip(uncached_indices, response.data)):
        embedding = embedding_data.embedding
        embeddings_list[text_idx] = embedding
        # Cache the embedding
        save_embedding_to_cache(uncached_texts[idx], kb.vector_model, embedding, kb)
    
    if cache_hits > 0:
      logger.debug(f"Cache hits: {cache_hits}/{len(chunks)}")
    
    # Filter out any None values (shouldn't happen but be safe)
    return [emb for emb in embeddings_list if emb is not None]
    
  except Exception as e:
    logger.error(f"Error getting embeddings for batch: {e}")
    return []

async def process_batch_and_update(kb: KnowledgeBase, index: faiss.IndexIDMap, 
                                 chunks: List[str], ids: List[int]) -> Set[int]:
  """
  Process a batch and update the database with success status.
  
  Args:
      kb: The KnowledgeBase instance.
      index: The FAISS index.
      chunks: List of text chunks.
      ids: List of corresponding IDs.
      
  Returns:
      Set of successfully processed IDs.
  """
  try:
    # Process the batch
    embeddings_list = await process_embedding_batch_async(kb, chunks)
    
    if not embeddings_list:
      logger.warning(f"No embeddings were generated for batch")
      return set()
    
    # Add embeddings to index
    embeddings = np.array(embeddings_list, dtype=np.float32)
    ids_array = np.array(ids, dtype=np.int64)
    index.add_with_ids(embeddings, ids_array)
    
    # Return successfully processed IDs
    return set(ids)
  except Exception as e:
    logger.error(f"Error processing batch: {e}")
    return set()

async def process_all_batches_with_checkpoints(kb: KnowledgeBase, index: faiss.IndexIDMap, 
                                          all_chunks: List[List[str]], all_ids: List[List[int]]) -> Set[int]:
  """
  Process all batches of embeddings with checkpointing and concurrency.
  
  Args:
      kb: The KnowledgeBase instance.
      index: The FAISS index.
      all_chunks: List of batches of text chunks.
      all_ids: List of batches of corresponding IDs.
      
  Returns:
      Set of all successfully processed IDs.
  """
  all_processed_ids = set()
  checkpoint_counter = 0
  
  # Set concurrency limit based on dataset size
  # For larger datasets, process more batches concurrently
  dataset_size = sum(len(chunk_batch) for chunk_batch in all_chunks)
  max_concurrency = getattr(kb, 'api_max_concurrency', 8)
  min_concurrency = getattr(kb, 'api_min_concurrency', 3)
  concurrency_limit = min(max_concurrency, max(min_concurrency, dataset_size // 1000))
  logger.info(f"Using concurrency limit of {concurrency_limit} for embedding API calls")
  
  # Process batches in parallel with a semaphore to limit concurrent API calls
  semaphore = asyncio.Semaphore(concurrency_limit)
  
  async def process_batch_with_semaphore(i, chunks, ids):
    async with semaphore:
      logger.info(f"Processing batch {i+1}/{len(all_chunks)}")
      return await process_batch_and_update(kb, index, chunks, ids)
  
  # Create tasks for batch processing with the semaphore for concurrent processing
  # Process batches in smaller groups to allow checkpointing
  checkpoint_interval = getattr(kb, 'checkpoint_interval', 10)
  for batch_group_idx in range(0, len(all_chunks), checkpoint_interval):
    # Get a group of batches to process
    batch_group = all_chunks[batch_group_idx:batch_group_idx + checkpoint_interval]
    id_group = all_ids[batch_group_idx:batch_group_idx + checkpoint_interval]
    
    # Process this group of batches concurrently
    tasks = []
    for i, (chunks, ids) in enumerate(zip(batch_group, id_group)):
      tasks.append(process_batch_with_semaphore(batch_group_idx + i, chunks, ids))
    
    # Wait for all tasks in this group to complete
    batch_results = await asyncio.gather(*tasks)
    
    # Combine results
    for successful_ids in batch_results:
      all_processed_ids.update(successful_ids)
    
    # Checkpoint after each group
    logger.info(f"Checkpoint: saving progress after {len(batch_group)} batches")
    
    # Save FAISS index
    faiss.write_index(index, kb.knowledge_base_vector)
    
    # Update database to mark processed chunks as embedded
    if all_processed_ids:
      # Process in smaller batches to avoid "too many SQL variables" error
      sql_batch_size = getattr(kb, 'sql_batch_size', 500)  # SQLite typically has a limit of 999 variables
      processed_ids_list = list(all_processed_ids)
      
      for i in range(0, len(processed_ids_list), sql_batch_size):
        batch = processed_ids_list[i:i+sql_batch_size]
        from utils.security_utils import safe_sql_in_query
        # Use safe SQL execution for ID list
        query_template = "UPDATE docs SET embedded=1 WHERE id IN ({placeholders})"
        safe_sql_in_query(kb.sql_cursor, query_template, batch)
      
      kb.sql_connection.commit()
  
  # Final save
  faiss.write_index(index, kb.knowledge_base_vector)
  return all_processed_ids

def process_embeddings(args: argparse.Namespace, logger) -> str:
  """
  Generate embeddings for the text data stored in the CustomKB knowledge base.
  
  Takes text chunks from the database and generates vector embeddings using AI models.
  Features optimizations including checkpoint updating, batch processing, forced delays 
  between API calls, and local embedding caching for redundant texts.
  
  Args:
      args: Command-line arguments containing:
          config_file: Path to knowledge base configuration
          reset_database: Flag to reset already-embedded status in database
          verbose: Enable verbose output
          debug: Enable debug output
      logger: Initialized logger instance

  Returns:
      A status message indicating the number of embeddings created and saved.
  """
  # Get configuration file
  config_file = get_fq_cfg_filename(args.config_file)
  if not config_file:
    return "Error: Configuration file not found."
    
  logger.info(f"{config_file=}")
  reset_database = args.reset_database

  # Initialize knowledge base
  kb = KnowledgeBase(config_file)
  if args.verbose:
    kb.save_config()

  # Configure cache manager with KB settings
  configure_cache_manager(kb)

  from utils.logging_utils import log_model_operation, OperationLogger
  
  # Log embedding operation start
  log_model_operation(logger, "embedding_start", kb.vector_model,
                     vector_dimensions=kb.vector_dimensions,
                     vector_chunks=kb.vector_chunks)

  logger.info(f"Embedding data from database {kb.knowledge_base_db} to vector file {kb.knowledge_base_vector}.")

  # Check if database exists
  if not os.path.exists(kb.knowledge_base_db):
    return f"Error: Database {kb.knowledge_base_db} does not yet exist!"

  # Connect to database
  connect_to_database(kb)

  # Reset embedded flag if requested or if vector file doesn't exist
  if not os.path.exists(kb.knowledge_base_vector) or reset_database:
    kb.sql_cursor.execute("UPDATE docs SET embedded=0;")
    kb.sql_connection.commit()

  # Get rows to embed
  kb.sql_cursor.execute("SELECT id, embedtext FROM docs WHERE embedded=0 and embedtext != '';")
  rows = kb.sql_cursor.fetchall()
  if not rows:
    close_database(kb)
    return f"No rows were found to embed."

  logger.info(f"{len(rows)} found for embedding.")

  # Initialize or load FAISS index
  if os.path.exists(kb.knowledge_base_vector):
    logger.info(f'Opening existing {kb.knowledge_base_vector} embeddings.')
    index = faiss.read_index(kb.knowledge_base_vector)
  else:
    logger.info(f'Creating new {kb.knowledge_base_vector} embeddings file.')
    # Get embedding dimensions from a sample or from cache
    cached_embedding = get_cached_embedding(rows[0][1], kb.vector_model)
    if cached_embedding:
      embedding_dimensions = len(cached_embedding)
    else:
      response = openai_client.embeddings.create(input=rows[0][1], model=kb.vector_model)
      embedding_dimensions = len(response.data[0].embedding)
      # Cache this embedding
      save_embedding_to_cache(rows[0][1], kb.vector_model, response.data[0].embedding, kb)
    
    logger.info(f"Using {embedding_dimensions=}")
    index = get_optimal_faiss_index(embedding_dimensions, len(rows), kb)
    reset_database = True

  # Reset embedded flag if needed
  if reset_database:
    logger.info('Resetting already-embedded flag in database')
    kb.sql_cursor.execute("UPDATE docs SET embedded=0;")
    kb.sql_connection.commit()

  # Prepare batches for async processing
  all_chunks = []
  all_ids = []
  current_chunks = []
  current_ids = []
  progress_counter = 0
  
  # Calculate optimal batch size
  sample_chunks = [row[1] for row in rows[:min(100, len(rows))]]
  configured_batch_size = kb.vector_chunks
  optimal_batch_size = calculate_optimal_batch_size(sample_chunks, kb.vector_model, configured_batch_size, kb)
  logger.info(f"Using optimal batch size of {optimal_batch_size} (configured max: {configured_batch_size})")
  
  # Deduplicate texts to avoid embedding the same text multiple times
  text_to_ids: Dict[str, List[int]] = {}
  for row_id, text in rows:
    if text in text_to_ids:
      text_to_ids[text].append(row_id)
    else:
      text_to_ids[text] = [row_id]
  
  unique_rows = [(text_to_ids[text][0], text) for text in text_to_ids]
  logger.info(f"Reduced to {len(unique_rows)} unique texts (from {len(rows)} total)")
  
  for row_id, text in unique_rows:
    progress_counter += 1
    
    if (progress_counter % optimal_batch_size) == 0:
      logger.info(f"{progress_counter}/{len(unique_rows)} ~{time_to_finish(kb.start_time, progress_counter, len(unique_rows))}")
    
    current_chunks.append(text)
    current_ids.append(row_id)
    
    if len(current_chunks) >= optimal_batch_size:
      all_chunks.append(current_chunks)
      all_ids.append(current_ids)
      current_chunks = []
      current_ids = []
  
  # Add any remaining chunks
  if current_chunks:
    all_chunks.append(current_chunks)
    all_ids.append(current_ids)
  
  # Train the index if it requires training (IVF indexes need training)
  if hasattr(index.index, 'is_trained') and not index.index.is_trained:
    logger.info("Training FAISS index on sample data...")
    # Collect sample texts for training
    train_sample_size = min(getattr(kb, 'faiss_train_sample_size', 10000), len(unique_rows))
    train_texts = [text for _, text in unique_rows[:train_sample_size]]
    
    # Get embeddings for training samples
    logger.info(f"Getting embeddings for {len(train_texts)} training samples...")
    train_embeddings = []
    
    # Process training samples in batches
    for i in range(0, len(train_texts), optimal_batch_size):
      batch = train_texts[i:i + optimal_batch_size]
      batch_embeddings = asyncio.run(get_embeddings_for_batch(kb, batch))
      if batch_embeddings:
        train_embeddings.extend(batch_embeddings)
    
    if train_embeddings:
      # Convert to numpy array
      train_embeddings_np = np.array(train_embeddings, dtype=np.float32)
      logger.info(f"Training index with {len(train_embeddings_np)} samples...")
      index.index.train(train_embeddings_np)
      logger.info("Index training completed")
    else:
      logger.error("Failed to get training embeddings - index may not work properly")
  
  # Process all batches using asyncio with checkpointing
  all_processed_ids = asyncio.run(process_all_batches_with_checkpoints(kb, index, all_chunks, all_ids))
  
  # Handle duplicated texts - ensure all copies get marked as embedded
  all_duplicate_ids = set()
  for text, id_list in text_to_ids.items():
    # Add all ids from duplicate texts if the primary was successfully processed
    if id_list[0] in all_processed_ids:
      all_duplicate_ids.update(id_list)
  
  # Mark all rows as embedded
  if all_duplicate_ids:
    # Process in smaller batches to avoid "too many SQL variables" error
    batch_size = 500  # SQLite typically has a limit of 999 variables
    duplicate_ids_list = list(all_duplicate_ids)
    
    for i in range(0, len(duplicate_ids_list), batch_size):
      batch = duplicate_ids_list[i:i+batch_size]
      from utils.security_utils import safe_sql_in_query
      # Use safe SQL execution for ID list
      query_template = "UPDATE docs SET embedded=1 WHERE id IN ({placeholders})"
      safe_sql_in_query(kb.sql_cursor, query_template, batch)
    
    kb.sql_connection.commit()
  
  # Save final index
  if hasattr(index.index, 'train_mode'):
    index.index.train_mode = False  # Disable training mode for final usage
  faiss.write_index(index, kb.knowledge_base_vector)

  # Close database connection
  close_database(kb)

  return f"{len(all_duplicate_ids)} embeddings (from {len(rows)} total rows with {len(unique_rows)} unique texts) saved to {kb.knowledge_base_vector}"

#fin