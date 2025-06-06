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

# New configuration for rate limiting and checkpointing
API_CALL_DELAY = 0.05  # Forced delay of 50ms between API calls
MAX_BATCH_SIZE = 100   # Maximum number of chunks to send in a single API call
CHECKPOINT_INTERVAL = 10  # Save progress after every 10 successful batches

# Memory cache configuration
MEMORY_CACHE_SIZE = 10000  # Number of embeddings to keep in memory

# In-memory embedding cache
embedding_memory_cache = {}
embedding_memory_cache_keys = []

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
  
  # First check in-memory cache (faster)
  if cache_key in embedding_memory_cache:
    return embedding_memory_cache[cache_key]
  
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

def add_to_memory_cache(cache_key: str, embedding: List[float]) -> None:
  """
  Add an embedding to the in-memory cache with LRU eviction.
  
  Args:
      cache_key: The cache key.
      embedding: The embedding vector.
  """
  # If cache is full, remove oldest item
  if len(embedding_memory_cache_keys) >= MEMORY_CACHE_SIZE:
    oldest_key = embedding_memory_cache_keys.pop(0)
    if oldest_key in embedding_memory_cache:
      del embedding_memory_cache[oldest_key]
  
  # Add to memory cache
  embedding_memory_cache[cache_key] = embedding
  embedding_memory_cache_keys.append(cache_key)

def save_embedding_to_cache(text: str, model: str, embedding: List[float]) -> None:
  """
  Save an embedding to both memory and disk cache.
  
  Args:
      text: The text that was embedded.
      model: The model used for embedding.
      embedding: The embedding vector.
  """
  cache_key = get_cache_key(text, model)
  
  # Save to memory cache for immediate access
  add_to_memory_cache(cache_key, embedding)
  
  # Save to disk asynchronously using a thread to avoid blocking
  cache_file = os.path.join(CACHE_DIR, f"{cache_key}.json")
  
  def save_to_disk():
    try:
      with open(cache_file, 'w') as f:
        json.dump(embedding, f)
    except IOError as e:
      logger.warning(f"Failed to cache embedding: {e}")
  
  # Use thread pool for disk I/O to avoid blocking the main process
  executor = ThreadPoolExecutor(max_workers=4)
  executor.submit(save_to_disk)
  executor.shutdown(wait=False)

def get_optimal_faiss_index(dimensions: int, dataset_size: int) -> faiss.Index:
  """
  Create an optimal FAISS index based on dataset size.
  
  Args:
      dimensions: The dimensions of the embedding vectors.
      dataset_size: The expected size of the dataset.
      
  Returns:
      A FAISS index optimized for the dataset.
  """
  # For high-dimensional vectors (>1536), use a flat index 
  # which doesn't require training and works with any dimensionality
  if dimensions > 1536:
    logger.info(f"Using IndexFlatIP due to high dimensionality: {dimensions}")
    index = faiss.IndexFlatIP(dimensions)
    return faiss.IndexIDMap(index)
    
  # For smaller datasets, use exact search
  if dataset_size < 1000:
    index = faiss.IndexFlatIP(dimensions)
  elif dataset_size < 100000:
    # For medium datasets, use IVF with 4*sqrt(n) centroids
    n_centroids = min(int(4 * (dataset_size ** 0.5)), 256)  # Limit number of centroids
    quantizer = faiss.IndexFlatIP(dimensions)
    index = faiss.IndexIVFFlat(quantizer, dimensions, n_centroids, faiss.METRIC_INNER_PRODUCT)
    index.train_mode = True  # Enable training mode initially
  else:
    # For large datasets, use IVF with PQ for compression
    n_centroids = min(int(4 * (dataset_size ** 0.5)), 512)  # Limit number of centroids
    quantizer = faiss.IndexFlatIP(dimensions)
    # Use 8-bit quantization with 16 subquantizers
    n_subquantizers = min(16, dimensions // 64)  # Ensure subquantizers fit dimensions
    index = faiss.IndexIVFPQ(quantizer, dimensions, n_centroids, n_subquantizers, 8, faiss.METRIC_INNER_PRODUCT)
    index.train_mode = True  # Enable training mode initially
  
  return faiss.IndexIDMap(index)

def calculate_optimal_batch_size(chunks: List[str], model: str, max_batch_size: int) -> int:
  """
  Calculate the optimal batch size based on token limits.
  
  Args:
      chunks: The text chunks to embed.
      model: The embedding model.
      max_batch_size: The maximum batch size.
      
  Returns:
      An optimal batch size.
  """
  # More efficient token estimation
  # Sample a few chunks rather than processing all of them
  sample_size = min(10, len(chunks))
  sample_chunks = chunks[:sample_size]
  avg_tokens = sum(len(chunk.split()) * 1.3 for chunk in sample_chunks) / sample_size
  
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
  max_tries = 20
  tries = 0
  cached_embeddings: List[Optional[List[float]]] = [get_cached_embedding(chunk, kb.vector_model) for chunk in chunks]
  uncached_indices = [i for i, emb in enumerate(cached_embeddings) if emb is None]
  
  if not uncached_indices:
    # All embeddings found in cache
    return [emb for emb in cached_embeddings if emb is not None]
  
  uncached_chunks = [chunks[i] for i in uncached_indices]
  
  # Split into smaller sub-batches to improve reliability
  sub_batch_size = min(MAX_BATCH_SIZE, len(uncached_chunks))
  sub_batches = [uncached_chunks[i:i+sub_batch_size] for i in range(0, len(uncached_chunks), sub_batch_size)]
  sub_indices = [uncached_indices[i:i+sub_batch_size] for i in range(0, len(uncached_indices), sub_batch_size)]
  
  all_embeddings = []
  all_indices = []
  
  for sub_batch, sub_idx in zip(sub_batches, sub_indices):
    tries = 0
    while True:
      try:
        # Add forced delay to avoid rate limiting
        await asyncio.sleep(API_CALL_DELAY)
        
        response = await async_openai_client.embeddings.create(
          input=sub_batch, 
          model=kb.vector_model
        )
        
        new_embeddings = [data.embedding for data in response.data]
        
        # Cache the new embeddings
        for i, chunk_idx, emb in zip(range(len(sub_batch)), sub_idx, new_embeddings):
          save_embedding_to_cache(chunks[chunk_idx], kb.vector_model, emb)
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
        backoff = (tries ** 2) + (0.1 * np.random.random())
        logger.info(f"Rate limit hit. Backing off for {backoff:.2f} seconds")
        await asyncio.sleep(backoff)
  
  # Merge cached and new embeddings
  result = cached_embeddings.copy()
  for idx, emb in all_embeddings:
    result[idx] = emb
      
  return [emb for emb in result if emb is not None]

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
  concurrency_limit = min(8, max(3, dataset_size // 1000))
  logger.info(f"Using concurrency limit of {concurrency_limit} for embedding API calls")
  
  # Process batches in parallel with a semaphore to limit concurrent API calls
  semaphore = asyncio.Semaphore(concurrency_limit)
  
  async def process_batch_with_semaphore(i, chunks, ids):
    async with semaphore:
      logger.info(f"Processing batch {i+1}/{len(all_chunks)}")
      return await process_batch_and_update(kb, index, chunks, ids)
  
  # Create tasks for batch processing with the semaphore for concurrent processing
  # Process batches in smaller groups to allow checkpointing
  for batch_group_idx in range(0, len(all_chunks), CHECKPOINT_INTERVAL):
    # Get a group of batches to process
    batch_group = all_chunks[batch_group_idx:batch_group_idx + CHECKPOINT_INTERVAL]
    id_group = all_ids[batch_group_idx:batch_group_idx + CHECKPOINT_INTERVAL]
    
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
      batch_size = 500  # SQLite typically has a limit of 999 variables
      processed_ids_list = list(all_processed_ids)
      
      for i in range(0, len(processed_ids_list), batch_size):
        batch = processed_ids_list[i:i+batch_size]
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
      save_embedding_to_cache(rows[0][1], kb.vector_model, response.data[0].embedding)
    
    logger.info(f"Using {embedding_dimensions=}")
    index = get_optimal_faiss_index(embedding_dimensions, len(rows))
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
  optimal_batch_size = calculate_optimal_batch_size(sample_chunks, kb.vector_model, configured_batch_size)
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