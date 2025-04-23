#!/usr/bin/env python
"""
Embedding management for CustomKB.
Handles the generation and storage of vector embeddings.
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
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any, Set
from concurrent.futures import ThreadPoolExecutor

from utils.logging_utils import setup_logging, get_logger, time_to_finish
from config.config_manager import KnowledgeBase, get_fq_cfg_filename
from database.db_manager import connect_to_database, close_database

# Import OpenAI client
from openai import OpenAI, AsyncOpenAI
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
  raise EnvironmentError("OPENAI_API_KEY environment variable not set.")
openai_client = OpenAI(api_key=OPENAI_API_KEY)
async_openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

logger = get_logger(__name__)

# Embedding cache directory
CACHE_DIR = os.path.join(os.getenv('VECTORDBS', '/var/lib/vectordbs'), '.embedding_cache')
os.makedirs(CACHE_DIR, exist_ok=True)

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
  Retrieve a cached embedding if it exists.
  
  Args:
      text: The text to embed.
      model: The model used for embedding.
      
  Returns:
      The cached embedding or None if not found.
  """
  cache_key = get_cache_key(text, model)
  cache_file = os.path.join(CACHE_DIR, f"{cache_key}.json")
  
  if os.path.exists(cache_file):
    try:
      with open(cache_file, 'r') as f:
        return json.load(f)
    except (json.JSONDecodeError, IOError):
      return None
  
  return None

def save_embedding_to_cache(text: str, model: str, embedding: List[float]) -> None:
  """
  Save an embedding to the cache.
  
  Args:
      text: The text that was embedded.
      model: The model used for embedding.
      embedding: The embedding vector.
  """
  cache_key = get_cache_key(text, model)
  cache_file = os.path.join(CACHE_DIR, f"{cache_key}.json")
  
  try:
    with open(cache_file, 'w') as f:
      json.dump(embedding, f)
  except IOError as e:
    logger.warning(f"Failed to cache embedding: {e}")

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
  # Approximate tokens (rough estimate)
  avg_tokens = sum(len(chunk.split()) * 1.3 for chunk in chunks) / len(chunks)
  
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
  
  while True:
    try:
      response = await async_openai_client.embeddings.create(
        input=uncached_chunks, 
        model=kb.vector_model
      )
      new_embeddings = [data.embedding for data in response.data]
      
      # Cache the new embeddings
      for i, emb in zip(uncached_indices, new_embeddings):
        save_embedding_to_cache(chunks[i], kb.vector_model, emb)
      
      # Merge cached and new embeddings
      result = cached_embeddings.copy()
      for i, emb in zip(uncached_indices, new_embeddings):
        result[i] = emb
        
      return [emb for emb in result if emb is not None]
    except Exception as e:
      logger.error(e)
      logger.error(f"{tries=} {kb.vector_model=}")
      tries += 1
      if tries > max_tries:
        logger.error(e)
        logger.error(f"{tries=}\n{kb.vector_model=}\n{uncached_chunks=}\n")
        sys.exit(1)
      # Exponential backoff with jitter
      backoff = (tries ** 2) + (0.1 * np.random.random())
      await asyncio.sleep(backoff)

async def process_all_batches(kb: KnowledgeBase, index: faiss.IndexIDMap, 
                             all_chunks: List[List[str]], all_ids: List[List[int]]) -> None:
  """
  Process all batches of embeddings concurrently.
  
  Args:
      kb: The KnowledgeBase instance.
      index: The FAISS index.
      all_chunks: List of batches of text chunks.
      all_ids: List of batches of corresponding IDs.
  """
  tasks = []
  
  # Create async tasks for each batch
  for chunks in all_chunks:
    tasks.append(process_embedding_batch_async(kb, chunks))
  
  # Process batches concurrently
  embeddings_batches = await asyncio.gather(*tasks)
  
  # Add embeddings to index
  for embeddings_list, ids in zip(embeddings_batches, all_ids):
    embeddings = np.array(embeddings_list, dtype=np.float32)
    ids_array = np.array(ids, dtype=np.int64)
    index.add_with_ids(embeddings, ids_array)
  
  # Save the index
  faiss.write_index(index, kb.knowledge_base_vector)

def process_embeddings(args: argparse.Namespace) -> str:
  """
  Generate embeddings for the text data stored in the CustomKB knowledge base.

  Args:
      args: Command-line arguments.

  Returns:
      A status message indicating the result of the operation.
  """
  global logger
  logger = setup_logging(args.verbose, args.debug)

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
    return f"Error: No rows were found to embed."

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
  optimal_batch_size = calculate_optimal_batch_size(sample_chunks, kb.vector_model, kb.vector_chunks)
  logger.info(f"Using optimal batch size of {optimal_batch_size} (configured max: {kb.vector_chunks})")
  
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
  
  # Process all batches using asyncio
  asyncio.run(process_all_batches(kb, index, all_chunks, all_ids))
  
  # Handle duplicated texts - ensure all copies get marked as embedded
  all_processed_ids = set()
  for text, id_list in text_to_ids.items():
    # Add all ids from duplicate texts
    for row_id in id_list:
      all_processed_ids.add(row_id)
  
  # Mark all rows as embedded
  placeholders = ','.join(['?'] * len(all_processed_ids))
  kb.sql_cursor.execute(f"UPDATE docs SET embedded=1 WHERE id IN ({placeholders});", 
                      list(all_processed_ids))
  kb.sql_connection.commit()
  
  # Save final index
  if hasattr(index.index, 'train_mode'):
    index.index.train_mode = False  # Disable training mode for final usage
  faiss.write_index(index, kb.knowledge_base_vector)

  # Close database connection
  close_database(kb)

  return f"{len(rows)} embeddings (with {len(unique_rows)} unique texts) saved to {kb.knowledge_base_vector}"

# Legacy method kept for backward compatibility
def process_embedding_batch(kb: KnowledgeBase, index: faiss.IndexIDMap,
                           chunks: List[str], ids: List[int]) -> None:
  """
  Process a batch of text chunks to generate embeddings and add to the FAISS index.

  Args:
      kb: The KnowledgeBase instance.
      index: The FAISS index.
      chunks: List of text chunks to embed.
      ids: List of corresponding IDs.
  """
  max_tries = 20
  tries = 0

  # Check cache first
  embeddings_list = []
  uncached_indices = []
  uncached_chunks = []
  
  for i, chunk in enumerate(chunks):
    cached_emb = get_cached_embedding(chunk, kb.vector_model)
    if cached_emb:
      embeddings_list.append(cached_emb)
    else:
      embeddings_list.append(None)
      uncached_indices.append(i)
      uncached_chunks.append(chunk)
  
  if uncached_chunks:
    while True:
      try:
        response = openai_client.embeddings.create(input=uncached_chunks, model=kb.vector_model)
        for i, data in enumerate(response.data):
          idx = uncached_indices[i]
          embeddings_list[idx] = data.embedding
          save_embedding_to_cache(chunks[idx], kb.vector_model, data.embedding)
        break
      except Exception as e:
        logger.error(e)
        logger.error(f"{tries=} {kb.vector_model=}")
        tries += 1
        if tries > max_tries:
          logger.error(e)
          logger.error(f"{tries=}\n{kb.vector_model=}\n{chunks=}\n")
          sys.exit(1)
        backoff = (tries**2) + (0.1 * np.random.random())
        time.sleep(backoff)

  embeddings = np.array(embeddings_list, dtype=np.float32)
  ids_array = np.array(ids, dtype=np.int64)
  index.add_with_ids(embeddings, ids_array)
  faiss.write_index(index, kb.knowledge_base_vector)

#fin
