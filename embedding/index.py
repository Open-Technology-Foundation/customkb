#!/usr/bin/env python
"""
FAISS index management for CustomKB.

This module handles FAISS index creation, optimization,
persistence, and similarity search operations.
"""

import os
import json
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path

# Load FAISS with proper GPU initialization
from utils.faiss_loader import get_faiss
faiss, FAISS_GPU_AVAILABLE = get_faiss()

from utils.logging_config import get_logger
from utils.exceptions import IndexError as CustomIndexError

logger = get_logger(__name__)


def get_optimal_faiss_index(dimensions: int, dataset_size: int, kb=None) -> faiss.Index:
  """
  Create an optimal FAISS index based on dataset characteristics.
  
  Args:
      dimensions: Embedding dimensions
      dataset_size: Expected number of vectors
      kb: Optional KnowledgeBase for configuration
      
  Returns:
      Configured FAISS index
  """
  # Get configuration from KB if available
  use_gpu = getattr(kb, 'use_gpu_faiss', False) if kb else False
  index_type = getattr(kb, 'faiss_index_type', 'auto') if kb else 'auto'
  
  # Auto-select index type based on dataset size
  if index_type == 'auto':
    if dataset_size < 10000:
      index_type = 'flat'  # Exact search for small datasets
    elif dataset_size < 100000:
      index_type = 'ivf'  # IVF for medium datasets
    else:
      index_type = 'hnsw'  # HNSW for large datasets
  
  logger.info(f"Creating FAISS index: type={index_type}, dimensions={dimensions}, "
             f"dataset_size={dataset_size}, gpu={use_gpu}")
  
  # Create base index
  if index_type == 'flat':
    # Flat index for exact search
    index = faiss.IndexFlatL2(dimensions)
    
  elif index_type == 'ivf':
    # IVF index for faster search with some accuracy loss
    nlist = min(int(np.sqrt(dataset_size)), 4096)  # Number of clusters
    quantizer = faiss.IndexFlatL2(dimensions)
    index = faiss.IndexIVFFlat(quantizer, dimensions, nlist)
    
  elif index_type == 'hnsw':
    # HNSW index for very fast approximate search
    M = 32  # Number of connections per layer
    index = faiss.IndexHNSWFlat(dimensions, M)
    
  elif index_type == 'pq':
    # Product Quantization for memory efficiency
    M = 8  # Number of subquantizers
    nbits = 8  # Bits per subquantizer
    index = faiss.IndexPQ(dimensions, M, nbits)
    
  else:
    # Default to flat index
    logger.warning(f"Unknown index type: {index_type}, using flat index")
    index = faiss.IndexFlatL2(dimensions)
  
  # Wrap in IDMap for document ID tracking
  index = faiss.IndexIDMap(index)
  
  # Move to GPU if available and requested
  if use_gpu and faiss.get_num_gpus() > 0:
    try:
      logger.info("Moving FAISS index to GPU")
      gpu_resource = faiss.StandardGpuResources()
      index = faiss.index_cpu_to_gpu(gpu_resource, 0, index)
    except Exception as e:
      logger.warning(f"Failed to move index to GPU: {e}, using CPU")
  
  return index


def train_index(index: faiss.Index, training_vectors: np.ndarray) -> None:
  """
  Train index if it requires training (e.g., IVF, PQ).
  
  Args:
      index: FAISS index to train
      training_vectors: Vectors to use for training
  """
  # Check if index needs training
  if hasattr(index, 'is_trained') and not index.is_trained:
    logger.info(f"Training index with {len(training_vectors)} vectors")
    
    # Ensure we have enough training data
    min_training = getattr(index, 'nlist', 100) * 40  # IVF needs ~40 vectors per cluster
    
    if len(training_vectors) < min_training:
      logger.warning(f"Insufficient training data: {len(training_vectors)} < {min_training}")
      # Duplicate vectors if needed
      while len(training_vectors) < min_training:
        training_vectors = np.vstack([training_vectors, training_vectors])
      training_vectors = training_vectors[:min_training]
    
    # Train the index
    index.train(training_vectors)
    logger.info("Index training completed")


def add_vectors_to_index(index: faiss.Index, vectors: np.ndarray, 
                        ids: Optional[np.ndarray] = None) -> int:
  """
  Add vectors to the index.
  
  Args:
      index: FAISS index
      vectors: Vectors to add (shape: [n, d])
      ids: Optional IDs for vectors
      
  Returns:
      Number of vectors added
  """
  if len(vectors) == 0:
    return 0
  
  # Ensure vectors are float32
  if vectors.dtype != np.float32:
    vectors = vectors.astype(np.float32)
  
  # Generate IDs if not provided
  if ids is None:
    start_id = index.ntotal
    ids = np.arange(start_id, start_id + len(vectors), dtype=np.int64)
  else:
    ids = ids.astype(np.int64)
  
  # Add to index
  if isinstance(index, faiss.IndexIDMap):
    index.add_with_ids(vectors, ids)
  else:
    index.add(vectors)
  
  logger.debug(f"Added {len(vectors)} vectors to index (total: {index.ntotal})")
  return len(vectors)


def search_index(index: faiss.Index, query_vectors: np.ndarray, 
                k: int = 10, nprobe: int = 10) -> Tuple[np.ndarray, np.ndarray]:
  """
  Search the index for similar vectors.
  
  Args:
      index: FAISS index
      query_vectors: Query vectors (shape: [n, d])
      k: Number of nearest neighbors
      nprobe: Number of clusters to search (for IVF indexes)
      
  Returns:
      Tuple of (distances, indices)
  """
  # Ensure query vectors are float32
  if query_vectors.dtype != np.float32:
    query_vectors = query_vectors.astype(np.float32)
  
  # Ensure query is 2D
  if query_vectors.ndim == 1:
    query_vectors = query_vectors.reshape(1, -1)
  
  # Set search parameters for IVF indexes
  if hasattr(index, 'nprobe'):
    index.nprobe = nprobe
  
  # Search
  distances, indices = index.search(query_vectors, k)
  
  return distances, indices


def save_index(index: faiss.Index, index_path: str, metadata: Dict[str, Any] = None) -> None:
  """
  Save FAISS index to disk.
  
  Args:
      index: FAISS index to save
      index_path: Path to save the index
      metadata: Optional metadata to save alongside
  """
  try:
    # Create directory if needed
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    
    # Save index
    faiss.write_index(index, index_path)
    logger.info(f"Saved FAISS index to {index_path} ({index.ntotal} vectors)")
    
    # Save metadata if provided
    if metadata:
      metadata_path = index_path + '.meta'
      with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
      logger.debug(f"Saved index metadata to {metadata_path}")
    
  except Exception as e:
    logger.error(f"Failed to save index: {e}")
    raise CustomIndexError(f"Index save failed: {e}") from e


def load_index(index_path: str) -> Tuple[faiss.Index, Optional[Dict[str, Any]]]:
  """
  Load FAISS index from disk.
  
  Args:
      index_path: Path to the index file
      
  Returns:
      Tuple of (index, metadata)
  """
  try:
    # Load index
    if not os.path.exists(index_path):
      raise FileNotFoundError(f"Index file not found: {index_path}")
    
    index = faiss.read_index(index_path)
    logger.info(f"Loaded FAISS index from {index_path} ({index.ntotal} vectors)")
    
    # Load metadata if exists
    metadata = None
    metadata_path = index_path + '.meta'
    if os.path.exists(metadata_path):
      with open(metadata_path, 'r') as f:
        metadata = json.load(f)
      logger.debug(f"Loaded index metadata from {metadata_path}")
    
    return index, metadata
    
  except Exception as e:
    logger.error(f"Failed to load index: {e}")
    raise CustomIndexError(f"Index load failed: {e}") from e


def merge_indexes(indexes: List[faiss.Index]) -> faiss.Index:
  """
  Merge multiple FAISS indexes into one.
  
  Args:
      indexes: List of indexes to merge
      
  Returns:
      Merged index
  """
  if not indexes:
    raise ValueError("No indexes to merge")
  
  if len(indexes) == 1:
    return indexes[0]
  
  # Get dimensions from first index
  dimensions = indexes[0].d
  
  # Create new index of same type
  merged = get_optimal_faiss_index(
    dimensions, 
    sum(idx.ntotal for idx in indexes)
  )
  
  # Merge all indexes
  for idx in indexes:
    if idx.ntotal > 0:
      # Extract vectors and IDs
      vectors = idx.reconstruct_n(0, idx.ntotal)
      
      # Add to merged index
      add_vectors_to_index(merged, vectors)
  
  logger.info(f"Merged {len(indexes)} indexes into one with {merged.ntotal} vectors")
  return merged


def optimize_index(index: faiss.Index, optimization_level: str = 'medium') -> faiss.Index:
  """
  Optimize index for better performance.
  
  Args:
      index: Index to optimize
      optimization_level: Level of optimization ('low', 'medium', 'high')
      
  Returns:
      Optimized index
  """
  logger.info(f"Optimizing index with level: {optimization_level}")
  
  if optimization_level == 'low':
    # Just ensure index is compacted
    if hasattr(index, 'compact'):
      index.compact()
    
  elif optimization_level == 'medium':
    # Rebuild with better parameters
    if index.ntotal > 10000 and isinstance(index, faiss.IndexFlatL2):
      # Convert flat index to IVF for better performance
      dimensions = index.d
      vectors = index.reconstruct_n(0, index.ntotal)
      
      # Create IVF index
      nlist = min(int(np.sqrt(index.ntotal)), 4096)
      quantizer = faiss.IndexFlatL2(dimensions)
      new_index = faiss.IndexIVFFlat(quantizer, dimensions, nlist)
      new_index.train(vectors)
      new_index.add(vectors)
      
      index = faiss.IndexIDMap(new_index)
      logger.info(f"Converted flat index to IVF with {nlist} clusters")
    
  elif optimization_level == 'high':
    # Maximum optimization with compression
    if index.ntotal > 100000:
      dimensions = index.d
      vectors = index.reconstruct_n(0, index.ntotal)
      
      # Create HNSW index for very fast search
      M = 48
      index = faiss.IndexHNSWFlat(dimensions, M)
      index.add(vectors)
      index = faiss.IndexIDMap(index)
      
      logger.info(f"Created optimized HNSW index with M={M}")
  
  return index


def get_index_stats(index: faiss.Index) -> Dict[str, Any]:
  """
  Get statistics about the index.
  
  Args:
      index: FAISS index
      
  Returns:
      Dictionary with index statistics
  """
  stats = {
    'total_vectors': index.ntotal,
    'dimensions': index.d,
    'index_type': type(index).__name__,
    'is_trained': getattr(index, 'is_trained', True),
    'memory_usage_mb': 0
  }
  
  # Estimate memory usage
  if hasattr(index, 'sa_code_size'):
    # For binary indexes
    stats['memory_usage_mb'] = (index.sa_code_size() * index.ntotal) / (1024 * 1024)
  else:
    # Estimate based on float32 vectors
    stats['memory_usage_mb'] = (index.ntotal * index.d * 4) / (1024 * 1024)
  
  # Add index-specific stats
  if hasattr(index, 'nlist'):
    stats['num_clusters'] = index.nlist
  if hasattr(index, 'nprobe'):
    stats['search_nprobe'] = index.nprobe
  
  return stats


def remove_vectors_from_index(index: faiss.Index, ids_to_remove: List[int]) -> int:
  """
  Remove vectors from the index by ID.
  
  Args:
      index: FAISS index (must be IDMap)
      ids_to_remove: List of IDs to remove
      
  Returns:
      Number of vectors removed
  """
  if not isinstance(index, faiss.IndexIDMap):
    raise ValueError("Index must be IndexIDMap to remove by ID")
  
  if not ids_to_remove:
    return 0
  
  # Convert to numpy array
  ids_array = np.array(ids_to_remove, dtype=np.int64)
  
  # Remove from index
  removed = index.remove_ids(ids_array)
  
  logger.info(f"Removed {removed} vectors from index")
  return removed


#fin