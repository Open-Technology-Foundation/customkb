#!/usr/bin/env python
"""
GPU index configuration for large FAISS indexes.

This module provides advanced GPU configurations for handling large indexes
that may not fit entirely in GPU memory.
"""

import faiss
import logging

logger = logging.getLogger(__name__)

def configure_gpu_resources_for_large_index(temp_memory_gb=2.0, pinned_memory_gb=4.0):
  """
  Configure GPU resources optimized for large indexes.
  
  Args:
      temp_memory_gb: Temporary memory allocation in GB
      pinned_memory_gb: Pinned memory allocation in GB
      
  Returns:
      Configured StandardGpuResources object
  """
  res = faiss.StandardGpuResources()
  
  # Limit temporary memory to prevent OOM
  res.setTempMemory(int(temp_memory_gb * 1024 * 1024 * 1024))
  
  # Set pinned memory for CPU-GPU transfers
  res.setPinnedMemory(int(pinned_memory_gb * 1024 * 1024 * 1024))
  
  # Don't use unified memory (can be slower)
  res.noTempMemory()
  
  return res

def create_gpu_index_config(use_float16=True, use_precomputed_tables=True):
  """
  Create GPU index configuration options.
  
  Args:
      use_float16: Use 16-bit floats to save memory
      use_precomputed_tables: Use precomputed tables for speed
      
  Returns:
      GpuClonerOptions or GpuIndexIVFConfig
  """
  config = faiss.GpuClonerOptions()
  config.useFloat16 = use_float16
  config.usePrecomputed = use_precomputed_tables
  config.indicesOptions = faiss.INDICES_32_BIT  # Use 32-bit indices
  config.storeTransposed = True  # Better memory access pattern
  
  return config

def estimate_gpu_memory_usage(index_size_bytes, use_float16=True):
  """
  Estimate GPU memory requirements for a FAISS index.
  
  Args:
      index_size_bytes: Size of the index file in bytes
      use_float16: Whether Float16 will be used
      
  Returns:
      Tuple of (index_memory_gb, temp_memory_gb, total_memory_gb)
  """
  # Base index memory
  index_memory = index_size_bytes
  
  # Adjust for Float16 (roughly 60% of original size)
  if use_float16:
    index_memory *= 0.6
    
  # Temporary memory overhead (typically 50-100% of index size)
  temp_memory = index_memory * 0.5
  
  # Convert to GB
  index_memory_gb = index_memory / (1024**3)
  temp_memory_gb = temp_memory / (1024**3)
  total_memory_gb = index_memory_gb + temp_memory_gb
  
  return index_memory_gb, temp_memory_gb, total_memory_gb

def use_gpu_index_subset(index, gpu_resources, subset_size=1000000):
  """
  For very large indexes, keep only a subset on GPU.
  
  This is useful when the full index doesn't fit but you want
  to accelerate searches for the most common vectors.
  
  Args:
      index: CPU index
      gpu_resources: Configured GPU resources
      subset_size: Number of vectors to keep on GPU
      
  Returns:
      GPU index with subset of vectors
  """
  if index.ntotal <= subset_size:
    # Full index fits
    return faiss.index_cpu_to_gpu(gpu_resources, 0, index)
  
  # Create a subset index
  logger.info(f"Creating GPU index subset with {subset_size} vectors")
  
  # For now, return None - this would require more complex implementation
  # involving index sharding or hierarchical search
  return None

# Example usage in query_manager.py:
"""
from gpu_index_config import (
    configure_gpu_resources_for_large_index,
    create_gpu_index_config,
    estimate_gpu_memory_usage
)

# Check if index fits in GPU memory
index_size_bytes = os.path.getsize(kb.knowledge_base_vector)
index_gb, temp_gb, total_gb = estimate_gpu_memory_usage(index_size_bytes)

if total_gb < 22:  # Leave 1GB buffer on 23GB GPU
    res = configure_gpu_resources_for_large_index(
        temp_memory_gb=min(temp_gb, 4.0),
        pinned_memory_gb=2.0
    )
    config = create_gpu_index_config(use_float16=True)
    index = faiss.index_cpu_to_gpu(res, 0, index, config)
"""