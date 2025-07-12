"""
GPU memory detection and management utilities.

This module provides functions to detect GPU memory capacity and make
intelligent decisions about GPU usage for FAISS operations.
"""

import os
import subprocess
import re
from typing import Optional, Tuple
from utils.logging_utils import get_logger

logger = get_logger(__name__)

# Cache for GPU memory to avoid repeated detection
_gpu_memory_cache = None


def get_gpu_memory_mb() -> Optional[int]:
  """
  Detect total GPU memory in MB.
  
  Returns:
      Total GPU memory in MB, or None if no GPU is detected.
  """
  global _gpu_memory_cache
  
  # Return cached value if available
  if _gpu_memory_cache is not None:
    return _gpu_memory_cache
  
  # Check if GPU is disabled via environment
  if os.environ.get('FAISS_NO_GPU', '').lower() in ['1', 'true']:
    logger.info("GPU disabled via FAISS_NO_GPU environment variable")
    return None
  
  # Try PyTorch first (most reliable)
  try:
    import torch
    if torch.cuda.is_available():
      device_props = torch.cuda.get_device_properties(0)
      memory_bytes = device_props.total_memory
      memory_mb = memory_bytes // (1024 * 1024)
      logger.info(f"Detected GPU memory via PyTorch: {memory_mb} MB")
      _gpu_memory_cache = memory_mb
      return memory_mb
  except ImportError:
    logger.debug("PyTorch not available for GPU detection")
  except Exception as e:
    logger.warning(f"PyTorch GPU detection failed: {e}")
  
  # Fallback to nvidia-smi
  try:
    result = subprocess.run(
      ['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits'],
      capture_output=True,
      text=True,
      timeout=5
    )
    
    if result.returncode == 0:
      # Parse output (might have multiple GPUs)
      lines = result.stdout.strip().split('\n')
      if lines and lines[0]:
        # Use first GPU's memory
        memory_mb = int(lines[0])
        logger.info(f"Detected GPU memory via nvidia-smi: {memory_mb} MB")
        _gpu_memory_cache = memory_mb
        return memory_mb
  except (subprocess.SubprocessError, ValueError) as e:
    logger.debug(f"nvidia-smi GPU detection failed: {e}")
  except FileNotFoundError:
    logger.debug("nvidia-smi not found")
  
  # No GPU detected
  logger.info("No GPU detected for FAISS operations")
  _gpu_memory_cache = None
  return None


def get_safe_gpu_memory_limit_mb(buffer_gb: float = 4.0) -> Optional[int]:
  """
  Get safe GPU memory limit for FAISS operations.
  
  Leaves a buffer for other GPU operations and prevents OOM errors.
  
  Args:
      buffer_gb: Safety buffer in GB to reserve (default 4GB).
      
  Returns:
      Safe memory limit in MB, or None if no GPU.
  """
  # Check for manual override
  override = os.environ.get('FAISS_GPU_MEMORY_LIMIT_MB')
  if override:
    try:
      limit = int(override)
      logger.info(f"Using manual GPU memory limit: {limit} MB")
      return limit
    except ValueError:
      logger.warning(f"Invalid FAISS_GPU_MEMORY_LIMIT_MB: {override}")
  
  # Detect GPU memory
  total_memory_mb = get_gpu_memory_mb()
  if total_memory_mb is None:
    return None
  
  # Calculate safe limit
  buffer_mb = int(buffer_gb * 1024)
  safe_limit_mb = max(0, total_memory_mb - buffer_mb)
  
  logger.info(f"GPU memory: {total_memory_mb} MB total, {safe_limit_mb} MB available for FAISS (buffer: {buffer_gb} GB)")
  
  return safe_limit_mb


def should_use_gpu_for_index(index_size_mb: float, kb_config: Optional[object] = None) -> Tuple[bool, str]:
  """
  Determine if GPU should be used for a FAISS index.
  
  Args:
      index_size_mb: Size of the FAISS index in MB.
      kb_config: Optional KnowledgeBase config object for custom settings.
      
  Returns:
      Tuple of (should_use_gpu, reason_message)
  """
  # Get buffer size from config or use default
  buffer_gb = 4.0
  if kb_config and hasattr(kb_config, 'faiss_gpu_memory_buffer_gb'):
    buffer_gb = float(kb_config.faiss_gpu_memory_buffer_gb)
  
  # Get safe memory limit
  safe_limit_mb = get_safe_gpu_memory_limit_mb(buffer_gb)
  
  if safe_limit_mb is None:
    return False, "No GPU detected"
  
  if safe_limit_mb <= 0:
    return False, f"Insufficient GPU memory (buffer: {buffer_gb} GB)"
  
  if index_size_mb > safe_limit_mb:
    return False, f"Index too large ({index_size_mb:.1f} MB > {safe_limit_mb} MB limit)"
  
  # Check if GPU is explicitly disabled for this KB
  if kb_config and hasattr(kb_config, 'disable_gpu_faiss'):
    if kb_config.disable_gpu_faiss:
      return False, "GPU disabled in knowledge base configuration"
  
  return True, f"Index fits in GPU memory ({index_size_mb:.1f} MB < {safe_limit_mb} MB limit)"


def get_gpu_info_string() -> str:
  """
  Get a human-readable string with GPU information.
  
  Returns:
      String describing GPU status and memory.
  """
  memory_mb = get_gpu_memory_mb()
  
  if memory_mb is None:
    return "No GPU detected"
  
  memory_gb = memory_mb / 1024.0
  return f"GPU detected: {memory_gb:.1f} GB memory"


def reset_gpu_memory_cache():
  """Reset the GPU memory cache. Useful for testing."""
  global _gpu_memory_cache
  _gpu_memory_cache = None


#fin