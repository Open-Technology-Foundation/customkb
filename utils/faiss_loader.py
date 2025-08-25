#!/usr/bin/env python
"""
FAISS loader with proper GPU initialization handling.

This module ensures FAISS loads correctly even when CUDA has initialization issues.
It attempts to use GPU acceleration when available but falls back gracefully.
"""

import os
import sys
import logging
import warnings

logger = logging.getLogger(__name__)

# Global flag to track if we've already attempted GPU initialization
_GPU_INIT_ATTEMPTED = False
_FAISS_MODULE = None
_GPU_AVAILABLE = False

def initialize_faiss_gpu():
  """
  Attempt to initialize FAISS with GPU support.
  
  Returns:
      tuple: (faiss module, gpu_available boolean)
  """
  global _GPU_INIT_ATTEMPTED, _FAISS_MODULE, _GPU_AVAILABLE
  
  if _GPU_INIT_ATTEMPTED:
    return _FAISS_MODULE, _GPU_AVAILABLE
  
  _GPU_INIT_ATTEMPTED = True
  
  # First, check if GPU should be explicitly disabled
  if os.getenv('FAISS_NO_GPU', '').lower() in ('1', 'true', 'yes'):
    logger.info("FAISS GPU disabled by environment variable")
    import faiss
    _FAISS_MODULE = faiss
    _GPU_AVAILABLE = False
    return faiss, False
  
  # Try to detect CUDA issues before importing FAISS
  cuda_available = check_cuda_availability()
  
  if not cuda_available:
    logger.warning("CUDA not available, using CPU-only FAISS")
    # Set environment to prevent FAISS from trying GPU
    os.environ['FAISS_NO_GPU'] = '1'
    import faiss
    _FAISS_MODULE = faiss
    _GPU_AVAILABLE = False
    return faiss, False
  
  # CUDA seems available, try to import FAISS with GPU
  try:
    # Suppress warnings during import
    with warnings.catch_warnings():
      warnings.simplefilter("ignore")
      import faiss
    
    # Test GPU functionality
    try:
      ngpus = faiss.get_num_gpus()
      if ngpus > 0:
        # Try to create a GPU resource to ensure it works
        res = faiss.StandardGpuResources()
        logger.info(f"FAISS GPU initialized successfully with {ngpus} GPU(s)")
        _FAISS_MODULE = faiss
        _GPU_AVAILABLE = True
        return faiss, True
      else:
        logger.info("FAISS loaded but no GPUs detected")
        _FAISS_MODULE = faiss
        _GPU_AVAILABLE = False
        return faiss, False
        
    except RuntimeError as e:
      error_str = str(e)
      if "999" in error_str or "unknown error" in error_str.lower():
        logger.error("CUDA error 999 detected - GPU driver/runtime mismatch")
        logger.error("This is a system-level issue that requires:")
        logger.error("  1. Reboot the server")
        logger.error("  2. Reinstall CUDA toolkit")
        logger.error("  3. Update NVIDIA drivers")
        logger.error("Falling back to CPU mode")
      else:
        logger.error(f"FAISS GPU initialization failed: {e}")
      
      _FAISS_MODULE = faiss
      _GPU_AVAILABLE = False
      return faiss, False
      
  except ImportError as e:
    logger.error(f"Failed to import FAISS: {e}")
    raise
  except Exception as e:
    logger.error(f"Unexpected error during FAISS initialization: {e}")
    import faiss
    _FAISS_MODULE = faiss
    _GPU_AVAILABLE = False
    return faiss, False

def check_cuda_availability():
  """
  Check if CUDA is available and working.
  
  Returns:
      bool: True if CUDA is available and working
  """
  try:
    # First check if GPU hardware exists using nvidia-ml
    try:
      import pynvml
      pynvml.nvmlInit()
      device_count = pynvml.nvmlDeviceGetCount()
      if device_count > 0:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(name, bytes):
          name = name.decode()
        logger.info(f"GPU hardware detected: {name}")
      pynvml.nvmlShutdown()
    except:
      pass  # NVML not available or failed
    
    # Try to check CUDA using ctypes
    import ctypes
    
    # Try to load CUDA driver
    try:
      cuda_driver = ctypes.CDLL('libcuda.so.1')
      cuda_driver.cuInit.restype = ctypes.c_int
      cuda_driver.cuInit.argtypes = [ctypes.c_uint]
      
      # Try to initialize CUDA driver
      result = cuda_driver.cuInit(0)
      if result != 0:
        if result == 999:
          logger.critical("CUDA ERROR 999: Driver/runtime mismatch or corruption!")
          logger.critical("The GPU hardware exists but CUDA cannot initialize")
          logger.critical("This requires system administrator action:")
          logger.critical("  1. Try rebooting the server")
          logger.critical("  2. Reinstall CUDA toolkit matching driver version")
          logger.critical("  3. Check for conflicting CUDA installations")
        else:
          logger.debug(f"CUDA driver initialization failed with code {result}")
        return False
        
      # Check device count
      cuda_driver.cuDeviceGetCount.restype = ctypes.c_int
      cuda_driver.cuDeviceGetCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
      
      count = ctypes.c_int()
      result = cuda_driver.cuDeviceGetCount(ctypes.byref(count))
      if result != 0 or count.value == 0:
        logger.debug(f"No CUDA devices found or error {result}")
        return False
        
      logger.info(f"CUDA driver successfully initialized with {count.value} device(s)")
      return True
      
    except OSError:
      logger.debug("CUDA driver library not found")
      return False
      
  except Exception as e:
    logger.debug(f"Error checking CUDA availability: {e}")
    return False

def get_faiss():
  """
  Get FAISS module with automatic GPU/CPU selection.
  
  Returns:
      tuple: (faiss module, gpu_available boolean)
  """
  return initialize_faiss_gpu()

# Convenience function for backward compatibility
def load_faiss():
  """
  Load FAISS module (backward compatibility).
  
  Returns:
      faiss module
  """
  faiss, _ = get_faiss()
  return faiss

#fin