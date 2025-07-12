#!/usr/bin/env python
"""
CUDA initialization helper for FAISS GPU.
Ensures CUDA is properly initialized before FAISS attempts to use it.
"""

import os
import sys
import ctypes
import logging

logger = logging.getLogger(__name__)

def init_cuda():
  """
  Initialize CUDA runtime to prevent CUDA error 999.
  
  Returns:
      bool: True if CUDA initialized successfully, False otherwise.
  """
  try:
    # Try to load CUDA runtime library
    cuda_lib = None
    for lib_name in ['libcudart.so.12', 'libcudart.so.11', 'libcudart.so']:
      try:
        cuda_lib = ctypes.CDLL(lib_name)
        break
      except OSError:
        continue
    
    if not cuda_lib:
      logger.warning("CUDA runtime library not found")
      return False
    
    # Define CUDA functions
    cuda_lib.cudaSetDevice.restype = ctypes.c_int
    cuda_lib.cudaSetDevice.argtypes = [ctypes.c_int]
    
    cuda_lib.cudaGetDeviceCount.restype = ctypes.c_int
    cuda_lib.cudaGetDeviceCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
    
    cuda_lib.cudaDeviceReset.restype = ctypes.c_int
    cuda_lib.cudaDeviceReset.argtypes = []
    
    # Note: cudaInit is for the driver API, not runtime API
    # For runtime API, any CUDA call initializes the runtime
    
    # Get device count
    device_count = ctypes.c_int()
    result = cuda_lib.cudaGetDeviceCount(ctypes.byref(device_count))
    if result != 0:
      logger.warning(f"Failed to get CUDA device count: {result}")
      return False
    
    if device_count.value == 0:
      logger.warning("No CUDA devices found")
      return False
    
    # Set device 0 as current
    result = cuda_lib.cudaSetDevice(0)
    if result != 0:
      logger.warning(f"Failed to set CUDA device: {result}")
      return False
    
    logger.info(f"CUDA initialized successfully with {device_count.value} device(s)")
    return True
    
  except Exception as e:
    logger.warning(f"CUDA initialization failed: {e}")
    return False

def get_faiss_with_fallback():
  """
  Import FAISS with automatic GPU fallback.
  
  Returns:
      tuple: (faiss module, use_gpu boolean)
  """
  import faiss
  
  # Check if GPU should be disabled
  if os.getenv('FAISS_NO_GPU', '').lower() in ('1', 'true', 'yes'):
    logger.info("FAISS GPU disabled by environment variable")
    return faiss, False
  
  # Try to initialize CUDA first
  cuda_ok = init_cuda()
  
  if not cuda_ok:
    logger.warning("CUDA initialization failed, FAISS will use CPU")
    return faiss, False
  
  # Now try to get GPU count from FAISS
  try:
    ngpus = faiss.get_num_gpus()
    if ngpus > 0:
      logger.info(f"FAISS detected {ngpus} GPU(s)")
      return faiss, True
    else:
      logger.info("FAISS detected no GPUs")
      return faiss, False
  except Exception as e:
    logger.warning(f"FAISS GPU detection failed: {e}")
    return faiss, False

#fin