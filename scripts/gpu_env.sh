#!/bin/bash
# GPU Environment Configuration for CustomKB
# Source this file before running CustomKB for optimal GPU performance

# Force GPU 0 (L4)
export CUDA_VISIBLE_DEVICES=0

# L4 compute capability
export TORCH_CUDA_ARCH_LIST="8.9"

# Optimize CUDA memory allocation
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Enable TF32 for better performance on newer GPUs
export TORCH_ALLOW_TF32_ON_CUDA=1
export TORCH_ALLOW_TF32_ON_CUDNN=1

# FAISS GPU memory settings
export FAISS_GPU_USE_FLOAT16=1

echo "GPU environment configured for CustomKB"
echo "CUDA Device: $CUDA_VISIBLE_DEVICES"
echo "Compute capability: $TORCH_CUDA_ARCH_LIST"