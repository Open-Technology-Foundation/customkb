# GPU Memory Management in CustomKB

CustomKB now includes intelligent GPU memory detection and management for optimal FAISS performance across different GPU configurations.

## Overview

The system automatically:
- Detects available GPU memory
- Calculates safe memory limits with configurable buffers
- Determines whether to use GPU for FAISS operations
- Adjusts batch sizes and float16 mode based on GPU memory
- Provides clear logging of GPU decisions

## Key Features

### 1. Dynamic GPU Memory Detection

CustomKB automatically detects GPU memory using multiple methods:
- **PyTorch** (primary): Most accurate, provides detailed GPU properties
- **nvidia-smi** (fallback): Works when PyTorch CUDA is unavailable
- **FAISS** (last resort): Basic detection with conservative estimates

### 2. Intelligent Index Placement

The system decides whether to place FAISS indexes on GPU based on:
- Index size (including float16 compression if enabled)
- Available GPU memory
- Configured safety buffer (default 4GB)
- Temporary memory requirements (~20% of index size)

### 3. Configuration Options

#### Environment Variables
```bash
# Override automatic GPU memory detection
export FAISS_GPU_MEMORY_LIMIT_MB=16384  # Force 16GB limit

# Disable GPU completely
export FAISS_NO_GPU=1
```

#### Configuration File Settings
```ini
[ALGORITHMS]
# GPU batch size for FAISS operations
faiss_gpu_batch_size = 1024

# Use float16 to reduce memory usage by ~40%
faiss_gpu_use_float16 = true

# Safety buffer in GB (reserved for other operations)
faiss_gpu_memory_buffer_gb = 4.0

# Manual memory limit override (0 = auto-detect)
faiss_gpu_memory_limit_mb = 0
```

## GPU Memory Tiers

The optimize command automatically adjusts settings based on GPU memory:

| GPU Memory | Batch Size | Float16 | Suitable Index Size |
|------------|------------|---------|-------------------|
| < 8GB      | 512        | Always  | < 3GB             |
| 8-16GB     | 1024       | Always  | < 10GB            |
| 16-24GB    | 2048       | Optional| < 16GB            |
| > 24GB     | 4096       | Optional| < 20GB+           |

## Usage Examples

### Check GPU Status
```bash
# Show optimization tiers with GPU info
customkb optimize --show-tiers

# Analyze specific KB with GPU recommendations
customkb optimize myproject --analyze
```

### Manual Configuration
```bash
# Force specific GPU memory limit
export FAISS_GPU_MEMORY_LIMIT_MB=8192
customkb query myproject "search query"

# Disable GPU for troubleshooting
export FAISS_NO_GPU=1
customkb query myproject "search query"
```

### Optimization Examples

For a system with 24GB GPU:
```bash
# Preview optimizations
customkb optimize myproject --dry-run

# Apply GPU-aware optimizations
customkb optimize myproject
```

## Troubleshooting

### GPU Not Detected
1. Check CUDA installation: `nvidia-smi`
2. Verify PyTorch CUDA: `python -c "import torch; print(torch.cuda.is_available())"`
3. Check environment: `echo $CUDA_VISIBLE_DEVICES`

### Out of Memory Errors
1. Increase buffer: Set `faiss_gpu_memory_buffer_gb = 6.0`
2. Enable float16: Set `faiss_gpu_use_float16 = true`
3. Reduce batch size: Set `faiss_gpu_batch_size = 512`
4. Force CPU: Set `FAISS_NO_GPU=1`

### Performance Issues
1. Check GPU utilization: `nvidia-smi dmon -s u`
2. Monitor memory: `watch -n 1 nvidia-smi`
3. Adjust batch sizes in config

## Log Messages

The system provides clear logging:
```
INFO - GPU detected via PyTorch: NVIDIA RTX 4070 with 23648MB memory
INFO - GPU memory: 23648MB total, 4096MB buffer, 19552MB safe limit
INFO - GPU suitable for FAISS: GPU suitable for index (7373MB needed < 19552MB available)
INFO - FAISS index loaded on GPU
```

Or when GPU is not suitable:
```
INFO - GPU not suitable for FAISS index: Index too large for GPU (22118MB needed > 19552MB available)
INFO - Using CPU for FAISS search
```

## Best Practices

1. **Leave adequate buffer**: Default 4GB buffer prevents OOM errors
2. **Use float16 for large indexes**: Reduces memory by ~40%
3. **Monitor GPU memory**: Use `nvidia-smi` during queries
4. **Test with dry-run**: Preview changes before applying
5. **Start conservative**: Begin with smaller batch sizes

## Performance Impact

GPU acceleration typically provides:
- **2-10x faster** vector searches
- **Lower latency** for real-time queries
- **Higher throughput** for batch processing
- **Better scaling** with index size

The actual speedup depends on:
- Index size and dimensions
- GPU model and memory bandwidth
- Query batch size
- CPU baseline performance

#fin