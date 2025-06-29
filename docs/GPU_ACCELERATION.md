# GPU Acceleration in CustomKB

## Overview

CustomKB supports GPU acceleration for the reranking step using CUDA-compatible GPUs. This feature can improve query performance when reranking large numbers of documents.

## Configuration

To enable GPU acceleration, set the following in your knowledge base configuration:

```ini
[ALGORITHMS]
# Enable GPU acceleration for reranking
reranking_device = cuda

# Increase batch size for better GPU utilization
reranking_batch_size = 64

# Increase top_k for more documents to rerank
reranking_top_k = 50
```

## Performance Considerations

### When GPU Acceleration Helps

1. **Large reranking sets**: GPU benefits increase with more documents
   - 10 documents: ~0% speedup
   - 50 documents: ~5-10% speedup  
   - 100 documents: ~10-15% speedup
   - 500+ documents: ~20-30% speedup

2. **Batch processing**: Larger batch sizes utilize GPU better
   - Recommended: 64-128 for consumer GPUs
   - Recommended: 128-256 for datacenter GPUs

3. **Model complexity**: Heavier models benefit more from GPU
   - MiniLM models: Modest speedup (10-20%)
   - Base models: Better speedup (20-40%)
   - Large models: Best speedup (40-60%)

### GPU Memory Requirements

The cross-encoder models used for reranking have modest memory requirements:
- MiniLM-L-6: ~500MB
- MiniLM-L-12: ~800MB
- Base models: ~1.5GB

Most modern GPUs (4GB+ VRAM) can handle these models comfortably.

## Monitoring GPU Usage

Use the provided monitoring script to check GPU utilization:

```bash
./scripts/gpu_monitor.sh "your query" knowledge_base_name
```

This will show:
- GPU utilization percentage
- Memory usage
- Peak values during query processing

## Benchmarking

Compare GPU vs CPU performance for your specific workload:

```bash
python scripts/benchmark_gpu.py kb_config --iterations 5
```

## Example Configurations

### Development Machine (RTX 4070)
```ini
reranking_device = cuda
reranking_batch_size = 64
reranking_top_k = 50
```

### Production Server (NVIDIA L4)
```ini
reranking_device = cuda
reranking_batch_size = 128
reranking_top_k = 100
```

### CPU-Only Fallback
```ini
reranking_device = cpu
reranking_batch_size = 32
reranking_top_k = 20
```

## Troubleshooting

1. **CUDA not available**: Ensure PyTorch is installed with CUDA support
   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   ```

2. **Out of memory**: Reduce batch size or reranking_top_k

3. **No speedup**: Check if documents being reranked > 50

4. **Slow first query**: Model loading to GPU takes time, subsequent queries are faster

#fin