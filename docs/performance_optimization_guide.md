# CustomKB Performance Optimization Guide

## Memory-Based Optimization Tiers

CustomKB automatically optimizes settings based on available system memory. Here's a guide to the different optimization tiers:

### Memory Tiers

| Tier | Memory Range | Use Case |
|------|-------------|----------|
| **Low** | < 16 GB | Development, testing, small datasets |
| **Medium** | 16-64 GB | Standard workloads, moderate datasets |
| **High** | 64-128 GB | Production workloads, large datasets |
| **Very High** | > 128 GB | Enterprise deployments, maximum performance |

### Key Settings by Tier

| Setting | Low | Medium | High | Very High |
|---------|-----|--------|------|-----------|
| **memory_cache_size** | 125,000 | 250,000 | 375,000 | 500,000 |
| **reference_batch_size** | 25 | 37 | 50 | 75 |
| **io_thread_pool_size** | 16 | 24 | 32 | 48 |
| **api_max_concurrency** | 16 | 24 | 32 | 48 |
| **embedding_batch_size** | 500 | 750 | 1,000 | 1,500 |
| **file_processing_batch_size** | 2,500 | 3,750 | 5,000 | 7,500 |
| **enable_hybrid_search** | false | true | true | true |
| **reranking_device** | cpu | cpu | cuda | cuda |

### Usage Examples

#### 1. Analyze your system and KBs:
```bash
customkb optimize --analyze
```

#### 2. Preview optimizations (dry run):
```bash
# For current system memory
customkb optimize myknowledgebase --dry-run

# Simulate different memory tier
customkb optimize myknowledgebase --dry-run --memory-gb 128
```

#### 3. Apply optimizations:
```bash
# Optimize for current system
customkb optimize myknowledgebase

# Optimize all KBs
customkb optimize
```

#### 4. Test specific memory configurations:
```bash
# Test low memory settings (8GB)
customkb optimize myknowledgebase --memory-gb 8

# Test high memory settings (96GB)
customkb optimize myknowledgebase --memory-gb 96
```

### Performance Impact

The most significant performance improvements come from:

1. **reference_batch_size**: Controls how many documents are fetched per database query
   - Low: 25 (more queries, slower)
   - Very High: 75 (fewer queries, 3x faster)

2. **memory_cache_size**: Number of embeddings cached in memory
   - Low: 125K (more cache misses)
   - Very High: 500K (better hit rate)

3. **Thread pools**: Parallel processing capacity
   - Low: 16 threads
   - Very High: 48 threads

### Recommendations by Use Case

#### Development Machine (16-32 GB RAM)
- Use default medium tier settings
- Disable GPU features if no CUDA available
- Consider reducing cache sizes for multiple concurrent projects

#### Production Server (64-256 GB RAM)
- Use high or very high tier settings
- Enable all performance features (hybrid search, reranking)
- Monitor memory usage and adjust if needed

#### Cloud/Container Deployments
- Explicitly set memory tier based on container limits
- Use `--memory-gb` flag to match container allocation
- Consider lower settings to leave headroom for other processes

### Monitoring Performance

After optimization, monitor performance using:

```bash
source .venv/bin/activate
python utils/performance_analyzer.py myknowledgebase --benchmark
```

This will show:
- Memory usage statistics
- Cache hit rates
- Query performance metrics
- Optimization recommendations

### Manual Tuning

If automatic optimization doesn't meet your needs, you can manually edit the configuration file. Key parameters to adjust:

1. **For faster queries**: Increase `reference_batch_size`
2. **For better cache performance**: Increase `memory_cache_size`
3. **For faster embedding generation**: Increase `embedding_batch_size` and `api_max_concurrency`
4. **For memory constraints**: Reduce all cache and batch sizes proportionally

### Troubleshooting

**Out of Memory Errors:**
- Use a lower memory tier: `--memory-gb 16`
- Reduce `memory_cache_size` manually
- Check for memory leaks with performance analyzer

**Slow Query Performance:**
- Increase `reference_batch_size` (biggest impact)
- Enable hybrid search for better accuracy
- Check cache hit rates with performance analyzer

**API Rate Limits:**
- Reduce `api_max_concurrency`
- Increase `api_call_delay_seconds`

#fin