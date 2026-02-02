#!/usr/bin/env python
"""Show optimization settings for all memory tiers."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.optimize_kb_performance import get_optimized_settings


def main():
    """Show optimization settings for different memory tiers."""

    # Test different memory sizes
    test_sizes = [
        (8, "Low - Development/Testing"),
        (32, "Medium - Standard Workloads"),
        (96, "High - Production"),
        (256, "Very High - Enterprise")
    ]

    print("CustomKB Optimization Tiers")
    print("=" * 80)

    for memory_gb, description in test_sizes:
        settings = get_optimized_settings(memory_gb)
        tier = settings['tier']
        opts = settings['optimizations']

        print(f"\n{description}")
        print(f"Memory: {memory_gb} GB | Tier: {tier.upper()}")
        print("-" * 60)

        # Show key settings
        print(f"  Memory cache size: {opts['LIMITS']['memory_cache_size']:>10}")
        print(f"  Reference batch size: {opts['PERFORMANCE']['reference_batch_size']:>7}")
        print(f"  IO thread pool: {opts['PERFORMANCE']['io_thread_pool_size']:>12}")
        print(f"  API concurrency: {opts['API']['api_max_concurrency']:>11}")
        print(f"  Embedding batch size: {opts['PERFORMANCE']['embedding_batch_size']:>7}")
        print(f"  File batch size: {opts['PERFORMANCE']['file_processing_batch_size']:>11}")
        print(f"  Hybrid search: {opts['ALGORITHMS']['enable_hybrid_search']:>13}")
        print(f"  Fusion method: {opts['ALGORITHMS'].get('hybrid_fusion_method', 'rrf'):>13}")
        print(f"  RRF k: {opts['ALGORITHMS'].get('rrf_k', '60'):>21}")
        print(f"  Reranking device: {opts['ALGORITHMS']['reranking_device']:>10}")
        print(f"  FAISS GPU batch: {opts['ALGORITHMS'].get('faiss_gpu_batch_size', 'N/A'):>11}")
        print(f"  FAISS use float16: {opts['ALGORITHMS'].get('faiss_gpu_use_float16', 'N/A'):>9}")

    print("\n" + "=" * 80)
    print("\nTo apply these settings to your knowledgebase:")
    print("  ./optimize-kb <kb-name> --memory-gb <memory>")
    print("\nTo see current system memory:")
    print("  ./optimize-kb --analyze")

if __name__ == '__main__':
    main()

#fin
