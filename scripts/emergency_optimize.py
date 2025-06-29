#!/usr/bin/env python
"""
Emergency optimization script to prevent system crashes from large result sets.
Applies conservative settings to prevent memory exhaustion.
"""

import sys
import os
import configparser

def apply_emergency_settings(config_path):
    """Apply emergency settings to prevent crashes."""
    config = configparser.ConfigParser()
    config.read(config_path)
    
    # Emergency settings to prevent crashes
    emergency_settings = {
        'LIMITS': {
            # Reduce memory usage
            'memory_cache_size': '50000',  # Drastically reduced
            'max_query_length': '5000',    # Limit query size
        },
        'PERFORMANCE': {
            # Reduce batch sizes
            'reference_batch_size': '10',  # Much smaller batches
            'io_thread_pool_size': '8',    # Fewer threads
            'cache_thread_pool_size': '4', # Fewer threads
        },
        'ALGORITHMS': {
            # Limit BM25 results
            'bm25_max_results': '500',     # Add new setting
            'reranking_top_k': '20',       # Reduce reranking load
            'reranking_device': 'cpu',     # Force CPU to free GPU memory
            'enable_hybrid_search': 'false', # Disable for now
            'enable_query_enhancement': 'true',    # Keep enhancement on
            'query_enhancement_synonyms': 'false',  # Disable synonyms - can harm precision
            'query_enhancement_spelling': 'true',   # Keep spelling correction
            # Conservative GPU settings for FAISS in emergencies
            'faiss_gpu_batch_size': '512',  # Smaller batches
            'faiss_gpu_use_float16': 'true',  # Enable float16 to save memory
        }
    }
    
    # Apply emergency settings
    for section, settings in emergency_settings.items():
        if section not in config:
            config.add_section(section)
        for key, value in settings.items():
            config.set(section, key, value)
    
    # Backup and save
    import shutil
    from datetime import datetime
    backup_name = f"{config_path}.pre_emergency.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    shutil.copy2(config_path, backup_name)
    
    with open(config_path, 'w') as f:
        config.write(f)
    
    print(f"Emergency settings applied to {config_path}")
    print(f"Backup saved to {backup_name}")
    print("\nKey changes:")
    print("- Disabled hybrid search to prevent BM25 overload")
    print("- Reduced memory cache to 50K")
    print("- Reduced batch sizes")
    print("- Disabled query synonyms")
    print("- Forced CPU mode for reranking and FAISS")

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python emergency_optimize.py <config_path>")
        sys.exit(1)
    
    config_path = sys.argv[1]
    if not os.path.exists(config_path):
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)
    
    apply_emergency_settings(config_path)

#fin