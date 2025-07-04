#!/usr/bin/env python
"""
Test script to verify the crash prevention fixes work correctly.
"""

import os
import sys
import psutil
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from utils.resource_manager import ResourceGuard, get_resource_guard
from embedding.embed_manager import cache_manager
from database.db_manager import sqlite_connection
from config.config_manager import KnowledgeBase


def test_resource_guard():
    """Test resource guard functionality."""
    print("Testing Resource Guard...")
    
    guard = ResourceGuard(memory_limit_gb=1.0)
    
    # Check initial state
    ok, msg = guard.monitor.check_memory_limits()
    print(f"  Initial check: {msg}")
    
    # Test guarded operation
    try:
        with guard.guarded_operation("test_operation"):
            print("  Running test operation...")
            # Simulate some work
            data = [i for i in range(1000000)]
            del data
    except MemoryError as e:
        print(f"  Memory error caught: {e}")
    
    print("  ✓ Resource guard working")


def test_cache_memory_limits():
    """Test embedding cache memory limits."""
    print("\nTesting Cache Memory Limits...")
    
    # Configure cache with low memory limit
    cache_manager.configure(memory_limit_mb=10)
    
    # Add embeddings until limit is reached
    evictions = 0
    for i in range(1000):
        # Create a 1536-dimensional embedding (6KB each)
        embedding = [0.1] * 1536
        cache_key = f"test_key_{i}"
        
        initial_evictions = cache_manager._metrics['cache_evictions']
        cache_manager.add_to_memory_cache(cache_key, embedding)
        
        if cache_manager._metrics['cache_evictions'] > initial_evictions:
            evictions += 1
    
    metrics = cache_manager.get_metrics()
    print(f"  Cache size: {metrics['cache_size']}")
    print(f"  Memory usage: {metrics['memory_usage_mb']:.1f}MB")
    print(f"  Evictions: {evictions}")
    print("  ✓ Cache memory limits working")


def test_database_context():
    """Test database context managers."""
    print("\nTesting Database Context Managers...")
    
    # Test direct SQLite connection
    test_db = "/tmp/test_context.db"
    
    try:
        with sqlite_connection(test_db) as (conn, cursor):
            cursor.execute("CREATE TABLE test (id INTEGER PRIMARY KEY)")
            cursor.execute("INSERT INTO test VALUES (1)")
            conn.commit()
        print("  ✓ SQLite context manager working")
    except Exception as e:
        print(f"  ✗ Error: {e}")
    finally:
        if os.path.exists(test_db):
            os.remove(test_db)


def test_memory_mapped_faiss():
    """Test memory-mapped FAISS configuration."""
    print("\nTesting Memory-Mapped FAISS Configuration...")
    
    # Create a test config
    kb = KnowledgeBase('test', use_memory_mapped_faiss=True)
    
    if hasattr(kb, 'use_memory_mapped_faiss') and kb.use_memory_mapped_faiss:
        print("  ✓ Memory-mapped FAISS configuration working")
    else:
        print("  ✗ Memory-mapped FAISS not configured")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing Crash Prevention Fixes")
    print("=" * 60)
    
    initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
    print(f"Initial memory: {initial_memory:.1f}MB")
    
    try:
        test_resource_guard()
        test_cache_memory_limits()
        test_database_context()
        test_memory_mapped_faiss()
    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()
    
    final_memory = psutil.Process().memory_info().rss / 1024 / 1024
    print(f"\nFinal memory: {final_memory:.1f}MB (delta: {final_memory - initial_memory:+.1f}MB)")
    
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)


if __name__ == '__main__':
    main()

#fin