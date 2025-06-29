#!/usr/bin/env python
"""
Performance analyzer for CustomKB queries.
Helps identify bottlenecks and optimize query performance.
"""

import time
import psutil
import os
import sys
import sqlite3
import faiss
import numpy as np
from typing import Dict, List, Tuple, Optional
import argparse
import asyncio

from config.config_manager import KnowledgeBase, get_fq_cfg_filename
from database.db_manager import connect_to_database, close_database
from utils.logging_utils import setup_logging, get_logger

logger = get_logger(__name__)

class PerformanceAnalyzer:
  """Analyze CustomKB performance metrics."""
  
  def __init__(self, kb: KnowledgeBase):
    self.kb = kb
    self.metrics = {
      'memory_usage': [],
      'query_times': [],
      'cache_stats': {},
      'db_stats': {},
      'index_stats': {}
    }
  
  def get_memory_info(self) -> Dict[str, float]:
    """Get current memory usage information."""
    process = psutil.Process()
    memory_info = process.memory_info()
    
    return {
      'rss_mb': memory_info.rss / 1024 / 1024,
      'vms_mb': memory_info.vms / 1024 / 1024,
      'percent': process.memory_percent(),
      'available_mb': psutil.virtual_memory().available / 1024 / 1024,
      'total_mb': psutil.virtual_memory().total / 1024 / 1024
    }
  
  def analyze_database(self) -> Dict[str, any]:
    """Analyze database performance characteristics."""
    stats = {}
    
    try:
      # Get table statistics
      self.kb.sql_cursor.execute("SELECT COUNT(*) FROM docs")
      stats['total_documents'] = self.kb.sql_cursor.fetchone()[0]
      
      self.kb.sql_cursor.execute("SELECT COUNT(DISTINCT sourcedoc) FROM docs")
      stats['unique_files'] = self.kb.sql_cursor.fetchone()[0]
      
      self.kb.sql_cursor.execute("SELECT COUNT(*) FROM docs WHERE embedded = 1")
      stats['embedded_documents'] = self.kb.sql_cursor.fetchone()[0]
      
      # Check index usage
      self.kb.sql_cursor.execute("PRAGMA index_list(docs)")
      indexes = self.kb.sql_cursor.fetchall()
      stats['indexes'] = [idx[1] for idx in indexes]
      
      # Analyze query plan for common queries
      self.kb.sql_cursor.execute("""
        EXPLAIN QUERY PLAN 
        SELECT * FROM docs 
        WHERE sourcedoc = ? AND sid >= ? AND sid <= ?
      """, ('dummy.txt', 0, 10))
      stats['context_query_plan'] = self.kb.sql_cursor.fetchall()
      
      # Get database size
      db_file = self.kb.knowledge_base_db
      if os.path.exists(db_file):
        stats['db_size_mb'] = os.path.getsize(db_file) / 1024 / 1024
      
    except sqlite3.Error as e:
      logger.error(f"Database analysis error: {e}")
    
    return stats
  
  def analyze_faiss_index(self) -> Dict[str, any]:
    """Analyze FAISS index characteristics."""
    stats = {}
    
    try:
      if os.path.exists(self.kb.knowledge_base_vector):
        # Load index
        index = faiss.read_index(self.kb.knowledge_base_vector)
        
        stats['index_size_mb'] = os.path.getsize(self.kb.knowledge_base_vector) / 1024 / 1024
        stats['num_vectors'] = index.ntotal
        stats['dimension'] = index.d
        stats['index_type'] = str(type(index).__name__)
        
        # Check if it's an IVF index
        if hasattr(index, 'nlist'):
          stats['num_centroids'] = index.nlist
          stats['nprobe'] = getattr(index, 'nprobe', 'N/A')
        
        # Memory usage estimate
        if hasattr(index, 'sa_code_size'):
          stats['code_size'] = index.sa_code_size()
        
    except Exception as e:
      logger.error(f"FAISS index analysis error: {e}")
    
    return stats
  
  def analyze_cache_performance(self) -> Dict[str, any]:
    """Analyze cache performance metrics."""
    from embedding.embed_manager import cache_manager
    
    return cache_manager.get_metrics()
  
  def benchmark_query(self, query_text: str, iterations: int = 5) -> Dict[str, float]:
    """Benchmark query performance."""
    times = []
    memory_before = self.get_memory_info()
    
    for i in range(iterations):
      start_time = time.time()
      
      # Run query synchronously for timing
      from query.query_manager import process_query
      args = argparse.Namespace(
        config_file=self.kb.config_file,
        query_text=query_text,
        query_file=None,
        context_only=True,
        role=None,
        model=None,
        top_k=None,
        context_scope=None,
        temperature=None,
        max_tokens=None,
        verbose=False,
        debug=False
      )
      
      result = process_query(args, logger)
      
      elapsed = time.time() - start_time
      times.append(elapsed)
      
      # Skip first iteration (cold start)
      if i > 0:
        self.metrics['query_times'].append(elapsed)
    
    memory_after = self.get_memory_info()
    
    return {
      'avg_time': np.mean(times[1:]),
      'min_time': np.min(times[1:]),
      'max_time': np.max(times[1:]),
      'std_time': np.std(times[1:]),
      'memory_delta_mb': memory_after['rss_mb'] - memory_before['rss_mb']
    }
  
  def generate_report(self) -> str:
    """Generate a comprehensive performance report."""
    report = []
    report.append("=== CustomKB Performance Analysis Report ===\n")
    
    # System Information
    memory = self.get_memory_info()
    report.append("System Memory:")
    report.append(f"  Total: {memory['total_mb']:.1f} MB")
    report.append(f"  Available: {memory['available_mb']:.1f} MB")
    report.append(f"  Process RSS: {memory['rss_mb']:.1f} MB ({memory['percent']:.1f}%)\n")
    
    # Configuration
    report.append("Current Configuration:")
    report.append(f"  Reference batch size: {self.kb.reference_batch_size}")
    report.append(f"  Embedding batch size: {self.kb.embedding_batch_size}")
    report.append(f"  Memory cache size: {self.kb.memory_cache_size}")
    report.append(f"  IO thread pool size: {self.kb.io_thread_pool_size}")
    report.append(f"  Cache thread pool size: {self.kb.cache_thread_pool_size}")
    report.append(f"  API max concurrency: {self.kb.api_max_concurrency}\n")
    
    # Database Analysis
    db_stats = self.analyze_database()
    report.append("Database Statistics:")
    report.append(f"  Total documents: {db_stats.get('total_documents', 'N/A')}")
    report.append(f"  Unique files: {db_stats.get('unique_files', 'N/A')}")
    report.append(f"  Embedded documents: {db_stats.get('embedded_documents', 'N/A')}")
    report.append(f"  Database size: {db_stats.get('db_size_mb', 0):.1f} MB")
    report.append(f"  Indexes: {', '.join(db_stats.get('indexes', []))}\n")
    
    # FAISS Index Analysis
    index_stats = self.analyze_faiss_index()
    report.append("FAISS Index Statistics:")
    report.append(f"  Index type: {index_stats.get('index_type', 'N/A')}")
    report.append(f"  Number of vectors: {index_stats.get('num_vectors', 'N/A')}")
    report.append(f"  Dimension: {index_stats.get('dimension', 'N/A')}")
    report.append(f"  Index size: {index_stats.get('index_size_mb', 0):.1f} MB")
    if 'num_centroids' in index_stats:
      report.append(f"  Number of centroids: {index_stats['num_centroids']}")
      report.append(f"  nprobe: {index_stats.get('nprobe', 'N/A')}\n")
    
    # Cache Performance
    cache_stats = self.analyze_cache_performance()
    report.append("Cache Performance:")
    report.append(f"  Cache hits: {cache_stats.get('cache_hits', 0)}")
    report.append(f"  Cache misses: {cache_stats.get('cache_misses', 0)}")
    report.append(f"  Hit ratio: {cache_stats.get('cache_hit_ratio', 0):.2%}")
    report.append(f"  Cache size: {cache_stats.get('cache_size', 0)}/{cache_stats.get('max_cache_size', 0)}")
    report.append(f"  Evictions: {cache_stats.get('cache_evictions', 0)}\n")
    
    # Query Performance
    if self.metrics['query_times']:
      report.append("Query Performance:")
      report.append(f"  Average time: {np.mean(self.metrics['query_times']):.3f}s")
      report.append(f"  Min time: {np.min(self.metrics['query_times']):.3f}s")
      report.append(f"  Max time: {np.max(self.metrics['query_times']):.3f}s\n")
    
    # Recommendations
    report.append("Performance Recommendations:")
    
    # Memory recommendations
    if memory['available_mb'] > 100000:  # More than 100GB available
      if self.kb.memory_cache_size < 100000:
        report.append("  - Increase memory_cache_size to utilize available RAM")
      if self.kb.reference_batch_size < 20:
        report.append("  - Increase reference_batch_size for better query performance")
    
    # Database recommendations
    if db_stats.get('total_documents', 0) > 100000:
      if self.kb.sql_batch_size < 1000:
        report.append("  - Increase sql_batch_size for large databases")
    
    # Index recommendations
    if index_stats.get('num_vectors', 0) > 1000000:
      if index_stats.get('index_type') == 'IndexFlatL2':
        report.append("  - Consider using IVF index for large vector sets")
    
    # Cache recommendations
    if cache_stats.get('cache_hit_ratio', 0) < 0.5:
      report.append("  - Low cache hit ratio - consider increasing cache size")
    
    return '\n'.join(report)

def main():
  """Main entry point for performance analyzer."""
  parser = argparse.ArgumentParser(description='Analyze CustomKB performance')
  parser.add_argument('config_file', help='Path to knowledge base configuration')
  parser.add_argument('--benchmark', help='Benchmark with test query', action='store_true')
  parser.add_argument('--query', help='Query text for benchmarking', default='test query')
  parser.add_argument('--iterations', help='Number of benchmark iterations', type=int, default=5)
  
  args = parser.parse_args()
  
  # Initialize
  logger = setup_logging("performance_analyzer", verbose=True)
  cfgfile = get_fq_cfg_filename(args.config_file)
  
  if not cfgfile:
    logger.error(f"Configuration file not found: {args.config_file}")
    return 1
  
  kb = KnowledgeBase(cfgfile)
  connect_to_database(kb)
  
  # Create analyzer
  analyzer = PerformanceAnalyzer(kb)
  
  # Run benchmark if requested
  if args.benchmark:
    logger.info(f"Running benchmark with query: '{args.query}'")
    benchmark_results = analyzer.benchmark_query(args.query, args.iterations)
    logger.info(f"Benchmark results: {benchmark_results}")
  
  # Generate report
  report = analyzer.generate_report()
  print(report)
  
  # Cleanup
  close_database(kb)
  
  return 0

if __name__ == '__main__':
  sys.exit(main())

#fin