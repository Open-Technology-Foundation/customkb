"""
Performance tests for BM25 hybrid search functionality.
Tests BM25 index building, loading, searching, and memory usage under various loads.
"""

import pytest
import time
import psutil
import os
import tempfile
import sqlite3
import pickle
import numpy as np
from unittest.mock import patch, Mock
from typing import List, Tuple

from config.config_manager import KnowledgeBase
from embedding.bm25_manager import (
  build_bm25_index, 
  load_bm25_index, 
  get_bm25_scores,
  ensure_bm25_index
)
from query.query_manager import perform_hybrid_search
from utils.text_utils import tokenize_for_bm25


@pytest.mark.performance
@pytest.mark.slow
class TestBM25IndexPerformance:
  """Test BM25 index building and loading performance."""
  
  def test_bm25_index_building_performance(self, temp_data_manager):
    """Test BM25 index building performance with varying dataset sizes."""
    # Test different dataset sizes
    test_sizes = [100, 500, 1000, 2000]
    
    for size in test_sizes:
      kb_dir = temp_data_manager.create_temp_dir()
      config_file = os.path.join(kb_dir, f'perf_test_{size}.cfg')
      
      # Create config
      with open(config_file, 'w') as f:
        f.write("""[DEFAULT]
vector_model = text-embedding-3-small
vector_dimensions = 1536

[ALGORITHMS]
enable_hybrid_search = true
bm25_k1 = 1.2
bm25_b = 0.75
""")
      
      # Create test database
      kb = KnowledgeBase(config_file)
      self._create_test_database(kb, size)
      
      # Measure index building time
      kb.sql_connection = sqlite3.connect(kb.knowledge_base_db)
      kb.sql_cursor = kb.sql_connection.cursor()
      
      start_time = time.time()
      bm25_index = build_bm25_index(kb)
      build_time = time.time() - start_time
      
      kb.sql_connection.close()
      
      assert bm25_index is not None
      
      # Performance assertions based on dataset size
      if size <= 500:
        assert build_time < 2.0  # Small datasets should build quickly
      elif size <= 1000:
        assert build_time < 5.0  # Medium datasets
      else:
        assert build_time < 15.0  # Large datasets
      
      # Calculate throughput
      docs_per_second = size / build_time
      assert docs_per_second > 50  # Should process at least 50 docs/sec
      
      print(f"Built BM25 index for {size} docs in {build_time:.3f}s ({docs_per_second:.1f} docs/sec)")
  
  def test_bm25_index_loading_performance(self, temp_data_manager):
    """Test BM25 index loading performance."""
    kb_dir = temp_data_manager.create_temp_dir()
    config_file = os.path.join(kb_dir, 'load_test.cfg')
    
    with open(config_file, 'w') as f:
      f.write("""[DEFAULT]
vector_model = text-embedding-3-small

[ALGORITHMS]
enable_hybrid_search = true
""")
    
    # Create and build index
    kb = KnowledgeBase(config_file)
    self._create_test_database(kb, 1000)
    
    kb.sql_connection = sqlite3.connect(kb.knowledge_base_db)
    kb.sql_cursor = kb.sql_connection.cursor()
    build_bm25_index(kb)
    kb.sql_connection.close()
    
    # Test loading performance multiple times
    load_times = []
    for i in range(10):
      start_time = time.time()
      data = load_bm25_index(kb)
      load_time = time.time() - start_time
      load_times.append(load_time)
      
      assert data is not None
      assert data['total_docs'] == 1000
    
    avg_load_time = sum(load_times) / len(load_times)
    max_load_time = max(load_times)
    
    # Loading should be fast and consistent
    assert avg_load_time < 0.5  # Average under 500ms
    assert max_load_time < 1.0   # Maximum under 1 second
    
    print(f"BM25 index loading - Avg: {avg_load_time:.3f}s, Max: {max_load_time:.3f}s")
  
  def test_bm25_index_file_size_efficiency(self, temp_data_manager):
    """Test BM25 index file size efficiency."""
    dataset_sizes = [100, 500, 1000, 2000]
    
    for size in dataset_sizes:
      kb_dir = temp_data_manager.create_temp_dir()
      config_file = os.path.join(kb_dir, f'size_test_{size}.cfg')
      
      with open(config_file, 'w') as f:
        f.write("""[DEFAULT]
vector_model = text-embedding-3-small

[ALGORITHMS]
enable_hybrid_search = true
""")
      
      # Create test data with realistic content
      kb = KnowledgeBase(config_file)
      self._create_realistic_test_database(kb, size)
      
      # Build index
      kb.sql_connection = sqlite3.connect(kb.knowledge_base_db)
      kb.sql_cursor = kb.sql_connection.cursor()
      build_bm25_index(kb)
      kb.sql_connection.close()
      
      # Check file size
      index_path = kb.knowledge_base_vector.replace('.faiss', '.bm25')
      file_size = os.path.getsize(index_path)
      
      # Size should scale reasonably with dataset size
      size_per_doc = file_size / size
      
      # Should be efficient storage (less than 10KB per document on average)
      assert size_per_doc < 10 * 1024
      
      print(f"{size} docs: {file_size / 1024:.1f} KB ({size_per_doc:.1f} bytes/doc)")
  
  def test_bm25_index_memory_usage(self, temp_data_manager):
    """Test BM25 index memory usage during building and loading."""
    import psutil
    
    process = psutil.Process()
    initial_memory = process.memory_info().rss
    
    kb_dir = temp_data_manager.create_temp_dir()
    config_file = os.path.join(kb_dir, 'memory_test.cfg')
    
    with open(config_file, 'w') as f:
      f.write("""[DEFAULT]
vector_model = text-embedding-3-small

[ALGORITHMS]
enable_hybrid_search = true
""")
    
    # Create moderate-sized dataset
    kb = KnowledgeBase(config_file)
    self._create_test_database(kb, 1500)
    
    # Monitor memory during index building
    kb.sql_connection = sqlite3.connect(kb.knowledge_base_db)
    kb.sql_cursor = kb.sql_connection.cursor()
    
    pre_build_memory = process.memory_info().rss
    bm25_index = build_bm25_index(kb)
    post_build_memory = process.memory_info().rss
    
    build_memory_increase = post_build_memory - pre_build_memory
    
    # Load index and monitor memory
    pre_load_memory = process.memory_info().rss
    loaded_data = load_bm25_index(kb)
    post_load_memory = process.memory_info().rss
    
    load_memory_increase = post_load_memory - pre_load_memory
    
    kb.sql_connection.close()
    
    # Memory usage should be reasonable
    assert build_memory_increase < 200 * 1024 * 1024  # Less than 200MB for building
    assert load_memory_increase < 100 * 1024 * 1024   # Less than 100MB for loading
    
    print(f"Memory usage - Build: {build_memory_increase / 1024 / 1024:.1f}MB, Load: {load_memory_increase / 1024 / 1024:.1f}MB")
  
  def _create_test_database(self, kb: KnowledgeBase, num_docs: int):
    """Create test database with specified number of documents."""
    conn = sqlite3.connect(kb.knowledge_base_db)
    cursor = conn.cursor()
    
    cursor.execute('''
      CREATE TABLE docs (
        id INTEGER PRIMARY KEY,
        sid INTEGER,
        sourcedoc VARCHAR(255),
        originaltext TEXT,
        embedtext TEXT,
        embedded INTEGER DEFAULT 0,
        language TEXT DEFAULT "en",
        metadata TEXT,
        keyphrase_processed INTEGER DEFAULT 0,
        bm25_tokens TEXT,
        doc_length INTEGER DEFAULT 0
      )
    ''')
    
    # Generate simple test documents
    for i in range(num_docs):
      text = f"Document {i} contains machine learning algorithms and data processing techniques."
      tokens, length = tokenize_for_bm25(text, 'en')
      
      cursor.execute('''
        INSERT INTO docs (id, sid, sourcedoc, originaltext, embedtext, embedded, 
                         language, metadata, keyphrase_processed, bm25_tokens, doc_length)
        VALUES (?, 0, ?, ?, ?, 1, 'en', '{}', 1, ?, ?)
      ''', (i+1, f'doc{i+1}.txt', text, text.lower(), tokens, length))
    
    conn.commit()
    conn.close()
  
  def _create_realistic_test_database(self, kb: KnowledgeBase, num_docs: int):
    """Create test database with realistic, varied content."""
    conn = sqlite3.connect(kb.knowledge_base_db)
    cursor = conn.cursor()
    
    cursor.execute('''
      CREATE TABLE docs (
        id INTEGER PRIMARY KEY,
        sid INTEGER,
        sourcedoc VARCHAR(255),
        originaltext TEXT,
        embedtext TEXT,
        embedded INTEGER DEFAULT 0,
        language TEXT DEFAULT "en",
        metadata TEXT,
        keyphrase_processed INTEGER DEFAULT 0,
        bm25_tokens TEXT,
        doc_length INTEGER DEFAULT 0
      )
    ''')
    
    # Templates for realistic content
    templates = [
      "Machine learning algorithms using {method} achieve {accuracy}% accuracy on {dataset}.",
      "Deep learning models with {architecture} process {datatype} for {application} tasks.",
      "Natural language processing techniques like {technique} understand {language} text.",
      "Computer vision systems detect {objects} using {approach} on {hardware} platforms.",
      "Reinforcement learning agents learn {behavior} through {environment} simulations."
    ]
    
    import random
    methods = ['supervised learning', 'unsupervised learning', 'semi-supervised learning']
    accuracies = ['89.2', '92.7', '95.1', '97.3']
    datasets = ['CIFAR-10', 'ImageNet', 'COCO', 'MNIST']
    architectures = ['transformer', 'CNN', 'RNN', 'attention']
    datatypes = ['images', 'text', 'audio', 'video']
    applications = ['classification', 'detection', 'segmentation', 'generation']
    
    for i in range(num_docs):
      template = random.choice(templates)
      text = template.format(
        method=random.choice(methods),
        accuracy=random.choice(accuracies),
        dataset=random.choice(datasets),
        architecture=random.choice(architectures),
        datatype=random.choice(datatypes),
        application=random.choice(applications),
        technique=random.choice(['BERT', 'GPT', 'T5', 'RoBERTa']),
        language=random.choice(['English', 'Spanish', 'French']),
        objects=random.choice(['faces', 'cars', 'buildings']),
        approach=random.choice(['YOLO', 'R-CNN', 'SSD']),
        hardware=random.choice(['GPU', 'TPU', 'CPU']),
        behavior=random.choice(['navigation', 'planning', 'control']),
        environment=random.choice(['simulation', 'real-world', 'virtual'])
      )
      
      tokens, length = tokenize_for_bm25(text, 'en')
      
      cursor.execute('''
        INSERT INTO docs (id, sid, sourcedoc, originaltext, embedtext, embedded, 
                         language, metadata, keyphrase_processed, bm25_tokens, doc_length)
        VALUES (?, 0, ?, ?, ?, 1, 'en', '{}', 1, ?, ?)
      ''', (i+1, f'doc{i+1}.txt', text, text.lower(), tokens, length))
    
    conn.commit()
    conn.close()


@pytest.mark.performance
@pytest.mark.slow
class TestBM25SearchPerformance:
  """Test BM25 search performance."""
  
  def test_bm25_scoring_performance(self, temp_data_manager):
    """Test BM25 scoring performance with various query types."""
    kb_dir = temp_data_manager.create_temp_dir()
    config_file = os.path.join(kb_dir, 'scoring_test.cfg')
    
    with open(config_file, 'w') as f:
      f.write("""[DEFAULT]
vector_model = text-embedding-3-small

[ALGORITHMS]
enable_hybrid_search = true
""")
    
    # Create test dataset
    kb = KnowledgeBase(config_file)
    self._create_search_test_database(kb, 2000)
    
    # Build index
    kb.sql_connection = sqlite3.connect(kb.knowledge_base_db)
    kb.sql_cursor = kb.sql_connection.cursor()
    build_bm25_index(kb)
    bm25_data = load_bm25_index(kb)
    kb.sql_connection.close()
    
    # Test different query types
    test_queries = [
      "machine learning",               # Simple query
      "deep learning neural networks",  # Multi-term query
      "computer vision algorithms",     # Domain-specific query
      "artificial intelligence",        # Common terms
      "reinforcement learning agents"   # Longer query
    ]
    
    for query in test_queries:
      # Measure scoring time
      start_time = time.time()
      
      # Run scoring multiple times for accurate measurement
      for _ in range(100):
        scores = get_bm25_scores(kb, query, bm25_data)
      
      total_time = time.time() - start_time
      avg_time = total_time / 100
      
      assert len(scores) > 0
      assert avg_time < 0.01  # Should score within 10ms on average
      
      # Calculate throughput
      queries_per_second = 100 / total_time
      
      print(f"Query '{query}': {avg_time:.4f}s avg ({queries_per_second:.1f} queries/sec)")
  
  def test_concurrent_bm25_search_performance(self, temp_data_manager):
    """Test BM25 search performance under concurrent load."""
    import asyncio
    import threading
    
    kb_dir = temp_data_manager.create_temp_dir()
    config_file = os.path.join(kb_dir, 'concurrent_test.cfg')
    
    with open(config_file, 'w') as f:
      f.write("""[DEFAULT]
vector_model = text-embedding-3-small

[ALGORITHMS]
enable_hybrid_search = true
""")
    
    # Setup test data
    kb = KnowledgeBase(config_file)
    self._create_search_test_database(kb, 1000)
    
    kb.sql_connection = sqlite3.connect(kb.knowledge_base_db)
    kb.sql_cursor = kb.sql_connection.cursor()
    build_bm25_index(kb)
    bm25_data = load_bm25_index(kb)
    kb.sql_connection.close()
    
    # Test concurrent searches
    test_queries = [
      "machine learning algorithms",
      "deep learning models",
      "natural language processing",
      "computer vision systems",
      "reinforcement learning"
    ] * 10  # 50 total queries
    
    def search_worker(query: str, results: list):
      """Worker function for concurrent searches."""
      start_time = time.time()
      scores = get_bm25_scores(kb, query, bm25_data)
      search_time = time.time() - start_time
      results.append((query, len(scores), search_time))
    
    # Run concurrent searches
    start_time = time.time()
    threads = []
    results = []
    
    for query in test_queries:
      thread = threading.Thread(target=search_worker, args=(query, results))
      threads.append(thread)
      thread.start()
    
    # Wait for all threads to complete
    for thread in threads:
      thread.join()
    
    total_time = time.time() - start_time
    
    # Performance assertions
    assert len(results) == len(test_queries)
    assert total_time < 10.0  # Should complete all searches within 10 seconds
    
    # Calculate throughput
    searches_per_second = len(test_queries) / total_time
    avg_search_time = sum(r[2] for r in results) / len(results)
    
    assert searches_per_second > 10  # Should handle at least 10 searches/sec
    assert avg_search_time < 0.1     # Average search time under 100ms
    
    print(f"Concurrent search performance: {searches_per_second:.1f} searches/sec, avg {avg_search_time:.4f}s")
  
  def test_bm25_search_scalability(self, temp_data_manager):
    """Test BM25 search scalability with increasing dataset sizes."""
    dataset_sizes = [500, 1000, 2000, 4000]
    
    for size in dataset_sizes:
      kb_dir = temp_data_manager.create_temp_dir()
      config_file = os.path.join(kb_dir, f'scale_test_{size}.cfg')
      
      with open(config_file, 'w') as f:
        f.write("""[DEFAULT]
vector_model = text-embedding-3-small

[ALGORITHMS]
enable_hybrid_search = true
""")
      
      # Create and setup dataset
      kb = KnowledgeBase(config_file)
      self._create_search_test_database(kb, size)
      
      kb.sql_connection = sqlite3.connect(kb.knowledge_base_db)
      kb.sql_cursor = kb.sql_connection.cursor()
      build_bm25_index(kb)
      bm25_data = load_bm25_index(kb)
      kb.sql_connection.close()
      
      # Measure search time
      start_time = time.time()
      
      # Run multiple searches
      for _ in range(20):
        scores = get_bm25_scores(kb, "machine learning algorithms", bm25_data)
        assert len(scores) > 0
      
      total_time = time.time() - start_time
      avg_time = total_time / 20
      
      # Search time should not grow significantly with dataset size
      if size <= 1000:
        assert avg_time < 0.005  # Under 5ms for small datasets
      elif size <= 2000:
        assert avg_time < 0.010  # Under 10ms for medium datasets
      else:
        assert avg_time < 0.020  # Under 20ms for large datasets
      
      print(f"Dataset {size}: {avg_time:.4f}s per search")
  
  def _create_search_test_database(self, kb: KnowledgeBase, num_docs: int):
    """Create test database optimized for search testing."""
    conn = sqlite3.connect(kb.knowledge_base_db)
    cursor = conn.cursor()
    
    cursor.execute('''
      CREATE TABLE docs (
        id INTEGER PRIMARY KEY,
        sid INTEGER,
        sourcedoc VARCHAR(255),
        originaltext TEXT,
        embedtext TEXT,
        embedded INTEGER DEFAULT 0,
        language TEXT DEFAULT "en",
        metadata TEXT,
        keyphrase_processed INTEGER DEFAULT 0,
        bm25_tokens TEXT,
        doc_length INTEGER DEFAULT 0
      )
    ''')
    
    # Create diverse content for better search testing
    content_patterns = [
      "Machine learning algorithms use {} to process {} data efficiently.",
      "Deep learning neural networks with {} layers achieve high accuracy.",
      "Natural language processing models understand {} using {} techniques.",
      "Computer vision systems detect {} objects in {} environments.",
      "Reinforcement learning agents learn optimal {} through {}.",
      "Artificial intelligence research focuses on {} for {} applications.",
      "Data science workflows process {} using {} and {} tools.",
      "Statistical models predict {} based on {} features."
    ]
    
    import random
    terms = ['classification', 'regression', 'clustering', 'optimization', 'prediction',
             'analysis', 'recognition', 'detection', 'generation', 'transformation',
             'supervised', 'unsupervised', 'reinforcement', 'neural', 'statistical']
    
    for i in range(num_docs):
      pattern = random.choice(content_patterns)
      
      # Fill pattern with random terms
      filled_terms = random.sample(terms, min(3, len(terms)))
      try:
        text = pattern.format(*filled_terms)
      except (IndexError, KeyError) as e:
        # Not enough terms for pattern or format error
        text = f"Document {i} about machine learning and artificial intelligence algorithms."
      
      tokens, length = tokenize_for_bm25(text, 'en')
      
      cursor.execute('''
        INSERT INTO docs (id, sid, sourcedoc, originaltext, embedtext, embedded, 
                         language, metadata, keyphrase_processed, bm25_tokens, doc_length)
        VALUES (?, 0, ?, ?, ?, 1, 'en', '{}', 1, ?, ?)
      ''', (i+1, f'doc{i+1}.txt', text, text.lower(), tokens, length))
    
    conn.commit()
    conn.close()


@pytest.mark.performance
@pytest.mark.slow
class TestHybridSearchPerformance:
  """Test hybrid search performance combining vector and BM25."""
  
  @patch('query.query_manager.faiss.read_index')
  def test_hybrid_search_performance(self, mock_read_index, temp_data_manager):
    """Test performance of hybrid search combining vector and BM25 results."""
    # Setup mock vector index
    mock_index = Mock()
    mock_index.search.return_value = (
      np.array([[0.2, 0.4, 0.6, 0.8, 1.0]] * 100),  # distances
      np.array([list(range(100))])                    # indices
    )
    mock_read_index.return_value = mock_index
    
    kb_dir = temp_data_manager.create_temp_dir()
    config_file = os.path.join(kb_dir, 'hybrid_test.cfg')
    
    with open(config_file, 'w') as f:
      f.write("""[DEFAULT]
vector_model = text-embedding-3-small
vector_dimensions = 1536

[ALGORITHMS]
enable_hybrid_search = true
vector_weight = 0.7
""")
    
    # Create test data
    kb = KnowledgeBase(config_file)
    self._create_hybrid_test_database(kb, 1000)
    
    # Build BM25 index
    kb.sql_connection = sqlite3.connect(kb.knowledge_base_db)
    kb.sql_cursor = kb.sql_connection.cursor()
    build_bm25_index(kb)
    kb.sql_connection.close()
    
    # Test hybrid search performance
    query_vector = np.array([[0.1] * 1536])
    
    # Measure hybrid search time
    start_time = time.time()
    
    for _ in range(50):  # Multiple searches for accurate timing
      results = perform_hybrid_search(kb, "machine learning algorithms", 
                                    query_vector, mock_index)
      assert len(results) > 0
    
    total_time = time.time() - start_time
    avg_time = total_time / 50
    
    # Hybrid search should be fast
    assert avg_time < 0.1  # Under 100ms per search
    
    # Calculate throughput
    searches_per_second = 50 / total_time
    
    print(f"Hybrid search performance: {avg_time:.4f}s avg ({searches_per_second:.1f} searches/sec)")
  
  @patch('query.query_manager.faiss.read_index')
  def test_hybrid_search_different_weights(self, mock_read_index, temp_data_manager):
    """Test hybrid search performance with different vector/BM25 weight combinations."""
    mock_index = Mock()
    mock_index.search.return_value = (
      np.array([[0.3, 0.5, 0.7]]),
      np.array([[0, 1, 2]])
    )
    mock_read_index.return_value = mock_index
    
    # Test different weight combinations
    weight_configs = [
      (0.9, 0.1),  # Mostly vector
      (0.7, 0.3),  # Default
      (0.5, 0.5),  # Equal
      (0.3, 0.7),  # Mostly BM25
      (0.1, 0.9),  # Mostly BM25
    ]
    
    for vector_weight, bm25_weight in weight_configs:
      kb_dir = temp_data_manager.create_temp_dir()
      config_file = os.path.join(kb_dir, f'weight_test_{vector_weight}.cfg')
      
      with open(config_file, 'w') as f:
        f.write(f"""[DEFAULT]
vector_model = text-embedding-3-small
vector_dimensions = 1536

[ALGORITHMS]
enable_hybrid_search = true
vector_weight = {vector_weight}
""")
      
      kb = KnowledgeBase(config_file)
      self._create_hybrid_test_database(kb, 500)
      
      kb.sql_connection = sqlite3.connect(kb.knowledge_base_db)
      kb.sql_cursor = kb.sql_connection.cursor()
      build_bm25_index(kb)
      kb.sql_connection.close()
      
      # Measure performance
      query_vector = np.array([[0.1] * 1536])
      
      start_time = time.time()
      results = perform_hybrid_search(kb, "machine learning", query_vector, mock_index)
      search_time = time.time() - start_time
      
      assert len(results) > 0
      assert search_time < 0.1  # Should be fast regardless of weights
      
      print(f"Weight {vector_weight:.1f}/{bm25_weight:.1f}: {search_time:.4f}s")
  
  def test_hybrid_search_memory_efficiency(self, temp_data_manager):
    """Test memory efficiency of hybrid search operations."""
    import psutil
    
    process = psutil.Process()
    initial_memory = process.memory_info().rss
    
    kb_dir = temp_data_manager.create_temp_dir()
    config_file = os.path.join(kb_dir, 'memory_test.cfg')
    
    with open(config_file, 'w') as f:
      f.write("""[DEFAULT]
vector_model = text-embedding-3-small

[ALGORITHMS]
enable_hybrid_search = true
""")
    
    # Create substantial dataset
    kb = KnowledgeBase(config_file)
    self._create_hybrid_test_database(kb, 2000)
    
    # Build index
    kb.sql_connection = sqlite3.connect(kb.knowledge_base_db)
    kb.sql_cursor = kb.sql_connection.cursor()
    build_bm25_index(kb)
    kb.sql_connection.close()
    
    pre_search_memory = process.memory_info().rss
    
    # Perform many searches to test memory usage
    with patch('query.query_manager.faiss.read_index') as mock_read_index:
      mock_index = Mock()
      mock_index.search.return_value = (
        np.random.random((1, 100)),  # distances
        np.array([list(range(100))])  # indices
      )
      mock_read_index.return_value = mock_index
      
      query_vector = np.array([[0.1] * 1536])
      
      # Run many searches
      for i in range(100):
        results = perform_hybrid_search(kb, f"test query {i % 10}", 
                                      query_vector, mock_index)
        assert len(results) > 0
        
        # Check memory periodically
        if i % 20 == 0:
          current_memory = process.memory_info().rss
          memory_increase = current_memory - pre_search_memory
          
          # Memory should not grow excessively during searches
          assert memory_increase < 100 * 1024 * 1024  # Less than 100MB
    
    final_memory = process.memory_info().rss
    total_memory_increase = final_memory - initial_memory
    
    # Total memory increase should be reasonable
    assert total_memory_increase < 200 * 1024 * 1024  # Less than 200MB total
    
    print(f"Memory increase during hybrid search testing: {total_memory_increase / 1024 / 1024:.1f}MB")
  
  def _create_hybrid_test_database(self, kb: KnowledgeBase, num_docs: int):
    """Create database for hybrid search testing."""
    conn = sqlite3.connect(kb.knowledge_base_db)
    cursor = conn.cursor()
    
    cursor.execute('''
      CREATE TABLE docs (
        id INTEGER PRIMARY KEY,
        sid INTEGER,
        sourcedoc VARCHAR(255),
        originaltext TEXT,
        embedtext TEXT,
        embedded INTEGER DEFAULT 0,
        language TEXT DEFAULT "en",
        metadata TEXT,
        keyphrase_processed INTEGER DEFAULT 0,
        bm25_tokens TEXT,
        doc_length INTEGER DEFAULT 0
      )
    ''')
    
    # Generate content with good vocabulary overlap for testing
    base_terms = ['machine', 'learning', 'algorithm', 'data', 'model', 'neural', 
                  'network', 'deep', 'artificial', 'intelligence', 'computer', 
                  'vision', 'language', 'processing', 'classification']
    
    import random
    
    for i in range(num_docs):
      # Create document with random selection of terms
      selected_terms = random.sample(base_terms, random.randint(3, 8))
      text = f"Document {i} discusses {' '.join(selected_terms)} and related concepts."
      
      tokens, length = tokenize_for_bm25(text, 'en')
      
      cursor.execute('''
        INSERT INTO docs (id, sid, sourcedoc, originaltext, embedtext, embedded, 
                         language, metadata, keyphrase_processed, bm25_tokens, doc_length)
        VALUES (?, 0, ?, ?, ?, 1, 'en', '{}', 1, ?, ?)
      ''', (i+1, f'doc{i+1}.txt', text, text.lower(), tokens, length))
    
    conn.commit()
    conn.close()


@pytest.mark.performance
@pytest.mark.slow
class TestBM25TokenizationPerformance:
  """Test BM25 tokenization performance."""
  
  def test_tokenization_speed_with_various_content(self):
    """Test tokenization speed with different types of content."""
    # Test content of different types and sizes
    test_contents = [
      # Short technical content
      "Machine learning algorithms achieve 95% accuracy.",
      
      # Medium content with mixed elements
      "The state-of-the-art deep learning model processes natural language using transformer architectures with 175 billion parameters.",
      
      # Long content
      " ".join([f"Sentence {i} contains various technical terms like machine-learning, AI/ML, and data-processing." for i in range(100)]),
      
      # Content with many special characters
      "API endpoints: https://api.example.com/v2/models, user@company.com, version 3.14.15, accuracy 95.7%",
      
      # Mixed language and technical content
      "Python 3.9 with TensorFlow 2.8 and PyTorch 1.12 supports CUDA 11.6 for GPU acceleration on NVIDIA RTX 3090"
    ]
    
    for i, content in enumerate(test_contents):
      # Measure tokenization time
      start_time = time.time()
      
      # Run tokenization multiple times for accurate measurement
      for _ in range(1000):
        tokens, length = tokenize_for_bm25(content, 'en')
        assert tokens is not None
        assert length > 0
      
      total_time = time.time() - start_time
      avg_time = total_time / 1000
      
      # Tokenization should be very fast
      assert avg_time < 0.001  # Under 1ms per tokenization
      
      chars_per_second = len(content) * 1000 / total_time
      
      print(f"Content type {i+1}: {avg_time:.6f}s avg ({chars_per_second:.0f} chars/sec)")
  
  def test_bulk_tokenization_performance(self):
    """Test performance of bulk tokenization operations."""
    # Generate bulk content for testing
    bulk_content = []
    for i in range(1000):  # Reduced from 5000 to prevent memory issues
      content = f"Document {i} about machine learning, deep learning, and artificial intelligence applications."
      bulk_content.append(content)
    
    # Measure bulk tokenization time
    start_time = time.time()
    
    tokenized_results = []
    for content in bulk_content:
      tokens, length = tokenize_for_bm25(content, 'en')
      tokenized_results.append((tokens, length))
    
    total_time = time.time() - start_time
    
    # Performance assertions
    docs_per_second = len(bulk_content) / total_time
    assert docs_per_second > 1000  # Should process over 1000 docs/sec
    assert total_time < 10.0       # Should complete within 10 seconds
    
    # Verify all tokenizations succeeded
    assert len(tokenized_results) == len(bulk_content)
    for tokens, length in tokenized_results:
      assert tokens is not None
      assert length > 0
    
    print(f"Bulk tokenization: {docs_per_second:.0f} docs/sec ({total_time:.2f}s total)")
  
  def test_tokenization_memory_usage(self):
    """Test memory usage during tokenization operations."""
    import psutil
    
    process = psutil.Process()
    initial_memory = process.memory_info().rss
    
    # Create large amount of content
    large_contents = []
    for i in range(1000):
      # Vary content size
      multiplier = (i % 10) + 1
      content = ("Machine learning and artificial intelligence research focuses on deep learning algorithms. " * multiplier)
      large_contents.append(content)
    
    # Monitor memory during tokenization
    pre_tokenization_memory = process.memory_info().rss
    
    tokenized = []
    for content in large_contents:
      tokens, length = tokenize_for_bm25(content, 'en')
      tokenized.append((tokens, length))
    
    post_tokenization_memory = process.memory_info().rss
    
    memory_increase = post_tokenization_memory - pre_tokenization_memory
    
    # Memory usage should be reasonable
    assert memory_increase < 50 * 1024 * 1024  # Less than 50MB for 1000 docs
    
    # Cleanup and check memory is released
    del tokenized
    del large_contents
    
    import gc
    gc.collect()
    
    final_memory = process.memory_info().rss
    memory_after_cleanup = final_memory - initial_memory
    
    # Should release most memory after cleanup
    assert memory_after_cleanup < memory_increase / 2
    
    print(f"Tokenization memory usage: {memory_increase / 1024 / 1024:.1f}MB peak, {memory_after_cleanup / 1024 / 1024:.1f}MB after cleanup")


@pytest.mark.performance
@pytest.mark.slow
class TestBM25RebuildPerformance:
  """Test BM25 index rebuild performance and thresholds."""
  
  def test_rebuild_threshold_performance(self, temp_data_manager):
    """Test performance of rebuild threshold checking."""
    kb_dir = temp_data_manager.create_temp_dir()
    config_file = os.path.join(kb_dir, 'rebuild_test.cfg')
    
    with open(config_file, 'w') as f:
      f.write("""[DEFAULT]
vector_model = text-embedding-3-small

[ALGORITHMS]
enable_hybrid_search = true
bm25_rebuild_threshold = 100
""")
    
    # Create initial dataset
    kb = KnowledgeBase(config_file)
    self._create_rebuild_test_database(kb, 500)
    
    # Build initial index
    kb.sql_connection = sqlite3.connect(kb.knowledge_base_db)
    kb.sql_cursor = kb.sql_connection.cursor()
    build_bm25_index(kb)
    
    # Test threshold checking performance
    from embedding.bm25_manager import rebuild_bm25_if_needed
    
    # Measure threshold check time (should be very fast)
    start_time = time.time()
    
    for _ in range(100):
      result = rebuild_bm25_if_needed(kb)
      assert result is True
    
    total_time = time.time() - start_time
    avg_time = total_time / 100
    
    assert avg_time < 0.001  # Should check threshold under 1ms
    
    kb.sql_connection.close()
    
    print(f"Rebuild threshold check: {avg_time:.6f}s avg")
  
  def test_incremental_rebuild_performance(self, temp_data_manager):
    """Test performance of rebuilding index with incremental data."""
    kb_dir = temp_data_manager.create_temp_dir()
    config_file = os.path.join(kb_dir, 'incremental_test.cfg')
    
    with open(config_file, 'w') as f:
      f.write("""[DEFAULT]
vector_model = text-embedding-3-small

[ALGORITHMS]
enable_hybrid_search = true
bm25_rebuild_threshold = 50
""")
    
    # Start with small dataset
    kb = KnowledgeBase(config_file)
    self._create_rebuild_test_database(kb, 200)
    
    kb.sql_connection = sqlite3.connect(kb.knowledge_base_db)
    kb.sql_cursor = kb.sql_connection.cursor()
    
    # Build initial index
    start_time = time.time()
    initial_index = build_bm25_index(kb)
    initial_build_time = time.time() - start_time
    
    assert initial_index is not None
    
    # Add more data to trigger rebuild
    cursor = kb.sql_cursor
    for i in range(200, 300):  # Add 100 more docs
      text = f"Additional document {i} with machine learning content."
      tokens, length = tokenize_for_bm25(text, 'en')
      
      cursor.execute('''
        INSERT INTO docs (id, sid, sourcedoc, originaltext, embedtext, embedded, 
                         language, metadata, keyphrase_processed, bm25_tokens, doc_length)
        VALUES (?, 0, ?, ?, ?, 1, 'en', '{}', 1, ?, ?)
      ''', (i+1, f'doc{i+1}.txt', text, text.lower(), tokens, length))
    
    kb.sql_connection.commit()
    
    # Measure rebuild time
    start_time = time.time()
    rebuilt_index = build_bm25_index(kb)
    rebuild_time = time.time() - start_time
    
    kb.sql_connection.close()
    
    assert rebuilt_index is not None
    
    # Rebuild should be reasonably fast even with more data
    assert rebuild_time < initial_build_time * 2  # Should not be more than 2x slower
    assert rebuild_time < 5.0  # Should complete within 5 seconds
    
    print(f"Initial build (200 docs): {initial_build_time:.3f}s")
    print(f"Rebuild (300 docs): {rebuild_time:.3f}s")
  
  def _create_rebuild_test_database(self, kb: KnowledgeBase, num_docs: int):
    """Create database for rebuild testing."""
    conn = sqlite3.connect(kb.knowledge_base_db)
    cursor = conn.cursor()
    
    cursor.execute('''
      CREATE TABLE docs (
        id INTEGER PRIMARY KEY,
        sid INTEGER,
        sourcedoc VARCHAR(255),
        originaltext TEXT,
        embedtext TEXT,
        embedded INTEGER DEFAULT 0,
        language TEXT DEFAULT "en",
        metadata TEXT,
        keyphrase_processed INTEGER DEFAULT 0,
        bm25_tokens TEXT,
        doc_length INTEGER DEFAULT 0
      )
    ''')
    
    for i in range(num_docs):
      text = f"Test document {i} contains machine learning and data science concepts."
      tokens, length = tokenize_for_bm25(text, 'en')
      
      cursor.execute('''
        INSERT INTO docs (id, sid, sourcedoc, originaltext, embedtext, embedded, 
                         language, metadata, keyphrase_processed, bm25_tokens, doc_length)
        VALUES (?, 0, ?, ?, ?, 1, 'en', '{}', 1, ?, ?)
      ''', (i+1, f'doc{i+1}.txt', text, text.lower(), tokens, length))
    
    conn.commit()
    conn.close()

#fin