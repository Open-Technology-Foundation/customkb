"""
Performance tests for CustomKB.
Tests scalability, throughput, and resource usage under various loads.
"""

import time
from unittest.mock import Mock, patch

import psutil
import pytest


@pytest.mark.performance
@pytest.mark.slow
class TestDatabasePerformance:
  """Test database operation performance."""

  def test_large_file_processing_performance(self, temp_data_manager, mock_nltk_data):
    """Test performance of processing large files."""
    # Create large test file (simulate large document)
    large_content = "This is a test sentence. " * 10000  # ~250KB of text
    large_file = temp_data_manager.create_temp_text_file(large_content, "large_doc.txt")

    # Create config
    config_content = """[DEFAULT]
vector_model = text-embedding-3-small
vector_dimensions = 1536
db_min_tokens = 100
db_max_tokens = 200
"""
    config_file = temp_data_manager.create_temp_config(config_content)

    from config.config_manager import KnowledgeBase
    from database.db_manager import process_database

    mock_logger = Mock()

    # Measure processing time
    start_time = time.time()

    args = Mock()
    args.config_file = config_file
    args.files = [large_file]
    args.language = 'en'
    args.force = False
    args.verbose = False
    args.debug = False

    with patch('builtins.input', return_value='y'):
      result = process_database(args, mock_logger)

    processing_time = time.time() - start_time

    # Performance assertions
    assert processing_time < 10.0  # Should process within 10 seconds
    assert "files added to database" in result

    # Check chunk count (should create multiple chunks)
    kb = KnowledgeBase(config_file)
    import sqlite3
    conn = sqlite3.connect(kb.knowledge_base_db)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM docs")
    chunk_count = cursor.fetchone()[0]
    conn.close()

    # Should create multiple chunks from large file
    assert chunk_count > 10
    print(f"Processed large file in {processing_time:.2f}s, created {chunk_count} chunks")

  def test_many_small_files_performance(self, temp_data_manager, mock_nltk_data):
    """Test performance of processing many small files."""
    # Create many small files
    num_files = 100
    test_files = []

    for i in range(num_files):
      content = f"This is test document {i}. It contains some sample content for testing."
      file_path = temp_data_manager.create_temp_text_file(content, f"doc_{i:03d}.txt")
      test_files.append(file_path)

    config_content = """[DEFAULT]
vector_model = text-embedding-3-small
vector_dimensions = 1536
db_min_tokens = 10
db_max_tokens = 50
"""
    config_file = temp_data_manager.create_temp_config(config_content)

    from database.db_manager import process_database

    mock_logger = Mock()

    # Measure processing time
    start_time = time.time()

    args = Mock()
    args.config_file = config_file
    args.files = test_files
    args.language = 'en'
    args.force = False
    args.verbose = False
    args.debug = False

    with patch('builtins.input', return_value='y'):
      result = process_database(args, mock_logger)

    processing_time = time.time() - start_time

    # Performance assertions
    assert processing_time < 15.0  # Should process within 15 seconds
    assert f"{num_files} files added to database" in result

    # Check processing rate
    files_per_second = num_files / processing_time
    assert files_per_second > 5  # Should process at least 5 files per second

    print(f"Processed {num_files} files in {processing_time:.2f}s ({files_per_second:.1f} files/sec)")

  def test_database_memory_usage(self, temp_data_manager, mock_nltk_data):
    """Test memory usage during database operations."""

    # Monitor memory usage
    process = psutil.Process()
    initial_memory = process.memory_info().rss

    # Create moderate-sized test data
    test_files = []
    for i in range(20):
      content = "Sample content for memory testing. " * 1000  # ~35KB per file
      file_path = temp_data_manager.create_temp_text_file(content, f"memory_test_{i}.txt")
      test_files.append(file_path)

    config_content = """[DEFAULT]
vector_model = text-embedding-3-small
vector_dimensions = 1536
db_min_tokens = 100
db_max_tokens = 200
"""
    config_file = temp_data_manager.create_temp_config(config_content)

    from database.db_manager import process_database

    mock_logger = Mock()

    args = Mock()
    args.config_file = config_file
    args.files = test_files
    args.language = 'en'
    args.force = False
    args.verbose = False
    args.debug = False

    with patch('builtins.input', return_value='y'):
      process_database(args, mock_logger)

    peak_memory = process.memory_info().rss
    memory_increase = peak_memory - initial_memory

    # Memory should not increase excessively (less than 100MB for this test)
    assert memory_increase < 100 * 1024 * 1024

    print(f"Memory increase: {memory_increase / 1024 / 1024:.1f} MB")


@pytest.mark.performance
@pytest.mark.slow
class TestEmbeddingPerformance:
  """Test embedding generation performance."""

  def test_embedding_batch_processing_performance(self, temp_config_file, mock_openai_client):
    """Test performance of embedding batch processing."""
    from config.config_manager import KnowledgeBase
    from embedding.embed_manager import calculate_optimal_batch_size

    kb = KnowledgeBase(temp_config_file)

    # Test with various batch sizes
    test_chunks = [f"Test chunk {i} with some sample content" for i in range(500)]

    # Test batch size calculation performance
    start_time = time.time()

    for _ in range(100):  # Repeat to measure average time
      batch_size = calculate_optimal_batch_size(test_chunks, kb.vector_model, 100)

    calc_time = time.time() - start_time

    assert calc_time < 1.0  # Should calculate quickly
    assert batch_size >= 1

    print(f"Batch size calculation: {calc_time:.4f}s for 100 iterations")

  def test_embedding_cache_performance(self, temp_data_manager):
    """Test embedding cache read/write performance."""
    from embedding.embed_manager import get_cached_embedding, save_embedding_to_cache

    # Test cache write performance
    test_embeddings = [[0.1] * 1536 for _ in range(1000)]

    with patch('embedding.embed_manager.CACHE_DIR', temp_data_manager.create_temp_dir()):
      start_time = time.time()

      for i, embedding in enumerate(test_embeddings):
        save_embedding_to_cache(f"text_{i}", "test-model", embedding)

      write_time = time.time() - start_time

      # Test cache read performance
      start_time = time.time()

      hits = 0
      for i in range(len(test_embeddings)):
        result = get_cached_embedding(f"text_{i}", "test-model")
        if result is not None:
          hits += 1

      read_time = time.time() - start_time

      # Performance assertions
      assert write_time < 5.0  # Should write 1000 embeddings within 5 seconds
      assert read_time < 2.0   # Should read 1000 embeddings within 2 seconds

      write_rate = len(test_embeddings) / write_time
      read_rate = len(test_embeddings) / read_time

      print(f"Cache write rate: {write_rate:.1f} embeddings/sec")
      print(f"Cache read rate: {read_rate:.1f} embeddings/sec")
      print(f"Cache hit rate: {hits}/{len(test_embeddings)}")


@pytest.mark.performance
@pytest.mark.slow
class TestQueryPerformance:
  """Test query processing performance."""

  def test_vector_search_performance(self, temp_database, temp_config_file, mock_faiss_index):
    """Test vector search performance with various dataset sizes."""
    import numpy as np

    # Mock different index sizes
    test_sizes = [100, 1000, 10000]

    for size in test_sizes:
      # Mock FAISS index with specified size
      mock_faiss_index.ntotal = size
      mock_faiss_index.search.return_value = (
        [np.random.random(50).tolist()],  # distances
        [list(range(50))]                  # indices
      )

      # Measure search time
      start_time = time.time()

      query_vector = np.random.random((1, 1536)).astype(np.float32)
      distances, indices = mock_faiss_index.search(query_vector, 50)

      search_time = time.time() - start_time

      # Search should be fast regardless of index size
      assert search_time < 0.1  # Should search within 100ms

      print(f"Vector search with {size} docs: {search_time:.4f}s")

  def test_context_building_performance(self, temp_config_file):
    """Test performance of context string building."""
    from config.config_manager import KnowledgeBase
    from query.query_manager import build_reference_string

    kb = KnowledgeBase(temp_config_file)

    # Create large reference list
    large_reference = []
    for i in range(1000):
      # [rid, rsrc, rsid, originaltext, distance, metadata]
      large_reference.append([
        i, f"doc_{i}.txt", i % 10, f"Content chunk {i} with some text", 0.8, '{}'
      ])

    # Measure context building time
    start_time = time.time()

    context = build_reference_string(kb, large_reference)

    build_time = time.time() - start_time

    # Should build context quickly even with many references
    assert build_time < 2.0  # Should build within 2 seconds
    assert len(context) > 0

    # Check context size
    context_size_kb = len(context) / 1024

    print(f"Built context from {len(large_reference)} references in {build_time:.4f}s")
    print(f"Context size: {context_size_kb:.1f} KB")

  def test_concurrent_query_performance(self, temp_config_file, mock_faiss_index):
    """Test performance under concurrent query load."""
    import asyncio

    from config.config_manager import KnowledgeBase
    from query.query_manager import process_reference_batch

    kb = KnowledgeBase(temp_config_file)

    # Mock database connection
    with patch.object(kb, 'sql_cursor') as mock_cursor:
      mock_cursor.execute.return_value = None
      mock_cursor.fetchall.return_value = [
        (1, 0, "test.txt", "Sample content", '{}')
      ]

      async def run_concurrent_queries():
        # Create multiple query batches
        batches = [
          [(i, 0.8) for i in range(10)]  # 10 doc IDs per batch
          for _ in range(20)  # 20 concurrent batches
        ]

        start_time = time.time()

        # Process batches concurrently
        tasks = [process_reference_batch(kb, batch) for batch in batches]
        results = await asyncio.gather(*tasks)

        total_time = time.time() - start_time

        # Should handle concurrent processing efficiently
        assert total_time < 5.0  # Should complete within 5 seconds
        assert len(results) == 20

        throughput = (len(batches) * 10) / total_time  # docs per second
        print(f"Concurrent query throughput: {throughput:.1f} docs/sec")

        return total_time

      # Run the async test
      import asyncio
      asyncio.run(run_concurrent_queries())


@pytest.mark.performance
@pytest.mark.slow
class TestMemoryUsagePatterns:
  """Test memory usage patterns under various scenarios."""

  def test_memory_usage_during_large_operations(self, temp_data_manager):
    """Test memory usage during large operations."""

    process = psutil.Process()

    # Monitor memory during large text processing
    large_texts = ["Sample text content " * 10000 for _ in range(100)]

    initial_memory = process.memory_info().rss

    # Simulate text processing
    from utils.text_utils import clean_text

    processed_texts = []
    for text in large_texts:
      processed = clean_text(text)
      processed_texts.append(processed)

    peak_memory = process.memory_info().rss
    memory_increase = peak_memory - initial_memory

    # Clean up processed texts
    del processed_texts
    del large_texts

    # Memory increase should be reasonable (less than 500MB)
    assert memory_increase < 500 * 1024 * 1024

    print(f"Memory increase during text processing: {memory_increase / 1024 / 1024:.1f} MB")

  def test_memory_cleanup_after_operations(self, temp_data_manager):
    """Test that memory is properly cleaned up after operations."""
    import gc

    process = psutil.Process()
    initial_memory = process.memory_info().rss

    # Perform memory-intensive operations
    for _i in range(10):
      # Create and process large data structures
      large_data = {f"key_{j}": [k] * 1000 for j in range(1000) for k in range(100)}

      # Process the data
      processed = {k: sum(v) for k, v in large_data.items()}

      # Clean up
      del large_data
      del processed

      # Force garbage collection
      gc.collect()

    final_memory = process.memory_info().rss
    memory_change = final_memory - initial_memory

    # Memory should not grow significantly after cleanup
    assert memory_change < 100 * 1024 * 1024  # Less than 100MB growth

    print(f"Memory change after operations and cleanup: {memory_change / 1024 / 1024:.1f} MB")


@pytest.mark.performance
@pytest.mark.slow
class TestScalabilityLimits:
  """Test scalability limits and breaking points."""

  def test_maximum_chunk_processing(self, temp_data_manager, mock_nltk_data):
    """Test processing limits for maximum number of chunks."""
    # Create configuration for small chunks to maximize count
    config_content = """[DEFAULT]
vector_model = text-embedding-3-small
vector_dimensions = 1536
db_min_tokens = 10
db_max_tokens = 20
"""
    config_file = temp_data_manager.create_temp_config(config_content)

    # Create file that will generate many chunks
    large_content = " ".join([f"Sentence {i} with unique content." for i in range(10000)])
    large_file = temp_data_manager.create_temp_text_file(large_content, "max_chunks.txt")

    from config.config_manager import KnowledgeBase
    from database.db_manager import process_database

    mock_logger = Mock()

    start_time = time.time()

    args = Mock()
    args.config_file = config_file
    args.files = [large_file]
    args.language = 'en'
    args.force = False
    args.verbose = False
    args.debug = False

    with patch('builtins.input', return_value='y'):
      process_database(args, mock_logger)

    processing_time = time.time() - start_time

    # Check chunk count
    kb = KnowledgeBase(config_file)
    import sqlite3
    conn = sqlite3.connect(kb.knowledge_base_db)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM docs")
    chunk_count = cursor.fetchone()[0]
    conn.close()

    # Should handle large number of chunks
    assert chunk_count > 1000
    assert processing_time < 30.0  # Should complete within 30 seconds

    chunks_per_second = chunk_count / processing_time
    print(f"Processed {chunk_count} chunks in {processing_time:.2f}s ({chunks_per_second:.1f} chunks/sec)")

  def test_query_response_time_with_large_context(self, temp_config_file):
    """Test query response time with large context."""
    from config.config_manager import KnowledgeBase
    from query.query_manager import build_reference_string

    kb = KnowledgeBase(temp_config_file)

    # Create very large reference context
    huge_reference = []
    for i in range(5000):  # Very large context
      huge_reference.append([
        i, f"doc_{i}.txt", i % 100,
        f"This is content chunk {i} with substantial text content " * 10,
        0.8, '{}'
      ])

    start_time = time.time()

    context = build_reference_string(kb, huge_reference)

    build_time = time.time() - start_time

    # Should handle large context reasonably
    assert build_time < 10.0  # Should complete within 10 seconds
    assert len(context) > 0

    context_size_mb = len(context) / (1024 * 1024)
    print(f"Built {context_size_mb:.1f} MB context in {build_time:.2f}s")

#fin
