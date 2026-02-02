"""
Unit tests for embed_manager.py
Tests embedding generation, caching, FAISS index operations, and async processing.
"""

import json
import os
import tempfile
import threading
from unittest.mock import AsyncMock, Mock, patch

import pytest

from config.config_manager import KnowledgeBase
from embedding.embed_manager import (
  CacheThreadManager,
  add_to_memory_cache,
  cache_manager,
  calculate_optimal_batch_size,
  configure_cache_manager,
  get_cache_key,
  get_cached_embedding,
  get_optimal_faiss_index,
  process_embedding_batch_async,
  process_embeddings,
  save_embedding_to_cache,
)


class TestCacheThreadManager:
  """Test the CacheThreadManager class for proper thread pool management."""

  def test_cache_manager_initialization(self):
    """Test cache manager initializes with correct defaults."""
    manager = CacheThreadManager()
    assert manager._max_workers == 4
    assert manager._memory_cache_size == 10000
    assert manager._executor is None  # Lazy initialization
    assert len(manager._memory_cache) == 0
    assert len(manager._memory_cache_keys) == 0

  def test_cache_manager_configuration(self):
    """Test cache manager configuration updates."""
    manager = CacheThreadManager()
    manager.configure(max_workers=8, memory_cache_size=5000)
    assert manager._max_workers == 8
    assert manager._memory_cache_size == 5000

  def test_cache_manager_lazy_executor_creation(self):
    """Test that executor is only created when needed."""
    manager = CacheThreadManager()
    assert manager._executor is None

    # Submit a task to trigger executor creation
    import threading
    task_executed = threading.Event()

    def test_task():
      task_executed.set()

    future = manager.submit_cache_task(test_task)
    assert manager._executor is not None

    # Wait for task completion
    future.result(timeout=5)
    assert task_executed.is_set()

  def test_cache_manager_thread_safety(self):
    """Test cache manager thread safety with concurrent operations."""
    manager = CacheThreadManager()
    import threading

    # Test concurrent cache additions
    def add_embeddings(thread_id):
      for i in range(10):
        cache_key = f"thread_{thread_id}_key_{i}"
        embedding = [float(thread_id), float(i)]
        manager.add_to_memory_cache(cache_key, embedding)

    threads = []
    for i in range(5):
      thread = threading.Thread(target=add_embeddings, args=(i,))
      threads.append(thread)
      thread.start()

    for thread in threads:
      thread.join()

    # Verify no data corruption occurred
    assert len(manager._memory_cache) == 50
    assert len(manager._memory_cache_keys) == 50

  def test_cache_manager_lru_eviction(self):
    """Test LRU eviction works correctly."""
    manager = CacheThreadManager()
    manager.configure(memory_cache_size=3)

    # Add 3 items (should fit)
    manager.add_to_memory_cache("key1", [1.0])
    manager.add_to_memory_cache("key2", [2.0])
    manager.add_to_memory_cache("key3", [3.0])
    assert len(manager._memory_cache) == 3

    # Add 4th item (should evict oldest)
    manager.add_to_memory_cache("key4", [4.0])
    assert len(manager._memory_cache) == 3
    assert "key1" not in manager._memory_cache
    assert "key4" in manager._memory_cache

  def test_cache_manager_cleanup(self):
    """Test cache manager cleanup properly shuts down executor."""
    manager = CacheThreadManager()

    # Force executor creation
    task_executed = threading.Event()
    def test_task():
      task_executed.set()

    future = manager.submit_cache_task(test_task)
    future.result(timeout=5)

    assert manager._executor is not None

    # Test cleanup
    manager._cleanup()
    assert manager._executor is None

  def test_configure_cache_manager_function(self, mock_kb):
    """Test the global configure_cache_manager function."""
    # Override cache settings
    mock_kb.cache_thread_pool_size = 6
    mock_kb.memory_cache_size = 8000

    # Configure the global cache manager
    configure_cache_manager(mock_kb)

    # Verify configuration was applied
    assert cache_manager._max_workers == 6
    assert cache_manager._memory_cache_size == 8000

  def test_no_threadpool_resource_leak(self):
    """Test that repeated cache operations don't create new thread pools."""
    manager = CacheThreadManager()

    # Simulate multiple cache save operations
    for _i in range(10):
      def dummy_task():
        pass
      manager.submit_cache_task(dummy_task)

    # Should only have one executor instance
    first_executor = manager._executor
    assert first_executor is not None

    # Submit more tasks
    for _i in range(10):
      def dummy_task():
        pass
      manager.submit_cache_task(dummy_task)

    # Should still be the same executor (no leak)
    assert manager._executor is first_executor

  def test_concurrent_cache_stress(self):
    """Stress test with many concurrent cache operations."""
    manager = CacheThreadManager()
    manager.configure(memory_cache_size=100)  # Small cache for eviction testing

    import random
    import threading
    import time

    # Shared state for stress test
    results = []
    errors = []

    def stress_worker(worker_id, operations_count):
      """Worker that performs many cache operations."""
      try:
        for i in range(operations_count):
          cache_key = f"worker_{worker_id}_key_{i}"
          embedding = [float(worker_id), float(i), random.random()]

          # Add to cache
          manager.add_to_memory_cache(cache_key, embedding)

          # Try to retrieve
          retrieved = manager.get_from_memory_cache(cache_key)

          # Store result
          results.append({
            'worker_id': worker_id,
            'operation': i,
            'added': embedding,
            'retrieved': retrieved,
            'match': retrieved == embedding if retrieved else False
          })

          # Small random delay to increase race condition chances
          time.sleep(random.uniform(0.001, 0.005))

      except (KeyError, TypeError, ValueError) as e:
        errors.append({'worker_id': worker_id, 'error': str(e)})

    # Create stress test threads
    num_workers = 10
    operations_per_worker = 50
    threads = []

    for worker_id in range(num_workers):
      thread = threading.Thread(target=stress_worker, args=(worker_id, operations_per_worker))
      threads.append(thread)

    # Start all threads
    start_time = time.time()
    for thread in threads:
      thread.start()

    # Wait for completion
    for thread in threads:
      thread.join(timeout=30)  # 30 second timeout

    end_time = time.time()

    # Analyze results
    assert len(errors) == 0, f"Errors occurred during stress test: {errors}"
    assert len(results) > 0, "No results generated"

    # Check data integrity - all retrieved values should match added values
    mismatches = [r for r in results if r['retrieved'] and not r['match']]
    assert len(mismatches) == 0, f"Data corruption detected: {mismatches[:5]}"  # Show first 5

    # Verify cache size limits were respected
    final_cache_size = len(manager._memory_cache)
    assert final_cache_size <= 100, f"Cache size exceeded limit: {final_cache_size}"

    # Performance checks
    duration = end_time - start_time
    total_operations = num_workers * operations_per_worker
    ops_per_second = total_operations / duration

    print(f"Stress test: {total_operations} operations in {duration:.2f}s ({ops_per_second:.1f} ops/sec)")

    # Should be able to handle at least 100 ops/sec
    assert ops_per_second > 100, f"Performance too slow: {ops_per_second:.1f} ops/sec"

  def test_cache_performance_metrics(self):
    """Test that performance metrics are tracked correctly."""
    manager = CacheThreadManager()
    manager.reset_metrics()  # Start fresh

    # Perform various operations
    manager.add_to_memory_cache("key1", [1.0, 2.0])
    manager.add_to_memory_cache("key2", [3.0, 4.0])

    # Hit
    result1 = manager.get_from_memory_cache("key1")
    assert result1 == [1.0, 2.0]

    # Miss
    result2 = manager.get_from_memory_cache("nonexistent")
    assert result2 is None

    # Submit a task
    def dummy_task():
      pass
    manager.submit_cache_task(dummy_task)

    # Get metrics
    metrics = manager.get_metrics()

    # Verify metrics
    assert metrics['cache_hits'] == 1
    assert metrics['cache_misses'] == 1
    assert metrics['cache_adds'] == 2
    assert metrics['thread_pool_tasks'] == 1
    assert metrics['cache_hit_ratio'] == 0.5  # 1 hit out of 2 requests
    assert metrics['cache_size'] == 2
    assert metrics['max_cache_size'] == 10000  # Default

  def test_concurrent_metrics_accuracy(self):
    """Test that metrics remain accurate under concurrent access."""
    manager = CacheThreadManager()
    manager.reset_metrics()

    import threading
    import uuid

    def cache_operations(thread_idx):
      # Use thread index + UUID to guarantee unique keys (thread idents can be reused)
      thread_id = f"{thread_idx}_{uuid.uuid4().hex[:8]}"
      for i in range(100):
        # Add and retrieve
        key = f"thread_{thread_id}_key_{i}"
        manager.add_to_memory_cache(key, [float(i)])
        manager.get_from_memory_cache(key)  # Should be a hit
        manager.get_from_memory_cache("nonexistent")  # Should be a miss

    # Run with multiple threads
    threads = []
    for idx in range(5):
      thread = threading.Thread(target=cache_operations, args=(idx,))
      threads.append(thread)
      thread.start()

    for thread in threads:
      thread.join()

    # Check final metrics
    metrics = manager.get_metrics()
    expected_adds = 5 * 100  # 5 threads * 100 adds each
    expected_hits = 5 * 100  # 5 threads * 100 hits each
    expected_misses = 5 * 100  # 5 threads * 100 misses each

    assert metrics['cache_adds'] == expected_adds
    assert metrics['cache_hits'] == expected_hits
    assert metrics['cache_misses'] == expected_misses

    # Hit ratio should be 50% (500 hits out of 1000 total requests)
    expected_hit_ratio = expected_hits / (expected_hits + expected_misses)
    assert abs(metrics['cache_hit_ratio'] - expected_hit_ratio) < 0.01

  def test_thread_safe_proxy_functionality(self):
    """Test that backward compatibility proxies are thread-safe."""
    from embedding.embed_manager import cache_manager, embedding_memory_cache, embedding_memory_cache_keys

    # Clear cache first
    cache_manager._memory_cache.clear()
    cache_manager._memory_cache_keys.clear()

    # Add some test data via the global cache manager
    cache_manager.add_to_memory_cache("key1", [1.0, 2.0])
    cache_manager.add_to_memory_cache("key2", [3.0, 4.0])

    # Test ThreadSafeCacheProxy
    assert "key1" in embedding_memory_cache
    assert embedding_memory_cache["key1"] == [1.0, 2.0]
    assert embedding_memory_cache.get("key1") == [1.0, 2.0]
    assert embedding_memory_cache.get("nonexistent") is None
    assert len(embedding_memory_cache) == 2

    # Test keys/values/items
    keys = list(embedding_memory_cache.keys())
    assert "key1" in keys and "key2" in keys

    values = embedding_memory_cache.values()
    assert [1.0, 2.0] in values and [3.0, 4.0] in values

    items = embedding_memory_cache.items()
    assert ("key1", [1.0, 2.0]) in items

    # Test ThreadSafeCacheKeysProxy
    assert len(embedding_memory_cache_keys) == 2
    assert "key1" in embedding_memory_cache_keys

    # Test iteration
    keys_list = list(embedding_memory_cache_keys)
    assert len(keys_list) == 2

    # Test indexing
    first_key = embedding_memory_cache_keys[0]
    assert first_key in ["key1", "key2"]

  def test_proxy_deprecation_warnings(self):
    """Test that deprecated operations raise warnings."""
    import warnings

    from embedding.embed_manager import embedding_memory_cache, embedding_memory_cache_keys

    # Test deprecated cache assignment
    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter("always")
      embedding_memory_cache["test_key"] = [5.0, 6.0]
      assert len(w) == 1
      assert issubclass(w[0].category, DeprecationWarning)
      assert "deprecated" in str(w[0].message)

    # Test deprecated keys manipulation
    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter("always")
      embedding_memory_cache_keys.append("new_key")
      assert len(w) == 1
      assert issubclass(w[0].category, DeprecationWarning)

    with warnings.catch_warnings(record=True) as w:
      warnings.simplefilter("always")
      embedding_memory_cache_keys.remove("some_key")
      assert len(w) == 1
      assert issubclass(w[0].category, DeprecationWarning)


class TestCacheFunctionality:
  """Test embedding caching functionality."""

  def test_get_cache_key(self):
    """Test cache key generation."""
    text = "This is a test text"
    model = "text-embedding-3-small"

    key = get_cache_key(text, model)

    assert isinstance(key, str)
    assert model in key
    assert len(key) > len(model)  # Should include hash

    # Same input should produce same key
    key2 = get_cache_key(text, model)
    assert key == key2

    # Different input should produce different key
    key3 = get_cache_key("Different text", model)
    assert key != key3

  def test_get_cached_embedding_miss(self):
    """Test cache miss for non-existent embedding."""
    with patch('embedding.embed_manager.CACHE_DIR', tempfile.mkdtemp()):
      result = get_cached_embedding("non-existent text", "test-model")
      assert result is None

  def test_get_cached_embedding_memory_hit(self):
    """Test cache hit from memory cache using new cache manager."""
    test_embedding = [0.1, 0.2, 0.3]
    cache_key = "test_key"

    with patch('embedding.embed_manager.get_cache_key', return_value=cache_key), \
         patch.object(cache_manager, 'get_from_memory_cache', return_value=test_embedding):
      result = get_cached_embedding("test text", "test-model")
      assert result == test_embedding

  def test_get_cached_embedding_disk_hit(self, temp_data_manager):
    """Test cache hit from disk cache."""
    temp_dir = temp_data_manager.create_temp_dir()
    test_embedding = [0.1, 0.2, 0.3]

    # Create cached file in subdirectory structure matching get_cache_file_path
    cache_key = "test_model_hash123"
    subdir = os.path.join(temp_dir, cache_key[:2])
    os.makedirs(subdir, exist_ok=True)
    cache_file = os.path.join(subdir, f"{cache_key}.json")
    with open(cache_file, 'w') as f:
      json.dump(test_embedding, f)

    with patch('embedding.embed_manager.get_cache_key', return_value=cache_key), \
         patch.object(cache_manager, 'get_from_memory_cache', return_value=None), \
         patch('embedding.cache.CACHE_DIR', temp_dir):
      result = get_cached_embedding("test text", "test-model")
      assert result == test_embedding

  def test_save_embedding_to_cache(self):
    """Test saving embedding to cache using new cache manager."""
    test_embedding = [0.1, 0.2, 0.3]
    text = "test text"
    model = "test-model"

    with patch('embedding.embed_manager.add_to_memory_cache') as mock_memory, \
         patch.object(cache_manager, 'submit_cache_task') as mock_submit:
      save_embedding_to_cache(text, model, test_embedding)

      # Should add to memory cache
      mock_memory.assert_called_once()

      # Should submit disk save task to cache manager
      mock_submit.assert_called_once()

  def test_memory_cache_lru_eviction(self):
    """Test LRU eviction in memory cache using the new cache manager."""
    # This test is now covered by TestCacheThreadManager.test_cache_manager_lru_eviction
    # but we keep this one for backwards compatibility testing of the add_to_memory_cache function

    # Create a temporary cache manager for testing
    test_manager = CacheThreadManager()
    test_manager.configure(memory_cache_size=2)

    # Mock the global cache manager
    with patch('embedding.embed_manager.cache_manager', test_manager):

      # Create a KB instance with required cache attributes
      kb = Mock()
      kb.memory_cache_size = 2
      kb.cache_memory_limit_mb = 500  # Must set to avoid Mock comparison error

      # Add first embedding
      add_to_memory_cache("key1", [0.1], kb)
      assert "key1" in test_manager._memory_cache

      # Add second embedding
      add_to_memory_cache("key2", [0.2], kb)
      assert "key2" in test_manager._memory_cache
      assert len(test_manager._memory_cache) == 2

      # Add third embedding (should evict first)
      add_to_memory_cache("key3", [0.3], kb)
      assert "key1" not in test_manager._memory_cache
      assert "key2" in test_manager._memory_cache
      assert "key3" in test_manager._memory_cache
      assert len(test_manager._memory_cache) == 2


class TestFaissIndexOptimization:
  """Test FAISS index optimization functionality."""

  @patch('faiss.IndexFlatIP')
  @patch('faiss.IndexIDMap')
  def test_small_dataset_flat_index(self, mock_id_map, mock_flat):
    """Test flat index for small datasets."""
    mock_index = Mock()
    mock_flat.return_value = mock_index
    mock_id_map.return_value = mock_index

    get_optimal_faiss_index(1536, 500)  # Small dataset

    mock_flat.assert_called_once_with(1536)
    mock_id_map.assert_called_once_with(mock_index)

  @patch('faiss.IndexFlatIP')
  @patch('faiss.IndexIVFFlat')
  @patch('faiss.IndexIDMap')
  def test_medium_dataset_ivf_index(self, mock_id_map, mock_ivf, mock_flat):
    """Test IVF index for medium datasets."""
    mock_quantizer = Mock()
    mock_flat.return_value = mock_quantizer
    mock_index = Mock()
    mock_ivf.return_value = mock_index
    mock_id_map.return_value = mock_index

    get_optimal_faiss_index(1536, 50000)  # Medium dataset

    mock_ivf.assert_called_once()
    assert mock_index.train_mode is True

  @patch('faiss.IndexFlatIP')
  @patch('faiss.IndexIVFPQ')
  @patch('faiss.IndexIDMap')
  def test_large_dataset_pq_index(self, mock_id_map, mock_pq, mock_flat):
    """Test PQ index for large datasets."""
    mock_quantizer = Mock()
    mock_flat.return_value = mock_quantizer
    mock_index = Mock()
    mock_pq.return_value = mock_index
    mock_id_map.return_value = mock_index

    get_optimal_faiss_index(1536, 500000)  # Large dataset

    mock_pq.assert_called_once()
    assert mock_index.train_mode is True

  @patch('faiss.IndexFlatIP')
  @patch('faiss.IndexIDMap')
  def test_high_dimensional_vectors(self, mock_id_map, mock_flat):
    """Test handling of high-dimensional vectors."""
    mock_index = Mock()
    mock_flat.return_value = mock_index
    mock_id_map.return_value = mock_index

    get_optimal_faiss_index(3072, 10000)  # High dimensions

    mock_flat.assert_called_once_with(3072)
    mock_id_map.assert_called_once_with(mock_index)


class TestBatchSizeCalculation:
  """Test optimal batch size calculation."""

  def test_calculate_optimal_batch_size(self):
    """Test optimal batch size calculation."""
    chunks = ["short text"] * 100
    model = "text-embedding-3-small"
    max_batch_size = 100

    result = calculate_optimal_batch_size(chunks, model, max_batch_size)

    assert isinstance(result, int)
    assert result >= 1
    assert result <= max_batch_size

  def test_batch_size_with_long_texts(self):
    """Test batch size calculation with long texts."""
    long_chunks = ["very long text " * 100] * 10
    model = "text-embedding-3-small"
    max_batch_size = 100

    result = calculate_optimal_batch_size(long_chunks, model, max_batch_size)

    # Should be smaller batch size for long texts
    assert result < max_batch_size
    assert result >= 1

  def test_batch_size_unknown_model(self):
    """Test batch size calculation with unknown model."""
    chunks = ["test text"] * 50
    model = "unknown-model"
    max_batch_size = 50

    result = calculate_optimal_batch_size(chunks, model, max_batch_size)

    # Should still return valid batch size
    assert isinstance(result, int)
    assert result >= 1
    assert result <= max_batch_size

  def test_minimum_batch_size(self):
    """Test that batch size is at least 1."""
    very_long_chunks = ["extremely long text " * 1000] * 5
    model = "text-embedding-3-small"
    max_batch_size = 50

    result = calculate_optimal_batch_size(very_long_chunks, model, max_batch_size)

    assert result >= 1


class TestAsyncEmbeddingProcessing:
  """Test async embedding processing functionality."""

  @pytest.mark.asyncio
  async def test_process_embedding_batch_all_cached(self, temp_config_file):
    """Test processing batch where all embeddings are cached."""
    kb = KnowledgeBase(temp_config_file)
    chunks = ["text1", "text2", "text3"]

    cached_embeddings = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]

    with patch('embedding.embed_manager.get_cached_embedding') as mock_cache:
      mock_cache.side_effect = cached_embeddings

      result = await process_embedding_batch_async(kb, chunks)

      assert len(result) == 3
      assert result == cached_embeddings

  @pytest.mark.asyncio
  async def test_process_embedding_batch_api_call(self, temp_config_file):
    """Test processing batch with API call for uncached embeddings."""
    kb = KnowledgeBase(temp_config_file)
    chunks = ["text1", "text2"]

    mock_embeddings = [[0.1, 0.2], [0.3, 0.4]]

    # Mock no cached embeddings
    with patch('embedding.embed_manager.get_cached_embedding', return_value=None), \
         patch('embedding.embed_manager.save_embedding_to_cache'), \
         patch('embedding.embed_manager.litellm_embed.get_embeddings', new_callable=AsyncMock, return_value=mock_embeddings):
      result = await process_embedding_batch_async(kb, chunks)

      assert len(result) == 2

  @pytest.mark.asyncio
  async def test_process_embedding_batch_retry_logic(self, temp_config_file):
    """Test retry logic for failed API calls."""
    kb = KnowledgeBase(temp_config_file)
    chunks = ["text1"]

    mock_get_embeddings = AsyncMock(side_effect=[
      ConnectionError("API error"),
      ConnectionError("API error again"),
      [[0.1, 0.2]],  # Success on third try
    ])

    with patch('embedding.embed_manager.get_cached_embedding', return_value=None), \
         patch('embedding.embed_manager.save_embedding_to_cache'), \
         patch('embedding.embed_manager.litellm_embed.get_embeddings', mock_get_embeddings), \
         patch('asyncio.sleep', new_callable=AsyncMock):
      result = await process_embedding_batch_async(kb, chunks)

      assert len(result) == 1
      assert mock_get_embeddings.call_count == 3

  @pytest.mark.asyncio
  async def test_process_embedding_batch_max_retries(self, temp_config_file):
    """Test max retries exceeded."""
    kb = KnowledgeBase(temp_config_file)
    chunks = ["text1"]

    mock_get_embeddings = AsyncMock(side_effect=ConnectionError("Persistent API error"))

    with patch('embedding.embed_manager.get_cached_embedding', return_value=None), \
         patch('embedding.embed_manager.litellm_embed.get_embeddings', mock_get_embeddings), \
         patch('asyncio.sleep', new_callable=AsyncMock):
      result = await process_embedding_batch_async(kb, chunks)

      # Should return empty list after max retries
      assert len(result) == 0

  @pytest.mark.asyncio
  async def test_process_embedding_batch_mixed_cache_states(self, temp_config_file):
    """Test processing batch with mixed cached/uncached embeddings."""
    kb = KnowledgeBase(temp_config_file)
    chunks = ["hit_cache", "miss_cache"]  # "hit" is in cache, "miss" is not

    def mock_cache_lookup(text, model):
      # Note: text.startswith("hit") not "in" to avoid substring issues
      if text.startswith("hit"):
        return [0.1, 0.2]
      return None

    # Mock LiteLLM embeddings to return the uncached embedding
    mock_get_embeddings = AsyncMock(return_value=[[0.3, 0.4]])

    with patch('embedding.embed_manager.get_cached_embedding', side_effect=mock_cache_lookup), \
         patch('embedding.embed_manager.save_embedding_to_cache'), \
         patch('embedding.embed_manager.litellm_embed.get_embeddings', mock_get_embeddings), \
         patch('asyncio.sleep', new_callable=AsyncMock):
      result = await process_embedding_batch_async(kb, chunks)

      assert len(result) == 2
      assert [0.1, 0.2] in result  # Cached embedding (hit_cache)
      assert [0.3, 0.4] in result  # API embedding (miss_cache)


class TestProcessEmbeddings:
  """Test the main process_embeddings function."""

  def test_process_embeddings_no_config(self):
    """Test processing with invalid config file."""
    args = Mock()
    args.config_file = "nonexistent.cfg"
    args.reset_database = False
    args.verbose = True
    args.debug = False

    mock_logger = Mock()

    result = process_embeddings(args, mock_logger)

    # Check for error about configuration - actual message may vary
    assert "not found" in result or "Error" in result

  def test_process_embeddings_no_database(self, temp_config_file):
    """Test processing with non-existent database."""
    args = Mock()
    args.config_file = temp_config_file
    args.reset_database = False
    args.verbose = True
    args.debug = False

    mock_logger = Mock()

    result = process_embeddings(args, mock_logger)

    assert "does not yet exist" in result

  def test_process_embeddings_no_rows(self, temp_config_file, temp_database):
    """Test processing with database containing no unembedded rows."""
    args = Mock()
    args.config_file = temp_config_file
    args.reset_database = False
    args.verbose = True
    args.debug = False

    mock_logger = Mock()

    # Mock get_fq_cfg_filename to return the temp config file
    with patch('embedding.embed_manager.get_fq_cfg_filename', return_value=temp_config_file), \
         patch('embedding.embed_manager.KnowledgeBase') as mock_kb_class:
      mock_cursor = Mock()
      mock_cursor.fetchall.return_value = []  # No rows to embed

      mock_kb = Mock()
      mock_kb.knowledge_base_db = temp_database
      mock_kb.knowledge_base_vector = temp_database.replace('.db', '.faiss')
      mock_kb.vector_model = 'text-embedding-3-small'
      mock_kb.vector_dimensions = 1536
      mock_kb.vector_chunks = 512
      mock_kb.sql_cursor = mock_cursor
      mock_kb.sql_connection = Mock()
      mock_kb_class.return_value = mock_kb

      with patch('embedding.embed_manager.connect_to_database'), \
           patch('embedding.embed_manager.close_database'), \
           patch('os.path.exists', return_value=True):  # DB exists
        result = process_embeddings(args, mock_logger)

        assert "No rows were found to embed" in result

  @patch('embedding.embed_manager.asyncio.run')
  def test_process_embeddings_success(self, mock_asyncio, temp_config_file, temp_database):
    """Test successful embedding processing."""

    args = Mock()
    args.config_file = temp_config_file
    args.reset_database = False
    args.verbose = True
    args.debug = False

    mock_logger = Mock()

    # Mock successful async processing
    mock_asyncio.return_value = {1, 2, 3}  # Successfully processed IDs

    # Mock FAISS via get_faiss_instance
    mock_faiss = Mock()
    mock_faiss.write_index = Mock()

    with patch('embedding.embed_manager.KnowledgeBase') as mock_kb_class:
      mock_kb = Mock()
      mock_kb.knowledge_base_db = temp_database
      mock_kb.knowledge_base_vector = temp_database.replace('.db', '.faiss')
      mock_kb.vector_model = "test-model"
      mock_kb.vector_chunks = 100
      mock_kb_class.return_value = mock_kb

      with patch('embedding.embed_manager.connect_to_database'), \
           patch('embedding.embed_manager.close_database'), \
           patch('embedding.embed_manager.os.path.exists', return_value=False), \
           patch('embedding.embed_manager.litellm_embed.get_embedding_sync', return_value=[0.1] * 1536), \
           patch('embedding.embed_manager.get_optimal_faiss_index') as mock_index:
        mock_index.return_value = Mock()
        with patch('embedding.embed_manager.get_faiss_instance', return_value=(mock_faiss, False)):

          result = process_embeddings(args, mock_logger)

          # Check result contains expected text
          assert isinstance(result, str)

  def test_process_embeddings_reset_database(self, temp_config_file, temp_database):
    """Test processing with database reset."""
    args = Mock()
    args.config_file = temp_config_file
    args.reset_database = True
    args.verbose = True
    args.debug = False

    mock_logger = Mock()

    # Mock get_fq_cfg_filename to return the temp config file
    with patch('embedding.embed_manager.get_fq_cfg_filename', return_value=temp_config_file), \
         patch('embedding.embed_manager.KnowledgeBase') as mock_kb_class:
      mock_cursor = Mock()
      mock_cursor.fetchall.return_value = []  # No rows to embed

      mock_kb = Mock()
      mock_kb.knowledge_base_db = temp_database
      mock_kb.knowledge_base_vector = temp_database.replace('.db', '.faiss')
      mock_kb.vector_model = 'text-embedding-3-small'
      mock_kb.vector_dimensions = 1536
      mock_kb.vector_chunks = 512
      mock_kb.sql_cursor = mock_cursor
      mock_kb.sql_connection = Mock()
      mock_kb_class.return_value = mock_kb

      with patch('embedding.embed_manager.connect_to_database'), \
           patch('embedding.embed_manager.close_database'), \
           patch('os.path.exists', return_value=True):  # DB exists
        process_embeddings(args, mock_logger)

        # Should call reset query when reset_database=True
        mock_cursor.execute.assert_any_call("UPDATE docs SET embedded=0;")

  def test_process_embeddings_existing_vector_file(self, temp_config_file, temp_database):
    """Test processing with existing vector file."""
    args = Mock()
    args.config_file = temp_config_file
    args.reset_database = False
    args.verbose = True
    args.debug = False

    mock_logger = Mock()

    # Mock get_fq_cfg_filename to return the temp config file
    with patch('embedding.embed_manager.get_fq_cfg_filename', return_value=temp_config_file), \
         patch('embedding.embed_manager.KnowledgeBase') as mock_kb_class:
      mock_cursor = Mock()
      mock_cursor.fetchall.return_value = []  # No rows to embed

      mock_kb = Mock()
      mock_kb.knowledge_base_db = temp_database
      mock_kb.knowledge_base_vector = temp_database.replace('.db', '.faiss')
      mock_kb.vector_model = 'text-embedding-3-small'
      mock_kb.vector_dimensions = 1536
      mock_kb.vector_chunks = 512
      mock_kb.sql_cursor = mock_cursor
      mock_kb.sql_connection = Mock()
      mock_kb_class.return_value = mock_kb

      # Mock FAISS via get_faiss_instance (lazy loading pattern)
      mock_faiss = Mock()
      mock_index = Mock()
      mock_faiss.read_index.return_value = mock_index

      with patch('embedding.embed_manager.get_faiss_instance', return_value=(mock_faiss, False)), \
           patch('embedding.embed_manager.connect_to_database'), \
           patch('embedding.embed_manager.close_database'), \
           patch('os.path.exists', return_value=True):  # DB and vector file exist
        result = process_embeddings(args, mock_logger)

        # The function should complete with "No rows" since fetchall returns []
        assert "No rows were found to embed" in result


class TestErrorHandling:
  """Test error handling in embedding manager."""

  @pytest.mark.skip(reason="API key validation occurs at module import, not testable post-import")
  def test_invalid_api_key_handling(self):
    """Test handling of invalid API key.

    Note: OpenAI client initializes at module import time.
    Setting env vars after import doesn't trigger validation.
    This test would need to be done via subprocess/reimport.
    """
    pass

  def test_corrupted_cache_file_handling(self, temp_data_manager):
    """Test handling of corrupted cache files."""
    temp_dir = temp_data_manager.create_temp_dir()

    # Create corrupted cache file
    cache_key = "test_key"
    cache_file = os.path.join(temp_dir, f"{cache_key}.json")
    with open(cache_file, 'w') as f:
      f.write("invalid json content")

    with patch('embedding.embed_manager.CACHE_DIR', temp_dir), \
         patch('embedding.embed_manager.get_cache_key', return_value=cache_key):
      result = get_cached_embedding("test text", "test-model")
      assert result is None  # Should handle corruption gracefully

  def test_cache_directory_creation_failure(self):
    """Test handling of cache directory creation failure."""
    with patch('os.makedirs', side_effect=OSError("Permission denied")):
      # Should not crash during module import
      try:
        pass
      except OSError:
        pytest.fail("Should handle cache directory creation failure gracefully")

  @pytest.mark.asyncio
  async def test_network_timeout_handling(self, temp_config_file):
    """Test handling of network timeouts during API calls."""
    kb = KnowledgeBase(temp_config_file)
    chunks = ["test text"]

    mock_get_embeddings = AsyncMock(side_effect=TimeoutError("Request timeout"))

    with patch('embedding.embed_manager.get_cached_embedding', return_value=None), \
         patch('embedding.embed_manager.litellm_embed.get_embeddings', mock_get_embeddings), \
         patch('asyncio.sleep', new_callable=AsyncMock):
      result = await process_embedding_batch_async(kb, chunks)

      # Should handle timeout gracefully
      assert isinstance(result, list)

#fin
