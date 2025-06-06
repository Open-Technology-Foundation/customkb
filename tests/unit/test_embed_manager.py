"""
Unit tests for embed_manager.py
Tests embedding generation, caching, FAISS index operations, and async processing.
"""

import pytest
import os
import tempfile
import numpy as np
import asyncio
import json
from unittest.mock import patch, Mock, AsyncMock, MagicMock
from pathlib import Path

from embedding.embed_manager import (
  get_cache_key,
  get_cached_embedding,
  save_embedding_to_cache,
  get_optimal_faiss_index,
  calculate_optimal_batch_size,
  process_embedding_batch_async,
  process_embeddings
)
from config.config_manager import KnowledgeBase


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
    """Test cache hit from memory cache."""
    test_embedding = [0.1, 0.2, 0.3]
    cache_key = "test_key"
    
    with patch('embedding.embed_manager.embedding_memory_cache', {cache_key: test_embedding}):
      with patch('embedding.embed_manager.get_cache_key', return_value=cache_key):
        result = get_cached_embedding("test text", "test-model")
        assert result == test_embedding
  
  def test_get_cached_embedding_disk_hit(self, temp_data_manager):
    """Test cache hit from disk cache."""
    temp_dir = temp_data_manager.create_temp_dir()
    test_embedding = [0.1, 0.2, 0.3]
    
    # Create cached file
    cache_key = "test_model_hash123"
    cache_file = os.path.join(temp_dir, f"{cache_key}.json")
    with open(cache_file, 'w') as f:
      json.dump(test_embedding, f)
    
    with patch('embedding.embed_manager.CACHE_DIR', temp_dir):
      with patch('embedding.embed_manager.get_cache_key', return_value=cache_key):
        with patch('embedding.embed_manager.embedding_memory_cache', {}):
          result = get_cached_embedding("test text", "test-model")
          assert result == test_embedding
  
  def test_save_embedding_to_cache(self):
    """Test saving embedding to cache."""
    test_embedding = [0.1, 0.2, 0.3]
    text = "test text"
    model = "test-model"
    
    with patch('embedding.embed_manager.add_to_memory_cache') as mock_memory:
      with patch('embedding.embed_manager.ThreadPoolExecutor') as mock_executor:
        save_embedding_to_cache(text, model, test_embedding)
        
        # Should add to memory cache
        mock_memory.assert_called_once()
        
        # Should submit disk save task
        mock_executor.assert_called_once()
  
  def test_memory_cache_lru_eviction(self):
    """Test LRU eviction in memory cache."""
    from embedding.embed_manager import add_to_memory_cache, embedding_memory_cache, embedding_memory_cache_keys
    from config.config_manager import KnowledgeBase
    
    # Clear cache
    embedding_memory_cache.clear()
    embedding_memory_cache_keys.clear()
    
    # Create a KB instance with small cache size for testing
    kb = KnowledgeBase("test", memory_cache_size=2)
    
    # Add first embedding
    add_to_memory_cache("key1", [0.1], kb)
    assert "key1" in embedding_memory_cache
    
    # Add second embedding
    add_to_memory_cache("key2", [0.2], kb)
    assert "key2" in embedding_memory_cache
    assert len(embedding_memory_cache) == 2
    
    # Add third embedding (should evict first)
    add_to_memory_cache("key3", [0.3], kb)
    assert "key1" not in embedding_memory_cache
    assert "key2" in embedding_memory_cache
    assert "key3" in embedding_memory_cache
    assert len(embedding_memory_cache) == 2


class TestFaissIndexOptimization:
  """Test FAISS index optimization functionality."""
  
  @patch('faiss.IndexFlatIP')
  @patch('faiss.IndexIDMap')
  def test_small_dataset_flat_index(self, mock_id_map, mock_flat):
    """Test flat index for small datasets."""
    mock_index = Mock()
    mock_flat.return_value = mock_index
    mock_id_map.return_value = mock_index
    
    result = get_optimal_faiss_index(1536, 500)  # Small dataset
    
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
    
    result = get_optimal_faiss_index(1536, 50000)  # Medium dataset
    
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
    
    result = get_optimal_faiss_index(1536, 500000)  # Large dataset
    
    mock_pq.assert_called_once()
    assert mock_index.train_mode is True
  
  @patch('faiss.IndexFlatIP')
  @patch('faiss.IndexIDMap')
  def test_high_dimensional_vectors(self, mock_id_map, mock_flat):
    """Test handling of high-dimensional vectors."""
    mock_index = Mock()
    mock_flat.return_value = mock_index
    mock_id_map.return_value = mock_index
    
    result = get_optimal_faiss_index(3072, 10000)  # High dimensions
    
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
  async def test_process_embedding_batch_api_call(self, temp_config_file, mock_openai_client):
    """Test processing batch with API call for uncached embeddings."""
    kb = KnowledgeBase(temp_config_file)
    chunks = ["text1", "text2"]
    
    # Mock no cached embeddings
    with patch('embedding.embed_manager.get_cached_embedding', return_value=None):
      with patch('embedding.embed_manager.save_embedding_to_cache'):
        with patch('embedding.embed_manager.async_openai_client', mock_openai_client['async']):
          result = await process_embedding_batch_async(kb, chunks)
          
          assert len(result) == 1  # Mock returns one embedding
          mock_openai_client['async'].embeddings.create.assert_called_once()
  
  @pytest.mark.asyncio
  async def test_process_embedding_batch_retry_logic(self, temp_config_file):
    """Test retry logic for failed API calls."""
    kb = KnowledgeBase(temp_config_file)
    chunks = ["text1"]
    
    mock_client = AsyncMock()
    mock_client.embeddings.create.side_effect = [
      Exception("API error"),
      Exception("API error again"),
      Mock(data=[Mock(embedding=[0.1, 0.2])])  # Success on third try
    ]
    
    with patch('embedding.embed_manager.get_cached_embedding', return_value=None):
      with patch('embedding.embed_manager.save_embedding_to_cache'):
        with patch('embedding.embed_manager.async_openai_client', mock_client):
          with patch('asyncio.sleep', new_callable=AsyncMock):
            result = await process_embedding_batch_async(kb, chunks)
            
            assert len(result) == 1
            assert mock_client.embeddings.create.call_count == 3
  
  @pytest.mark.asyncio
  async def test_process_embedding_batch_max_retries(self, temp_config_file):
    """Test max retries exceeded."""
    kb = KnowledgeBase(temp_config_file)
    chunks = ["text1"]
    
    mock_client = AsyncMock()
    mock_client.embeddings.create.side_effect = Exception("Persistent API error")
    
    with patch('embedding.embed_manager.get_cached_embedding', return_value=None):
      with patch('embedding.embed_manager.async_openai_client', mock_client):
        with patch('asyncio.sleep', new_callable=AsyncMock):
          result = await process_embedding_batch_async(kb, chunks)
          
          # Should return empty list after max retries
          assert len(result) == 0
  
  @pytest.mark.asyncio
  async def test_process_embedding_batch_mixed_cache_states(self, temp_config_file):
    """Test processing batch with mixed cached/uncached embeddings."""
    kb = KnowledgeBase(temp_config_file)
    chunks = ["cached_text", "uncached_text"]
    
    def mock_cache_lookup(text, model):
      if "cached" in text:
        return [0.1, 0.2]
      return None
    
    mock_response = Mock()
    mock_response.data = [Mock(embedding=[0.3, 0.4])]
    
    with patch('embedding.embed_manager.get_cached_embedding', side_effect=mock_cache_lookup):
      with patch('embedding.embed_manager.save_embedding_to_cache'):
        with patch('embedding.embed_manager.async_openai_client.embeddings.create', 
                  new_callable=AsyncMock, return_value=mock_response):
          result = await process_embedding_batch_async(kb, chunks)
          
          assert len(result) == 2
          assert [0.1, 0.2] in result  # Cached embedding
          assert [0.3, 0.4] in result  # API embedding


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
    
    assert "Configuration file not found" in result
  
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
    # Mark all rows as embedded
    conn = sqlite3.connect(temp_database)
    cursor = conn.cursor()
    cursor.execute("UPDATE docs SET embedded=1")
    conn.commit()
    conn.close()
    
    args = Mock()
    args.config_file = temp_config_file
    args.reset_database = False
    args.verbose = True
    args.debug = False
    
    mock_logger = Mock()
    
    # Mock KnowledgeBase to use our test database
    with patch('embedding.embed_manager.KnowledgeBase') as mock_kb_class:
      mock_kb = Mock()
      mock_kb.knowledge_base_db = temp_database
      mock_kb.knowledge_base_vector = temp_database.replace('.db', '.faiss')
      mock_kb_class.return_value = mock_kb
      
      with patch('embedding.embed_manager.connect_to_database'):
        with patch('embedding.embed_manager.close_database'):
          result = process_embeddings(args, mock_logger)
          
          assert "No rows were found to embed" in result
  
  @patch('embedding.embed_manager.asyncio.run')
  @patch('embedding.embed_manager.faiss.write_index')
  def test_process_embeddings_success(self, mock_write, mock_asyncio, temp_config_file, temp_database):
    """Test successful embedding processing."""
    import sqlite3
    
    args = Mock()
    args.config_file = temp_config_file
    args.reset_database = False
    args.verbose = True
    args.debug = False
    
    mock_logger = Mock()
    
    # Mock successful async processing
    mock_asyncio.return_value = {1, 2, 3}  # Successfully processed IDs
    
    with patch('embedding.embed_manager.KnowledgeBase') as mock_kb_class:
      mock_kb = Mock()
      mock_kb.knowledge_base_db = temp_database
      mock_kb.knowledge_base_vector = temp_database.replace('.db', '.faiss')
      mock_kb.vector_model = "test-model"
      mock_kb.vector_chunks = 100
      mock_kb_class.return_value = mock_kb
      
      with patch('embedding.embed_manager.connect_to_database'):
        with patch('embedding.embed_manager.close_database'):
          with patch('embedding.embed_manager.os.path.exists', return_value=False):
            with patch('embedding.embed_manager.openai_client.embeddings.create') as mock_openai:
              mock_response = Mock()
              mock_response.data = [Mock(embedding=[0.1] * 1536)]
              mock_openai.return_value = mock_response
              
              with patch('embedding.embed_manager.get_optimal_faiss_index') as mock_index:
                mock_index.return_value = Mock()
                
                result = process_embeddings(args, mock_logger)
                
                assert "embeddings" in result
                assert "saved to" in result
  
  def test_process_embeddings_reset_database(self, temp_config_file, temp_database):
    """Test processing with database reset."""
    args = Mock()
    args.config_file = temp_config_file
    args.reset_database = True
    args.verbose = True
    args.debug = False
    
    mock_logger = Mock()
    
    with patch('embedding.embed_manager.KnowledgeBase') as mock_kb_class:
      mock_kb = Mock()
      mock_kb.knowledge_base_db = temp_database
      mock_kb.knowledge_base_vector = temp_database.replace('.db', '.faiss')
      mock_kb.sql_cursor = Mock()
      mock_kb.sql_connection = Mock()
      mock_kb_class.return_value = mock_kb
      
      with patch('embedding.embed_manager.connect_to_database'):
        with patch('embedding.embed_manager.close_database'):
          with patch('embedding.embed_manager.os.path.exists', return_value=True):
            # Mock empty result to avoid full processing
            mock_kb.sql_cursor.fetchall.return_value = []
            
            result = process_embeddings(args, mock_logger)
            
            # Should call reset query
            mock_kb.sql_cursor.execute.assert_any_call("UPDATE docs SET embedded=0;")
  
  def test_process_embeddings_existing_vector_file(self, temp_config_file, temp_database):
    """Test processing with existing vector file."""
    args = Mock()
    args.config_file = temp_config_file
    args.reset_database = False
    args.verbose = True
    args.debug = False
    
    mock_logger = Mock()
    
    with patch('embedding.embed_manager.KnowledgeBase') as mock_kb_class:
      mock_kb = Mock()
      mock_kb.knowledge_base_db = temp_database
      mock_kb.knowledge_base_vector = temp_database.replace('.db', '.faiss')
      mock_kb_class.return_value = mock_kb
      
      with patch('embedding.embed_manager.connect_to_database'):
        with patch('embedding.embed_manager.close_database'):
          with patch('embedding.embed_manager.os.path.exists', return_value=True):
            with patch('embedding.embed_manager.faiss.read_index') as mock_read:
              mock_index = Mock()
              mock_read.return_value = mock_index
              
              # Mock empty result to avoid full processing
              mock_kb.sql_cursor = Mock()
              mock_kb.sql_cursor.fetchall.return_value = []
              
              result = process_embeddings(args, mock_logger)
              
              mock_read.assert_called_once_with(mock_kb.knowledge_base_vector)


class TestErrorHandling:
  """Test error handling in embedding manager."""
  
  def test_invalid_api_key_handling(self):
    """Test handling of invalid API key."""
    with patch.dict(os.environ, {'OPENAI_API_KEY': 'invalid_key'}):
      with pytest.raises((EnvironmentError, ValueError)):
        # This should fail during module import validation
        from embedding import embed_manager
  
  def test_corrupted_cache_file_handling(self, temp_data_manager):
    """Test handling of corrupted cache files."""
    temp_dir = temp_data_manager.create_temp_dir()
    
    # Create corrupted cache file
    cache_key = "test_key"
    cache_file = os.path.join(temp_dir, f"{cache_key}.json")
    with open(cache_file, 'w') as f:
      f.write("invalid json content")
    
    with patch('embedding.embed_manager.CACHE_DIR', temp_dir):
      with patch('embedding.embed_manager.get_cache_key', return_value=cache_key):
        result = get_cached_embedding("test text", "test-model")
        assert result is None  # Should handle corruption gracefully
  
  def test_cache_directory_creation_failure(self):
    """Test handling of cache directory creation failure."""
    with patch('os.makedirs', side_effect=OSError("Permission denied")):
      # Should not crash during module import
      try:
        from embedding import embed_manager
      except OSError:
        pytest.fail("Should handle cache directory creation failure gracefully")
  
  @pytest.mark.asyncio
  async def test_network_timeout_handling(self, temp_config_file):
    """Test handling of network timeouts during API calls."""
    kb = KnowledgeBase(temp_config_file)
    chunks = ["test text"]
    
    mock_client = AsyncMock()
    mock_client.embeddings.create.side_effect = asyncio.TimeoutError("Request timeout")
    
    with patch('embedding.embed_manager.get_cached_embedding', return_value=None):
      with patch('embedding.embed_manager.async_openai_client', mock_client):
        with patch('asyncio.sleep', new_callable=AsyncMock):
          result = await process_embedding_batch_async(kb, chunks)
          
          # Should handle timeout gracefully
          assert isinstance(result, list)

#fin