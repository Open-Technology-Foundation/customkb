"""
Unit tests for the reranking manager module.
"""

import os
import sys
from unittest.mock import Mock, patch

import pytest

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from embedding.rerank_manager import (
  CACHE_DIR,
  batch_predict,
  cache_score,
  clear_reranking_cache,
  get_cached_score,
  get_reranking_cache_key,
  rerank_documents,
  rerank_search_results,
)


class TestRerankingCache:
  """Test caching functionality."""

  def test_cache_key_generation(self):
    """Test that cache keys are generated consistently."""
    query = "What is machine learning?"
    doc = "Machine learning is a type of artificial intelligence..."

    key1 = get_reranking_cache_key(query, doc)
    key2 = get_reranking_cache_key(query, doc)

    assert key1 == key2
    assert len(key1) == 32  # MD5 hash length

    # Different inputs should produce different keys
    key3 = get_reranking_cache_key("Different query", doc)
    assert key1 != key3

  def test_cache_and_retrieve_score(self):
    """Test caching and retrieving scores."""
    query = "test query"
    doc = "test document"
    score = 0.85

    # Clear cache first
    clear_reranking_cache()

    # Initially, no cached score
    assert get_cached_score(query, doc) is None

    # Cache the score
    cache_score(query, doc, score)

    # Retrieve from cache
    cached = get_cached_score(query, doc)
    assert cached == score

  def test_cache_persistence(self):
    """Test that cache persists to disk."""
    query = "persistent query"
    doc = "persistent document"
    score = 0.92

    # Clear and cache
    clear_reranking_cache()
    cache_score(query, doc, score)

    # Check disk file exists (uses .json format now)
    cache_key = get_reranking_cache_key(query, doc)
    cache_file = os.path.join(CACHE_DIR, f"{cache_key}.json")
    assert os.path.exists(cache_file)

    # Clear memory cache and verify disk cache works
    import embedding.rerank_manager
    embedding.rerank_manager._memory_cache.clear()

    # Should still retrieve from disk
    cached = get_cached_score(query, doc)
    assert cached == score


class TestBatchPredict:
  """Test batch prediction functionality."""

  def test_batch_predict_success(self):
    """Test successful batch prediction."""
    # Mock model - use side_effect to return correct number of scores per batch
    mock_model = Mock()
    # With batch_size=2 and 3 pairs: first batch (2 items), second batch (1 item)
    mock_model.predict.side_effect = [
      [0.8, 0.6],  # First batch: 2 scores
      [0.9]        # Second batch: 1 score
    ]

    pairs = [
      ("query1", "doc1"),
      ("query2", "doc2"),
      ("query3", "doc3")
    ]

    scores = batch_predict(mock_model, pairs, batch_size=2)

    assert len(scores) == 3
    assert scores == [0.8, 0.6, 0.9]
    assert mock_model.predict.call_count == 2  # Two batches (2 + 1)

  def test_batch_predict_error_handling(self):
    """Test error handling in batch prediction."""
    # Mock model that raises error
    mock_model = Mock()
    mock_model.predict.side_effect = RuntimeError("Model error")

    pairs = [("query", "doc")]

    scores = batch_predict(mock_model, pairs, batch_size=1)

    assert len(scores) == 1
    assert scores[0] == 0.0  # Default score on error


class TestRerankDocuments:
  """Test document reranking functionality."""

  @pytest.mark.asyncio
  async def test_rerank_documents_basic(self):
    """Test basic document reranking."""
    # Mock KB configuration
    mock_kb = Mock()
    mock_kb.reranking_model = 'test-model'
    mock_kb.reranking_device = 'cpu'
    mock_kb.reranking_batch_size = 2
    mock_kb.reranking_top_k = 3
    mock_kb.reranking_cache_size = 100

    query = "test query"
    documents = [
      (1, "relevant document", 0.5),
      (2, "very relevant document", 0.6),
      (3, "not relevant", 0.3),
      (4, "somewhat relevant", 0.4)
    ]

    # Mock the model loading and prediction
    with patch('embedding.rerank_manager.load_reranking_model') as mock_load:
      mock_model = Mock()
      mock_model.predict.return_value = [0.9, 0.95, 0.2]  # Reranking scores
      mock_load.return_value = mock_model

      # Clear cache first
      clear_reranking_cache()

      reranked = await rerank_documents(mock_kb, query, documents, top_k=3)

      # Check that top 3 were reranked
      assert len(reranked) == 4

      # Check ordering (highest score first)
      assert reranked[0][0] == 2  # doc 2 had highest reranking score (0.95)
      assert reranked[1][0] == 1  # doc 1 had second highest (0.9)
      assert reranked[2][0] == 3  # doc 3 had lowest (0.2)
      assert reranked[3][0] == 4  # doc 4 was not reranked

  @pytest.mark.asyncio
  async def test_rerank_with_cache(self):
    """Test that caching works during reranking."""
    mock_kb = Mock()
    mock_kb.reranking_model = 'test-model'
    mock_kb.reranking_device = 'cpu'
    mock_kb.reranking_batch_size = 2
    mock_kb.reranking_top_k = 2
    mock_kb.reranking_cache_size = 100

    query = "cached query"
    documents = [
      (1, "doc one", 0.5),
      (2, "doc two", 0.6)
    ]

    # Pre-cache one score
    cache_score(query, "doc one", 0.85)

    with patch('embedding.rerank_manager.load_reranking_model') as mock_load:
      mock_model = Mock()
      mock_model.predict.return_value = [0.75]  # Only for doc two
      mock_load.return_value = mock_model

      reranked = await rerank_documents(mock_kb, query, documents)

      # Model should only be called for uncached document
      assert mock_model.predict.call_count == 1

      # Check ordering
      assert reranked[0][0] == 1  # doc 1 with cached score 0.85
      assert reranked[1][0] == 2  # doc 2 with new score 0.75


class TestRerankSearchResults:
  """Test integration with search results."""

  @pytest.mark.asyncio
  async def test_rerank_search_results(self):
    """Test reranking of search results format."""
    # Mock KB with database
    mock_kb = Mock()
    mock_kb.enable_reranking = True
    mock_kb.reranking_top_k = 2
    mock_kb.reranking_model = 'test-model'
    mock_kb.reranking_device = 'cpu'
    mock_kb.reranking_batch_size = 2
    mock_kb.reranking_cache_size = 100

    # Mock database cursor
    mock_cursor = Mock()
    mock_cursor.fetchone.side_effect = [
      ("Document text 1",),
      ("Document text 2",),
    ]
    mock_kb.sql_cursor = mock_cursor

    query = "search query"
    search_results = [
      (101, 0.8),  # doc_id, distance
      (102, 0.7),
      (103, 0.9)   # Won't be reranked (top_k=2)
    ]

    with patch('embedding.rerank_manager.load_reranking_model') as mock_load:
      mock_model = Mock()
      mock_model.predict.return_value = [0.6, 0.9]  # Scores for docs
      mock_load.return_value = mock_model

      clear_reranking_cache()
      reranked = await rerank_search_results(mock_kb, query, search_results)

      # Check database queries
      assert mock_cursor.execute.call_count == 2

      # Check result format and ordering
      assert len(reranked) == 3
      assert reranked[0][0] == 102  # Higher reranking score
      assert reranked[1][0] == 101
      assert reranked[2][0] == 103  # Not reranked

  @pytest.mark.asyncio
  async def test_rerank_disabled(self):
    """Test that reranking is skipped when disabled."""
    mock_kb = Mock()
    mock_kb.enable_reranking = False

    search_results = [(1, 0.5), (2, 0.6)]

    reranked = await rerank_search_results(mock_kb, "query", search_results)

    # Should return original results unchanged
    assert reranked == search_results


def test_clear_cache():
  """Test cache clearing functionality."""
  # Add some cached items
  cache_score("query1", "doc1", 0.8)
  cache_score("query2", "doc2", 0.9)

  # Verify cache has items
  assert get_cached_score("query1", "doc1") == 0.8

  # Clear cache
  clear_reranking_cache()

  # Verify cache is empty
  assert get_cached_score("query1", "doc1") is None
  assert get_cached_score("query2", "doc2") is None


if __name__ == "__main__":
  pytest.main([__file__, "-v"])


#fin
