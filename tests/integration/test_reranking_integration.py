"""
Integration tests for reranking functionality in the query pipeline.
"""

import os
import sys
from unittest.mock import Mock, patch

import pytest

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from query.query_manager import process_query_async
from tests.fixtures.mock_data import create_mock_knowledge_base
from utils.logging_config import get_logger

logger = get_logger(__name__)


class TestRerankingIntegration:
  """Test reranking integration with the query pipeline."""

  @pytest.fixture
  def temp_config_file(self, tmp_path):
    """Create a temporary config file with reranking enabled."""
    config_content = """
[DEFAULT]
vector_model = text-embedding-3-small
vector_dimensions = 1536
db_min_tokens = 50
db_max_tokens = 100
query_model = gpt-4o-mini
query_top_k = 10
query_context_scope = 2

[ALGORITHMS]
enable_hybrid_search = true
enable_reranking = true
reranking_model = cross-encoder/ms-marco-MiniLM-L-6-v2
reranking_top_k = 5
reranking_batch_size = 16
reranking_device = cpu
"""
    config_file = tmp_path / "test_reranking.cfg"
    config_file.write_text(config_content)
    return str(config_file)

  @pytest.mark.asyncio
  async def test_query_with_reranking_enabled(self, temp_config_file, tmp_path):
    """Test that reranking is applied when enabled in config."""
    # Create mock KB with test data
    create_mock_knowledge_base(temp_config_file, tmp_path)

    # Mock query arguments
    mock_args = Mock()
    mock_args.config_file = temp_config_file
    mock_args.query_text = "What is artificial intelligence?"
    mock_args.query_file = None
    mock_args.context_only = True  # Only get context, not LLM response
    mock_args.verbose = False
    mock_args.debug = True

    # Mock the reranking model
    with patch('embedding.rerank_manager.load_reranking_model') as mock_load_model:
      # Mock cross-encoder model
      mock_model = Mock()
      # Return higher scores for more relevant documents
      mock_model.predict.return_value = [0.9, 0.7, 0.5, 0.3, 0.1]
      mock_load_model.return_value = mock_model

      # Mock vector search
      with patch('query.processing.get_query_embedding') as mock_embed:
        mock_embed.return_value = Mock()  # Mock embedding

        with patch('faiss.read_index') as mock_faiss:
          mock_index = Mock()
          mock_faiss.return_value = mock_index

          # Run query
          result = await process_query_async(mock_args, logger)

          # Verify reranking was called
          assert mock_load_model.called
          assert mock_model.predict.called

          # Check that result contains context
          assert result is not None
          assert isinstance(result, str)

  @pytest.mark.asyncio
  async def test_query_with_reranking_disabled(self, tmp_path):
    """Test that reranking is skipped when disabled."""
    # Create config with reranking disabled
    config_content = """
[DEFAULT]
vector_model = text-embedding-3-small
query_model = gpt-4o-mini

[ALGORITHMS]
enable_reranking = false
"""
    config_file = tmp_path / "test_no_reranking.cfg"
    config_file.write_text(config_content)

    mock_args = Mock()
    mock_args.config_file = str(config_file)
    mock_args.query_text = "Test query"
    mock_args.query_file = None
    mock_args.context_only = True
    mock_args.verbose = False
    mock_args.debug = False

    with patch('embedding.rerank_manager.load_reranking_model') as mock_load_model, \
         patch('query.processing.get_query_embedding') as mock_embed:
      mock_embed.return_value = Mock()

      with patch('faiss.read_index') as mock_faiss:
        mock_faiss.return_value = Mock()

        # Create mock KB database
        with patch('os.path.exists', return_value=True), \
             patch('query.processing.perform_hybrid_search') as mock_search:
          mock_search.return_value = [(1, 0.5), (2, 0.6)]

          with patch('query.processing.connect_to_database'), \
               patch('query.processing.process_reference_batch') as mock_process:
            mock_process.return_value = []

            await process_query_async(mock_args, logger)

            # Verify reranking was NOT called
            assert not mock_load_model.called

  @pytest.mark.asyncio
  async def test_reranking_error_handling(self, temp_config_file, tmp_path):
    """Test that query continues with original results if reranking fails."""
    mock_args = Mock()
    mock_args.config_file = temp_config_file
    mock_args.query_text = "Test query with error"
    mock_args.query_file = None
    mock_args.context_only = True
    mock_args.verbose = False
    mock_args.debug = True

    # Mock reranking to raise an error
    with patch('embedding.rerank_manager.load_reranking_model') as mock_load_model:
      mock_load_model.side_effect = Exception("Model loading failed")

      with patch('query.processing.get_query_embedding') as mock_embed:
        mock_embed.return_value = Mock()

        with patch('faiss.read_index') as mock_faiss:
          mock_faiss.return_value = Mock()

          with patch('os.path.exists', return_value=True), \
               patch('query.processing.perform_hybrid_search') as mock_search:
            original_results = [(1, 0.5), (2, 0.6)]
            mock_search.return_value = original_results

            with patch('query.processing.connect_to_database'), \
                 patch('query.processing.process_reference_batch') as mock_process:
              mock_process.return_value = [
                [1, "doc1.txt", 0, "Test content", 0.5, "{}"]
              ]

              # Should not raise error, should continue with original results
              result = await process_query_async(mock_args, logger)

              assert result is not None
              assert "Test content" in result


class TestRerankingPerformance:
  """Test reranking performance characteristics."""

  @pytest.mark.asyncio
  async def test_reranking_caching_performance(self, temp_config_file):
    """Test that caching improves reranking performance."""
    import time

    from embedding.rerank_manager import clear_reranking_cache, rerank_documents

    # Clear cache first
    clear_reranking_cache()

    mock_kb = Mock()
    mock_kb.reranking_model = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
    mock_kb.reranking_device = 'cpu'
    mock_kb.reranking_batch_size = 5
    mock_kb.reranking_top_k = 5
    mock_kb.reranking_cache_size = 100

    query = "performance test query"
    documents = [
      (i, f"Document {i} content for testing", 0.5 + i*0.01)
      for i in range(10)
    ]

    with patch('embedding.rerank_manager.load_reranking_model') as mock_load:
      mock_model = Mock()
      mock_model.predict.return_value = [0.8 - i*0.1 for i in range(5)]
      mock_load.return_value = mock_model

      # First run - no cache
      start_time = time.time()
      await rerank_documents(mock_kb, query, documents[:5])
      time.time() - start_time

      # Second run - with cache
      start_time = time.time()
      await rerank_documents(mock_kb, query, documents[:5])
      time.time() - start_time

      # Cache should make second run faster (no model prediction needed)
      assert mock_model.predict.call_count == 1  # Only called once
      # Note: Can't reliably test timing in unit tests, but we verify
      # that prediction was only called once


if __name__ == "__main__":
  pytest.main([__file__, "-v"])


#fin
