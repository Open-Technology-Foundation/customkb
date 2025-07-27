#!/usr/bin/env python
"""
Unit tests for Google AI embedding integration.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import numpy as np

from embedding.embed_manager import process_embedding_batch_async, process_embedding_batch
from query.query_manager import get_query_embedding


class TestGoogleEmbeddingIntegration:
  """Test Google AI embedding functionality."""
  
  @pytest.fixture
  def mock_google_client(self):
    """Create a mock Google AI client."""
    mock_client = Mock()
    
    # Mock embedding response
    mock_embedding = Mock()
    mock_embedding.values = [0.1] * 1536  # 1536 dimensions
    
    mock_response = Mock()
    mock_response.embeddings = [mock_embedding]
    
    # Set up the embed_content method
    mock_client.models.embed_content.return_value = mock_response
    
    return mock_client
  
  @pytest.fixture
  def mock_kb_google(self):
    """Create a mock KnowledgeBase configured for Google embeddings."""
    kb = Mock()
    kb.vector_model = "gemini-embedding-001"
    kb.api_call_delay_seconds = 0.05
    kb.embedding_batch_size = 100
    return kb
  
  @pytest.mark.asyncio
  async def test_process_embedding_batch_async_google(self, mock_kb_google, mock_google_client):
    """Test async batch embedding with Google AI."""
    chunks = ["test chunk 1", "test chunk 2", "test chunk 3"]
    
    with patch('embedding.embed_manager.google_client', mock_google_client):
      with patch('embedding.embed_manager.get_cached_embedding', return_value=None):
        with patch('embedding.embed_manager.save_embedding_to_cache'):
          result = await process_embedding_batch_async(mock_kb_google, chunks)
    
    # Verify Google client was called
    mock_google_client.models.embed_content.assert_called()
    
    # Check result structure
    assert len(result) == 3
    assert all(len(emb) == 1536 for emb in result)
  
  def test_process_embedding_batch_google(self, mock_kb_google, mock_google_client):
    """Test sync batch embedding with Google AI."""
    chunks = ["test chunk 1", "test chunk 2"]
    
    with patch('embedding.embed_manager.google_client', mock_google_client):
      with patch('embedding.embed_manager.get_cached_embedding', return_value=None):
        with patch('embedding.embed_manager.save_embedding_to_cache'):
          result = process_embedding_batch(chunks, mock_kb_google)
    
    # Verify Google client was called
    mock_google_client.models.embed_content.assert_called_with(
      model="gemini-embedding-001",
      contents=chunks
    )
    
    # Check result
    assert len(result) == 2
    assert all(len(emb) == 1536 for emb in result)
  
  @pytest.mark.asyncio
  async def test_get_query_embedding_google(self, mock_kb_google, mock_google_client):
    """Test query embedding generation with Google AI."""
    query_text = "test query"
    
    with patch('query.query_manager.google_client', mock_google_client):
      with patch('query.query_manager.get_cached_query_embedding', return_value=None):
        with patch('query.query_manager.save_query_embedding_to_cache'):
          result = await get_query_embedding(query_text, "gemini-embedding-001", mock_kb_google)
    
    # Verify Google client was called
    mock_google_client.models.embed_content.assert_called_with(
      model="gemini-embedding-001",
      contents=query_text
    )
    
    # Check result shape
    assert result.shape == (1, 1536)
    assert isinstance(result, np.ndarray)
  
  def test_google_client_not_initialized_error(self, mock_kb_google):
    """Test error when Google client is not initialized."""
    chunks = ["test chunk"]
    
    with patch('embedding.embed_manager.google_client', None):
      with pytest.raises(ValueError, match="Google AI client not initialized"):
        process_embedding_batch(chunks, mock_kb_google)
  
  def test_model_limits_include_google(self):
    """Test that model limits include Google models."""
    from embedding.embed_manager import calculate_optimal_batch_size
    
    kb = Mock()
    kb.vector_model = "gemini-embedding-001"
    
    chunks = ["test"] * 100
    result = calculate_optimal_batch_size(chunks, "gemini-embedding-001", 1000, kb)
    
    # Should not raise KeyError and should return a valid batch size
    assert result > 0
  
  def test_provider_detection(self, mock_kb_google):
    """Test that Google models are correctly detected."""
    assert mock_kb_google.vector_model.startswith('gemini-')
    
    # Test OpenAI model detection
    mock_kb_openai = Mock()
    mock_kb_openai.vector_model = "text-embedding-3-small"
    assert not mock_kb_openai.vector_model.startswith('gemini-')


if __name__ == "__main__":
  pytest.main([__file__, "-v"])

#fin