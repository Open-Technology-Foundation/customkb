"""
Unit tests for query_manager.py
Tests query processing, semantic search, AI response generation, and context building.
"""

import pytest
import os
import tempfile
import sqlite3
import numpy as np
import asyncio
import json
import time
from unittest.mock import patch, Mock, AsyncMock, MagicMock
from pathlib import Path

from query.query_manager import (
  get_cache_key,
  get_cached_query_embedding,
  save_query_embedding_to_cache,
  get_context_range,
  get_query_embedding,
  read_context_file,
  fetch_document_by_id,
  process_reference_batch,
  build_reference_string,
  generate_ai_response,
  process_query_async,
  process_query
)
from config.config_manager import KnowledgeBase


class TestQueryCaching:
  """Test query embedding caching functionality."""
  
  def test_get_cache_key_consistency(self):
    """Test that cache key generation is consistent."""
    query = "What is machine learning?"
    model = "text-embedding-3-small"
    
    key1 = get_cache_key(query, model)
    key2 = get_cache_key(query, model)
    
    assert key1 == key2
    assert model in key1
    assert len(key1) > len(model)
  
  def test_get_cached_query_embedding_miss(self, temp_data_manager):
    """Test cache miss for non-existent query embedding."""
    temp_dir = temp_data_manager.create_temp_dir()
    
    with patch('query.query_manager.CACHE_DIR', temp_dir):
      result = get_cached_query_embedding("non-existent query", "test-model")
      assert result is None
  
  def test_get_cached_query_embedding_hit(self, temp_data_manager):
    """Test cache hit for existing query embedding."""
    temp_dir = temp_data_manager.create_temp_dir()
    test_embedding = [0.1, 0.2, 0.3]
    
    # Create cached file
    cache_key = "test_model_hash123"
    cache_file = os.path.join(temp_dir, f"{cache_key}.json")
    with open(cache_file, 'w') as f:
      json.dump(test_embedding, f)
    
    with patch('query.query_manager.CACHE_DIR', temp_dir):
      with patch('query.query_manager.get_cache_key', return_value=cache_key):
        with patch('time.time', return_value=1000):
          with patch('os.path.getmtime', return_value=999):  # Recent file
            result = get_cached_query_embedding("test query", "test-model")
            assert result == test_embedding
  
  def test_get_cached_query_embedding_expired(self, temp_data_manager):
    """Test cache miss for expired query embedding."""
    temp_dir = temp_data_manager.create_temp_dir()
    test_embedding = [0.1, 0.2, 0.3]
    
    # Create cached file
    cache_key = "test_model_hash123"
    cache_file = os.path.join(temp_dir, f"{cache_key}.json")
    with open(cache_file, 'w') as f:
      json.dump(test_embedding, f)
    
    with patch('query.query_manager.CACHE_DIR', temp_dir):
      with patch('query.query_manager.get_cache_key', return_value=cache_key):
        with patch('time.time', return_value=1000000):  # Much later time
          with patch('os.path.getmtime', return_value=1000):  # Old file
            result = get_cached_query_embedding("test query", "test-model")
            assert result is None
            assert not os.path.exists(cache_file)  # Should be deleted
  
  def test_save_query_embedding_to_cache(self, temp_data_manager):
    """Test saving query embedding to cache."""
    temp_dir = temp_data_manager.create_temp_dir()
    test_embedding = [0.1, 0.2, 0.3]
    
    with patch('query.query_manager.CACHE_DIR', temp_dir):
      with patch('query.query_manager.get_cache_key', return_value="test_key"):
        save_query_embedding_to_cache("test query", "test-model", test_embedding)
        
        # Check that file was created
        cache_file = os.path.join(temp_dir, "test_key.json")
        assert os.path.exists(cache_file)
        
        with open(cache_file, 'r') as f:
          saved_embedding = json.load(f)
          assert saved_embedding == test_embedding


class TestContextRange:
  """Test context range calculation."""
  
  def test_basic_context_range(self):
    """Test basic context range calculation."""
    start, end = get_context_range(10, 5)
    
    assert start == 8  # 10 - 2 (half of 4, which is 5-1)
    assert end == 12   # start + 5 - 1
  
  def test_context_range_at_beginning(self):
    """Test context range at the beginning of document."""
    start, end = get_context_range(1, 5)
    
    assert start == 0  # Can't go below 0
    assert end == 4    # 0 + 5 - 1
  
  def test_context_range_single_item(self):
    """Test context range with single item."""
    start, end = get_context_range(5, 1)
    
    assert start == 5
    assert end == 5
  
  def test_context_range_even_number(self):
    """Test context range with even number of items."""
    start, end = get_context_range(10, 4)
    
    assert start == 9  # 10 - 1 (half of 3, which is 4-1)
    assert end == 12   # start + 4 - 1 = 9 + 4 - 1 = 12
  
  def test_context_range_zero_or_negative(self):
    """Test context range with zero or negative context size."""
    start, end = get_context_range(10, 0)
    assert start == 10
    assert end == 10
    
    start, end = get_context_range(10, -1)
    assert start == 10
    assert end == 10


class TestQueryEmbedding:
  """Test query embedding generation."""
  
  @pytest.mark.asyncio
  async def test_get_query_embedding_cached(self):
    """Test getting query embedding from cache."""
    test_embedding = [0.1, 0.2, 0.3]
    
    with patch('query.query_manager.get_cached_query_embedding', return_value=test_embedding):
      with patch('query.query_manager.clean_text', return_value="cleaned query"):
        result = await get_query_embedding("test query", "test-model")
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (1, 3)
        np.testing.assert_array_almost_equal(result[0], test_embedding)
  
  @pytest.mark.asyncio
  async def test_get_query_embedding_api_call(self):
    """Test getting query embedding via API call."""
    test_embedding = [0.1, 0.2, 0.3]
    
    mock_response = Mock()
    mock_response.data = [Mock(embedding=test_embedding)]
    
    mock_client = AsyncMock()
    mock_client.embeddings.create.return_value = mock_response
    
    with patch('query.query_manager.get_cached_query_embedding', return_value=None):
      with patch('query.query_manager.save_query_embedding_to_cache'):
        with patch('query.query_manager.async_openai_client', mock_client):
          with patch('query.query_manager.clean_text', return_value="cleaned query"):
            result = await get_query_embedding("test query", "test-model")
            
            assert isinstance(result, np.ndarray)
            assert result.shape == (1, 3)
            mock_client.embeddings.create.assert_called_once()


class TestContextFileReading:
  """Test context file reading functionality."""
  
  def test_read_context_file_success(self, temp_data_manager):
    """Test successful context file reading."""
    content = "This is context content with <special> characters & symbols."
    context_file = temp_data_manager.create_temp_text_file(content, "context.txt")
    
    file_content, base_name = read_context_file(context_file)
    
    assert "This is context content" in file_content
    assert "&lt;special&gt;" in file_content  # Should be XML escaped
    assert "&amp;" in file_content  # Should be XML escaped
    assert base_name == "context"
  
  def test_read_context_file_nonexistent(self):
    """Test reading non-existent context file."""
    file_content, base_name = read_context_file("/nonexistent/file.txt")
    
    assert file_content == ""
    assert base_name == ""
  
  def test_read_context_file_permission_error(self, temp_data_manager):
    """Test reading context file with permission error."""
    context_file = temp_data_manager.create_temp_text_file("content", "restricted.txt")
    
    with patch('builtins.open', side_effect=PermissionError("Access denied")):
      file_content, base_name = read_context_file(context_file)
      
      assert file_content == ""
      assert base_name == ""


class TestDocumentFetching:
  """Test document fetching from database."""
  
  def test_fetch_document_by_id_success(self, temp_database, temp_config_file):
    """Test successful document fetching."""
    kb = KnowledgeBase(temp_config_file)
    kb.knowledge_base_db = temp_database
    
    # Connect to database
    import sqlite3
    kb.sql_connection = sqlite3.connect(temp_database)
    kb.sql_cursor = kb.sql_connection.cursor()
    
    result = fetch_document_by_id(kb, 1)
    
    assert result is not None
    assert len(result) == 3  # (id, sid, sourcedoc)
    assert result[0] == 1    # id
    
    kb.sql_connection.close()
  
  def test_fetch_document_by_id_not_found(self, temp_database, temp_config_file):
    """Test fetching non-existent document."""
    kb = KnowledgeBase(temp_config_file)
    kb.knowledge_base_db = temp_database
    
    import sqlite3
    kb.sql_connection = sqlite3.connect(temp_database)
    kb.sql_cursor = kb.sql_connection.cursor()
    
    result = fetch_document_by_id(kb, 999)
    
    assert result is None
    
    kb.sql_connection.close()
  
  def test_fetch_document_database_error(self, temp_config_file):
    """Test handling of database errors during document fetching."""
    kb = KnowledgeBase(temp_config_file)
    kb.sql_cursor = Mock()
    kb.sql_cursor.execute.side_effect = sqlite3.Error("Database error")
    
    result = fetch_document_by_id(kb, 1)
    
    assert result is None


class TestReferenceBatchProcessing:
  """Test reference batch processing."""
  
  @pytest.mark.asyncio
  async def test_process_reference_batch_success(self, temp_database, temp_config_file):
    """Test successful reference batch processing."""
    kb = KnowledgeBase(temp_config_file)
    kb.knowledge_base_db = temp_database
    kb.query_context_scope = 2
    
    import sqlite3
    kb.sql_connection = sqlite3.connect(temp_database)
    kb.sql_cursor = kb.sql_connection.cursor()
    
    batch = [(1, 0.8), (2, 0.7)]  # (doc_id, distance) pairs
    
    result = await process_reference_batch(kb, batch)
    
    assert isinstance(result, list)
    assert len(result) > 0
    # Each result should have: [rid, rsrc, rsid, originaltext, distance, metadata]
    for item in result:
      assert len(item) == 6
    
    kb.sql_connection.close()
  
  @pytest.mark.asyncio
  async def test_process_reference_batch_adaptive_scope(self, temp_database, temp_config_file):
    """Test adaptive context scope based on similarity."""
    kb = KnowledgeBase(temp_config_file)
    kb.knowledge_base_db = temp_database
    kb.query_context_scope = 4
    
    import sqlite3
    kb.sql_connection = sqlite3.connect(temp_database)
    kb.sql_cursor = kb.sql_connection.cursor()
    
    # High similarity should use smaller scope
    batch = [(1, 0.5)]  # High similarity (low distance)
    
    with patch('query.query_manager.fetch_document_by_id', return_value=(1, 0, "test.txt")):
      result = await process_reference_batch(kb, batch)
      
      # Should use smaller context scope for high similarity
      assert isinstance(result, list)
    
    kb.sql_connection.close()
  
  @pytest.mark.asyncio
  async def test_process_reference_batch_missing_documents(self, temp_config_file):
    """Test handling of missing documents in batch processing."""
    kb = KnowledgeBase(temp_config_file)
    kb.query_context_scope = 2
    
    batch = [(999, 0.8)]  # Non-existent document
    
    with patch('query.query_manager.fetch_document_by_id', return_value=None):
      result = await process_reference_batch(kb, batch)
      
      assert result == []  # Should return empty list


class TestReferenceStringBuilding:
  """Test reference string building functionality."""
  
  def test_build_reference_string_basic(self, temp_config_file):
    """Test basic reference string building."""
    kb = KnowledgeBase(temp_config_file)
    
    # Mock reference data: [rid, rsrc, rsid, originaltext, distance, metadata]
    reference = [
      [1, "test.txt", 0, "First chunk of text", 0.8, '{"char_length": 20}'],
      [2, "test.txt", 1, "Second chunk of text", 0.7, '{"char_length": 21}']
    ]
    
    result = build_reference_string(kb, reference)
    
    assert '<context src="test.txt:0">' in result
    assert "First chunk of text" in result
    assert "Second chunk of text" in result
    assert "</context>" in result
  
  def test_build_reference_string_with_context_files(self, temp_config_file):
    """Test reference string building with context files."""
    kb = KnowledgeBase(temp_config_file)
    
    context_files_content = [
      ("Context file content", "context"),
      ("Additional context", "extra")
    ]
    
    reference = [
      [1, "test.txt", 0, "Main content", 0.8, '{}']
    ]
    
    result = build_reference_string(kb, reference, context_files_content)
    
    assert '<reference src="context">' in result
    assert "Context file content" in result
    assert '<reference src="extra">' in result
    assert "Additional context" in result
    assert "Main content" in result
  
  def test_build_reference_string_metadata_parsing(self, temp_config_file):
    """Test metadata parsing in reference string building."""
    kb = KnowledgeBase(temp_config_file)
    
    metadata = {
      "heading": "Test Heading",
      "section_type": "code_block",
      "char_length": 100,
      "word_count": 20
    }
    
    reference = [
      [1, "test.txt", 0, "Content with metadata", 0.8, json.dumps(metadata)]
    ]
    
    result = build_reference_string(kb, reference, debug=True)
    
    assert '<meta name="heading">Test Heading</meta>' in result
    assert '<meta name="section_type">code_block</meta>' in result
    assert '<meta name="similarity">0.2000</meta>' in result  # 1.0 - 0.8
  
  def test_build_reference_string_xml_escaping(self, temp_config_file):
    """Test XML escaping in reference string building."""
    kb = KnowledgeBase(temp_config_file)
    
    reference = [
      [1, "test.txt", 0, "Content with <tags> & special chars", 0.8, '{}']
    ]
    
    result = build_reference_string(kb, reference)
    
    assert "&lt;tags&gt;" in result
    assert "&amp;" in result
    assert "<tags>" not in result
  
  def test_build_reference_string_invalid_metadata(self, temp_config_file):
    """Test handling of invalid metadata in reference string building."""
    kb = KnowledgeBase(temp_config_file)
    
    reference = [
      [1, "test.txt", 0, "Content", 0.8, "invalid json metadata"]
    ]
    
    result = build_reference_string(kb, reference)
    
    # Should still build string without metadata
    assert "Content" in result
    assert '<context src="test.txt:0">' in result


class TestOpenAIResponsesAPIHelpers:
  """Test OpenAI Responses API helper functions."""
  
  def test_is_reasoning_model(self):
    """Test reasoning model detection."""
    from query.query_manager import _is_reasoning_model
    
    # Test reasoning models
    assert _is_reasoning_model("gpt-5") == True
    assert _is_reasoning_model("o1") == True
    assert _is_reasoning_model("o1-preview") == True
    assert _is_reasoning_model("o3") == True
    assert _is_reasoning_model("o3-mini") == True
    assert _is_reasoning_model("o4") == True
    assert _is_reasoning_model("o4-mini") == True
    
    # Test non-reasoning models
    assert _is_reasoning_model("gpt-5-chat-latest") == False
    assert _is_reasoning_model("gpt-4o") == False
    assert _is_reasoning_model("gpt-4o-mini") == False
    assert _is_reasoning_model("claude-3-sonnet") == False
  
  def test_format_messages_for_responses_api(self):
    """Test message formatting for Responses API."""
    from query.query_manager import format_messages_for_responses_api
    
    messages = [
      {"role": "system", "content": "You are helpful"},
      {"role": "user", "content": "Hello"},
      {"role": "assistant", "content": "Hi there!"},
      {"role": "user", "content": "How are you?"}
    ]
    
    formatted = format_messages_for_responses_api(messages)
    
    # Check system -> developer mapping
    assert formatted[0]["role"] == "developer"
    assert formatted[0]["content"][0]["type"] == "input_text"
    assert formatted[0]["content"][0]["text"] == "You are helpful"
    
    # Check user messages
    assert formatted[1]["role"] == "user"
    assert formatted[1]["content"][0]["type"] == "input_text"
    assert formatted[1]["content"][0]["text"] == "Hello"
    
    # Check assistant messages use output_text
    assert formatted[2]["role"] == "assistant"
    assert formatted[2]["content"][0]["type"] == "output_text"
    assert formatted[2]["content"][0]["text"] == "Hi there!"
    
    # Check second user message
    assert formatted[3]["role"] == "user"
    assert formatted[3]["content"][0]["type"] == "input_text"
    assert formatted[3]["content"][0]["text"] == "How are you?"
  
  def test_extract_content_from_response(self):
    """Test content extraction from Responses API response."""
    from query.query_manager import _extract_content_from_response
    
    # Test proper Responses API format
    response_data = {
      "output": [{
        "type": "message",
        "content": [{
          "type": "output_text",
          "text": "Extracted text"
        }]
      }]
    }
    
    content = _extract_content_from_response(response_data)
    assert content == "Extracted text"
    
    # Test with text field fallback
    response_data_alt = {
      "output": [{
        "type": "message",
        "content": [{
          "text": "Alternative text"
        }]
      }]
    }
    
    content_alt = _extract_content_from_response(response_data_alt)
    assert content_alt == "Alternative text"
    
    # Test empty response
    assert _extract_content_from_response({}) == ""
    assert _extract_content_from_response(None) == ""

class TestAIResponseGeneration:
  """Test AI response generation functionality."""
  
  @pytest.mark.asyncio
  async def test_generate_ai_response_openai(self, temp_config_file):
    """Test AI response generation with OpenAI model using Responses API."""
    kb = KnowledgeBase(temp_config_file)
    kb.query_model = "gpt-4o"
    kb.query_role = "You are a helpful assistant."
    kb.query_temperature = 0.7
    kb.query_max_tokens = 1000
    
    mock_response = Mock()
    mock_response.model_dump.return_value = {
      "output": [{
        "type": "message",
        "content": [{
          "type": "output_text",
          "text": "Test AI response"
        }]
      }]
    }
    
    mock_client = AsyncMock()
    mock_client.responses.create.return_value = mock_response
    
    with patch('query.query_manager.async_openai_client', mock_client):
      with patch('query.query_manager.elapsed_time', return_value="5s"):
        result = await generate_ai_response(kb, "Reference context", "Test query")
        
        assert result == "Test AI response"
        mock_client.responses.create.assert_called_once()
  
  @pytest.mark.asyncio
  async def test_generate_ai_response_claude(self, temp_config_file):
    """Test AI response generation with Claude model."""
    kb = KnowledgeBase(temp_config_file)
    kb.query_model = "claude-3-sonnet-20240229"
    kb.query_role = "You are a helpful assistant."
    kb.query_temperature = 0.7
    kb.query_max_tokens = 1000
    
    mock_response = Mock()
    mock_response.content = [Mock(text="Test Claude response")]
    
    mock_client = AsyncMock()
    mock_client.messages.create.return_value = mock_response
    
    with patch('query.query_manager.async_anthropic_client', mock_client):
      with patch('query.query_manager.elapsed_time', return_value="5s"):
        result = await generate_ai_response(kb, "Reference context", "Test query")
        
        assert result == "Test Claude response"
        mock_client.messages.create.assert_called_once()
  
  @pytest.mark.asyncio
  async def test_generate_ai_response_o1_model(self, temp_config_file):
    """Test AI response generation with O1 model using Responses API."""
    kb = KnowledgeBase(temp_config_file)
    kb.query_model = "o1-preview"
    kb.query_role = "You are a helpful assistant."
    kb.query_max_tokens = 1000
    
    mock_response = Mock()
    mock_response.model_dump.return_value = {
      "output": [{
        "type": "message",
        "content": [{
          "type": "output_text",
          "text": "Test O1 response"
        }]
      }]
    }
    
    mock_client = AsyncMock()
    mock_client.responses.create.return_value = mock_response
    
    with patch('query.query_manager.async_openai_client', mock_client):
      with patch('query.query_manager.elapsed_time', return_value="5s"):
        result = await generate_ai_response(kb, "Reference context", "Test query")
        
        assert result == "Test O1 response"
        # O1 model should use Responses API with reasoning parameters
        call_args = mock_client.responses.create.call_args
        payload = call_args[1]
        assert 'reasoning' in payload
        assert payload['reasoning']['effort'] == 'minimal'
        assert payload['text']['verbosity'] == 'low'
  
  @pytest.mark.asyncio
  async def test_generate_ai_response_error_handling(self, temp_config_file):
    """Test error handling in AI response generation."""
    kb = KnowledgeBase(temp_config_file)
    kb.query_model = "gpt-4o"
    kb.query_max_tokens = 1000
    kb.query_temperature = 0.7
    
    mock_client = AsyncMock()
    mock_client.responses.create.side_effect = Exception("API Error")
    
    with patch('query.query_manager.async_openai_client', mock_client):
      result = await generate_ai_response(kb, "Reference context", "Test query")
      
      assert "Error: Failed to generate response" in result
  
  @pytest.mark.asyncio
  async def test_generate_ai_response_datetime_replacement(self, temp_config_file):
    """Test datetime placeholder replacement in query role."""
    kb = KnowledgeBase(temp_config_file)
    kb.query_model = "gpt-4o"
    kb.query_role = "Current time: {{datetime}}"
    kb.query_max_tokens = 1000
    kb.query_temperature = 0.7
    
    mock_response = Mock()
    mock_response.model_dump.return_value = {
      "output": [{
        "type": "message",
        "content": [{
          "type": "output_text",
          "text": "Response"
        }]
      }]
    }
    
    mock_client = AsyncMock()
    mock_client.responses.create.return_value = mock_response
    
    with patch('query.query_manager.async_openai_client', mock_client):
      with patch('query.query_manager.elapsed_time', return_value="5s"):
        await generate_ai_response(kb, "Context", "Query")
        
        # Check that datetime was replaced in the formatted input
        call_args = mock_client.responses.create.call_args
        # The system message is now in the formatted input
        input_messages = call_args[1]['input']
        # Find the developer (previously system) message
        system_content = None
        for msg in input_messages:
          if msg['role'] == 'developer':
            system_content = msg['content'][0]['text']
            break
        assert system_content is not None
        assert "{{datetime}}" not in system_content
        assert "Current time:" in system_content
  
  @pytest.mark.asyncio
  async def test_generate_ai_response_grok(self, temp_config_file):
    """Test AI response generation with Grok model."""
    kb = KnowledgeBase(temp_config_file)
    kb.query_model = "grok-4"
    kb.query_role = "You are Grok, a helpful AI assistant."
    kb.query_temperature = 0.7
    kb.query_max_tokens = 1000
    
    # Mock the xAI response using OpenAI-compatible format
    mock_response = Mock()
    mock_response.choices = [Mock(message=Mock(content="Test Grok response"))]
    
    mock_client = AsyncMock()
    mock_client.chat.completions.create.return_value = mock_response
    
    with patch('query.query_manager.async_xai_client', mock_client):
      with patch('query.query_manager.elapsed_time', return_value="5s"):
        result = await generate_ai_response(kb, "Reference context", "Test query")
        
        assert result == "Test Grok response"
        mock_client.chat.completions.create.assert_called_once()
        call_args = mock_client.chat.completions.create.call_args
        # Model might be resolved to a specific version like "grok-4-0709"
        assert call_args[1]['model'].startswith("grok-4")
        assert call_args[1]['temperature'] == 0.7
        assert call_args[1]['max_tokens'] == 1000
  
  @pytest.mark.asyncio
  async def test_generate_ai_response_grok_no_client(self, temp_config_file):
    """Test error when xAI client is not initialized."""
    kb = KnowledgeBase(temp_config_file)
    kb.query_model = "grok-4"
    
    with patch('query.query_manager.async_xai_client', None):
      result = await generate_ai_response(kb, "Reference context", "Test query")
      
      assert "xAI client not initialized" in result


class TestProcessQuery:
  """Test the main process_query functions."""
  
  @pytest.mark.asyncio
  async def test_process_query_async_success(self, temp_database, temp_config_file, mock_faiss_index, mock_db_connection):
    """Test successful async query processing."""
    args = Mock()
    args.config_file = temp_config_file
    args.query_text = "What is machine learning?"
    args.query_file = ""
    args.context_only = False
    args.verbose = True
    args.format = "xml"
    args.debug = False
    
    mock_logger = Mock()
    
    kb = KnowledgeBase(temp_config_file)
    kb.knowledge_base_db = temp_database
    kb.knowledge_base_vector = temp_database.replace('.db', '.faiss')
    kb.sql_connection, kb.sql_cursor = mock_db_connection
    
    with patch('query.query_manager.KnowledgeBase', return_value=kb):
      with patch('query.query_manager.connect_to_database'):
        with patch('query.query_manager.close_database'):
          with patch('os.path.exists', return_value=True):
            with patch('os.path.getsize', return_value=1024):
              with patch('query.query_manager.faiss.read_index', return_value=mock_faiss_index):
                with patch('query.query_manager.get_query_embedding') as mock_embedding:
                  mock_embedding.return_value = np.array([[0.1, 0.2, 0.3]])
                  
                  with patch('query.query_manager.process_reference_batch') as mock_batch:
                    mock_batch.return_value = [
                      [1, "test.txt", 0, "Test content", 0.8, "{}"]
                    ]
                    
                    with patch('query.query_manager.generate_ai_response') as mock_ai:
                      mock_ai.return_value = "AI generated response"
                      
                      result = await process_query_async(args, mock_logger)
                      
                      assert result == "AI generated response"
  
  @pytest.mark.asyncio
  async def test_process_query_async_context_only(self, temp_database, temp_config_file, mock_faiss_index, mock_db_connection):
    """Test async query processing with context-only flag."""
    args = Mock()
    args.config_file = temp_config_file
    args.query_text = "What is machine learning?"
    args.query_file = ""
    args.context_only = True
    args.verbose = True
    
    mock_logger = Mock()
    
    kb = KnowledgeBase(temp_config_file)
    kb.knowledge_base_db = temp_database
    kb.knowledge_base_vector = temp_database.replace('.db', '.faiss')
    kb.sql_connection, kb.sql_cursor = mock_db_connection
    
    with patch('query.query_manager.KnowledgeBase', return_value=kb):
      with patch('query.query_manager.connect_to_database'):
        with patch('query.query_manager.close_database'):
          with patch('os.path.exists', return_value=True):
            with patch('os.path.getsize', return_value=1024):
              with patch('query.query_manager.faiss.read_index', return_value=mock_faiss_index):
                with patch('query.query_manager.get_query_embedding') as mock_embedding:
                  mock_embedding.return_value = np.array([[0.1, 0.2, 0.3]])
                  
                  with patch('query.query_manager.process_reference_batch') as mock_batch:
                    mock_batch.return_value = [
                      [1, "test.txt", 0, "Test content", 0.8, "{}"]
                    ]
                    
                    with patch('query.query_manager.build_reference_string') as mock_build:
                      mock_build.return_value = "Reference context"
                      
                      result = await process_query_async(args, mock_logger)
                      
                      assert result == "Reference context"
  
  def test_process_query_sync_wrapper(self, temp_config_file):
    """Test sync wrapper for process_query."""
    args = Mock()
    args.config_file = temp_config_file
    
    mock_logger = Mock()
    
    with patch('query.query_manager.asyncio.run') as mock_run:
      mock_run.return_value = "Query result"
      
      result = process_query(args, mock_logger)
      
      assert result == "Query result"
      mock_run.assert_called_once()
  
  @pytest.mark.asyncio
  async def test_process_query_with_query_file(self, temp_config_file, temp_data_manager, mock_db_connection):
    """Test query processing with additional query file."""
    # Create query file
    query_file_content = "Additional context from file"
    query_file = temp_data_manager.create_temp_text_file(query_file_content, "query.txt")
    
    args = Mock()
    args.config_file = temp_config_file
    args.query_text = "Main query"
    args.query_file = query_file
    args.context_only = True
    args.verbose = True
    args.format = "xml"
    args.debug = False
    
    mock_logger = Mock()
    
    with patch('query.query_manager.get_fq_cfg_filename', return_value=temp_config_file):
      with patch('query.query_manager.KnowledgeBase') as mock_kb_class:
        mock_kb = Mock()
        mock_kb.knowledge_base_db = "/tmp/test.db"
        mock_kb.knowledge_base_vector = "/tmp/test.faiss"
        mock_kb.sql_connection, mock_kb.sql_cursor = mock_db_connection
        mock_kb.max_query_file_size_mb = 1  # Add the missing attribute
        mock_kb.max_query_length = 10000  # Add max_query_length attribute
        mock_kb.query_top_k = 50  # Add query_top_k attribute
        mock_kb.reference_batch_size = 5  # Add reference_batch_size attribute
        mock_kb.enable_reranking = False  # Add enable_reranking attribute
        mock_kb.query_context_files = []  # Add query_context_files attribute
        mock_kb.io_thread_pool_size = 4  # Add io_thread_pool_size attribute
        mock_kb.start_time = int(time.time())  # Add start_time attribute
        mock_kb_class.return_value = mock_kb
        
        with patch('query.query_manager.connect_to_database'):
          with patch('query.query_manager.close_database'):
            with patch('os.path.exists', return_value=True):
              with patch('os.path.getsize', return_value=1024):
                # Create a mock FAISS index with search method
                mock_index = Mock()
                mock_index.search.return_value = (
                  np.array([[0.9, 0.8, 0.7]]),  # distances
                  np.array([[0, 1, 2]])          # indices
                )
                with patch('query.query_manager.faiss.read_index', return_value=mock_index):
                  with patch('query.query_manager.get_query_embedding', return_value=np.array([[0.1, 0.2, 0.3]])):
                    # Mock process_reference_batch to return proper reference data
                    # Format: (id, sourcedoc, sid, text, distance, metadata)
                    mock_refs = [(0, "doc1.txt", 1, "Sample text 1", 0.9, {"similarity": 0.9}),
                                 (1, "doc2.txt", 2, "Sample text 2", 0.8, {"similarity": 0.8})]
                    with patch('query.query_manager.process_reference_batch', return_value=mock_refs):
                      with patch('query.query_manager.build_reference_string', return_value="context"):
                        result = await process_query_async(args, mock_logger)
                        
                        # Should have combined query text with file content
                        assert result == "context"
  
  @pytest.mark.asyncio
  async def test_process_query_invalid_config(self):
    """Test query processing with invalid knowledgebase."""
    args = Mock()
    args.config_file = "nonexistent_kb"
    
    mock_logger = Mock()
    
    result = await process_query_async(args, mock_logger)
    
    assert "Knowledgebase 'nonexistent_kb' not found" in result or "Configuration file not found" in result
  
  @pytest.mark.asyncio
  async def test_process_query_missing_database(self, temp_config_file):
    """Test query processing with missing database."""
    args = Mock()
    args.config_file = temp_config_file
    args.query_text = "Test query"
    args.query_file = ""
    args.verbose = True
    
    mock_logger = Mock()
    
    with patch('query.query_manager.KnowledgeBase') as mock_kb_class:
      mock_kb = Mock()
      mock_kb.knowledge_base_db = "/nonexistent/test.db"
      mock_kb_class.return_value = mock_kb
      
      result = await process_query_async(args, mock_logger)
      
      assert "does not exist" in result
  
  @pytest.mark.asyncio
  async def test_process_query_missing_vector_file(self, temp_database, temp_config_file, mock_db_connection):
    """Test query processing with missing vector file."""
    args = Mock()
    args.config_file = temp_config_file
    args.query_text = "Test query"
    args.query_file = ""
    args.verbose = True
    
    mock_logger = Mock()
    
    kb = KnowledgeBase(temp_config_file)
    kb.knowledge_base_db = temp_database
    kb.knowledge_base_vector = "/nonexistent/test.faiss"
    kb.sql_connection, kb.sql_cursor = mock_db_connection
    
    with patch('query.query_manager.KnowledgeBase', return_value=kb):
      with patch('query.query_manager.connect_to_database'):
        with patch('query.query_manager.close_database'):
          result = await process_query_async(args, mock_logger)
          
          assert "does not yet exist" in result


class TestQueryEnhancement:
  """Test query enhancement functionality."""
  
  def test_normalize_query_basic(self):
    """Test basic query normalization."""
    from query.query_manager import normalize_query
    
    # Test whitespace normalization
    query = "  what   is  db  config   "
    normalized = normalize_query(query)
    assert normalized == "what is database configuration"
    
    # Test abbreviation expansion
    query = "api ui ml docs"
    normalized = normalize_query(query)
    assert "API" in normalized
    assert "user interface" in normalized
    assert "machine learning" in normalized
    assert "documentation" in normalized
  
  def test_get_synonyms_for_word_basic(self):
    """Test synonym generation for common words."""
    from query.query_manager import get_synonyms_for_word
    
    # Test with common word
    synonyms = get_synonyms_for_word("car", max_synonyms=3)
    assert isinstance(synonyms, list)
    assert len(synonyms) <= 3
    
    # Test with technical term that may not have synonyms
    synonyms = get_synonyms_for_word("xyzabc123", max_synonyms=2)
    assert synonyms == []
  
  def test_apply_spelling_correction(self):
    """Test spelling correction functionality."""
    from query.query_manager import apply_spelling_correction
    from config.config_manager import KnowledgeBase
    
    # Create mock KB with spelling correction enabled
    kb = Mock()
    kb.query_enhancement_spelling = True
    
    # Test common technical misspellings
    query = "databse cofigure querry performace"
    corrected = apply_spelling_correction(query, kb)
    
    assert "database" in corrected
    assert "configure" in corrected
    assert "query" in corrected
    assert "performance" in corrected
    
    # Test case preservation
    query = "Databse QUERRY"
    corrected = apply_spelling_correction(query, kb)
    assert "Database" in corrected
    assert "QUERY" in corrected
  
  def test_apply_spelling_correction_disabled(self):
    """Test spelling correction when disabled."""
    from query.query_manager import apply_spelling_correction
    
    # Mock KB with spelling correction disabled
    kb = Mock()
    kb.query_enhancement_spelling = False
    
    query = "databse cofigure querry"
    corrected = apply_spelling_correction(query, kb)
    
    # Should return original query unchanged
    assert corrected == query
  
  def test_expand_synonyms_basic(self):
    """Test synonym expansion functionality."""
    from query.query_manager import expand_synonyms
    
    # Mock KB with synonym expansion enabled
    kb = Mock()
    kb.query_enhancement_synonyms = True
    kb.max_synonyms_per_word = 2
    kb.synonym_relevance_threshold = 0.6
    
    # Test with expandable words
    query = "computer algorithm"
    expanded = expand_synonyms(query, kb)
    
    # Should contain original words
    assert "computer" in expanded
    assert "algorithm" in expanded
    
    # Should be longer than original (synonyms added)
    assert len(expanded.split()) >= len(query.split())
  
  def test_expand_synonyms_skip_common_words(self):
    """Test that common words are skipped in synonym expansion."""
    from query.query_manager import expand_synonyms
    
    kb = Mock()
    kb.query_enhancement_synonyms = True
    kb.max_synonyms_per_word = 2
    kb.synonym_relevance_threshold = 0.6
    
    # Test with common/stop words
    query = "the and for"
    expanded = expand_synonyms(query, kb)
    
    # Should be unchanged (all are common words)
    assert expanded == query
  
  def test_expand_synonyms_disabled(self):
    """Test synonym expansion when disabled."""
    from query.query_manager import expand_synonyms
    
    kb = Mock()
    kb.query_enhancement_synonyms = False
    
    query = "computer algorithm"
    expanded = expand_synonyms(query, kb)
    
    # Should return original query unchanged
    assert expanded == query
  
  def test_enhance_query_full_pipeline(self):
    """Test complete query enhancement pipeline."""
    from query.query_manager import enhance_query
    
    # Mock KB with all enhancements enabled
    kb = Mock()
    kb.enable_query_enhancement = True
    kb.query_enhancement_synonyms = True
    kb.query_enhancement_spelling = True
    kb.max_synonyms_per_word = 2
    kb.synonym_relevance_threshold = 0.6
    
    # Mock caching functions to avoid actual file I/O
    with patch('query.query_manager.get_cached_enhanced_query', return_value=None):
      with patch('query.query_manager.save_enhanced_query_to_cache'):
        query = "  databse  performace  ml  "
        enhanced = enhance_query(query, kb)
        
        # Should be normalized, corrected, and potentially expanded
        assert "database" in enhanced.lower()
        assert "performance" in enhanced.lower()
        assert "machine learning" in enhanced.lower()
        
        # Should be normalized (no extra spaces)
        assert "  " not in enhanced
  
  def test_enhance_query_disabled(self):
    """Test query enhancement when disabled."""
    from query.query_manager import enhance_query
    
    kb = Mock()
    kb.enable_query_enhancement = False
    
    query = "  databse  performace  "
    enhanced = enhance_query(query, kb)
    
    # Should return original query unchanged
    assert enhanced == query
  
  def test_enhance_query_cached(self):
    """Test query enhancement with cached result."""
    from query.query_manager import enhance_query
    
    kb = Mock()
    kb.enable_query_enhancement = True
    
    cached_result = "cached enhanced query"
    
    with patch('query.query_manager.get_cached_enhanced_query', return_value=cached_result):
      query = "original query"
      enhanced = enhance_query(query, kb)
      
      # Should return cached result
      assert enhanced == cached_result
  
  def test_enhance_query_error_handling(self):
    """Test query enhancement error handling."""
    from query.query_manager import enhance_query
    
    kb = Mock()
    kb.enable_query_enhancement = True
    
    # Mock an exception in the enhancement process
    with patch('query.query_manager.get_cached_enhanced_query', return_value=None):
      with patch('query.query_manager.normalize_query', side_effect=Exception("Test error")):
        query = "test query"
        enhanced = enhance_query(query, kb)
        
        # Should fallback to original query on error
        assert enhanced == query
  
  def test_enhancement_caching_functions(self):
    """Test query enhancement caching functionality."""
    from query.query_manager import (
      get_enhancement_cache_key,
      get_cached_enhanced_query,
      save_enhanced_query_to_cache
    )
    
    # Test cache key generation
    query = "test query"
    key1 = get_enhancement_cache_key(query)
    key2 = get_enhancement_cache_key(query)
    
    assert key1 == key2
    assert "enhancement_" in key1
    
    # Test caching with temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
      with patch('query.query_manager.CACHE_DIR', temp_dir):
        original = "original query"
        enhanced = "enhanced query with synonyms"
        
        # Save to cache
        save_enhanced_query_to_cache(original, enhanced)
        
        # Retrieve from cache
        cached = get_cached_enhanced_query(original)
        assert cached == enhanced
        
        # Test cache miss
        missing = get_cached_enhanced_query("nonexistent query")
        assert missing is None

#fin