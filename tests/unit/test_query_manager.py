"""
Unit tests for query_manager.py
Tests query processing, semantic search, AI response generation, and context building.
"""

import json
import os
import sqlite3
import time
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import pytest

from config.config_manager import KnowledgeBase
from query.query_manager import (
  build_reference_string,
  fetch_document_by_id,
  generate_ai_response,
  get_cache_key,
  get_cached_query_embedding,
  get_context_range,
  get_query_embedding,
  process_query,
  process_query_async,
  process_reference_batch,
  read_context_file,
  save_query_embedding_to_cache,
)


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

    with patch('query.embedding.QUERY_CACHE_DIR', temp_dir):
      result = get_cached_query_embedding("non-existent query", "test-model")
      assert result is None

  def test_get_cached_query_embedding_hit(self, temp_data_manager):
    """Test cache hit for existing query embedding."""
    temp_dir = temp_data_manager.create_temp_dir()
    test_embedding = [0.1, 0.2, 0.3]

    # Create cached file in subdirectory (first 2 chars of cache_key)
    cache_key = "test_model_hash123"
    cache_subdir = os.path.join(temp_dir, cache_key[:2])
    os.makedirs(cache_subdir, exist_ok=True)
    cache_file = os.path.join(cache_subdir, f"{cache_key}.json")

    # Write cache in the new structured format
    cache_data = {
      'model': 'test-model',
      'query_hash': cache_key,
      'embedding': test_embedding,
      'query_preview': 'test query',
      'timestamp': 999
    }
    with open(cache_file, 'w') as f:
      json.dump(cache_data, f)

    with patch('query.embedding.QUERY_CACHE_DIR', temp_dir), \
         patch('query.embedding.get_cache_key', return_value=cache_key), \
         patch('time.time', return_value=1000), \
         patch('os.path.getmtime', return_value=999):  # Recent file
      result = get_cached_query_embedding("test query", "test-model")
      assert result == test_embedding

  def test_get_cached_query_embedding_expired(self, temp_data_manager):
    """Test cache miss for expired query embedding."""
    temp_dir = temp_data_manager.create_temp_dir()
    test_embedding = [0.1, 0.2, 0.3]

    # Create cached file in subdirectory (first 2 chars of cache_key)
    cache_key = "test_model_hash123"
    cache_subdir = os.path.join(temp_dir, cache_key[:2])
    os.makedirs(cache_subdir, exist_ok=True)
    cache_file = os.path.join(cache_subdir, f"{cache_key}.json")

    # Write cache in the new structured format
    cache_data = {
      'model': 'test-model',
      'query_hash': cache_key,
      'embedding': test_embedding,
      'query_preview': 'test query',
      'timestamp': 1000
    }
    with open(cache_file, 'w') as f:
      json.dump(cache_data, f)

    with patch('query.embedding.QUERY_CACHE_DIR', temp_dir), \
         patch('query.embedding.get_cache_key', return_value=cache_key), \
         patch('time.time', return_value=1000000), \
         patch('os.path.getmtime', return_value=1000):  # Old file
      result = get_cached_query_embedding("test query", "test-model")
      assert result is None
      assert not os.path.exists(cache_file)  # Should be deleted

  def test_save_query_embedding_to_cache(self, temp_data_manager):
    """Test saving query embedding to cache."""
    temp_dir = temp_data_manager.create_temp_dir()
    test_embedding = [0.1, 0.2, 0.3]

    with patch('query.embedding.QUERY_CACHE_DIR', temp_dir), \
         patch('query.embedding.get_cache_key', return_value="test_key"):
      save_query_embedding_to_cache("test query", "test-model", test_embedding)

      # Check that file was created in subdirectory
      cache_key = "test_key"
      cache_subdir = os.path.join(temp_dir, cache_key[:2])
      cache_file = os.path.join(cache_subdir, f"{cache_key}.json")
      assert os.path.exists(cache_file)

      with open(cache_file) as f:
        saved_data = json.load(f)
        # New format includes metadata
        assert saved_data['embedding'] == test_embedding
        assert saved_data['model'] == 'test-model'
        assert saved_data['query_hash'] == cache_key
        assert 'timestamp' in saved_data


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

    # Patch at the actual implementation module
    with patch('query.embedding.get_cached_query_embedding', return_value=test_embedding), \
         patch('query.embedding.clean_text', return_value="cleaned query"), \
         patch('query.embedding.enhance_query', return_value="cleaned query"):
      result = await get_query_embedding("test query", "test-model")

      assert isinstance(result, np.ndarray)
      assert len(result) == 3
      np.testing.assert_array_almost_equal(result, test_embedding)

  @pytest.mark.asyncio
  async def test_get_query_embedding_api_call(self):
    """Test getting query embedding via API call when cache misses."""
    test_embedding = np.array([0.1, 0.2, 0.3], dtype=np.float32)

    # Patch at the actual implementation module
    with patch('query.embedding.get_cached_query_embedding', return_value=None), \
         patch('query.embedding.save_query_embedding_to_cache'), \
         patch('query.embedding.clean_text', return_value="cleaned query"), \
         patch('query.embedding.enhance_query', return_value="cleaned query"), \
         patch('query.embedding.generate_query_embedding', return_value=test_embedding):
      result = await get_query_embedding("test query", "test-model")

      assert isinstance(result, np.ndarray)
      assert len(result) == 3


class TestContextFileReading:
  """Test context file reading functionality."""

  def test_read_context_file_success(self, temp_data_manager):
    """Test successful context file reading."""
    content = "This is context content with <special> characters & symbols."
    context_file = temp_data_manager.create_temp_text_file(content, "context.txt")

    file_content, file_name = read_context_file(context_file)

    assert "This is context content" in file_content
    assert "<special>" in file_content  # Raw content, not escaped
    assert "&" in file_content  # Raw content, not escaped
    assert file_name == "context.txt"

  def test_read_context_file_nonexistent(self):
    """Test reading non-existent context file raises error."""
    from utils.exceptions import ProcessingError
    with pytest.raises(ProcessingError):
      read_context_file("/nonexistent/file.txt")

  def test_read_context_file_permission_error(self, temp_data_manager):
    """Test reading context file with permission error raises exception."""
    from utils.exceptions import ProcessingError
    context_file = temp_data_manager.create_temp_text_file("content", "restricted.txt")

    with patch('query.processing.read_text_file', side_effect=PermissionError("Access denied")), \
         pytest.raises(ProcessingError):
      read_context_file(context_file)


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
    from utils.exceptions import DatabaseError

    kb = KnowledgeBase(temp_config_file)
    kb.sql_cursor = Mock()
    kb.sql_cursor.execute.side_effect = sqlite3.Error("Database error")

    # Implementation raises DatabaseError on database errors
    with pytest.raises(DatabaseError, match="Failed to fetch document"):
      fetch_document_by_id(kb, 1)


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
    # Each result should have: [rid, rsrc, rsid, originaltext, distance, metadata, primary_category, categories]
    for item in result:
      assert len(item) == 8

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
  async def test_process_reference_batch_missing_documents(self, temp_config_file, temp_database):
    """Test handling of missing documents in batch processing."""
    kb = KnowledgeBase(temp_config_file)
    kb.query_context_scope = 2
    kb.knowledge_base_db = temp_database

    import sqlite3
    kb.sql_connection = sqlite3.connect(temp_database)
    kb.sql_cursor = kb.sql_connection.cursor()

    batch = [(999, 0.8)]  # Non-existent document

    with patch('query.search.fetch_document_by_id', return_value=None):
      result = await process_reference_batch(kb, batch)

      assert result == []  # Should return empty list

    kb.sql_connection.close()


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

    # Reference structure: [rid, rsrc, rsid, originaltext, distance, metadata, primary_category, categories]
    reference = [
      [1, "test.txt", 0, "Content with metadata", 0.8, json.dumps(metadata), None, None]
    ]

    result = build_reference_string(kb, reference, debug=True)

    assert '<meta name="heading">Test Heading</meta>' in result
    assert '<meta name="section_type">code_block</meta>' in result
    # Similarity formula: 1.0 / (1.0 + distance) = 1.0 / 1.8 = 0.5556
    assert '<meta name="similarity">0.5556</meta>' in result

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

    # Test reasoning models (o1 series)
    assert _is_reasoning_model("o1-preview") is True
    assert _is_reasoning_model("o1-mini") is True

    # Test non-reasoning models
    assert _is_reasoning_model("gpt-5") is False
    assert _is_reasoning_model("gpt-4o") is False
    assert _is_reasoning_model("gpt-4o-mini") is False
    assert _is_reasoning_model("claude-3-sonnet") is False
    assert _is_reasoning_model("o1") is False  # Without suffix

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

    # Check system -> developer mapping (content passes through as string)
    assert formatted[0]["role"] == "developer"
    assert formatted[0]["content"] == "You are helpful"

    # Check user messages
    assert formatted[1]["role"] == "user"
    assert formatted[1]["content"] == "Hello"

    # Check assistant messages
    assert formatted[2]["role"] == "assistant"
    assert formatted[2]["content"] == "Hi there!"

    # Check second user message
    assert formatted[3]["role"] == "user"
    assert formatted[3]["content"] == "How are you?"

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
    """Test AI response generation with OpenAI model via LiteLLM."""
    kb = KnowledgeBase(temp_config_file)
    kb.query_model = "gpt-4o"
    kb.query_role = "You are a helpful assistant."
    kb.query_temperature = 0.7
    kb.query_max_tokens = 1000

    # Mock LiteLLM response (OpenAI-compatible format)
    mock_message = Mock()
    mock_message.content = "Test AI response"
    mock_choice = Mock()
    mock_choice.message = mock_message
    mock_response = Mock()
    mock_response.choices = [mock_choice]

    mock_model_info = {'model': 'gpt-4o', 'provider': 'openai'}
    with patch('query.llm.get_canonical_model', return_value=mock_model_info), \
         patch('query.llm.litellm.acompletion', new_callable=AsyncMock, return_value=mock_response):
      result = await generate_ai_response(kb, "Reference context", "Test query")
      assert result == "Test AI response"

  @pytest.mark.asyncio
  async def test_generate_ai_response_claude(self, temp_config_file):
    """Test AI response generation with Claude model via LiteLLM."""
    kb = KnowledgeBase(temp_config_file)
    kb.query_model = "claude-sonnet"
    kb.query_role = "You are a helpful assistant."
    kb.query_temperature = 0.7
    kb.query_max_tokens = 1000

    # Mock LiteLLM response
    mock_message = Mock()
    mock_message.content = "Test Claude response"
    mock_choice = Mock()
    mock_choice.message = mock_message
    mock_response = Mock()
    mock_response.choices = [mock_choice]

    mock_model_info = {'model': 'claude-sonnet-4-0', 'provider': 'anthropic'}
    with patch('query.llm.get_canonical_model', return_value=mock_model_info), \
         patch('query.llm.litellm.acompletion', new_callable=AsyncMock, return_value=mock_response):
      result = await generate_ai_response(kb, "Reference context", "Test query")
      assert result == "Test Claude response"

  @pytest.mark.asyncio
  async def test_generate_ai_response_o1_model(self, temp_config_file):
    """Test AI response generation with O1 reasoning model via LiteLLM."""
    kb = KnowledgeBase(temp_config_file)
    kb.query_model = "o1-preview"
    kb.query_role = "You are a helpful assistant."
    kb.query_max_tokens = 1000

    # Mock LiteLLM response
    mock_message = Mock()
    mock_message.content = "Test O1 response"
    mock_choice = Mock()
    mock_choice.message = mock_message
    mock_response = Mock()
    mock_response.choices = [mock_choice]

    mock_model_info = {'model': 'o1-preview', 'provider': 'openai'}
    with patch('query.llm.get_canonical_model', return_value=mock_model_info), \
         patch('query.llm.litellm.acompletion', new_callable=AsyncMock, return_value=mock_response):
      result = await generate_ai_response(kb, "Reference context", "Test query")
      assert result == "Test O1 response"

  @pytest.mark.asyncio
  async def test_generate_ai_response_error_handling(self, temp_config_file):
    """Test error handling in AI response generation via LiteLLM."""
    import litellm

    from utils.exceptions import ModelError

    kb = KnowledgeBase(temp_config_file)
    kb.query_model = "gpt-4o"
    kb.query_max_tokens = 1000
    kb.query_temperature = 0.7

    mock_model_info = {'model': 'gpt-4o', 'provider': 'openai'}
    with patch('query.llm.get_canonical_model', return_value=mock_model_info), \
         patch('query.llm.litellm.acompletion', new_callable=AsyncMock,
               side_effect=litellm.APIError(message="API Error", status_code=500, model="gpt-4o", llm_provider="openai")), \
         pytest.raises((ModelError, Exception)):
      await generate_ai_response(kb, "Reference context", "Test query")

  @pytest.mark.asyncio
  async def test_generate_ai_response_datetime_replacement(self, temp_config_file):
    """Test response generation with custom query role via LiteLLM."""
    kb = KnowledgeBase(temp_config_file)
    kb.query_model = "gpt-4o"
    kb.query_role = "Current time: {{datetime}}"
    kb.query_max_tokens = 1000
    kb.query_temperature = 0.7

    # Mock LiteLLM response
    mock_message = Mock()
    mock_message.content = "Response"
    mock_choice = Mock()
    mock_choice.message = mock_message
    mock_response = Mock()
    mock_response.choices = [mock_choice]

    mock_acompletion = AsyncMock(return_value=mock_response)
    mock_model_info = {'model': 'gpt-4o', 'provider': 'openai'}
    with patch('query.llm.get_canonical_model', return_value=mock_model_info), \
         patch('query.llm.litellm.acompletion', mock_acompletion):
      await generate_ai_response(kb, "Context", "Query")

      # Verify LiteLLM was called
      assert mock_acompletion.called
      # Verify messages were passed
      call_args = mock_acompletion.call_args
      messages = call_args[1]['messages']
      assert any(msg['role'] == 'system' for msg in messages)

  @pytest.mark.asyncio
  async def test_generate_ai_response_grok(self, temp_config_file):
    """Test AI response generation with Grok/xAI model via LiteLLM."""
    kb = KnowledgeBase(temp_config_file)
    kb.query_model = "grok-4"
    kb.query_role = "You are Grok, a helpful AI assistant."
    kb.query_temperature = 0.7
    kb.query_max_tokens = 1000

    # Mock LiteLLM response
    mock_message = Mock()
    mock_message.content = "Test Grok response"
    mock_choice = Mock()
    mock_choice.message = mock_message
    mock_response = Mock()
    mock_response.choices = [mock_choice]

    mock_model_info = {'model': 'grok-4', 'provider': 'xai'}
    with patch('query.llm.get_canonical_model', return_value=mock_model_info), \
         patch('query.llm.litellm.acompletion', new_callable=AsyncMock, return_value=mock_response):
      result = await generate_ai_response(kb, "Reference context", "Test query")
      assert result == "Test Grok response"

  @pytest.mark.asyncio
  async def test_generate_ai_response_grok_no_api_key(self, temp_config_file):
    """Test error when xAI API key is missing (LiteLLM auth error)."""
    import litellm

    from utils.exceptions import APIError

    kb = KnowledgeBase(temp_config_file)
    kb.query_model = "grok-4"
    kb.query_max_tokens = 1000
    kb.query_temperature = 0.7

    mock_model_info = {'model': 'grok-4', 'provider': 'xai'}
    with patch('query.llm.get_canonical_model', return_value=mock_model_info), \
         patch('query.llm.litellm.acompletion', new_callable=AsyncMock,
               side_effect=litellm.AuthenticationError(
                 message="No API key", model="xai/grok-4", llm_provider="xai")), \
         pytest.raises(APIError, match="Authentication failed"):
      await generate_ai_response(kb, "Reference context", "Test query")


class TestProcessQuery:
  """Test the main process_query functions."""

  @pytest.mark.skip(reason="Complex async test requires refactoring - mocking deeply nested async functions")
  @pytest.mark.asyncio
  async def test_process_query_async_success(self, temp_database, temp_config_file, mock_faiss_index, mock_db_connection):
    """Test successful async query processing."""
    pass

  @pytest.mark.asyncio
  async def test_process_query_async_context_only(self, temp_database, temp_config_file, mock_faiss_index, mock_db_connection):
    """Test async query processing with context-only flag."""
    args = Mock()
    args.config_file = temp_config_file
    args.query_text = "What is machine learning?"
    args.query_file = ""
    args.context_only = True
    args.verbose = True
    args.context_files = []  # Must be explicitly set to avoid Mock truthiness
    args.format = "xml"
    args.debug = False

    mock_logger = Mock()

    kb = KnowledgeBase(temp_config_file)
    kb.knowledge_base_db = temp_database
    kb.knowledge_base_vector = temp_database.replace('.db', '.faiss')
    kb.sql_connection, kb.sql_cursor = mock_db_connection

    with patch('query.processing.KnowledgeBase', return_value=kb), \
         patch('query.processing.connect_to_database'), \
         patch('query.processing.close_database'), \
         patch('os.path.exists', return_value=True), \
         patch('os.path.getsize', return_value=1024):
      # All these functions are async, need AsyncMock
      mock_embedding = AsyncMock(return_value=np.array([[0.1, 0.2, 0.3]]))
      mock_search = AsyncMock(return_value=[(1, 0.8)])
      mock_batch = AsyncMock(return_value=[[1, "test.txt", 0, "Test content", 0.8, "{}"]])

      with patch('query.processing.get_query_embedding', mock_embedding), \
           patch('query.processing.perform_hybrid_search', mock_search), \
           patch('query.processing.process_reference_batch', mock_batch), \
           patch('query.processing.build_reference_string') as mock_build:
        mock_build.return_value = "Reference context"
        result = await process_query_async(args, mock_logger)
        assert result == "Reference context"

  def test_process_query_sync_wrapper(self, temp_config_file):
    """Test sync wrapper for process_query."""
    args = Mock()
    args.config_file = temp_config_file

    mock_logger = Mock()

    with patch('query.processing.asyncio.run') as mock_run:
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
    args.context_files = []  # Must be explicitly set to avoid Mock truthiness

    mock_logger = Mock()

    with patch('query.processing.get_fq_cfg_filename', return_value=temp_config_file), \
         patch('query.processing.KnowledgeBase') as mock_kb_class:
        mock_kb = Mock()
        mock_kb.knowledge_base_db = "/tmp/test.db"
        mock_kb.knowledge_base_vector = "/tmp/test.faiss"
        mock_kb.sql_connection, mock_kb.sql_cursor = mock_db_connection
        mock_kb.max_query_file_size_mb = 1
        mock_kb.max_query_length = 10000
        mock_kb.query_top_k = 50
        mock_kb.reference_batch_size = 5
        mock_kb.enable_reranking = False
        mock_kb.query_context_files = []
        mock_kb.io_thread_pool_size = 4
        mock_kb.start_time = int(time.time())
        mock_kb.vector_model = "text-embedding-3-small"  # Needed for get_query_embedding
        mock_kb.query_context_scope = 4  # Needed for query processing
        mock_kb_class.return_value = mock_kb

        with patch('query.processing.connect_to_database'), \
             patch('query.processing.close_database'), \
             patch('os.path.exists', return_value=True), \
             patch('os.path.getsize', return_value=1024):
          # Mock the async functions in the processing pipeline
          mock_embedding = AsyncMock(return_value=np.array([[0.1, 0.2, 0.3]]))
          mock_search = AsyncMock(return_value=[(0, 0.9), (1, 0.8), (2, 0.7)])
          mock_refs = [(0, "doc1.txt", 1, "Sample text 1", 0.9, {"similarity": 0.9}),
                       (1, "doc2.txt", 2, "Sample text 2", 0.8, {"similarity": 0.8})]
          mock_batch = AsyncMock(return_value=mock_refs)

          with patch('query.processing.get_query_embedding', mock_embedding), \
               patch('query.processing.perform_hybrid_search', mock_search), \
               patch('query.processing.process_reference_batch', mock_batch), \
               patch('query.processing.build_reference_string', return_value="context"):
            result = await process_query_async(args, mock_logger)

            assert result == "context"

  @pytest.mark.asyncio
  async def test_process_query_invalid_config(self):
    """Test query processing with invalid knowledgebase raises appropriate error."""
    from utils.exceptions import QueryError

    args = Mock()
    args.config_file = "nonexistent_kb"
    args.query_text = "Test query"
    args.query_file = ""
    args.verbose = False
    args.context_files = []  # Must be explicitly set to avoid Mock truthiness

    mock_logger = Mock()

    # Should raise QueryError for invalid config
    with pytest.raises(QueryError) as exc_info:
      await process_query_async(args, mock_logger)

    # Check error message contains relevant info
    error_msg = str(exc_info.value).lower()
    assert "not found" in error_msg or "configuration" in error_msg or "error" in error_msg

  @pytest.mark.asyncio
  async def test_process_query_missing_database(self, temp_config_file):
    """Test query processing with missing database raises appropriate error."""
    from utils.exceptions import QueryError

    args = Mock()
    args.config_file = temp_config_file
    args.query_text = "Test query"
    args.query_file = ""
    args.verbose = True
    args.context_files = []  # Must be explicitly set to avoid Mock truthiness

    mock_logger = Mock()

    with patch('query.processing.KnowledgeBase') as mock_kb_class:
      mock_kb = Mock()
      mock_kb.knowledge_base_db = "/nonexistent/test.db"
      mock_kb_class.return_value = mock_kb

      # Should raise QueryError for missing database
      with pytest.raises(QueryError) as exc_info:
        await process_query_async(args, mock_logger)

      # Check error message contains relevant info
      error_msg = str(exc_info.value).lower()
      assert "database" in error_msg or "connect" in error_msg or "error" in error_msg

  @pytest.mark.asyncio
  async def test_process_query_missing_vector_file(self, temp_database, temp_config_file, mock_db_connection):
    """Test query processing with missing vector file raises appropriate error."""
    from utils.exceptions import QueryError

    args = Mock()
    args.config_file = temp_config_file
    args.query_text = "Test query"
    args.query_file = ""
    args.verbose = True
    args.context_files = []  # Must be explicitly set to avoid Mock truthiness

    mock_logger = Mock()

    kb = KnowledgeBase(temp_config_file)
    kb.knowledge_base_db = temp_database
    kb.knowledge_base_vector = "/nonexistent/test.faiss"
    kb.sql_connection, kb.sql_cursor = mock_db_connection

    # Mock get_query_embedding to prevent API call - the vector file check happens in perform_hybrid_search
    mock_embedding = AsyncMock(return_value=np.array([[0.1, 0.2, 0.3]]))

    with patch('query.processing.KnowledgeBase', return_value=kb), \
         patch('query.processing.connect_to_database'), \
         patch('query.processing.close_database'), \
         patch('query.processing.get_query_embedding', mock_embedding):
      # Should raise QueryError for missing vector file
      with pytest.raises(QueryError) as exc_info:
        await process_query_async(args, mock_logger)

      # The error message from search.py is "Vector index not found"
      error_msg = str(exc_info.value).lower()
      assert "vector" in error_msg or "not found" in error_msg or "error" in error_msg


class TestQueryEnhancement:
  """Test query enhancement functionality."""

  def test_normalize_query_basic(self):
    """Test basic query normalization."""
    from query.query_manager import normalize_query

    # Test whitespace normalization
    query = "  what   is  db  config   "
    normalized = normalize_query(query)
    # normalize_query does text cleaning: lowercasing and whitespace normalization
    assert normalized == "what is db config"

    # Test punctuation normalization
    query = "What???  Is...  This!!"
    normalized = normalize_query(query)
    # Reduces multiple punctuation to single
    assert "???" not in normalized
    assert "..." not in normalized
    assert "!!" not in normalized

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

    # Create mock KB with spelling correction enabled
    kb = Mock()
    kb.enable_spelling_correction = True

    # Test that function runs without error
    query = "databse cofigure querry performace"
    corrected = apply_spelling_correction(query, kb)

    # Function tries to use textblob or simple corrections
    # May or may not correct all words depending on textblob availability
    assert isinstance(corrected, str)
    assert len(corrected) > 0

  def test_apply_spelling_correction_disabled(self):
    """Test spelling correction when disabled."""
    from query.query_manager import apply_spelling_correction

    # Mock KB with spelling correction disabled
    kb = Mock()
    kb.enable_spelling_correction = False

    query = "databse cofigure querry"
    corrected = apply_spelling_correction(query, kb)

    # Should return original query unchanged
    assert corrected == query

  def test_expand_synonyms_basic(self):
    """Test synonym expansion functionality."""
    from query.query_manager import expand_synonyms

    # Mock KB with synonym expansion enabled
    kb = Mock()
    kb.enable_synonym_expansion = True
    kb.max_synonyms_per_word = 2
    kb.synonym_min_word_length = 4

    # Test with expandable words
    query = "computer algorithm"
    expanded = expand_synonyms(query, kb)

    # Should contain original words
    assert "computer" in expanded
    assert "algorithm" in expanded

    # Function may add synonyms or OR expressions
    assert isinstance(expanded, str)

  def test_expand_synonyms_skip_common_words(self):
    """Test that common words are skipped in synonym expansion."""
    from query.query_manager import expand_synonyms

    kb = Mock()
    kb.enable_synonym_expansion = True
    kb.max_synonyms_per_word = 2
    kb.synonym_min_word_length = 4

    # Test with common/stop words (too short and in stop words)
    query = "the and for"
    expanded = expand_synonyms(query, kb)

    # Should be unchanged (all are common/short words)
    assert expanded == query

  def test_expand_synonyms_disabled(self):
    """Test synonym expansion when disabled."""
    from query.query_manager import expand_synonyms

    kb = Mock()
    kb.enable_synonym_expansion = False

    query = "computer algorithm"
    expanded = expand_synonyms(query, kb)

    # Should return original query unchanged
    assert expanded == query

  def test_enhance_query_full_pipeline(self):
    """Test complete query enhancement pipeline."""
    from query.query_manager import enhance_query

    # Mock KB with all enhancements enabled
    kb = Mock()
    kb.enable_spelling_correction = True
    kb.enable_synonym_expansion = True
    kb.max_synonyms_per_word = 2
    kb.synonym_min_word_length = 4

    # Mock caching functions to avoid actual file I/O
    with patch('query.enhancement.get_cached_enhanced_query', return_value=None), \
         patch('query.enhancement.save_enhanced_query_to_cache'):
      query = "  databse  performace  ml  "
      enhanced = enhance_query(query, kb)

      # Should be normalized (no extra spaces)
      assert "  " not in enhanced
      # Original words should be present (possibly corrected)
      assert len(enhanced) > 0

  def test_enhance_query_disabled(self):
    """Test query enhancement when disabled."""
    from query.query_manager import enhance_query

    kb = Mock()
    kb.enable_spelling_correction = False
    kb.enable_synonym_expansion = False

    # Mock caching to avoid file I/O
    with patch('query.enhancement.get_cached_enhanced_query', return_value=None), \
         patch('query.enhancement.save_enhanced_query_to_cache'):
      query = "test query"
      enhanced = enhance_query(query, kb)

      # Should at least normalize but not spell correct or expand synonyms
      assert isinstance(enhanced, str)

  def test_enhance_query_cached(self):
    """Test query enhancement with cached result."""
    from query.query_manager import enhance_query

    kb = Mock()

    cached_result = "cached enhanced query"

    with patch('query.enhancement.get_cached_enhanced_query', return_value=cached_result):
      query = "original query"
      enhanced = enhance_query(query, kb)

      # Should return cached result
      assert enhanced == cached_result

  def test_enhance_query_error_handling(self):
    """Test query enhancement error handling."""
    from query.query_manager import enhance_query

    kb = Mock()
    kb.enable_spelling_correction = True
    kb.enable_synonym_expansion = True

    # Mock an exception in the enhancement process
    with patch('query.enhancement.get_cached_enhanced_query', return_value=None), \
         patch('query.enhancement.normalize_query', side_effect=ValueError("Test error")):
      query = "test query"
      enhanced = enhance_query(query, kb)

      # Should fallback to original query on error
      assert enhanced == query

  def test_enhancement_caching_functions(self):
    """Test query enhancement caching functionality."""
    from query.query_manager import get_cached_enhanced_query, get_enhancement_cache_key

    # Test cache key generation
    query = "test query"
    key1 = get_enhancement_cache_key(query)
    key2 = get_enhancement_cache_key(query)

    # Keys should be consistent for the same query
    assert key1 == key2
    # Key is a sha256 hash (64 hex chars)
    assert len(key1) == 64

    # Test different queries produce different keys
    key3 = get_enhancement_cache_key("different query")
    assert key1 != key3

    # Test cache miss for non-cached query
    # (Using the real cache dir - query won't be there)
    missing = get_cached_enhanced_query("definitely_nonexistent_query_xyz123")
    assert missing is None

#fin
