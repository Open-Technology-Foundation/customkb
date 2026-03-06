#!/usr/bin/env python
"""
Unit tests for utils/exceptions.py
Tests all custom exception classes, their formatting, and handle_exception utility.
"""

import builtins
import sqlite3
from unittest.mock import Mock

import pytest

from utils.exceptions import (
  APIError,
  APIResponseError,
  AuthenticationError,
  BatchError,
  CacheError,
  ChunkingError,
  ConfigurationError,
  CustomKBError,
  DatabaseConnectionError,
  DatabaseError,
  DatabaseIndexError,
  DiskSpaceError,
  DocumentProcessingError,
  EmbeddingError,
  EmbeddingGenerationError,
  FileSystemError,
  InputValidationError,
  InvalidConfigurationError,
  KBFileNotFoundError,
  KBMemoryError,
  KBPermissionError,
  KnowledgeBaseNotFoundError,
  ModelError,
  ModelNotAvailableError,
  NoResultsError,
  PermanentError,
  ProcessingError,
  QueryError,
  QueryProcessingError,
  RateLimitError,
  ResourceError,
  RetryableError,
  SearchError,
  SecurityValidationError,
  TableNotFoundError,
  TemporaryError,
  TokenLimitExceededError,
  ValidationError,
  handle_exception,
)


class TestCustomKBError:
  """Test base CustomKBError class."""

  def test_basic_message(self):
    err = CustomKBError('something broke')
    assert err.message == 'something broke'
    assert str(err) == 'something broke'
    assert err.details == {}

  def test_with_details(self):
    err = CustomKBError('fail', {'key': 'val', 'num': 42})
    assert err.details == {'key': 'val', 'num': 42}
    assert 'key=val' in str(err)
    assert 'num=42' in str(err)
    assert 'fail' in str(err)

  def test_empty_details_no_parens(self):
    err = CustomKBError('clean')
    assert '(' not in str(err)

  def test_is_exception(self):
    err = CustomKBError('test')
    assert isinstance(err, Exception)

  def test_details_default_to_empty_dict(self):
    err = CustomKBError('test', None)
    assert err.details == {}


class TestConfigurationErrors:
  """Test configuration error classes."""

  def test_configuration_error(self):
    err = ConfigurationError('bad config')
    assert isinstance(err, CustomKBError)
    assert str(err) == 'bad config'

  def test_invalid_configuration_error(self):
    err = InvalidConfigurationError('invalid value')
    assert isinstance(err, ConfigurationError)

  def test_kb_not_found_basic(self):
    err = KnowledgeBaseNotFoundError('mydb')
    assert isinstance(err, ConfigurationError)
    assert 'mydb' in str(err)
    assert 'not found' in str(err)
    assert err.details['kb_name'] == 'mydb'

  def test_kb_not_found_with_available(self):
    err = KnowledgeBaseNotFoundError('mydb', ['alpha', 'beta'])
    assert 'alpha' in str(err)
    assert 'beta' in str(err)
    assert err.details['available'] == ['alpha', 'beta']

  def test_kb_not_found_no_available(self):
    err = KnowledgeBaseNotFoundError('mydb', None)
    assert 'Available' not in str(err)
    assert 'available' not in err.details


class TestDatabaseErrors:
  """Test database error classes."""

  def test_database_error(self):
    err = DatabaseError('db fail')
    assert isinstance(err, CustomKBError)

  def test_connection_error(self):
    err = DatabaseConnectionError('conn fail')
    assert isinstance(err, DatabaseError)

  def test_query_error_basic(self):
    err = QueryError('query failed')
    assert isinstance(err, DatabaseError)
    assert str(err) == 'query failed'
    assert err.details == {}

  def test_query_error_with_query(self):
    long_query = 'SELECT ' + 'x' * 300
    err = QueryError('fail', query=long_query)
    assert len(err.details['query']) <= 200

  def test_query_error_with_params(self):
    long_params = ('a' * 200,)
    err = QueryError('fail', params=long_params)
    assert len(err.details['params']) <= 100

  def test_index_error(self):
    err = DatabaseIndexError('idx fail')
    assert isinstance(err, DatabaseError)

  def test_table_not_found(self):
    err = TableNotFoundError('users')
    assert isinstance(err, DatabaseError)
    assert 'users' in str(err)
    assert err.details['table'] == 'users'


class TestEmbeddingErrors:
  """Test embedding error classes."""

  def test_embedding_error(self):
    err = EmbeddingError('embed fail')
    assert isinstance(err, CustomKBError)

  def test_model_not_available_basic(self):
    err = ModelNotAvailableError('gpt-embed')
    assert 'gpt-embed' in str(err)
    assert err.details['model'] == 'gpt-embed'

  def test_model_not_available_with_reason(self):
    err = ModelNotAvailableError('gpt-embed', reason='no API key')
    assert 'no API key' in str(err)

  def test_embedding_generation_error(self):
    err = EmbeddingGenerationError('gen fail')
    assert isinstance(err, EmbeddingError)

  def test_cache_error(self):
    err = CacheError('cache broken')
    assert isinstance(err, EmbeddingError)


class TestAPIErrors:
  """Test API error classes."""

  def test_api_error(self):
    err = APIError('api fail')
    assert isinstance(err, CustomKBError)

  def test_authentication_error(self):
    err = AuthenticationError('openai')
    assert isinstance(err, APIError)
    assert 'openai' in str(err)
    assert err.details['service'] == 'openai'

  def test_rate_limit_no_retry(self):
    err = RateLimitError('openai')
    assert isinstance(err, APIError)
    assert 'openai' in str(err)
    assert 'retry_after' not in err.details

  def test_rate_limit_with_retry(self):
    err = RateLimitError('openai', retry_after=30)
    assert '30' in str(err)
    assert err.details['retry_after'] == 30

  def test_api_response_error_basic(self):
    err = APIResponseError('openai')
    assert isinstance(err, APIError)
    assert 'openai' in str(err)

  def test_api_response_error_with_status(self):
    err = APIResponseError('openai', status_code=500)
    assert '500' in str(err)
    assert err.details['status_code'] == 500

  def test_api_response_error_truncates_response(self):
    long_response = 'x' * 500
    err = APIResponseError('openai', response=long_response)
    assert len(err.details['response']) <= 200

  def test_model_error(self):
    err = ModelError('gpt-4', 'context too long')
    assert isinstance(err, APIError)
    assert 'gpt-4' in str(err)
    assert 'context too long' in str(err)
    assert err.details['model'] == 'gpt-4'
    assert err.details['reason'] == 'context too long'


class TestProcessingErrors:
  """Test processing error classes."""

  def test_processing_error(self):
    err = ProcessingError('proc fail')
    assert isinstance(err, CustomKBError)

  def test_document_processing_error(self):
    err = DocumentProcessingError('readme.md', 'encoding error')
    assert isinstance(err, ProcessingError)
    assert 'readme.md' in str(err)
    assert 'encoding error' in str(err)

  def test_chunking_error(self):
    err = ChunkingError('chunk fail')
    assert isinstance(err, ProcessingError)

  def test_batch_error_basic(self):
    err = BatchError('batch-1', 'timeout')
    assert isinstance(err, ProcessingError)
    assert 'batch-1' in str(err)
    assert 'timeout' in str(err)
    assert err.details['failed_items'] == 0
    assert err.details['total_items'] == 0

  def test_batch_error_with_counts(self):
    err = BatchError('batch-1', 'partial fail', failed_items=3, total_items=10)
    assert err.details['failed_items'] == 3
    assert err.details['total_items'] == 10
    assert 'success_rate' in err.details

  def test_batch_error_success_rate(self):
    err = BatchError('b', 'fail', failed_items=2, total_items=10)
    assert err.details['success_rate'] == 80.0
    assert '80.0%' in str(err)

  def test_batch_error_zero_total_no_rate(self):
    err = BatchError('b', 'fail', failed_items=0, total_items=0)
    assert 'success_rate' not in err.details

  def test_token_limit_exceeded(self):
    err = TokenLimitExceededError(tokens=5000, limit=4096)
    assert isinstance(err, ProcessingError)
    assert '5000' in str(err)
    assert '4096' in str(err)
    assert err.details['tokens'] == 5000
    assert err.details['limit'] == 4096


class TestQueryErrors:
  """Test query processing error classes."""

  def test_query_processing_error(self):
    err = QueryProcessingError('query fail')
    assert isinstance(err, CustomKBError)

  def test_no_results_error(self):
    err = NoResultsError('what is AI')
    assert isinstance(err, QueryProcessingError)
    assert 'what is AI' in str(err)

  def test_no_results_truncates_long_query(self):
    long_query = 'x' * 200
    err = NoResultsError(long_query)
    assert len(err.details['query']) <= 100

  def test_search_error(self):
    err = SearchError('search fail')
    assert isinstance(err, QueryProcessingError)


class TestFileSystemErrors:
  """Test file system error classes."""

  def test_file_system_error(self):
    err = FileSystemError('fs fail')
    assert isinstance(err, CustomKBError)

  def test_file_not_found_error(self):
    err = KBFileNotFoundError('/path/to/missing.txt')
    assert isinstance(err, FileSystemError)
    assert '/path/to/missing.txt' in str(err)

  def test_permission_error(self):
    err = KBPermissionError('/secret', 'write')
    assert isinstance(err, FileSystemError)
    assert '/secret' in str(err)
    assert 'write' in str(err)


class TestValidationErrors:
  """Test validation error classes."""

  def test_validation_error(self):
    err = ValidationError('invalid')
    assert isinstance(err, CustomKBError)

  def test_input_validation_error(self):
    err = InputValidationError('email', 'bad@', 'invalid format')
    assert isinstance(err, ValidationError)
    assert 'email' in str(err)
    assert 'invalid format' in str(err)
    assert err.details['field'] == 'email'

  def test_input_validation_truncates_value(self):
    long_val = 'x' * 200
    err = InputValidationError('field', long_val, 'too long')
    assert len(err.details['value']) <= 100

  def test_security_validation_error(self):
    err = SecurityValidationError('path traversal')
    assert isinstance(err, ValidationError)
    assert 'path traversal' in str(err)


class TestResourceErrors:
  """Test resource error classes."""

  def test_resource_error(self):
    err = ResourceError('resource fail')
    assert isinstance(err, CustomKBError)

  def test_memory_error(self):
    err = KBMemoryError(required=4096, available=2048)
    assert isinstance(err, ResourceError)
    assert '4096' in str(err)
    assert '2048' in str(err)

  def test_disk_space_error(self):
    err = DiskSpaceError(required=1000, available=50)
    assert isinstance(err, ResourceError)
    assert '1000' in str(err)
    assert '50' in str(err)


class TestRetryableError:
  """Test retryable error classes."""

  def test_retryable_error_defaults(self):
    err = RetryableError('retry me')
    assert isinstance(err, CustomKBError)
    assert err.retry_count == 0
    assert err.max_retries == 3

  def test_can_retry_true(self):
    err = RetryableError('retry', retry_count=1, max_retries=3)
    assert err.can_retry() is True

  def test_can_retry_false(self):
    err = RetryableError('retry', retry_count=3, max_retries=3)
    assert err.can_retry() is False

  def test_can_retry_exceeded(self):
    err = RetryableError('retry', retry_count=5, max_retries=3)
    assert err.can_retry() is False

  def test_temporary_error(self):
    err = TemporaryError('temp fail')
    assert isinstance(err, RetryableError)
    assert err.can_retry() is True

  def test_permanent_error(self):
    err = PermanentError('perm fail')
    assert isinstance(err, CustomKBError)
    assert not isinstance(err, RetryableError)


class TestHandleException:
  """Test handle_exception utility function."""

  def test_sqlite_database_error(self):
    with pytest.raises(DatabaseError, match='Database error'):
      handle_exception(sqlite3.DatabaseError('disk I/O'))

  def test_value_error_to_validation(self):
    with pytest.raises(ValidationError, match='Validation error'):
      handle_exception(ValueError('bad value'))

  def test_key_error_to_configuration(self):
    with pytest.raises(ConfigurationError, match='Missing configuration'):
      handle_exception(KeyError('missing_key'))

  def test_timeout_to_temporary(self):
    with pytest.raises(TemporaryError, match='timed out'):
      handle_exception(TimeoutError('timed out'))

  def test_builtin_file_not_found(self):
    """Bug 1 fix: builtin FileNotFoundError now correctly maps."""
    with pytest.raises(KBFileNotFoundError, match='missing.txt'):
      handle_exception(builtins.FileNotFoundError('missing.txt'))

  def test_builtin_permission_error(self):
    """Bug 1 fix: builtin PermissionError now correctly maps."""
    with pytest.raises(KBPermissionError, match='denied'):
      handle_exception(builtins.PermissionError('denied'))

  def test_builtin_memory_error(self):
    """Bug 1 fix: builtin MemoryError now correctly maps."""
    with pytest.raises(KBMemoryError):
      handle_exception(builtins.MemoryError('oom'))

  def test_builtin_connection_error(self):
    """Bug 1 fix: builtin ConnectionError now correctly maps."""
    with pytest.raises(DatabaseConnectionError, match='Connection failed'):
      handle_exception(builtins.ConnectionError('refused'))

  def test_sqlite_integrity_error(self):
    """Bug 2 fix: IntegrityError now matches before DatabaseError."""
    with pytest.raises(DatabaseError, match='integrity'):
      handle_exception(sqlite3.IntegrityError('UNIQUE constraint'))

  def test_unknown_to_custom_kb_error(self):
    with pytest.raises(CustomKBError, match='Unexpected error'):
      handle_exception(RuntimeError('unknown'))

  def test_no_raise_returns_error(self):
    result = handle_exception(ValueError('test'), raise_custom=False)
    assert isinstance(result, ValidationError)

  def test_with_logger(self):
    mock_logger = Mock()
    handle_exception(ValueError('test'), logger=mock_logger, raise_custom=False)
    mock_logger.error.assert_called_once()

  def test_chains_original_exception(self):
    original = ValueError('original')
    with pytest.raises(ValidationError) as exc_info:
      handle_exception(original)
    assert exc_info.value.__cause__ is original


class TestInheritanceChain:
  """Test that exception hierarchy is correct."""

  def test_all_inherit_from_base(self):
    """Every custom exception inherits from CustomKBError."""
    classes = [
      ConfigurationError,
      DatabaseError,
      EmbeddingError,
      APIError,
      ProcessingError,
      QueryProcessingError,
      FileSystemError,
      ValidationError,
      ResourceError,
      RetryableError,
      PermanentError,
    ]
    for cls in classes:
      assert issubclass(cls, CustomKBError)

  def test_sub_hierarchy(self):
    assert issubclass(KnowledgeBaseNotFoundError, ConfigurationError)
    assert issubclass(DatabaseConnectionError, DatabaseError)
    assert issubclass(QueryError, DatabaseError)
    assert issubclass(TableNotFoundError, DatabaseError)
    assert issubclass(ModelNotAvailableError, EmbeddingError)
    assert issubclass(CacheError, EmbeddingError)
    assert issubclass(AuthenticationError, APIError)
    assert issubclass(RateLimitError, APIError)
    assert issubclass(ModelError, APIError)
    assert issubclass(BatchError, ProcessingError)
    assert issubclass(TokenLimitExceededError, ProcessingError)
    assert issubclass(NoResultsError, QueryProcessingError)
    assert issubclass(SearchError, QueryProcessingError)
    assert issubclass(TemporaryError, RetryableError)


if __name__ == '__main__':
  pytest.main([__file__, '-v'])

# fin
