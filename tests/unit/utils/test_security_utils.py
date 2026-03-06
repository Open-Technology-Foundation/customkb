#!/usr/bin/env python
"""
Test security utilities - fixed to match current API.
"""

import json
import os
import tempfile
from unittest.mock import Mock, patch

import pytest

from utils.security_utils import (
  mask_sensitive_data,
  safe_json_loads,
  safe_log_error,
  safe_sql_in_query,
  sanitize_config_value,
  sanitize_query_text,
  validate_api_key,
  validate_database_name,
  validate_file_path,
  validate_safe_path,
  validate_table_name,
)


class TestValidateFilePath:
  """Test cases for validate_file_path function."""

  def test_validate_file_path_with_base_dir(self):
    """Test that paths within base directory are accepted."""
    with tempfile.TemporaryDirectory() as tmpdir:
      # Test path within base directory
      test_file = os.path.join(tmpdir, 'test.cfg')
      result = validate_file_path(test_file, ['.cfg'], base_dir=tmpdir, allow_absolute=True)
      assert result == test_file

      # Test relative path with base_dir
      result = validate_file_path('test.cfg', ['.cfg'], base_dir=tmpdir)
      assert result == 'test.cfg'

  def test_validate_file_path_vectordbs_auto_allowed(self):
    """Test that VECTORDBS paths are automatically allowed."""
    vectordb_file = '/var/lib/vectordbs/myproject.cfg'
    result = validate_file_path(vectordb_file, ['.cfg'], allow_absolute=True)
    assert result == vectordb_file

  def test_validate_file_path_absolute_not_allowed(self):
    """Test that absolute paths are rejected without allow_absolute."""
    with pytest.raises(ValueError, match='Absolute paths not allowed'):
      validate_file_path('/etc/passwd', None)

  def test_validate_file_path_outside_base_dir(self):
    """Test that paths outside base_dir are rejected."""
    with tempfile.TemporaryDirectory() as tmpdir, pytest.raises(ValueError, match='File path outside allowed directory'):
      validate_file_path('/etc/passwd', None, base_dir=tmpdir, allow_absolute=True)

  def test_validate_file_path_relative_paths(self):
    """Test that relative paths work correctly."""
    # Simple relative path
    result = validate_file_path('config.cfg', ['.cfg'])
    assert result == 'config.cfg'

    # Relative with subdirectory
    result = validate_file_path('configs/test.cfg', ['.cfg'])
    assert result == 'configs/test.cfg'

  def test_validate_file_path_empty_base_dir(self):
    """Test behavior with None base_dir."""
    # Absolute path without base_dir but with allow_absolute
    with tempfile.TemporaryDirectory() as tmpdir:
      test_file = os.path.join(tmpdir, 'test.cfg')
      result = validate_file_path(test_file, ['.cfg'], allow_absolute=True)
      assert result == test_file

  def test_validate_file_path_extension_validation(self):
    """Test file extension validation."""
    # Valid extension
    result = validate_file_path('test.cfg', ['.cfg'])
    assert result == 'test.cfg'

    # Invalid extension
    with pytest.raises(ValueError, match='Invalid file extension'):
      validate_file_path('test.txt', ['.cfg'])

    # No extension when required
    with pytest.raises(ValueError, match='Invalid file extension'):
      validate_file_path('test', ['.cfg'])

    # Empty string in allowed extensions (allows no extension)
    result = validate_file_path('test', ['', '.cfg'])
    assert result == 'test'

  def test_validate_file_path_traversal_prevention(self):
    """Test path traversal prevention."""
    # Without allow_relative_traversal, .. should be rejected
    with pytest.raises(ValueError, match='path traversal detected'):
      validate_file_path('../etc/passwd', None)

    # With allow_relative_traversal, .. is allowed
    result = validate_file_path('../configs/test.cfg', ['.cfg'], allow_relative_traversal=True)
    assert result == '../configs/test.cfg'

  def test_validate_file_path_dangerous_characters(self):
    """Test rejection of dangerous characters in filename."""
    # The function checks dangerous characters in filename only (basename), not full path
    # So 'test|cat /etc/passwd' won't fail because basename is 'passwd'
    dangerous_paths = [
      'test;cmd.txt',  # semicolon in filename
      'test|pipe.txt',  # pipe in filename
      'test`whoami`.txt',  # backticks in filename
      'test$(whoami).txt',  # command substitution in filename
      'test<script>.txt',  # angle bracket in filename
      'test>output.txt',  # output redirect in filename
    ]

    for path in dangerous_paths:
      with pytest.raises(ValueError, match='dangerous characters'):
        validate_file_path(path, None)

  def test_validate_file_path_null_bytes(self):
    """Test null byte handling."""
    # Null bytes are stripped, leaving valid filename
    result = validate_file_path('test\x00.cfg', ['.cfg'])
    assert result == 'test.cfg'

  def test_validate_file_path_empty_input(self):
    """Test empty input handling."""
    with pytest.raises(ValueError, match='empty'):
      validate_file_path('', None)

    with pytest.raises(ValueError, match='empty'):
      validate_file_path('   ', None)


class TestValidateSafePath:
  """Test cases for validate_safe_path function."""

  def test_validate_safe_path_within_base(self):
    """Test paths within base directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
      test_file = os.path.join(tmpdir, 'test.cfg')
      assert validate_safe_path(test_file, tmpdir) is True

      # Subdirectory
      subdir = os.path.join(tmpdir, 'sub', 'dir')
      assert validate_safe_path(subdir, tmpdir) is True

  def test_validate_safe_path_outside_base(self):
    """Test paths outside base directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
      assert validate_safe_path('/etc/passwd', tmpdir) is False
      assert validate_safe_path('/tmp/other', tmpdir) is False

  def test_validate_safe_path_relative(self):
    """Test relative path handling."""
    # Relative paths are converted to absolute for checking
    assert validate_safe_path('test.cfg', '/var/lib/vectordbs') is False


class TestValidateApiKey:
  """Test cases for validate_api_key function."""

  def test_valid_api_keys(self):
    """Test valid API key formats."""
    assert validate_api_key('sk-1234567890abcdef1234567890abcdef')
    assert validate_api_key('sk-1234567890abcdef1234567890abcdef', prefix='sk-')
    assert validate_api_key('1234567890abcdef1234567890abcdef', prefix=None)

  def test_invalid_api_keys(self):
    """Test invalid API key formats."""
    assert not validate_api_key('')
    assert not validate_api_key('short')
    assert not validate_api_key('sk-short', min_length=20)
    assert not validate_api_key('wrong-prefix-1234567890abcdef1234', prefix='sk-')
    assert not validate_api_key('key with spaces 1234567890')
    assert not validate_api_key('key@with#special$chars%1234')


class TestSanitizeQueryText:
  """Test cases for sanitize_query_text function."""

  def test_basic_sanitization(self):
    """Test basic query sanitization."""
    assert sanitize_query_text('Hello world') == 'Hello world'
    assert sanitize_query_text('  spaces  ') == 'spaces'
    # Newlines are preserved (not converted to spaces)
    assert sanitize_query_text('Line1\nLine2') == 'Line1\nLine2'

  def test_length_limiting(self):
    """Test query length limiting."""
    long_query = 'a' * 20000
    # Function raises error instead of truncating
    with pytest.raises(ValueError, match='Query too long'):
      sanitize_query_text(long_query, max_length=10000)

    # Valid length query passes
    valid_query = 'a' * 9999
    result = sanitize_query_text(valid_query, max_length=10000)
    assert result == valid_query

  def test_control_character_removal(self):
    """Test removal of control characters."""
    assert sanitize_query_text('Hello\x00World') == 'HelloWorld'
    # Tabs are preserved (not converted to spaces)
    assert sanitize_query_text('Tab\tSpace') == 'Tab\tSpace'
    # Newlines preserved, CR is also preserved, but final newline is stripped
    assert sanitize_query_text('CR\rLF\n') == 'CR\rLF'


class TestMaskSensitiveData:
  """Test cases for mask_sensitive_data function."""

  def test_api_key_masking(self):
    """Test API key masking in logs."""
    # Function only masks keys that are 40+ chars for OpenAI
    assert mask_sensitive_data('sk-' + 'a' * 40) == 'sk-***MASKED***'
    # Short keys are not masked
    assert mask_sensitive_data('sk-1234567890abcdef') == 'sk-1234567890abcdef'
    # Anthropic keys need 95+ chars after sk-ant-
    assert mask_sensitive_data('sk-ant-' + 'a' * 95) == 'sk-ant-***MASKED***'

  def test_no_masking_needed(self):
    """Test strings that don't need masking."""
    assert mask_sensitive_data('Regular log message') == 'Regular log message'
    assert mask_sensitive_data('Error code: 404') == 'Error code: 404'


class TestValidateTableName:
  """Test cases for validate_table_name function."""

  def test_valid_table_names(self):
    assert validate_table_name('docs') is True
    assert validate_table_name('my_table') is True
    assert validate_table_name('Table1') is True
    assert validate_table_name('_private') is True

  def test_empty_name(self):
    assert validate_table_name('') is False
    assert validate_table_name(None) is False

  def test_sql_injection_attempts(self):
    assert validate_table_name('docs; DROP TABLE users--') is False
    assert validate_table_name("docs' OR '1'='1") is False
    assert validate_table_name('docs UNION SELECT *') is False

  def test_special_characters_rejected(self):
    assert validate_table_name('my-table') is False
    assert validate_table_name('my.table') is False
    assert validate_table_name('my table') is False
    assert validate_table_name('table!') is False

  def test_starts_with_number_rejected(self):
    assert validate_table_name('1table') is False

  def test_dangerous_names(self):
    assert validate_table_name('sqlite_master') is False
    assert validate_table_name('sqlite_temp_master') is False
    assert validate_table_name('sqlite_sequence') is False
    assert validate_table_name('information_schema') is False
    assert validate_table_name('pg_tables') is False
    assert validate_table_name('sys') is False
    assert validate_table_name('master') is False
    assert validate_table_name('msdb') is False
    assert validate_table_name('tempdb') is False

  def test_length_limit(self):
    assert validate_table_name('a' * 64) is True
    assert validate_table_name('a' * 65) is False

  def test_case_insensitive_dangerous(self):
    assert validate_table_name('SQLITE_MASTER') is False
    assert validate_table_name('Sqlite_Master') is False


class TestSanitizeConfigValue:
  """Test cases for sanitize_config_value function."""

  def test_basic_value(self):
    assert sanitize_config_value('hello') == 'hello'

  def test_length_limit(self):
    with pytest.raises(ValueError, match='too long'):
      sanitize_config_value('x' * 1001)

  def test_custom_length_limit(self):
    with pytest.raises(ValueError, match='too long'):
      sanitize_config_value('x' * 51, max_length=50)

  def test_control_characters_removed(self):
    result = sanitize_config_value('hello\x00world\x07test')
    assert '\x00' not in result
    assert '\x07' not in result
    assert 'helloworld' in result

  def test_whitespace_stripped(self):
    assert sanitize_config_value('  hello  ') == 'hello'

  def test_newlines_and_tabs_preserved(self):
    # \n (\x0a) and \t (\x09) are NOT in the removed range
    result = sanitize_config_value('hello\nworld\ttab')
    assert '\n' in result
    assert '\t' in result

  def test_empty_string(self):
    assert sanitize_config_value('') == ''


class TestSafeSqlInQuery:
  """Test cases for safe_sql_in_query function."""

  def test_empty_list(self):
    cursor = Mock()
    safe_sql_in_query(cursor, 'SELECT * FROM docs WHERE id IN ({placeholders})', [])
    cursor.execute.assert_not_called()

  def test_placeholder_count(self):
    cursor = Mock()
    safe_sql_in_query(cursor, 'SELECT * FROM docs WHERE id IN ({placeholders})', [1, 2, 3])
    call_args = cursor.execute.call_args
    assert '?,?,?' in call_args[0][0]

  def test_single_id(self):
    cursor = Mock()
    safe_sql_in_query(cursor, 'SELECT * FROM docs WHERE id IN ({placeholders})', [42])
    call_args = cursor.execute.call_args
    assert '?' in call_args[0][0]
    assert call_args[0][1] == [42]

  def test_non_integer_rejection(self):
    cursor = Mock()
    with pytest.raises(ValueError, match='integers'):
      safe_sql_in_query(cursor, 'SELECT * FROM docs WHERE id IN ({placeholders})', [1, 'two', 3])

  def test_additional_params(self):
    cursor = Mock()
    safe_sql_in_query(
      cursor,
      'SELECT * FROM docs WHERE id IN ({placeholders}) AND status = ?',
      [1, 2],
      additional_params=('active',),
    )
    call_args = cursor.execute.call_args
    assert call_args[0][1] == [1, 2, 'active']

  def test_cursor_execute_called(self):
    cursor = Mock()
    safe_sql_in_query(cursor, 'DELETE FROM docs WHERE id IN ({placeholders})', [10, 20])
    cursor.execute.assert_called_once()


class TestSafeLogError:
  """Test cases for safe_log_error function."""

  @patch('utils.security_utils.logger')
  def test_basic_logging(self, mock_logger):
    safe_log_error('Something failed')
    mock_logger.error.assert_called_once_with('Something failed')

  @patch('utils.security_utils.logger')
  def test_sensitive_data_masked_in_message(self, mock_logger):
    api_key = 'sk-' + 'a' * 40
    safe_log_error(f'API error with key {api_key}')
    call_args = mock_logger.error.call_args[0][0]
    assert api_key not in call_args
    assert 'MASKED' in call_args

  @patch('utils.security_utils.logger')
  def test_sensitive_data_masked_in_kwargs(self, mock_logger):
    api_key = 'sk-' + 'a' * 40
    safe_log_error('Error occurred', api_key=api_key)
    call_args = mock_logger.error.call_args[0][0]
    assert api_key not in call_args
    assert 'Context' in call_args

  @patch('utils.security_utils.logger')
  def test_no_context_without_kwargs(self, mock_logger):
    safe_log_error('Simple error')
    call_args = mock_logger.error.call_args[0][0]
    assert 'Context' not in call_args


class TestSafeJsonLoads:
  """Test cases for safe_json_loads function."""

  def test_valid_json(self):
    result = safe_json_loads('{"key": "value"}')
    assert result == {'key': 'value'}

  def test_invalid_json(self):
    with pytest.raises(ValueError, match='Invalid JSON'):
      safe_json_loads('{invalid}')

  def test_size_limit(self):
    large_json = json.dumps({'data': 'x' * 20000})
    with pytest.raises(ValueError, match='too large'):
      safe_json_loads(large_json, max_size=100)

  def test_custom_size_limit(self):
    small_json = '{"a": 1}'
    result = safe_json_loads(small_json, max_size=50)
    assert result == {'a': 1}

  def test_nested_structures(self):
    nested = json.dumps({'a': {'b': {'c': [1, 2, 3]}}})
    result = safe_json_loads(nested)
    assert result['a']['b']['c'] == [1, 2, 3]

  def test_returns_dict(self):
    result = safe_json_loads('{"x": 1}')
    assert isinstance(result, dict)

  def test_array_json(self):
    # json.loads can parse arrays too, function returns Any but typed as dict
    result = safe_json_loads('[1, 2, 3]')
    assert result == [1, 2, 3]


class TestValidateDatabaseName:
  """Test cases for validate_database_name function."""

  def test_valid_names(self):
    assert validate_database_name('mydb') == 'mydb'
    assert validate_database_name('my_db') == 'my_db'
    assert validate_database_name('my-db') == 'my-db'
    assert validate_database_name('db.sqlite') == 'db.sqlite'
    assert validate_database_name('DB123') == 'DB123'

  def test_empty_name(self):
    with pytest.raises(ValueError, match='empty'):
      validate_database_name('')

  def test_invalid_characters(self):
    with pytest.raises(ValueError, match='invalid characters'):
      validate_database_name('my db')
    with pytest.raises(ValueError, match='invalid characters'):
      validate_database_name('db;drop')
    with pytest.raises(ValueError, match='invalid characters'):
      validate_database_name('db<script>')

  def test_path_traversal_with_slash(self):
    """Paths with / are caught by invalid characters check first."""
    with pytest.raises(ValueError, match='invalid characters'):
      validate_database_name('../etc/passwd')

  def test_path_traversal_no_slash(self):
    """Double dots without / are caught by path traversal check."""
    with pytest.raises(ValueError, match='path traversal'):
      validate_database_name('some..db')

  def test_absolute_path(self):
    with pytest.raises(ValueError, match='invalid characters'):
      validate_database_name('/etc/passwd')


if __name__ == '__main__':
  pytest.main([__file__, '-v'])

# fin
