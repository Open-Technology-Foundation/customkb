#!/usr/bin/env python
"""
Test security utilities - fixed to match current API.
"""

import os
import tempfile

import pytest

from utils.security_utils import (
  mask_sensitive_data,
  sanitize_query_text,
  validate_api_key,
  validate_file_path,
  validate_safe_path,
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
    with pytest.raises(ValueError, match="Absolute paths not allowed"):
      validate_file_path('/etc/passwd', None)

  def test_validate_file_path_outside_base_dir(self):
    """Test that paths outside base_dir are rejected."""
    with tempfile.TemporaryDirectory() as tmpdir, pytest.raises(ValueError, match="File path outside allowed directory"):
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
    with pytest.raises(ValueError, match="Invalid file extension"):
      validate_file_path('test.txt', ['.cfg'])

    # No extension when required
    with pytest.raises(ValueError, match="Invalid file extension"):
      validate_file_path('test', ['.cfg'])

    # Empty string in allowed extensions (allows no extension)
    result = validate_file_path('test', ['', '.cfg'])
    assert result == 'test'

  def test_validate_file_path_traversal_prevention(self):
    """Test path traversal prevention."""
    # Without allow_relative_traversal, .. should be rejected
    with pytest.raises(ValueError, match="path traversal detected"):
      validate_file_path('../etc/passwd', None)

    # With allow_relative_traversal, .. is allowed
    result = validate_file_path('../configs/test.cfg', ['.cfg'], allow_relative_traversal=True)
    assert result == '../configs/test.cfg'

  def test_validate_file_path_dangerous_characters(self):
    """Test rejection of dangerous characters in filename."""
    # The function checks dangerous characters in filename only (basename), not full path
    # So 'test|cat /etc/passwd' won't fail because basename is 'passwd'
    dangerous_paths = [
      'test;cmd.txt',        # semicolon in filename
      'test|pipe.txt',       # pipe in filename
      'test`whoami`.txt',    # backticks in filename
      'test$(whoami).txt',   # command substitution in filename
      'test<script>.txt',    # angle bracket in filename
      'test>output.txt'      # output redirect in filename
    ]

    for path in dangerous_paths:
      with pytest.raises(ValueError, match="dangerous characters"):
        validate_file_path(path, None)

  def test_validate_file_path_null_bytes(self):
    """Test null byte handling."""
    # Null bytes are stripped, leaving valid filename
    result = validate_file_path('test\x00.cfg', ['.cfg'])
    assert result == 'test.cfg'

  def test_validate_file_path_empty_input(self):
    """Test empty input handling."""
    with pytest.raises(ValueError, match="empty"):
      validate_file_path('', None)

    with pytest.raises(ValueError, match="empty"):
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
    with pytest.raises(ValueError, match="Query too long"):
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


if __name__ == '__main__':
  pytest.main([__file__, '-v'])

#fin
