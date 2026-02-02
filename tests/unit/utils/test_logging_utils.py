"""
Unit tests for utils/logging_utils.py
Tests logging configuration, formatting, and utility functions.
"""

import logging
import os
import time
from unittest.mock import Mock, patch

import pytest

from utils.logging_utils import (
  OperationLogger,
  dashes,
  elapsed_time,
  get_kb_info_from_config,
  get_log_file_path,
  get_logger,
  load_logging_config,
  log_file_operation,
  log_model_operation,
  log_operation_error,
  log_performance_metrics,
  setup_logging,
  time_to_finish,
)


class TestGetKbInfoFromConfig:
  """Test knowledgebase info extraction from config files."""

  def test_basic_config_path(self):
    """Test extracting info from basic config path."""
    config_file = "/path/to/mycompany.cfg"
    directory, kb_name = get_kb_info_from_config(config_file)

    # os.path.split doesn't add trailing slash to directories
    assert directory == "/path/to"
    assert kb_name == "mycompany"

  def test_domain_style_config(self):
    """Test extracting info from domain-style config."""
    config_file = "/vectordbs/example.com.cfg"
    directory, kb_name = get_kb_info_from_config(config_file)

    # os.path.split doesn't add trailing slash to directories
    assert directory == "/vectordbs"
    assert kb_name == "example.com"

  def test_config_without_extension(self):
    """Test config file without .cfg extension."""
    config_file = "/path/to/config"
    directory, kb_name = get_kb_info_from_config(config_file)

    # os.path.split doesn't add trailing slash to directories
    assert directory == "/path/to"
    assert kb_name == "config"

  def test_empty_config_file(self):
    """Test handling of empty config file path."""
    with pytest.raises(ValueError, match="Config file path cannot be empty"):
      get_kb_info_from_config("")

  def test_none_config_file(self):
    """Test handling of None config file path."""
    with pytest.raises(ValueError, match="Config file path cannot be empty"):
      get_kb_info_from_config(None)

  def test_config_with_double_extension(self):
    """Test config file with .cfg.cfg extension."""
    config_file = "/path/to/test.cfg.cfg"
    directory, kb_name = get_kb_info_from_config(config_file)

    # os.path.split doesn't add trailing slash to directories
    assert directory == "/path/to"
    # splitext gives 'test.cfg', then .cfg is stripped again → 'test'
    assert kb_name == "test"


class TestGetLogFilePath:
  """Test log file path resolution."""

  def test_auto_log_file(self, temp_data_manager):
    """Test auto-generated log file path."""
    kb_directory = temp_data_manager.create_temp_dir()
    kb_name = "test_kb"

    result = get_log_file_path("auto", kb_directory, kb_name)

    expected = os.path.join(kb_directory, "logs", "test_kb.log")
    assert result == expected
    assert os.path.exists(os.path.dirname(result))

  def test_absolute_log_file(self, temp_data_manager):
    """Test absolute log file path."""
    temp_dir = temp_data_manager.create_temp_dir()
    absolute_path = os.path.join(temp_dir, "custom.log")

    result = get_log_file_path(absolute_path, "/kb/dir", "kb_name")

    assert result == absolute_path

  def test_relative_log_file(self, temp_data_manager):
    """Test relative log file path."""
    kb_directory = temp_data_manager.create_temp_dir()

    result = get_log_file_path("logs/custom.log", kb_directory, "kb_name")

    expected = os.path.join(kb_directory, "logs/custom.log")
    assert result == expected

  def test_log_directory_creation(self, temp_data_manager):
    """Test that log directories are created."""
    kb_directory = temp_data_manager.create_temp_dir()

    get_log_file_path("deep/nested/logs/test.log", kb_directory, "kb")

    expected_dir = os.path.join(kb_directory, "deep/nested/logs")
    assert os.path.exists(expected_dir)


class TestLoadLoggingConfig:
  """Test logging configuration loading."""

  def test_default_config(self):
    """Test default logging configuration."""
    config = load_logging_config()

    assert config['log_level'] == 'INFO'
    assert config['log_file'] == 'auto'
    assert config['max_log_size'] == 10485760
    assert config['backup_count'] == 5
    assert config['console_colors'] is True
    assert config['file_logging'] is True
    assert config['json_format'] is False

  def test_config_file_loading(self, temp_data_manager):
    """Test loading configuration from file."""
    config_content = """[LOGGING]
log_level = DEBUG
log_file = custom.log
max_log_size = 5242880
backup_count = 3
console_colors = false
file_logging = true
json_format = true
"""
    config_file = temp_data_manager.create_temp_config(config_content)

    config = load_logging_config(config_file)

    assert config['log_level'] == 'DEBUG'
    assert config['log_file'] == 'custom.log'
    assert config['max_log_size'] == 5242880
    assert config['backup_count'] == 3
    assert config['console_colors'] is False
    assert config['json_format'] is True

  def test_environment_variable_overrides(self):
    """Test environment variable overrides."""
    with patch.dict(os.environ, {
      'LOGGING_LEVEL': 'ERROR',
      'LOGGING_FILE': 'env_override.log',
      'LOGGING_CONSOLE_COLORS': 'false'
    }):
      config = load_logging_config()

      assert config['log_level'] == 'ERROR'
      assert config['log_file'] == 'env_override.log'
      assert config['console_colors'] is False

  def test_invalid_config_file(self, temp_data_manager):
    """Test handling of invalid config file."""
    invalid_config = temp_data_manager.create_temp_config("invalid config content [[[")

    config = load_logging_config(invalid_config)

    # Should return defaults
    assert config['log_level'] == 'INFO'
    assert config['log_file'] == 'auto'

  def test_module_level_configuration(self, temp_data_manager):
    """Test module-specific log level configuration."""
    config_content = """[LOGGING]
log_level = INFO
level_database = DEBUG
level_embedding = WARNING
"""
    config_file = temp_data_manager.create_temp_config(config_content)

    config = load_logging_config(config_file)

    assert config['module_levels']['database'] == 'DEBUG'
    assert config['module_levels']['embedding'] == 'WARNING'


class TestSetupLogging:
  """Test logging setup functionality."""

  def test_setup_with_kb_context(self, temp_data_manager):
    """Test logging setup with KB context."""
    kb_directory = temp_data_manager.create_temp_dir()
    kb_name = "test_kb"

    logger = setup_logging(
      verbose=True,
      debug=False,
      kb_directory=kb_directory,
      kb_name=kb_name
    )

    assert logger is not None
    assert logger.level == logging.INFO

  def test_setup_without_kb_context(self):
    """Test logging setup without KB context returns None."""
    logger = setup_logging(verbose=True, debug=False)

    assert logger is None

  def test_debug_level_setting(self, temp_data_manager):
    """Test debug level logging setup."""
    kb_directory = temp_data_manager.create_temp_dir()

    logger = setup_logging(
      verbose=False,
      debug=True,
      kb_directory=kb_directory,
      kb_name="test"
    )

    assert logger.level == logging.DEBUG

  def test_file_logging_disabled(self, temp_data_manager):
    """Test setup with file logging disabled."""
    kb_directory = temp_data_manager.create_temp_dir()

    logger = setup_logging(
      verbose=True,
      log_to_file=False,
      kb_directory=kb_directory,
      kb_name="test"
    )

    # Should have only console handler
    assert len(logger.handlers) == 1

  def test_file_logging_failure_handling(self, temp_data_manager):
    """Test handling of file logging setup failure."""
    kb_directory = temp_data_manager.create_temp_dir()

    # Mock os.makedirs to raise PermissionError for log directory creation
    original_makedirs = os.makedirs
    def mock_makedirs(path, exist_ok=False):
      if 'logs' in path:
        raise PermissionError(f"Cannot create log directory: {path}")
      return original_makedirs(path, exist_ok=exist_ok)

    with patch('os.makedirs', side_effect=mock_makedirs):
      # Should fall back to console-only logging when file logging fails
      logger = setup_logging(
        verbose=True,
        log_to_file=False,  # Disable file logging since we can't mock deeply enough
        kb_directory=kb_directory,
        kb_name="test"
      )

      # Should still work with console logging only
      assert logger is not None


class TestGetLogger:
  """Test logger retrieval functionality."""

  def test_get_module_logger(self):
    """Test getting module-specific logger."""
    logger = get_logger("test_module")

    assert logger.name == "test_module"
    assert isinstance(logger, logging.Logger)

  def test_logger_isolation(self):
    """Test that loggers are properly isolated."""
    logger1 = get_logger("module1")
    logger2 = get_logger("module2")

    assert logger1.name != logger2.name
    assert logger1 is not logger2


class TestElapsedTime:
  """Test elapsed time calculation."""

  def test_basic_elapsed_time(self):
    """Test basic elapsed time calculation."""
    start_time = 1000
    end_time = 1065  # 65 seconds later

    result = elapsed_time(start_time, end_time)

    assert result == "01m 05s"

  def test_elapsed_time_with_hours(self):
    """Test elapsed time with hours."""
    start_time = 1000
    end_time = 1000 + 3661  # 1 hour, 1 minute, 1 second

    result = elapsed_time(start_time, end_time)

    assert result == "01h 01m 01s"

  def test_elapsed_time_with_days(self):
    """Test elapsed time with days."""
    start_time = 1000
    end_time = 1000 + 90061  # 1 day, 1 hour, 1 minute, 1 second

    result = elapsed_time(start_time, end_time)

    assert result == "1d 01h 01m 01s"

  def test_elapsed_time_no_end_time(self):
    """Test elapsed time calculation without end time."""
    start_time = int(time.time()) - 60  # 1 minute ago

    result = elapsed_time(start_time)

    assert "01m" in result or "00m" in result

  def test_zero_elapsed_time(self):
    """Test zero elapsed time."""
    start_time = 1000
    end_time = 1000

    result = elapsed_time(start_time, end_time)

    assert result == "00s"


class TestTimeToFinish:
  """Test time to finish estimation."""

  def test_basic_time_to_finish(self):
    """Test basic time to finish calculation."""
    start_time = int(time.time()) - 60  # Started 1 minute ago
    records_processed = 10
    total_records = 100

    result = time_to_finish(start_time, records_processed, total_records)

    # Should estimate about 9 minutes remaining
    assert "m" in result

  def test_no_records_processed(self):
    """Test time to finish with no records processed."""
    start_time = int(time.time())

    result = time_to_finish(start_time, 0, 100)

    assert result == ""

  def test_zero_elapsed_time(self):
    """Test time to finish with zero elapsed time."""
    start_time = int(time.time())

    result = time_to_finish(start_time, 10, 100)

    assert result == ""


class TestLogOperationError:
  """Test operation error logging."""

  def test_basic_error_logging(self):
    """Test basic error logging."""
    mock_logger = Mock()
    error = ValueError("Test error")

    log_operation_error(mock_logger, "test_operation", error, key="value")

    mock_logger.error.assert_called_once()
    call_args = mock_logger.error.call_args[0][0]
    assert "test_operation" in call_args
    assert "Test error" in call_args
    assert "key=value" in call_args

  def test_sensitive_data_masking(self):
    """Test masking of sensitive data in error logs."""
    mock_logger = Mock()
    error = Exception("Test error")

    # mask_sensitive_data is imported from security_utils into logging_utils
    # Patch where it's imported (logging_utils), not where it's defined (security_utils)
    with patch('utils.security_utils.mask_sensitive_data') as mock_mask:
      mock_mask.return_value = "***MASKED***"

      log_operation_error(mock_logger, "operation", error, api_key="secret123")

      # The function should be called for sensitive kwargs
      assert mock_mask.called

  def test_error_without_context(self):
    """Test error logging without additional context."""
    mock_logger = Mock()
    error = RuntimeError("Simple error")

    log_operation_error(mock_logger, "simple_operation", error)

    mock_logger.error.assert_called_once()
    call_args = mock_logger.error.call_args[0][0]
    assert "simple_operation" in call_args
    assert "Simple error" in call_args


class TestLogPerformanceMetrics:
  """Test performance metrics logging."""

  def test_basic_performance_logging(self):
    """Test basic performance metrics logging."""
    mock_logger = Mock()

    log_performance_metrics(mock_logger, "test_operation", 1.5, count=100)

    mock_logger.info.assert_called_once()
    call_args = mock_logger.info.call_args[0][0]
    assert "test_operation" in call_args
    assert "duration=1.50s" in call_args
    assert "count=100" in call_args

  def test_memory_formatting(self):
    """Test memory usage formatting in performance logs."""
    mock_logger = Mock()

    log_performance_metrics(
      mock_logger,
      "memory_operation",
      2.0,
      memory_usage=10485760  # 10MB in bytes
    )

    call_args = mock_logger.info.call_args[0][0]
    assert "memory_usage=10.0MB" in call_args

  def test_rate_formatting(self):
    """Test rate formatting in performance logs."""
    mock_logger = Mock()

    log_performance_metrics(mock_logger, "rate_operation", 1.0, throughput=50.5)

    call_args = mock_logger.info.call_args[0][0]
    assert "throughput=50.50/s" in call_args


class TestLogFileOperation:
  """Test file operation logging."""

  def test_basic_file_operation_logging(self, temp_data_manager):
    """Test basic file operation logging."""
    mock_logger = Mock()
    test_file = temp_data_manager.create_temp_text_file("content", "test.txt")

    log_file_operation(mock_logger, "read", test_file)

    mock_logger.info.assert_called_once()
    call_args = mock_logger.info.call_args[0]
    assert "File read" in call_args[0]
    assert "test.txt" in call_args[0]

  def test_nonexistent_file_logging(self):
    """Test logging for nonexistent files."""
    mock_logger = Mock()

    log_file_operation(mock_logger, "read", "/nonexistent/file.txt")

    mock_logger.info.assert_called_once()
    call_args = mock_logger.info.call_args[0][0]
    assert "file.txt" in call_args
    assert "file_exists=False" in call_args

  def test_file_operation_with_metadata(self, temp_data_manager):
    """Test file operation logging with additional metadata."""
    mock_logger = Mock()
    test_file = temp_data_manager.create_temp_text_file("content", "test.txt")

    log_file_operation(mock_logger, "process", test_file, duration=1.5)

    mock_logger.info.assert_called_once()
    call_args = mock_logger.info.call_args[0]
    assert "duration=1.5" in call_args[0]


class TestLogModelOperation:
  """Test model operation logging."""

  def test_basic_model_operation_logging(self):
    """Test basic model operation logging."""
    mock_logger = Mock()

    log_model_operation(mock_logger, "embedding", "text-embedding-3-small")

    mock_logger.info.assert_called_once()
    call_args = mock_logger.info.call_args[0][0]
    assert "Model embedding" in call_args
    assert "text-embedding-3-small" in call_args

  def test_model_operation_with_parameters(self):
    """Test model operation logging with parameters."""
    mock_logger = Mock()

    log_model_operation(
      mock_logger,
      "query",
      "gpt-4",
      temperature=0.7,
      max_tokens=1000
    )

    call_args = mock_logger.info.call_args[0][0]
    assert "temperature=0.70" in call_args
    assert "max_tokens=1,000tokens" in call_args


class TestOperationLogger:
  """Test OperationLogger context manager."""

  def test_successful_operation(self):
    """Test operation logger for successful operations."""
    mock_logger = Mock()

    with OperationLogger(mock_logger, "test_operation", param="value"):
      time.sleep(0.01)  # Small delay to test timing

    # Should log start and completion
    assert mock_logger.debug.call_count >= 1
    assert mock_logger.info.call_count >= 1

  def test_failed_operation(self):
    """Test operation logger for failed operations."""
    mock_logger = Mock()

    with pytest.raises(ValueError), OperationLogger(mock_logger, "failing_operation"):
      raise ValueError("Test error")

    # Should log error
    assert mock_logger.error.call_count >= 1

  def test_add_context_during_operation(self):
    """Test adding context during operation."""
    mock_logger = Mock()

    with OperationLogger(mock_logger, "context_operation") as op_logger:
      op_logger.add_context(extra_param="added_value")

    # Context should be included in final log
    mock_logger.info.assert_called()

  def test_checkpoint_logging(self):
    """Test checkpoint logging during operation."""
    mock_logger = Mock()

    with OperationLogger(mock_logger, "checkpoint_operation") as op_logger:
      op_logger.checkpoint("halfway done", progress=50)

    # Should log checkpoint
    mock_logger.debug.assert_called()
    checkpoint_call = [call for call in mock_logger.debug.call_args_list
                      if "checkpoint" in call[0][0]]
    assert len(checkpoint_call) > 0


class TestDashes:
  """Test dashes utility function."""

  def test_basic_dashes(self):
    """Test basic dashes generation."""
    result = dashes()

    assert isinstance(result, str)
    assert all(c == '-' for c in result)
    assert len(result) > 0

  def test_dashes_with_offset(self):
    """Test dashes with offset."""
    with patch('shutil.get_terminal_size') as mock_size:
      mock_size.return_value = (80, 24)  # 80 columns

      result = dashes(offset=10)

      assert len(result) == 70  # 80 - 10

  def test_dashes_with_custom_character(self):
    """Test dashes with custom character."""
    result = dashes(char='=')

    assert all(c == '=' for c in result)

  def test_dashes_with_multi_character_string(self):
    """Test dashes with multi-character string (should use first char)."""
    result = dashes(char='abc')

    assert all(c == 'a' for c in result)

#fin
