#!/usr/bin/env python3
"""
Unit tests for logging functionality.

Tests the logging setup and configuration, including the recently fixed
file handler bug and various logging scenarios.
"""

import pytest
import tempfile
import logging
import os
from unittest.mock import patch, Mock

# Import the module under test
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from email_processor import setup_logging


class TestLoggingSetup:
  """Test logging setup and configuration."""
  
  def test_setup_logging_console_only(self):
    """Test logging setup with console output only."""
    # Clear any existing handlers to avoid interference
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
      root_logger.removeHandler(handler)
    
    logger = setup_logging(verbose=False)
    
    assert isinstance(logger, logging.Logger)
    # Logger should have INFO level (effective level might be inherited)
    assert logger.getEffectiveLevel() <= logging.INFO
    assert len(logger.handlers) > 0
    
    # Should have at least one console handler
    console_handlers = [h for h in logger.handlers if isinstance(h, logging.StreamHandler)]
    assert len(console_handlers) > 0
    
    # Console handler should have INFO level
    assert console_handlers[0].level == logging.INFO
  
  def test_setup_logging_verbose_mode(self):
    """Test logging setup with verbose (debug) mode."""
    # Clear any existing handlers to avoid interference
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
      root_logger.removeHandler(handler)
    
    logger = setup_logging(verbose=True)
    
    assert logger.getEffectiveLevel() <= logging.DEBUG
    
    # Check that handler is set to DEBUG level
    console_handlers = [h for h in logger.handlers if isinstance(h, logging.StreamHandler)]
    assert len(console_handlers) > 0
    assert console_handlers[0].level == logging.DEBUG
  
  def test_setup_logging_with_file(self):
    """Test logging setup with file output."""
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
      temp_filename = temp_file.name
    
    try:
      logger = setup_logging(verbose=True, log_file=temp_filename)
      
      # Should have both console and file handlers
      assert len(logger.handlers) >= 2
      
      # Check for file handler
      file_handlers = [h for h in logger.handlers if isinstance(h, logging.FileHandler)]
      assert len(file_handlers) >= 1
      
      # Test that we can actually log to file
      test_message = "Test log message"
      logger.info(test_message)
      
      # Verify file was created and contains message
      assert os.path.exists(temp_filename)
      
      # Force flush handlers
      for handler in logger.handlers:
        if hasattr(handler, 'flush'):
          handler.flush()
      
      with open(temp_filename, 'r') as f:
        log_content = f.read()
        assert test_message in log_content
      
    finally:
      # Cleanup
      try:
        os.unlink(temp_filename)
      except:
        pass
  
  def test_setup_logging_file_handler_bug_fix(self):
    """Test that the file handler bug has been fixed."""
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
      temp_filename = temp_file.name
    
    try:
      # This should not raise an exception
      logger = setup_logging(verbose=True, log_file=temp_filename)
      
      # Verify file handler was added correctly
      file_handlers = [h for h in logger.handlers if isinstance(h, logging.FileHandler)]
      assert len(file_handlers) >= 1
      
      file_handler = file_handlers[0]
      assert file_handler.level == logging.DEBUG
      assert file_handler.formatter is not None
      
    finally:
      try:
        os.unlink(temp_filename)
      except:
        pass
  
  def test_logging_format_consistency(self):
    """Test that logging format is consistent across handlers."""
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
      temp_filename = temp_file.name
    
    try:
      logger = setup_logging(verbose=True, log_file=temp_filename)
      
      expected_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
      
      for handler in logger.handlers:
        if hasattr(handler, 'formatter') and handler.formatter:
          # Check format string if accessible
          if hasattr(handler.formatter, '_fmt'):
            assert handler.formatter._fmt == expected_format
      
    finally:
      try:
        os.unlink(temp_filename)
      except:
        pass
  
  def test_logging_levels(self):
    """Test different logging levels work correctly."""
    logger = setup_logging(verbose=True)
    
    # These should not raise exceptions
    logger.debug("Debug message")
    logger.info("Info message") 
    logger.warning("Warning message")
    logger.error("Error message")
    logger.critical("Critical message")
  
  def test_logging_with_invalid_file_path(self):
    """Test logging setup with invalid file path."""
    invalid_path = "/nonexistent/directory/logfile.log"
    
    # This should raise an exception due to invalid path
    with pytest.raises(FileNotFoundError):
      setup_logging(verbose=True, log_file=invalid_path)
  
  def test_logger_name_correct(self):
    """Test that logger has correct name."""
    logger = setup_logging()
    assert logger.name == 'email_processor'
  
  def test_multiple_setup_calls_cleanup(self):
    """Test that multiple setup calls don't create handler leaks."""
    # First setup
    logger1 = setup_logging(verbose=False)
    initial_handler_count = len(logger1.handlers)
    
    # Second setup (should clean up properly)
    logger2 = setup_logging(verbose=True)
    
    # Should be same logger instance but handlers might be reconfigured
    assert logger1.name == logger2.name
    
    # Verify no excessive handler accumulation
    assert len(logger2.handlers) >= initial_handler_count
    # Allow for reasonable handler count (console + possible file)
    assert len(logger2.handlers) <= initial_handler_count + 2


class TestLoggingIntegration:
  """Test logging integration with EmailProcessor."""
  
  def test_email_processor_logging_integration(self):
    """Test that EmailProcessor uses logging correctly."""
    from email_processor import EmailProcessor
    
    with patch('config_loader.get_config') as mock_get_config:
      # Mock config to avoid file dependencies
      mock_config = Mock()
      mock_config.get_base_dir.return_value = '/tmp'
      mock_config.get_timestamp_file.return_value = '.last_check'
      mock_config.get_drafts_dir.return_value = '/tmp/.Drafts/cur'
      mock_get_config.return_value = mock_config
      
      # Create processor with custom logger
      test_logger = setup_logging(verbose=True)
      processor = EmailProcessor(logger=test_logger)
      
      assert processor.logger is test_logger
      assert processor.logger.name == 'email_processor'
  
  def test_logging_in_error_scenarios(self):
    """Test logging behavior during error scenarios."""
    logger = setup_logging(verbose=True)
    
    # Test logging with exception info
    try:
      raise ValueError("Test exception")
    except ValueError as e:
      logger.error(f"Test error: {e}", exc_info=True)
    
    # Should not raise exceptions
    assert True  # If we get here, logging worked
  
  def test_logging_performance_basic(self):
    """Basic test to ensure logging doesn't significantly impact performance."""
    import time
    
    logger = setup_logging(verbose=False)
    
    # Time a bunch of log operations
    start_time = time.time()
    for i in range(1000):
      logger.info(f"Test message {i}")
    end_time = time.time()
    
    # Should complete quickly (less than 1 second for 1000 messages)
    duration = end_time - start_time
    assert duration < 1.0


class TestLoggingConfiguration:
  """Test logging configuration edge cases."""
  
  def test_logging_with_unicode_content(self):
    """Test logging with unicode content."""
    logger = setup_logging(verbose=True)
    
    # Test various unicode characters
    unicode_message = "Test message with unicode: ñáéíóú 中文 🚀"
    
    # Should not raise exceptions
    logger.info(unicode_message)
    logger.error(unicode_message)
  
  def test_logging_with_very_long_messages(self):
    """Test logging with very long messages."""
    logger = setup_logging(verbose=True)
    
    # Create a very long message
    long_message = "A" * 10000
    
    # Should not raise exceptions
    logger.info(long_message)
  
  def test_logging_thread_safety_basic(self):
    """Basic test for logging thread safety."""
    import threading
    
    logger = setup_logging(verbose=True)
    results = []
    
    def log_messages():
      for i in range(100):
        logger.info(f"Thread message {i}")
        results.append(i)
    
    # Create multiple threads
    threads = []
    for _ in range(3):
      thread = threading.Thread(target=log_messages)
      threads.append(thread)
      thread.start()
    
    # Wait for all threads to complete
    for thread in threads:
      thread.join()
    
    # All messages should have been logged
    assert len(results) == 300

#fin