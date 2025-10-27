#!/usr/bin/env python3
"""
Unit tests for EmailProcessor class.

Tests all functionality of the email processing system including
initialization, email discovery, parsing, AI integration, and reply generation.
"""

import pytest
import tempfile
import os
import time
import json
from unittest.mock import Mock, patch, MagicMock, call
from pathlib import Path

# Import test utilities
from tests.utils.test_helpers import (
  create_test_email_file, generate_maildir_filename, 
  create_sample_email_content, create_test_emails_with_scenarios,
  MockAIResponse, mock_customkb_response
)

# Import the module under test
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from email_processor import EmailProcessor, setup_logging


class TestEmailProcessorInitialization:
  """Test EmailProcessor initialization and setup."""
  
  def test_init_with_default_config(self, mock_config):
    """Test initialization with default configuration."""
    processor = EmailProcessor(config_path=mock_config.config_file)
    
    assert processor.config is not None
    assert processor.base_dir == mock_config.get_base_dir()
    assert processor.drafts_dir == mock_config.get_drafts_dir()
    assert isinstance(processor.stats, dict)
    assert all(key in processor.stats for key in [
      'total_found', 'already_replied', 'no_ticket_pattern', 
      'spam_detected', 'reply_generated', 'errors'
    ])
  
  def test_init_with_custom_logger(self, mock_config):
    """Test initialization with custom logger."""
    custom_logger = setup_logging(verbose=True)
    processor = EmailProcessor(logger=custom_logger, config_path=mock_config.config_file)
    
    assert processor.logger is custom_logger
  
  def test_init_ai_clients_with_keys(self, mock_config, monkeypatch):
    """Test AI client initialization with API keys present."""
    monkeypatch.setenv('OPENAI_API_KEY', 'test-openai-key')
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'test-anthropic-key')
    
    with patch('anthropic.Anthropic') as mock_anthropic, \
         patch('openai.OpenAI') as mock_openai:
      
      processor = EmailProcessor(config_path=mock_config.config_file)
      
      # Should attempt to initialize both clients
      mock_anthropic.assert_called_once_with(api_key='test-anthropic-key')
      mock_openai.assert_called_once_with(api_key='test-openai-key')
  
  def test_init_ai_clients_without_keys(self, mock_config):
    """Test AI client initialization without API keys."""
    with patch.dict(os.environ, {}, clear=True):
      processor = EmailProcessor(config_path=mock_config.config_file)
      
      # Should not have AI clients but not crash
      assert not hasattr(processor, 'openai_client')
      assert not hasattr(processor, 'anthropic_client')
  
  def test_init_ai_clients_import_error(self, mock_config, monkeypatch):
    """Test AI client initialization with import errors."""
    monkeypatch.setenv('OPENAI_API_KEY', 'test-key')
    
    with patch('builtins.__import__', side_effect=ImportError("Module not found")):
      processor = EmailProcessor(config_path=mock_config.config_file)
      
      # Should handle import errors gracefully
      assert not hasattr(processor, 'openai_client')


class TestTimestampManagement:
  """Test timestamp file management for incremental processing."""
  
  def test_get_last_check_timestamp_existing_file(self, email_processor_with_mocks, temp_email_dir):
    """Test reading existing timestamp file."""
    processor = email_processor_with_mocks
    timestamp_file = os.path.join(temp_email_dir, '.last_check')
    test_timestamp = time.time() - 3600  # 1 hour ago
    
    with open(timestamp_file, 'w') as f:
      f.write(str(test_timestamp))
    
    # Mock the config to return our temp directory timestamp file
    processor.config.get_timestamp_file = Mock(return_value='.last_check')
    
    with patch.object(processor, 'timestamp_file', timestamp_file):
      result = processor.get_last_check_timestamp()
      assert abs(result - test_timestamp) < 1  # Allow small floating point differences
  
  def test_get_last_check_timestamp_missing_file(self, email_processor_with_mocks):
    """Test reading non-existent timestamp file."""
    processor = email_processor_with_mocks
    
    with patch.object(processor, 'timestamp_file', '/nonexistent/timestamp'):
      result = processor.get_last_check_timestamp()
      assert result == 0
  
  def test_update_last_check_timestamp_default(self, email_processor_with_mocks, temp_email_dir):
    """Test updating timestamp with current time."""
    processor = email_processor_with_mocks
    timestamp_file = os.path.join(temp_email_dir, '.last_check')
    
    with patch.object(processor, 'timestamp_file', timestamp_file):
      before_time = time.time()
      processor.update_last_check_timestamp()
      after_time = time.time()
      
      assert os.path.exists(timestamp_file)
      with open(timestamp_file, 'r') as f:
        written_timestamp = float(f.read().strip())
        assert before_time <= written_timestamp <= after_time
  
  def test_update_last_check_timestamp_custom(self, email_processor_with_mocks, temp_email_dir):
    """Test updating timestamp with custom value."""
    processor = email_processor_with_mocks
    timestamp_file = os.path.join(temp_email_dir, '.last_check')
    custom_timestamp = 1234567890.0
    
    with patch.object(processor, 'timestamp_file', timestamp_file):
      processor.update_last_check_timestamp(custom_timestamp)
      
      with open(timestamp_file, 'r') as f:
        written_timestamp = float(f.read().strip())
        assert written_timestamp == custom_timestamp


class TestEmailDiscovery:
  """Test email discovery and filtering functionality."""
  
  def test_discover_emails_with_samples(self, email_processor_with_mocks, sample_maildir_files):
    """Test email discovery with sample files."""
    processor = email_processor_with_mocks
    
    # Mock subprocess.run to return our sample files
    mock_result = Mock()
    mock_result.returncode = 0
    mock_result.stdout = '\n'.join([
      sample_maildir_files['unreplied'],
      sample_maildir_files['spam']
    ])
    mock_result.stderr = ''
    
    with patch('subprocess.run', return_value=mock_result):
      emails = processor.discover_emails(since_timestamp=0, full_scan=True)
    
    assert len(emails) >= 1  # At least the unreplied one should be found
    assert all('path' in email and 'mtime' in email for email in emails)
  
  def test_discover_emails_since_timestamp(self, email_processor_with_mocks, sample_maildir_files):
    """Test email discovery with timestamp filtering."""
    processor = email_processor_with_mocks
    
    # Get file modification time
    sample_file = sample_maildir_files['unreplied']
    file_mtime = os.path.getmtime(sample_file)
    
    mock_result = Mock()
    mock_result.returncode = 0
    mock_result.stdout = sample_file
    mock_result.stderr = ''
    
    with patch('subprocess.run', return_value=mock_result):
      # Should find file if timestamp is before modification time
      emails = processor.discover_emails(since_timestamp=file_mtime - 10)
      assert len(emails) >= 1
      
      # Should not find file if timestamp is after modification time
      emails = processor.discover_emails(since_timestamp=file_mtime + 10)
      assert len(emails) == 0
  
  def test_discover_emails_filters_replied(self, email_processor_with_mocks, sample_maildir_files):
    """Test that discovery filters out already replied emails."""
    processor = email_processor_with_mocks
    
    mock_result = Mock()
    mock_result.returncode = 0
    mock_result.stdout = '\n'.join([
      sample_maildir_files['unreplied'],
      sample_maildir_files['replied']  # This should be filtered out
    ])
    mock_result.stderr = ''
    
    with patch('subprocess.run', return_value=mock_result):
      emails = processor.discover_emails(full_scan=True)
    
    # Should only include unreplied emails
    unreplied_paths = [email['path'] for email in emails]
    assert sample_maildir_files['unreplied'] in unreplied_paths
    assert sample_maildir_files['replied'] not in unreplied_paths
  
  def test_has_replied_flag(self, email_processor_with_mocks):
    """Test Maildir replied flag detection."""
    processor = email_processor_with_mocks
    
    assert processor._has_replied_flag('email:2,RS') is True
    assert processor._has_replied_flag('email:2,R') is True
    assert processor._has_replied_flag('email:2,SRT') is True
    assert processor._has_replied_flag('email:2,S') is False
    assert processor._has_replied_flag('email:2,') is False
    assert processor._has_replied_flag('email_no_flags') is False


class TestEmailParsing:
  """Test email parsing and content extraction."""
  
  def test_parse_email_file_valid(self, email_processor_with_mocks, sample_maildir_files):
    """Test parsing valid email file."""
    processor = email_processor_with_mocks
    email_file = sample_maildir_files['unreplied']
    
    email_data = processor.parse_email_file(email_file)
    
    assert email_data is not None
    assert isinstance(email_data, dict)
    assert 'subject' in email_data
    assert 'from' in email_data
    assert 'body' in email_data
    assert '[#:TEST001]' in email_data['subject']
  
  def test_parse_email_file_missing(self, email_processor_with_mocks):
    """Test parsing non-existent email file."""
    processor = email_processor_with_mocks
    
    email_data = processor.parse_email_file('/nonexistent/email/file')
    assert email_data is None
  
  def test_extract_body_plain_text(self, email_processor_with_mocks):
    """Test extracting body from plain text email."""
    processor = email_processor_with_mocks
    
    # Create a simple email message
    import email
    from email.mime.text import MIMEText
    
    msg = MIMEText("This is the email body content.")
    msg['Subject'] = "Test Subject"
    msg['From'] = "test@example.com"
    
    body = processor._extract_body(msg)
    assert body == "This is the email body content."
  
  def test_extract_body_multipart(self, email_processor_with_mocks):
    """Test extracting body from multipart email."""
    processor = email_processor_with_mocks
    
    import email
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText
    
    msg = MIMEMultipart()
    msg['Subject'] = "Test Subject"
    msg['From'] = "test@example.com"
    
    # Add plain text part
    text_part = MIMEText("This is the plain text content.", 'plain')
    msg.attach(text_part)
    
    # Add HTML part
    html_part = MIMEText("<p>This is HTML content.</p>", 'html')
    msg.attach(html_part)
    
    body = processor._extract_body(msg)
    assert body == "This is the plain text content."
  
  def test_extract_body_with_limit(self, email_processor_with_mocks):
    """Test body extraction respects body limit."""
    processor = email_processor_with_mocks
    processor.config.get_body_limit = Mock(return_value=50)
    
    import email
    from email.mime.text import MIMEText
    
    long_body = "A" * 1000  # Very long body
    msg = MIMEText(long_body)
    
    body = processor._extract_body(msg)
    assert len(body) == 50


class TestTicketPatternDetection:
  """Test ticket pattern detection functionality."""
  
  def test_has_ticket_pattern_valid(self, email_processor_with_mocks):
    """Test ticket pattern detection with valid patterns."""
    processor = email_processor_with_mocks
    
    assert processor.has_ticket_pattern('Re: [#:TEST001] Company inquiry') is True
    assert processor.has_ticket_pattern('[#:ABC123] Tax question') is True
    assert processor.has_ticket_pattern('Subject with [#:VISA001] in middle') is True
  
  def test_has_ticket_pattern_invalid(self, email_processor_with_mocks):
    """Test ticket pattern detection with invalid patterns."""
    processor = email_processor_with_mocks
    
    assert processor.has_ticket_pattern('Regular email subject') is False
    assert processor.has_ticket_pattern('[#] Missing colon and ID') is False
    assert processor.has_ticket_pattern('[TEST001] Missing #:') is False
    assert processor.has_ticket_pattern('') is False
    assert processor.has_ticket_pattern(None) is False


class TestSpamDetection:
  """Test AI-powered spam detection functionality."""
  
  def test_is_legitimate_email_openai_legitimate(self, email_processor_with_mocks, sample_email_data):
    """Test legitimate email detection with OpenAI."""
    processor = email_processor_with_mocks
    
    # Configure OpenAI client to return LEGITIMATE
    processor.openai_client.chat.completions.create.return_value = \
      MockAIResponse.openai_spam_response(is_legitimate=True)
    
    result = processor.is_legitimate_email(sample_email_data)
    assert result is True
    
    # Verify OpenAI was called
    processor.openai_client.chat.completions.create.assert_called_once()
  
  def test_is_legitimate_email_openai_spam(self, email_processor_with_mocks, sample_email_data):
    """Test spam email detection with OpenAI."""
    processor = email_processor_with_mocks
    
    # Configure OpenAI client to return SPAM
    processor.openai_client.chat.completions.create.return_value = \
      MockAIResponse.openai_spam_response(is_legitimate=False)
    
    result = processor.is_legitimate_email(sample_email_data)
    assert result is False
  
  def test_is_legitimate_email_anthropic_fallback(self, email_processor_with_mocks, sample_email_data):
    """Test Anthropic fallback when OpenAI unavailable."""
    processor = email_processor_with_mocks
    
    # Remove OpenAI client to trigger Anthropic fallback
    delattr(processor, 'openai_client')
    
    processor.anthropic_client.messages.create.return_value = \
      MockAIResponse.anthropic_spam_response(is_legitimate=True)
    
    result = processor.is_legitimate_email(sample_email_data)
    assert result is True
    
    # Verify Anthropic was called
    processor.anthropic_client.messages.create.assert_called_once()
  
  def test_is_legitimate_email_no_ai_clients(self, email_processor_with_mocks, sample_email_data):
    """Test spam detection when no AI clients available."""
    processor = email_processor_with_mocks
    
    # Remove both AI clients
    if hasattr(processor, 'openai_client'):
      delattr(processor, 'openai_client')
    if hasattr(processor, 'anthropic_client'):
      delattr(processor, 'anthropic_client')
    
    result = processor.is_legitimate_email(sample_email_data)
    # Should default to legitimate when no AI available
    assert result is True


class TestEmailEvaluation:
  """Test complete email evaluation pipeline."""
  
  def test_evaluate_email_all_criteria_pass(self, email_processor_with_mocks, sample_maildir_files):
    """Test email evaluation when all criteria pass."""
    processor = email_processor_with_mocks
    
    # Mock spam detection to return legitimate
    processor.is_legitimate_email = Mock(return_value=True)
    
    result = processor.evaluate_email(sample_maildir_files['unreplied'])
    
    assert result['needs_reply'] is True
    assert 'meets all criteria' in result['reason']
    assert result['email_data'] is not None
  
  def test_evaluate_email_already_replied(self, email_processor_with_mocks, sample_maildir_files):
    """Test email evaluation for already replied email."""
    processor = email_processor_with_mocks
    
    result = processor.evaluate_email(sample_maildir_files['replied'])
    
    assert result['needs_reply'] is False
    assert 'already has reply flag' in result['reason']
  
  def test_evaluate_email_no_ticket_pattern(self, email_processor_with_mocks, sample_maildir_files):
    """Test email evaluation without ticket pattern."""
    processor = email_processor_with_mocks
    
    result = processor.evaluate_email(sample_maildir_files['spam'])
    
    assert result['needs_reply'] is False
    assert 'does not contain ticket pattern' in result['reason']
  
  def test_evaluate_email_spam_detected(self, email_processor_with_mocks, temp_email_dir):
    """Test email evaluation when spam is detected."""
    processor = email_processor_with_mocks
    
    # Create email with ticket pattern but will be marked as spam
    spam_content = create_sample_email_content(
      subject="Spam with ticket pattern",
      sender="spam@example.com",
      body="This is spam content with promotional material.",
      ticket_id="SPAM001"
    )
    spam_file = create_test_email_file(
      os.path.join(temp_email_dir, 'cur'),
      generate_maildir_filename(flags='S'),
      spam_content
    )
    
    # Mock spam detection to return spam
    processor.is_legitimate_email = Mock(return_value=False)
    
    result = processor.evaluate_email(spam_file)
    
    assert result['needs_reply'] is False
    assert 'detected as spam' in result['reason']


class TestConsultantAssignment:
  """Test consultant assignment functionality."""
  
  def test_determine_consultant(self, email_processor_with_mocks, sample_email_data):
    """Test consultant determination based on email content."""
    processor = email_processor_with_mocks
    
    # Mock config method
    processor.config.determine_consultant_type = Mock(return_value='company')
    processor.config.get_consultant = Mock(return_value={
      'name': 'Test Consultant',
      'email': 'test@example.com'
    })
    
    consultant = processor.determine_consultant(sample_email_data)
    
    assert consultant is not None
    assert 'name' in consultant
    assert 'email' in consultant
    
    # Verify content was analyzed
    processor.config.determine_consultant_type.assert_called_once()


class TestCustomKBIntegration:
  """Test CustomKB integration and query construction."""
  
  def test_construct_query(self, email_processor_with_mocks, sample_email_data):
    """Test CustomKB query construction."""
    processor = email_processor_with_mocks
    
    # Mock config method
    processor.config.get_customkb_query_prompt = Mock(return_value="Formatted query")
    processor.config.get_country_name = Mock(return_value="United States")
    
    query = processor.construct_query(sample_email_data)
    
    assert isinstance(query, str)
    processor.config.get_customkb_query_prompt.assert_called_once()
  
  def test_call_customkb_success(self, email_processor_with_mocks, mock_subprocess):
    """Test successful CustomKB call."""
    processor = email_processor_with_mocks
    
    # Configure mock subprocess to return success
    mock_subprocess.return_value.returncode = 0
    mock_subprocess.return_value.stdout = "Mock CustomKB response"
    
    with patch('subprocess.run', mock_subprocess):
      result = processor.call_customkb("Test query")
    
    assert result == "Mock CustomKB response"
    mock_subprocess.assert_called_once()
  
  def test_call_customkb_failure(self, email_processor_with_mocks, mock_subprocess):
    """Test failed CustomKB call."""
    processor = email_processor_with_mocks
    
    # Configure mock subprocess to return failure
    mock_subprocess.return_value.returncode = 1
    mock_subprocess.return_value.stderr = "CustomKB error"
    
    with patch('subprocess.run', mock_subprocess):
      result = processor.call_customkb("Test query")
    
    assert result is None


class TestEmailDraftGeneration:
  """Test email draft generation and formatting."""
  
  def test_generate_email_draft(self, email_processor_with_mocks, sample_email_data):
    """Test email draft generation."""
    processor = email_processor_with_mocks
    
    # Mock config methods
    processor.config.get_email_signature = Mock(return_value="Test Signature")
    
    consultant = {
      'name': 'Test Consultant',
      'email': 'test@example.com'
    }
    reply_content = "This is the generated reply content."
    
    draft = processor.generate_email_draft(sample_email_data, reply_content, consultant)
    
    assert draft is not None
    assert isinstance(draft, str)
    assert 'From: Test Consultant <test@example.com>' in draft
    assert 'To: test@example.com' in draft
    assert 'Re: ' in draft
    assert reply_content in draft
    assert 'Test Signature' in draft
  
  def test_save_draft(self, email_processor_with_mocks, temp_email_dir):
    """Test saving email draft to drafts directory."""
    processor = email_processor_with_mocks
    
    # Update processor to use temp directory
    drafts_dir = os.path.join(temp_email_dir, '.Drafts', 'cur')
    os.makedirs(drafts_dir, exist_ok=True)
    processor.drafts_dir = drafts_dir
    
    draft_content = """From: test@example.com
To: recipient@example.com
Subject: Test Draft

This is a test draft email."""
    
    draft_path = processor.save_draft(draft_content, '/original/email/path')
    
    assert draft_path is not None
    assert os.path.exists(draft_path)
    assert ':2,D' in os.path.basename(draft_path)  # Draft flag
    
    with open(draft_path, 'r') as f:
      saved_content = f.read()
      assert saved_content == draft_content


class TestMaildirFlagManagement:
  """Test Maildir flag manipulation."""
  
  def test_mark_as_replied(self, email_processor_with_mocks, temp_email_dir):
    """Test marking email as replied."""
    processor = email_processor_with_mocks
    
    # Create test email file
    cur_dir = os.path.join(temp_email_dir, 'cur')
    os.makedirs(cur_dir, exist_ok=True)
    
    original_filename = generate_maildir_filename(flags='S')
    original_path = os.path.join(cur_dir, original_filename)
    
    with open(original_path, 'w') as f:
      f.write("Test email content")
    
    # Mock config method
    processor.config.add_maildir_flag = Mock(return_value=original_filename.replace(':2,S', ':2,SR'))
    
    new_path = processor.mark_as_replied(original_path)
    
    assert new_path != original_path
    assert ':2,SR' in new_path or ':2,RS' in new_path
    processor.config.add_maildir_flag.assert_called_once_with(original_filename, 'replied')


class TestStatisticsTracking:
  """Test processing statistics tracking."""
  
  def test_stats_initialization(self, email_processor_with_mocks):
    """Test that statistics are properly initialized."""
    processor = email_processor_with_mocks
    
    expected_keys = [
      'total_found', 'already_replied', 'no_ticket_pattern',
      'spam_detected', 'reply_generated', 'errors'
    ]
    
    for key in expected_keys:
      assert key in processor.stats
      assert processor.stats[key] == 0
  
  def test_print_summary(self, email_processor_with_mocks, capsys):
    """Test statistics summary printing."""
    processor = email_processor_with_mocks
    
    # Set some test statistics
    processor.stats.update({
      'total_found': 10,
      'already_replied': 2,
      'spam_detected': 3,
      'reply_generated': 5
    })
    
    processor.print_summary()
    
    captured = capsys.readouterr()
    assert 'Processing Summary' in captured.out
    assert '10' in captured.out  # total_found
    assert '5' in captured.out   # reply_generated


class TestErrorHandling:
  """Test error handling and edge cases."""
  
  def test_process_email_with_exception(self, email_processor_with_mocks):
    """Test email processing with exceptions."""
    processor = email_processor_with_mocks
    
    # Mock evaluate_email to raise exception
    processor.evaluate_email = Mock(side_effect=Exception("Test error"))
    
    email_info = {'path': '/test/email', 'mtime': time.time()}
    result = processor.process_email(email_info)
    
    assert result['processed'] is False
    assert 'error' in result['reason'].lower()
    assert processor.stats['errors'] > 0
  
  def test_graceful_degradation_no_customkb(self, email_processor_with_mocks, sample_email_data):
    """Test graceful degradation when CustomKB unavailable."""
    processor = email_processor_with_mocks
    
    # Mock CustomKB to fail
    processor.call_customkb = Mock(return_value=None)
    
    result = processor.generate_reply('/test/email')
    
    assert result['success'] is False
    assert 'CustomKB' in result['error']

#fin