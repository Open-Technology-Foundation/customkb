#!/usr/bin/env python3
"""
Integration tests for the complete email processing pipeline.

These tests verify that the entire system works together correctly,
from email discovery through reply generation and file management.
"""

import pytest
import tempfile
import os
import time
from unittest.mock import patch, Mock

# Import test utilities
from tests.utils.test_helpers import (
  create_test_emails_with_scenarios, create_temp_timestamp_file,
  MockAIResponse, mock_customkb_response, assert_maildir_file_exists,
  count_files_with_pattern
)

# Import the module under test
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from email_processor import EmailProcessor


@pytest.mark.integration
class TestCompleteEmailPipeline:
  """Test the complete email processing pipeline end-to-end."""
  
  def test_full_pipeline_legitimate_email(self, temp_email_dir, test_config_file, monkeypatch):
    """Test complete pipeline with legitimate business email."""
    # Set up environment
    monkeypatch.setenv('OPENAI_API_KEY', 'test-openai-key')
    
    # Create test emails
    scenarios = create_test_emails_with_scenarios(temp_email_dir)
    
    # Create drafts directory
    drafts_dir = os.path.join(temp_email_dir, '.Drafts', 'cur')
    os.makedirs(drafts_dir, exist_ok=True)
    
    # Mock external dependencies
    mock_subprocess = Mock()
    mock_subprocess.return_value.returncode = 0
    mock_subprocess.return_value.stdout = mock_customkb_response("company formation")
    mock_subprocess.return_value.stderr = ""
    
    with patch('subprocess.run', mock_subprocess), \
         patch('anthropic.Anthropic'), \
         patch('email_processor.OpenAI') as mock_openai_class:
      
      # Set up OpenAI mock
      mock_openai = Mock()
      mock_openai.chat.completions.create.return_value = \
        MockAIResponse.openai_spam_response(is_legitimate=True)
      mock_openai_class.return_value = mock_openai
      
      # Create processor and run pipeline
      processor = EmailProcessor(config_path=test_config_file)
      processor.openai_client = mock_openai
      
      # Process emails
      success = processor.process_emails(full_scan=True, dry_run=False)
      
      assert success is True
      assert processor.stats['reply_generated'] > 0
      
      # Verify draft was created
      draft_files = count_files_with_pattern(drafts_dir, '*:2,D')
      assert draft_files > 0
  
  def test_full_pipeline_spam_filtering(self, temp_email_dir, test_config_file, monkeypatch):
    """Test complete pipeline filters out spam correctly."""
    monkeypatch.setenv('OPENAI_API_KEY', 'test-openai-key')
    
    # Create test emails including spam
    scenarios = create_test_emails_with_scenarios(temp_email_dir)
    
    drafts_dir = os.path.join(temp_email_dir, '.Drafts', 'cur')
    os.makedirs(drafts_dir, exist_ok=True)
    
    with patch('subprocess.run'), \
         patch('anthropic.Anthropic'), \
         patch('email_processor.OpenAI') as mock_openai_class:
      
      # Set up OpenAI mock to detect spam
      mock_openai = Mock()
      mock_openai.chat.completions.create.return_value = \
        MockAIResponse.openai_spam_response(is_legitimate=False)
      mock_openai_class.return_value = mock_openai
      
      processor = EmailProcessor(config_path=test_config_file)
      processor.openai_client = mock_openai
      
      success = processor.process_emails(full_scan=True, dry_run=False)
      
      assert success is True
      assert processor.stats['spam_detected'] > 0
      assert processor.stats['reply_generated'] == 0  # No replies for spam
      
      # Verify no drafts were created
      draft_files = count_files_with_pattern(drafts_dir, '*:2,D')
      assert draft_files == 0
  
  def test_incremental_processing(self, temp_email_dir, test_config_file, monkeypatch):
    """Test incremental processing based on timestamps."""
    monkeypatch.setenv('OPENAI_API_KEY', 'test-openai-key')
    
    # Create timestamp file from 1 hour ago
    old_timestamp = time.time() - 3600
    create_temp_timestamp_file(temp_email_dir, old_timestamp)
    
    # Create test emails
    scenarios = create_test_emails_with_scenarios(temp_email_dir)
    
    drafts_dir = os.path.join(temp_email_dir, '.Drafts', 'cur')
    os.makedirs(drafts_dir, exist_ok=True)
    
    mock_subprocess = Mock()
    mock_subprocess.return_value.returncode = 0
    mock_subprocess.return_value.stdout = mock_customkb_response()
    
    with patch('subprocess.run', mock_subprocess), \
         patch('anthropic.Anthropic'), \
         patch('email_processor.OpenAI') as mock_openai_class:
      
      mock_openai = Mock()
      mock_openai.chat.completions.create.return_value = \
        MockAIResponse.openai_spam_response(is_legitimate=True)
      mock_openai_class.return_value = mock_openai
      
      processor = EmailProcessor(config_path=test_config_file)
      processor.openai_client = mock_openai
      
      # First run - should process emails
      success = processor.process_emails(dry_run=False)
      assert success is True
      
      first_run_stats = processor.stats.copy()
      
      # Second run immediately - should not reprocess
      processor.stats = {key: 0 for key in processor.stats}  # Reset stats
      success = processor.process_emails(dry_run=False)
      assert success is True
      
      # Should find fewer or no emails to process
      assert processor.stats['total_found'] <= first_run_stats['total_found']
  
  def test_dry_run_mode(self, temp_email_dir, test_config_file, monkeypatch):
    """Test dry run mode doesn't generate actual replies."""
    monkeypatch.setenv('OPENAI_API_KEY', 'test-openai-key')
    
    scenarios = create_test_emails_with_scenarios(temp_email_dir)
    drafts_dir = os.path.join(temp_email_dir, '.Drafts', 'cur')
    os.makedirs(drafts_dir, exist_ok=True)
    
    with patch('subprocess.run'), \
         patch('anthropic.Anthropic'), \
         patch('email_processor.OpenAI') as mock_openai_class:
      
      mock_openai = Mock()
      mock_openai.chat.completions.create.return_value = \
        MockAIResponse.openai_spam_response(is_legitimate=True)
      mock_openai_class.return_value = mock_openai
      
      processor = EmailProcessor(config_path=test_config_file)
      processor.openai_client = mock_openai
      
      success = processor.process_emails(full_scan=True, dry_run=True)
      
      assert success is True
      assert processor.stats['reply_generated'] == 0  # No actual replies in dry run
      
      # Verify no drafts were created
      draft_files = count_files_with_pattern(drafts_dir, '*:2,D')
      assert draft_files == 0
      
      # Verify no emails were marked as replied
      replied_files = count_files_with_pattern(
        os.path.join(temp_email_dir, 'cur'), '*:2,*R*'
      )
      assert replied_files == 0  # Should not mark emails as replied in dry run
  
  def test_ai_client_fallback(self, temp_email_dir, test_config_file, monkeypatch):
    """Test fallback from OpenAI to Anthropic when OpenAI fails."""
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'test-anthropic-key')
    # Don't set OpenAI key to trigger fallback
    
    scenarios = create_test_emails_with_scenarios(temp_email_dir)
    drafts_dir = os.path.join(temp_email_dir, '.Drafts', 'cur')
    os.makedirs(drafts_dir, exist_ok=True)
    
    mock_subprocess = Mock()
    mock_subprocess.return_value.returncode = 0
    mock_subprocess.return_value.stdout = mock_customkb_response()
    
    with patch('subprocess.run', mock_subprocess), \
         patch('anthropic.Anthropic') as mock_anthropic_class:
      
      mock_anthropic = Mock()
      mock_anthropic.messages.create.return_value = \
        MockAIResponse.anthropic_spam_response(is_legitimate=True)
      mock_anthropic_class.return_value = mock_anthropic
      
      processor = EmailProcessor(config_path=test_config_file)
      processor.anthropic_client = mock_anthropic
      
      success = processor.process_emails(full_scan=True, dry_run=False)
      
      assert success is True
      # Should use Anthropic when OpenAI unavailable
      mock_anthropic.messages.create.assert_called()
  
  def test_error_recovery(self, temp_email_dir, test_config_file, monkeypatch):
    """Test system continues processing after individual email errors."""
    monkeypatch.setenv('OPENAI_API_KEY', 'test-openai-key')
    
    scenarios = create_test_emails_with_scenarios(temp_email_dir)
    drafts_dir = os.path.join(temp_email_dir, '.Drafts', 'cur')
    os.makedirs(drafts_dir, exist_ok=True)
    
    # Mock subprocess to fail for CustomKB calls
    mock_subprocess = Mock()
    mock_subprocess.return_value.returncode = 1  # Failure
    mock_subprocess.return_value.stderr = "CustomKB error"
    
    with patch('subprocess.run', mock_subprocess), \
         patch('anthropic.Anthropic'), \
         patch('email_processor.OpenAI') as mock_openai_class:
      
      mock_openai = Mock()
      mock_openai.chat.completions.create.return_value = \
        MockAIResponse.openai_spam_response(is_legitimate=True)
      mock_openai_class.return_value = mock_openai
      
      processor = EmailProcessor(config_path=test_config_file)
      processor.openai_client = mock_openai
      
      success = processor.process_emails(full_scan=True, dry_run=False)
      
      # Should complete successfully despite individual email failures
      assert success is True
      assert processor.stats['errors'] > 0  # Should track errors
      assert processor.stats['reply_generated'] == 0  # No replies due to CustomKB failure
  
  def test_maildir_flag_consistency(self, temp_email_dir, test_config_file, monkeypatch):
    """Test Maildir flag consistency throughout processing."""
    monkeypatch.setenv('OPENAI_API_KEY', 'test-openai-key')
    
    scenarios = create_test_emails_with_scenarios(temp_email_dir)
    drafts_dir = os.path.join(temp_email_dir, '.Drafts', 'cur')
    os.makedirs(drafts_dir, exist_ok=True)
    
    mock_subprocess = Mock()
    mock_subprocess.return_value.returncode = 0
    mock_subprocess.return_value.stdout = mock_customkb_response()
    
    with patch('subprocess.run', mock_subprocess), \
         patch('anthropic.Anthropic'), \
         patch('email_processor.OpenAI') as mock_openai_class:
      
      mock_openai = Mock()
      mock_openai.chat.completions.create.return_value = \
        MockAIResponse.openai_spam_response(is_legitimate=True)
      mock_openai_class.return_value = mock_openai
      
      processor = EmailProcessor(config_path=test_config_file)
      processor.openai_client = mock_openai
      
      # Get original unreplied email path
      original_unreplied = scenarios['legitimate_company']
      assert ':2,S' in original_unreplied  # Should start as seen only
      
      success = processor.process_emails(full_scan=True, dry_run=False)
      assert success is True
      
      # Verify original email was marked as replied
      cur_dir = os.path.join(temp_email_dir, 'cur')
      replied_files = [f for f in os.listdir(cur_dir) if ':2,' in f and 'R' in f.split(':2,')[1]]
      assert len(replied_files) > 0
      
      # Verify draft has correct flag
      draft_files = [f for f in os.listdir(drafts_dir) if ':2,D' in f]
      assert len(draft_files) > 0


@pytest.mark.integration  
class TestConfigurationIntegration:
  """Test integration with different configuration scenarios."""
  
  def test_custom_consultant_routing(self, temp_email_dir, test_config_data, monkeypatch):
    """Test custom consultant routing configuration."""
    monkeypatch.setenv('OPENAI_API_KEY', 'test-openai-key')
    
    # Add custom consultant to config
    test_config_data['consultants']['custom'] = {
      'name': 'Custom Consultant',
      'email': 'custom@example.com',
      'title': 'Custom Title',
      'phone': '+1111111111'
    }
    test_config_data['keywords']['custom'] = ['custom', 'special']
    
    # Create config file with custom consultant
    import yaml
    config_file = os.path.join(temp_email_dir, 'custom_config.yaml')
    test_config_data['directories']['base_dir'] = temp_email_dir
    with open(config_file, 'w') as f:
      yaml.dump(test_config_data, f)
    
    # Create email with custom keyword
    from tests.utils.test_helpers import create_sample_email_content, create_test_email_file, generate_maildir_filename
    
    custom_content = create_sample_email_content(
      subject="Custom service inquiry",
      sender="client@example.com",
      body="I need help with your custom special services.",
      ticket_id="CUSTOM001"
    )
    
    cur_dir = os.path.join(temp_email_dir, 'cur')
    os.makedirs(cur_dir, exist_ok=True)
    
    custom_email = create_test_email_file(
      cur_dir,
      generate_maildir_filename(flags='S'),
      custom_content
    )
    
    drafts_dir = os.path.join(temp_email_dir, '.Drafts', 'cur')
    os.makedirs(drafts_dir, exist_ok=True)
    
    mock_subprocess = Mock()
    mock_subprocess.return_value.returncode = 0
    mock_subprocess.return_value.stdout = "Custom consultant response"
    
    with patch('subprocess.run', mock_subprocess), \
         patch('anthropic.Anthropic'), \
         patch('email_processor.OpenAI') as mock_openai_class:
      
      mock_openai = Mock()
      mock_openai.chat.completions.create.return_value = \
        MockAIResponse.openai_spam_response(is_legitimate=True)
      mock_openai_class.return_value = mock_openai
      
      processor = EmailProcessor(config_path=config_file)
      processor.openai_client = mock_openai
      
      success = processor.process_emails(full_scan=True, dry_run=False)
      assert success is True
      
      # Verify custom consultant was used
      draft_files = os.listdir(drafts_dir)
      assert len(draft_files) > 0
      
      # Check draft content includes custom consultant
      with open(os.path.join(drafts_dir, draft_files[0]), 'r') as f:
        draft_content = f.read()
        assert 'Custom Consultant' in draft_content
  
  def test_modified_processing_limits(self, temp_email_dir, test_config_data, monkeypatch):
    """Test with modified processing limits."""
    monkeypatch.setenv('OPENAI_API_KEY', 'test-openai-key')
    
    # Modify processing limits
    test_config_data['email']['body_limit'] = 100  # Very small limit
    test_config_data['email']['analysis_limit'] = 50
    
    # Create config file
    import yaml
    config_file = os.path.join(temp_email_dir, 'limits_config.yaml')
    test_config_data['directories']['base_dir'] = temp_email_dir
    with open(config_file, 'w') as f:
      yaml.dump(test_config_data, f)
    
    scenarios = create_test_emails_with_scenarios(temp_email_dir)
    drafts_dir = os.path.join(temp_email_dir, '.Drafts', 'cur')
    os.makedirs(drafts_dir, exist_ok=True)
    
    mock_subprocess = Mock()
    mock_subprocess.return_value.returncode = 0
    mock_subprocess.return_value.stdout = mock_customkb_response()
    
    with patch('subprocess.run', mock_subprocess), \
         patch('anthropic.Anthropic'), \
         patch('email_processor.OpenAI') as mock_openai_class:
      
      mock_openai = Mock()
      mock_openai.chat.completions.create.return_value = \
        MockAIResponse.openai_spam_response(is_legitimate=True)
      mock_openai_class.return_value = mock_openai
      
      processor = EmailProcessor(config_path=config_file)
      processor.openai_client = mock_openai
      
      # Verify limits are applied
      assert processor.config.get_body_limit() == 100
      assert processor.config.get_analysis_limit() == 50
      
      success = processor.process_emails(full_scan=True, dry_run=False)
      assert success is True


@pytest.mark.integration
@pytest.mark.slow
class TestPerformanceIntegration:
  """Test performance characteristics of the complete system."""
  
  def test_processing_multiple_emails(self, temp_email_dir, test_config_file, monkeypatch):
    """Test processing multiple emails efficiently."""
    import time
    
    monkeypatch.setenv('OPENAI_API_KEY', 'test-openai-key')
    
    # Create multiple test emails
    cur_dir = os.path.join(temp_email_dir, 'cur')
    os.makedirs(cur_dir, exist_ok=True)
    
    from tests.utils.test_helpers import create_sample_email_content, create_test_email_file, generate_maildir_filename
    
    num_emails = 10
    email_files = []
    
    for i in range(num_emails):
      content = create_sample_email_content(
        subject=f"Business inquiry {i}",
        sender=f"client{i}@example.com",
        body=f"This is business inquiry number {i} about company formation.",
        ticket_id=f"TEST{i:03d}"
      )
      
      email_file = create_test_email_file(
        cur_dir,
        generate_maildir_filename(flags='S'),
        content
      )
      email_files.append(email_file)
    
    drafts_dir = os.path.join(temp_email_dir, '.Drafts', 'cur')
    os.makedirs(drafts_dir, exist_ok=True)
    
    mock_subprocess = Mock()
    mock_subprocess.return_value.returncode = 0
    mock_subprocess.return_value.stdout = mock_customkb_response()
    
    with patch('subprocess.run', mock_subprocess), \
         patch('anthropic.Anthropic'), \
         patch('email_processor.OpenAI') as mock_openai_class:
      
      mock_openai = Mock()
      mock_openai.chat.completions.create.return_value = \
        MockAIResponse.openai_spam_response(is_legitimate=True)
      mock_openai_class.return_value = mock_openai
      
      processor = EmailProcessor(config_path=test_config_file)
      processor.openai_client = mock_openai
      
      start_time = time.time()
      success = processor.process_emails(full_scan=True, dry_run=False)
      end_time = time.time()
      
      assert success is True
      processing_time = end_time - start_time
      
      # Should process emails efficiently (less than 5 seconds for 10 emails)
      assert processing_time < 5.0
      
      # Verify all emails were processed
      assert processor.stats['reply_generated'] == num_emails
      
      # Verify all drafts were created
      draft_files = count_files_with_pattern(drafts_dir, '*:2,D')
      assert draft_files == num_emails

#fin