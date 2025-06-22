#!/usr/bin/env python3
"""
Unit tests for EmailConfig class.

Tests all functionality of the configuration management system including
YAML loading, validation, getter methods, and error handling.
"""

import pytest
import tempfile
import os
import yaml
from unittest.mock import patch, mock_open

# Import the module under test
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from config_loader import EmailConfig, get_config, reload_config


class TestEmailConfigInitialization:
  """Test EmailConfig initialization and configuration loading."""
  
  def test_init_with_default_config_file(self):
    """Test initialization with default config file path."""
    with patch('pathlib.Path.exists', return_value=True), \
         patch('builtins.open', mock_open(read_data='directories:\n  base_dir: /test')), \
         patch('yaml.safe_load', return_value={'directories': {'base_dir': '/test'}}):
      
      config = EmailConfig()
      assert config.config_file.name == 'email_config.yaml'
  
  def test_init_with_custom_config_file(self, test_config_file):
    """Test initialization with custom config file path."""
    config = EmailConfig(test_config_file)
    assert str(config.config_file) == test_config_file
    assert config.config is not None
  
  def test_init_with_missing_config_file(self):
    """Test initialization with non-existent config file."""
    with pytest.raises(FileNotFoundError):
      EmailConfig('/nonexistent/config.yaml')
  
  def test_init_with_invalid_yaml(self):
    """Test initialization with malformed YAML."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
      f.write('invalid: yaml: content: [')
      f.flush()
      
      try:
        with pytest.raises(yaml.YAMLError):
          EmailConfig(f.name)
      finally:
        os.unlink(f.name)
  
  def test_validation_missing_required_sections(self, test_config_data):
    """Test validation with missing required sections."""
    # Remove required section
    del test_config_data['directories']
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
      yaml.dump(test_config_data, f)
      f.flush()
      
      try:
        with pytest.raises(ValueError, match="Missing required configuration section: directories"):
          EmailConfig(f.name)
      finally:
        os.unlink(f.name)


class TestEmailConfigGetMethod:
  """Test the generic get() method with dot notation."""
  
  def test_get_existing_value(self, mock_config):
    """Test getting existing configuration value."""
    result = mock_config.get('directories.base_dir')
    assert result is not None
    assert isinstance(result, str)
  
  def test_get_nested_value(self, mock_config):
    """Test getting nested configuration value."""
    result = mock_config.get('models.openai.spam_detection')
    assert result == 'gpt-4o-mini'
  
  def test_get_nonexistent_value_with_default(self, mock_config):
    """Test getting non-existent value returns default."""
    result = mock_config.get('nonexistent.path', 'default_value')
    assert result == 'default_value'
  
  def test_get_nonexistent_value_without_default(self, mock_config):
    """Test getting non-existent value returns None."""
    result = mock_config.get('nonexistent.path')
    assert result is None


class TestDirectoryMethods:
  """Test directory and path related methods."""
  
  def test_get_base_dir(self, mock_config):
    """Test base directory retrieval."""
    result = mock_config.get_base_dir()
    assert isinstance(result, str)
    assert len(result) > 0  # Just verify it returns a non-empty string
  
  def test_get_drafts_dir(self, mock_config):
    """Test drafts directory path construction."""
    result = mock_config.get_drafts_dir()
    assert isinstance(result, str)
    assert result.endswith('.Drafts/cur')
    assert mock_config.get_base_dir() in result
  
  def test_get_temp_dir(self, mock_config):
    """Test temporary directory retrieval."""
    result = mock_config.get_temp_dir()
    assert result == '/tmp'
  
  def test_get_timestamp_file(self, mock_config):
    """Test timestamp file name retrieval."""
    result = mock_config.get_timestamp_file()
    assert result == '.last_check'


class TestEmailProcessingMethods:
  """Test email processing configuration methods."""
  
  def test_get_ticket_pattern(self, mock_config):
    """Test ticket pattern regex retrieval."""
    result = mock_config.get_ticket_pattern()
    assert isinstance(result, str)
    assert '\\[#:' in result
  
  def test_get_unreplied_pattern(self, mock_config):
    """Test unreplied email pattern retrieval."""
    result = mock_config.get_unreplied_pattern()
    assert result == '*:2,S'
  
  def test_get_body_limit(self, mock_config):
    """Test email body limit retrieval."""
    result = mock_config.get_body_limit()
    assert isinstance(result, int)
    assert result > 0
  
  def test_get_analysis_limit(self, mock_config):
    """Test analysis limit retrieval."""
    result = mock_config.get_analysis_limit()
    assert isinstance(result, int)
    assert result > 0
  
  def test_get_hostname(self, mock_config):
    """Test hostname retrieval."""
    result = mock_config.get_hostname()
    assert result == 'test-host'
  
  def test_get_process_delay(self, mock_config):
    """Test process delay retrieval."""
    result = mock_config.get_process_delay()
    assert isinstance(result, (int, float))
    assert result >= 0


class TestAIModelMethods:
  """Test AI model configuration methods."""
  
  def test_get_openai_model(self, mock_config):
    """Test OpenAI model name retrieval."""
    result = mock_config.get_openai_model()
    assert result == 'gpt-4o-mini'
  
  def test_get_anthropic_model(self, mock_config):
    """Test Anthropic model name retrieval."""
    result = mock_config.get_anthropic_model()
    assert result == 'claude-3-5-haiku-20241022'
  
  def test_get_customkb_settings(self, mock_config):
    """Test CustomKB settings retrieval."""
    result = mock_config.get_customkb_settings()
    assert isinstance(result, dict)
    assert 'knowledge_base' in result
    assert 'role' in result
    assert 'timeout' in result
    assert result['knowledge_base'] == 'test-kb'
  
  def test_get_openai_params(self, mock_config):
    """Test OpenAI API parameters retrieval."""
    result = mock_config.get_openai_params()
    assert isinstance(result, dict)
    assert 'max_tokens' in result
    assert 'temperature' in result
    assert result['max_tokens'] == 10
    assert result['temperature'] == 0.1
  
  def test_get_anthropic_params(self, mock_config):
    """Test Anthropic API parameters retrieval."""
    result = mock_config.get_anthropic_params()
    assert isinstance(result, dict)
    assert 'max_tokens' in result
    assert 'temperature' in result


class TestTimeoutMethods:
  """Test timeout configuration methods."""
  
  def test_get_customkb_timeout(self, mock_config):
    """Test CustomKB timeout retrieval."""
    result = mock_config.get_customkb_timeout()
    assert isinstance(result, int)
    assert result > 0
  
  def test_get_find_timeout(self, mock_config):
    """Test find command timeout retrieval."""
    result = mock_config.get_find_timeout()
    assert isinstance(result, int)
    assert result > 0


class TestConsultantMethods:
  """Test consultant assignment and retrieval methods."""
  
  def test_get_consultant_existing_type(self, mock_config):
    """Test getting existing consultant by type."""
    result = mock_config.get_consultant('company')
    assert isinstance(result, dict)
    assert 'name' in result
    assert 'email' in result
    assert 'title' in result
    assert 'phone' in result
  
  def test_get_consultant_nonexistent_type(self, mock_config):
    """Test getting non-existent consultant type."""
    result = mock_config.get_consultant('nonexistent')
    assert result is None
  
  def test_get_all_consultants(self, mock_config):
    """Test getting all consultants."""
    result = mock_config.get_all_consultants()
    assert isinstance(result, dict)
    assert 'company' in result
    assert 'default' in result
  
  def test_determine_consultant_type_company_keywords(self, mock_config):
    """Test consultant type determination with company keywords."""
    content = "I need help setting up a PMA company in Indonesia"
    result = mock_config.determine_consultant_type(content)
    assert result == 'company'
  
  def test_determine_consultant_type_tax_keywords(self, mock_config):
    """Test consultant type determination with tax keywords."""
    content = "We need assistance with corporate tax compliance"
    result = mock_config.determine_consultant_type(content)
    assert result == 'tax'
  
  def test_determine_consultant_type_no_match(self, mock_config):
    """Test consultant type determination with no keyword match."""
    content = "This is some generic content without specific keywords"
    result = mock_config.determine_consultant_type(content)
    assert result == 'default'
  
  def test_determine_consultant_type_case_insensitive(self, mock_config):
    """Test consultant type determination is case insensitive."""
    content = "I need help with COMPANY formation and PMA setup"
    result = mock_config.determine_consultant_type(content)
    assert result == 'company'


class TestCountryMethods:
  """Test country code conversion methods."""
  
  def test_get_country_name_existing_code(self, mock_config):
    """Test getting country name for existing code."""
    result = mock_config.get_country_name('US')
    assert result == 'United States'
  
  def test_get_country_name_case_insensitive(self, mock_config):
    """Test country name lookup is case insensitive."""
    result = mock_config.get_country_name('us')
    assert result == 'United States'
  
  def test_get_country_name_nonexistent_code(self, mock_config):
    """Test getting country name for non-existent code."""
    result = mock_config.get_country_name('XX')
    assert result == 'Country Code: XX'


class TestPromptMethods:
  """Test prompt template formatting methods."""
  
  def test_get_spam_detection_prompt(self, mock_config):
    """Test spam detection prompt formatting."""
    content = "Test email content"
    result = mock_config.get_spam_detection_prompt(content)
    assert isinstance(result, str)
    assert content in result
  
  def test_get_customkb_query_prompt(self, mock_config):
    """Test CustomKB query prompt formatting."""
    sender_context = "Sender Location: Australia"
    original_email = "Test email content"
    result = mock_config.get_customkb_query_prompt(sender_context, original_email)
    assert isinstance(result, str)
    assert sender_context in result
    assert original_email in result
  
  def test_get_email_signature(self, mock_config):
    """Test email signature formatting."""
    consultant = {
      'name': 'Test Consultant',
      'title': 'Test Title',
      'phone': '+1234567890',
      'email': 'test@example.com'
    }
    result = mock_config.get_email_signature(consultant)
    assert isinstance(result, str)
    assert consultant['name'] in result


class TestMaildirFlagMethods:
  """Test Maildir flag manipulation methods."""
  
  def test_get_maildir_flag(self, mock_config):
    """Test getting Maildir flag character."""
    result = mock_config.get_maildir_flag('seen')
    assert result == 'S'
    
    result = mock_config.get_maildir_flag('replied')
    assert result == 'R'
  
  def test_has_maildir_flag_true(self, mock_config):
    """Test checking for existing Maildir flag."""
    filename = 'email:2,RS'
    assert mock_config.has_maildir_flag(filename, 'replied') is True
    assert mock_config.has_maildir_flag(filename, 'seen') is True
  
  def test_has_maildir_flag_false(self, mock_config):
    """Test checking for non-existent Maildir flag."""
    filename = 'email:2,S'
    assert mock_config.has_maildir_flag(filename, 'replied') is False
  
  def test_has_maildir_flag_no_flags_section(self, mock_config):
    """Test checking flag on filename without flags section."""
    filename = 'email_without_flags'
    assert mock_config.has_maildir_flag(filename, 'replied') is False
  
  def test_add_maildir_flag_new_flag(self, mock_config):
    """Test adding new Maildir flag."""
    filename = 'email:2,S'
    result = mock_config.add_maildir_flag(filename, 'replied')
    assert result == 'email:2,SR'
  
  def test_add_maildir_flag_existing_flag(self, mock_config):
    """Test adding already existing Maildir flag."""
    filename = 'email:2,SR'
    result = mock_config.add_maildir_flag(filename, 'replied')
    assert result == 'email:2,SR'  # Should not duplicate
  
  def test_add_maildir_flag_no_flags_section(self, mock_config):
    """Test adding flag to filename without flags section."""
    filename = 'email_without_flags'
    result = mock_config.add_maildir_flag(filename, 'replied')
    assert result == 'email_without_flags:2,R'
  
  def test_remove_maildir_flag_existing(self, mock_config):
    """Test removing existing Maildir flag."""
    filename = 'email:2,RST'
    result = mock_config.remove_maildir_flag(filename, 'replied')
    assert result == 'email:2,ST'
  
  def test_remove_maildir_flag_nonexistent(self, mock_config):
    """Test removing non-existent Maildir flag."""
    filename = 'email:2,S'
    result = mock_config.remove_maildir_flag(filename, 'replied')
    assert result == 'email:2,S'  # Should remain unchanged
  
  def test_remove_maildir_flag_no_flags_section(self, mock_config):
    """Test removing flag from filename without flags section."""
    filename = 'email_without_flags'
    result = mock_config.remove_maildir_flag(filename, 'replied')
    assert result == 'email_without_flags'  # Should remain unchanged


class TestGlobalConfigFunctions:
  """Test global configuration management functions."""
  
  def test_get_config_singleton_behavior(self, test_config_file):
    """Test that get_config returns same instance."""
    config1 = get_config(test_config_file)
    config2 = get_config(test_config_file)
    assert config1 is config2
  
  def test_reload_config_creates_new_instance(self, test_config_file):
    """Test that reload_config creates new instance."""
    config1 = get_config(test_config_file)
    config2 = reload_config(test_config_file)
    assert config1 is not config2
    assert isinstance(config2, EmailConfig)


class TestConfigurationEdgeCases:
  """Test edge cases and error conditions."""
  
  def test_config_with_minimal_required_sections(self):
    """Test config with only required sections."""
    minimal_config = {
      'directories': {'base_dir': '/test'},
      'email': {},
      'models': {},
      'consultants': {},
      'keywords': {},
      'countries': {},
      'prompts': {},
      'maildir': {}
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
      yaml.dump(minimal_config, f)
      f.flush()
      
      try:
        config = EmailConfig(f.name)
        assert config.get_base_dir() == '/test'
      finally:
        os.unlink(f.name)
  
  def test_get_method_with_malformed_path(self, mock_config):
    """Test get method with malformed dot notation path."""
    # Empty path
    result = mock_config.get('')
    assert result is None
    
    # Path with empty segments
    result = mock_config.get('directories..base_dir')
    assert result is None
  
  def test_consultant_determination_with_empty_content(self, mock_config):
    """Test consultant type determination with empty content."""
    result = mock_config.determine_consultant_type('')
    assert result == 'default'
  
  def test_consultant_determination_with_multiple_matches(self, mock_config):
    """Test consultant type determination with multiple keyword matches."""
    content = "We need help with company formation and tax compliance"
    result = mock_config.determine_consultant_type(content)
    # Should return first match found
    assert result in ['company', 'tax']

#fin