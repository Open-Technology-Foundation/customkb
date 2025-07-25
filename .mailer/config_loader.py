#!/usr/bin/env python3
"""
Configuration loader for the Okusi Associates Email Auto-Reply System.

This module provides comprehensive configuration management for the email processing
system, loading settings from YAML files and providing type-safe access methods.

Key Features:
  - YAML-based configuration with validation
  - Hierarchical configuration access using dot notation
  - Type-safe configuration getters for all system components
  - Consultant assignment and routing logic
  - Maildir flag manipulation utilities
  - Prompt template formatting with variable substitution

Typical Usage:
  from config_loader import get_config
  
  config = get_config()
  base_dir = config.get_base_dir()
  consultant = config.get_consultant('company')
  model = config.get_openai_model()
"""

import os
import yaml
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class EmailConfig:
  """
  Configuration manager for the email processor system.
  
  Loads and manages all system settings from YAML configuration files,
  providing type-safe access methods and validation.
  
  The configuration is organized into sections:
  - directories: File paths and directories
  - email: Email processing settings
  - models: AI model configurations
  - consultants: Consultant database with contact information
  - keywords: Email classification keywords
  - prompts: Template strings for AI interactions
  - maildir: Maildir flag definitions
  
  Attributes:
    config_file (Path): Path to the YAML configuration file
    config (dict): Loaded configuration data
  """
  
  def __init__(self, config_file=None):
    """
    Initialize configuration loader.
    
    Args:
      config_file (str, optional): Path to configuration file.
                                  Defaults to email_config.yaml in script directory.
    
    Raises:
      FileNotFoundError: If configuration file doesn't exist
      yaml.YAMLError: If YAML parsing fails
      ValueError: If required configuration sections are missing
    """
    if config_file is None:
      # Default to config file in same directory as this script
      config_file = Path(__file__).parent / "email_config.yaml"
    
    self.config_file = Path(config_file)
    self.config = self._load_config()
    self._validate_config()
  
  def _load_config(self):
    """
    Load configuration from YAML file.
    
    Returns:
      dict: Parsed configuration data
    
    Raises:
      FileNotFoundError: If configuration file doesn't exist
      yaml.YAMLError: If YAML parsing fails
      Exception: For other file loading errors
    """
    try:
      with open(self.config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
      logger.info(f"Loaded configuration from {self.config_file}")
      return config
    except FileNotFoundError:
      logger.error(f"Configuration file not found: {self.config_file}")
      raise
    except yaml.YAMLError as e:
      logger.error(f"Error parsing YAML configuration: {e}")
      raise
    except Exception as e:
      logger.error(f"Error loading configuration: {e}")
      raise
  
  def _validate_config(self):
    """
    Validate that required configuration sections exist.
    
    This method ensures all critical configuration sections are present
    to prevent runtime errors from missing configuration.
    
    Raises:
      ValueError: If any required section is missing
    """
    required_sections = [
      'directories', 'email', 'models', 'consultants', 
      'keywords', 'countries', 'prompts', 'maildir'
    ]
    
    for section in required_sections:
      if section not in self.config:
        raise ValueError(f"Missing required configuration section: {section}")
    
    logger.debug("Configuration validation passed")
  
  def get(self, path, default=None):
    """
    Get configuration value using dot notation.
    
    Supports hierarchical access like 'models.openai.spam_detection' to
    access nested configuration values.
    
    Args:
      path (str): Dot-separated path to configuration value
      default: Default value if path not found
    
    Returns:
      Configuration value at path, or default if not found
    
    Example:
      >>> config.get('models.openai.spam_detection')
      'gpt-4o-mini'
      >>> config.get('nonexistent.path', 'fallback')
      'fallback'
    """
    try:
      keys = path.split('.')
      value = self.config
      for key in keys:
        value = value[key]
      return value
    except (KeyError, TypeError):
      return default
  
  # Directory and path methods
  def get_base_dir(self):
    """Get base email directory."""
    return self.get('directories.base_dir')
  
  def get_drafts_dir(self):
    """
    Get full path to drafts directory.
    
    Combines base_dir with drafts_dir to create absolute path.
    
    Returns:
      str: Absolute path to email drafts directory
    """
    base = self.get_base_dir()
    drafts = self.get('directories.drafts_dir')
    return os.path.join(base, drafts)
  
  def get_temp_dir(self):
    """Get temporary directory."""
    return self.get('directories.temp_dir')
  
  def get_timestamp_file(self):
    """Get timestamp file name."""
    return self.get('directories.timestamp_file')
  
  # Email processing methods
  def get_ticket_pattern(self):
    """Get regex pattern for ticket detection."""
    return self.get('email.ticket_pattern')
  
  def get_unreplied_pattern(self):
    """Get file pattern for unreplied emails."""
    return self.get('email.unreplied_pattern')
  
  def get_body_limit(self):
    """Get email body processing limit."""
    return self.get('email.body_limit', 2000)
  
  def get_analysis_limit(self):
    """Get content analysis limit."""
    return self.get('email.analysis_limit', 1000)
  
  def get_hostname(self):
    """Get system hostname."""
    return self.get('email.hostname', 'okusi0')
  
  def get_file_permissions(self):
    """Get file permissions for created files."""
    return self.get('email.file_permissions', '644')
  
  
  def get_process_delay(self):
    """Get delay between processing emails."""
    return self.get('email.process_delay', 0.5)
  
  # Model configuration methods
  def get_openai_model(self):
    """Get OpenAI model for spam detection."""
    return self.get('models.openai.spam_detection', 'gpt-4o-mini')
  
  def get_anthropic_model(self):
    """Get Anthropic model for spam detection."""
    return self.get('models.anthropic.spam_detection', 'claude-3-5-haiku-20241022')
  
  def get_customkb_settings(self):
    """Get CustomKB configuration."""
    return {
      'knowledge_base': self.get('models.customkb.knowledge_base', 'okusimail'),
      'role': self.get('models.customkb.role', 'You are a professional legal services consultant.'),
      'timeout': self.get('models.customkb.timeout', 120)
    }
  
  def get_openai_params(self):
    """Get OpenAI API parameters."""
    return {
      'max_tokens': self.get('models.openai.max_tokens', 10),
      'temperature': self.get('models.openai.temperature', 0.1)
    }
  
  def get_anthropic_params(self):
    """Get Anthropic API parameters."""
    return {
      'max_tokens': self.get('models.anthropic.max_tokens', 10),
      'temperature': self.get('models.anthropic.temperature', 0.1)
    }
  
  # Timeout methods
  def get_customkb_timeout(self):
    """Get CustomKB query timeout."""
    return self.get('timeouts.customkb_query', 120)
  
  def get_find_timeout(self):
    """Get find command timeout."""
    return self.get('timeouts.find_command', 30)
  
  # Consultant methods
  def get_consultant(self, consultant_type):
    """Get consultant information by type."""
    return self.get(f'consultants.{consultant_type}')
  
  def get_all_consultants(self):
    """Get all consultant information."""
    return self.get('consultants', {})
  
  def determine_consultant_type(self, content):
    """
    Determine consultant type based on email content.
    
    Analyzes email content against configured keywords to route
    inquiries to appropriate specialists.
    
    Args:
      content (str): Email subject and body content
    
    Returns:
      str: Consultant type ('company', 'tax', 'visa', 'director', or 'default')
    
    Example:
      >>> config.determine_consultant_type('PMA company setup')
      'company'
      >>> config.determine_consultant_type('visa application')
      'visa'
    """
    content_lower = content.lower()
    keywords = self.get('keywords', {})
    
    for consultant_type, keyword_list in keywords.items():
      if any(keyword in content_lower for keyword in keyword_list):
        return consultant_type
    
    return 'default'
  
  # Country methods
  def get_country_name(self, country_code):
    """
    Convert country code to country name.
    
    Args:
      country_code (str): Two-letter country code (e.g., 'US', 'ID')
    
    Returns:
      str: Full country name or formatted code if not found
    
    Example:
      >>> config.get_country_name('US')
      'United States'
      >>> config.get_country_name('XX')
      'Country Code: XX'
    """
    countries = self.get('countries', {})
    return countries.get(country_code.lower(), f"Country Code: {country_code.upper()}")
  
  # Prompt methods
  def get_spam_detection_prompt(self, content_for_analysis):
    """
    Get formatted spam detection prompt.
    
    Formats the spam detection prompt template with email content
    for AI analysis.
    
    Args:
      content_for_analysis (str): Email content to analyze
    
    Returns:
      str: Formatted prompt ready for AI processing
    """
    template = self.get('prompts.spam_detection', '')
    return template.format(content_for_analysis=content_for_analysis)
  
  def get_customkb_query_prompt(self, sender_context, original_email):
    """
    Get formatted CustomKB query prompt.
    
    Formats the CustomKB query template with sender context and
    original email for professional reply generation.
    
    Args:
      sender_context (str): Information about the email sender
      original_email (str): Original email content
    
    Returns:
      str: Formatted prompt for CustomKB query
    """
    template = self.get('prompts.customkb_query', '')
    return template.format(
      sender_context=sender_context,
      original_email=original_email
    )
  
  def get_email_signature(self, consultant):
    """
    Get formatted email signature.
    
    Formats the email signature template with consultant information.
    
    Args:
      consultant (dict): Consultant info with name, title, phone, email
    
    Returns:
      str: Formatted email signature
    """
    template = self.get('prompts.email_signature', '')
    return template.format(
      consultant_name=consultant['name'],
      consultant_title=consultant['title'],
      consultant_phone=consultant['phone'],
      consultant_email=consultant['email']
    )
  
  # Maildir flag methods
  def get_maildir_flag(self, flag_name):
    """Get Maildir flag character."""
    return self.get(f'maildir.{flag_name}')
  
  def has_maildir_flag(self, filename, flag_name):
    """
    Check if filename contains specific Maildir flag.
    
    Args:
      filename (str): Maildir filename
      flag_name (str): Flag name ('seen', 'replied', 'draft', etc.)
    
    Returns:
      bool: True if flag is present in filename
    
    Example:
      >>> config.has_maildir_flag('email:2,RS', 'replied')
      True
    """
    flag = self.get_maildir_flag(flag_name)
    if not flag or ':2,' not in filename:
      return False
    flags_part = filename.split(':2,')[-1]
    return flag in flags_part
  
  def add_maildir_flag(self, filename, flag_name):
    """
    Add Maildir flag to filename.
    
    Args:
      filename (str): Maildir filename
      flag_name (str): Flag name to add
    
    Returns:
      str: Filename with flag added
    
    Example:
      >>> config.add_maildir_flag('email:2,S', 'replied')
      'email:2,SR'
    """
    flag = self.get_maildir_flag(flag_name)
    if not flag:
      return filename
    
    if ':2,' in filename:
      base, flags_part = filename.rsplit(':2,', 1)
      if flag not in flags_part:
        flags_part += flag
      return f"{base}:2,{flags_part}"
    else:
      return f"{filename}:2,{flag}"
  
  def remove_maildir_flag(self, filename, flag_name):
    """
    Remove Maildir flag from filename.
    
    Args:
      filename (str): Maildir filename
      flag_name (str): Flag name to remove
    
    Returns:
      str: Filename with flag removed
    
    Example:
      >>> config.remove_maildir_flag('email:2,RS', 'replied')
      'email:2,S'
    """
    flag = self.get_maildir_flag(flag_name)
    if not flag or ':2,' not in filename:
      return filename
    
    base, flags_part = filename.rsplit(':2,', 1)
    flags_part = flags_part.replace(flag, '')
    return f"{base}:2,{flags_part}"

# Global configuration instance
_config_instance = None

def get_config(config_file=None):
  """
  Get global configuration instance.
  
  Uses singleton pattern to ensure only one configuration instance
  exists throughout the application.
  
  Args:
    config_file (str, optional): Path to configuration file
  
  Returns:
    EmailConfig: Global configuration instance
  """
  global _config_instance
  if _config_instance is None:
    _config_instance = EmailConfig(config_file)
  return _config_instance

def reload_config(config_file=None):
  """
  Reload configuration (useful for testing).
  
  Forces creation of new configuration instance, bypassing singleton.
  
  Args:
    config_file (str, optional): Path to configuration file
  
  Returns:
    EmailConfig: New configuration instance
  """
  global _config_instance
  _config_instance = EmailConfig(config_file)
  return _config_instance

#fin