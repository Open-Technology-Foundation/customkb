#!/usr/bin/env python3
"""
Pytest configuration and shared fixtures for email processor tests.

This module provides common fixtures and configuration for all tests
in the Okusi Associates Email Auto-Reply System test suite.
"""

import pytest
import tempfile
import os
import shutil
from pathlib import Path
from unittest.mock import Mock, MagicMock
import yaml

# Add parent directory to Python path for imports
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config_loader import EmailConfig
from email_processor import EmailProcessor


@pytest.fixture
def temp_email_dir():
  """
  Create temporary Maildir structure for testing.
  
  Returns:
    str: Path to temporary email directory with proper Maildir structure
  """
  temp_dir = tempfile.mkdtemp()
  
  # Create Maildir structure
  maildir_dirs = ['cur', 'new', 'tmp', '.Drafts/cur', '.Drafts/new', '.Drafts/tmp']
  for dir_name in maildir_dirs:
    os.makedirs(os.path.join(temp_dir, dir_name), exist_ok=True)
  
  yield temp_dir
  
  # Cleanup
  shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def test_config_data():
  """
  Provide test configuration data structure.
  
  Returns:
    dict: Complete test configuration matching production structure
  """
  return {
    'directories': {
      'base_dir': '/tmp/test_email',
      'drafts_dir': '.Drafts/cur',
      'temp_dir': '/tmp',
      'timestamp_file': '.last_check'
    },
    'email': {
      'unreplied_pattern': '*:2,S',
      'ticket_pattern': '\\[#:[a-zA-Z0-9]+\\]',
      'body_limit': 1000,
      'analysis_limit': 500,
      'hostname': 'test-host',
      'file_permissions': '644',
      'process_delay': 0.1
    },
    'models': {
      'openai': {
        'spam_detection': 'gpt-4o-mini',
        'max_tokens': 10,
        'temperature': 0.1
      },
      'anthropic': {
        'spam_detection': 'claude-3-5-haiku-20241022',
        'max_tokens': 10,
        'temperature': 0.1
      },
      'customkb': {
        'knowledge_base': 'test-kb',
        'role': 'Test role',
        'timeout': 30
      }
    },
    'timeouts': {
      'customkb_query': 30,
      'find_command': 10
    },
    'consultants': {
      'company': {
        'name': 'Test Consultant',
        'email': 'test@example.com',
        'title': 'Test Title',
        'phone': '+1234567890'
      },
      'default': {
        'name': 'Default Consultant',
        'email': 'default@example.com',
        'title': 'Default Title',
        'phone': '+0987654321'
      }
    },
    'keywords': {
      'company': ['company', 'pma', 'corporation'],
      'tax': ['tax', 'taxation', 'accounting']
    },
    'countries': {
      'us': 'United States',
      'id': 'Indonesia'
    },
    'prompts': {
      'spam_detection': 'Test spam prompt with {content_for_analysis}',
      'customkb_query': 'Test query: {sender_context} {original_email}',
      'email_signature': 'Test signature: {consultant_name}'
    },
    'maildir': {
      'seen': 'S',
      'replied': 'R',
      'draft': 'D',
      'flagged': 'F',
      'trashed': 'T'
    }
  }


@pytest.fixture
def test_config_file(test_config_data, temp_email_dir):
  """
  Create temporary configuration file for testing.
  
  Args:
    test_config_data: Configuration data from fixture
    temp_email_dir: Temporary email directory from fixture
    
  Returns:
    str: Path to temporary configuration file
  """
  # Update base_dir to use temp directory
  test_config_data['directories']['base_dir'] = temp_email_dir
  
  config_file = os.path.join(temp_email_dir, 'test_config.yaml')
  with open(config_file, 'w') as f:
    yaml.dump(test_config_data, f, default_flow_style=False)
  
  return config_file


@pytest.fixture
def mock_config(test_config_file):
  """
  Provide EmailConfig instance with test configuration.
  
  Args:
    test_config_file: Path to test configuration file
    
  Returns:
    EmailConfig: Configured EmailConfig instance for testing
  """
  return EmailConfig(test_config_file)


@pytest.fixture
def mock_openai_client():
  """
  Mock OpenAI client for testing.
  
  Returns:
    Mock: Mocked OpenAI client with predefined responses
  """
  mock_client = Mock()
  mock_response = Mock()
  mock_response.choices = [Mock()]
  mock_response.choices[0].message.content = "LEGITIMATE"
  mock_client.chat.completions.create.return_value = mock_response
  return mock_client


@pytest.fixture
def mock_anthropic_client():
  """
  Mock Anthropic client for testing.
  
  Returns:
    Mock: Mocked Anthropic client with predefined responses
  """
  mock_client = Mock()
  mock_response = Mock()
  mock_response.content = [Mock()]
  mock_response.content[0].text = "LEGITIMATE"
  mock_client.messages.create.return_value = mock_response
  return mock_client


@pytest.fixture
def mock_ai_clients(mock_openai_client, mock_anthropic_client):
  """
  Provide both mocked AI clients.
  
  Returns:
    dict: Dictionary with both mocked AI clients
  """
  return {
    'openai': mock_openai_client,
    'anthropic': mock_anthropic_client
  }


@pytest.fixture
def sample_email_data():
  """
  Provide sample email data for testing.
  
  Returns:
    dict: Sample email data structure
  """
  return {
    'subject': 'Re: [#:TEST001] Company formation inquiry',
    'from': 'test@example.com',
    'to': 'contact@okusi.id',
    'date': 'Mon, 16 Jun 2025 12:00:00 +0700',
    'message_id': '<test@example.com>',
    'body': 'I need help setting up a PMA company in Indonesia.',
    'country': 'US',
    'ip_address': '192.168.1.1',
    'priority': '',
    'importance': ''
  }


@pytest.fixture
def sample_maildir_files(temp_email_dir):
  """
  Create sample Maildir files for testing.
  
  Args:
    temp_email_dir: Temporary email directory
    
  Returns:
    dict: Dictionary mapping file types to file paths
  """
  files = {}
  
  # Create unreplied email (read but not replied)
  unreplied_content = """From: test@example.com
To: contact@okusi.id
Subject: Re: [#:TEST001] Company formation inquiry
Date: Mon, 16 Jun 2025 12:00:00 +0700
Message-ID: <test@example.com>

I need help setting up a PMA company in Indonesia.
"""
  
  unreplied_file = os.path.join(temp_email_dir, 'cur', '1710754744.V811I.test:2,S')
  with open(unreplied_file, 'w') as f:
    f.write(unreplied_content)
  files['unreplied'] = unreplied_file
  
  # Create replied email
  replied_file = os.path.join(temp_email_dir, 'cur', '1710754745.V812I.test:2,RS')
  with open(replied_file, 'w') as f:
    f.write(unreplied_content)
  files['replied'] = replied_file
  
  # Create spam email (no ticket pattern)
  spam_content = """From: spam@example.com
To: contact@okusi.id
Subject: Amazing SEO services for your business!
Date: Mon, 16 Jun 2025 12:00:00 +0700

We offer the best SEO services in the industry!
"""
  
  spam_file = os.path.join(temp_email_dir, 'cur', '1710754746.V813I.test:2,S')
  with open(spam_file, 'w') as f:
    f.write(spam_content)
  files['spam'] = spam_file
  
  return files


@pytest.fixture
def mock_subprocess():
  """
  Mock subprocess.run for testing external command calls.
  
  Returns:
    Mock: Mocked subprocess.run function
  """
  mock_result = Mock()
  mock_result.returncode = 0
  mock_result.stdout = "Mock CustomKB response"
  mock_result.stderr = ""
  
  mock_run = Mock(return_value=mock_result)
  return mock_run


@pytest.fixture
def email_processor_with_mocks(mock_config, mock_ai_clients, mock_subprocess, monkeypatch):
  """
  Create EmailProcessor instance with all external dependencies mocked.
  
  Args:
    mock_config: Mocked configuration
    mock_ai_clients: Mocked AI clients
    mock_subprocess: Mocked subprocess
    monkeypatch: Pytest monkeypatch fixture
    
  Returns:
    EmailProcessor: Fully mocked EmailProcessor instance for testing
  """
  # Mock environment variables
  monkeypatch.setenv('OPENAI_API_KEY', 'test-openai-key')
  monkeypatch.setenv('ANTHROPIC_API_KEY', 'test-anthropic-key')
  
  # Mock subprocess.run
  monkeypatch.setattr('subprocess.run', mock_subprocess)
  
  # Create processor instance
  processor = EmailProcessor(config_path=mock_config.config_file)
  
  # Inject mocked AI clients
  processor.openai_client = mock_ai_clients['openai']
  processor.anthropic_client = mock_ai_clients['anthropic']
  
  return processor


# Pytest configuration
def pytest_configure(config):
  """Configure pytest with custom markers."""
  config.addinivalue_line(
    "markers", "integration: mark test as integration test"
  )
  config.addinivalue_line(
    "markers", "unit: mark test as unit test"  
  )
  config.addinivalue_line(
    "markers", "slow: mark test as slow running"
  )


def pytest_collection_modifyitems(config, items):
  """Automatically mark tests based on their location."""
  for item in items:
    # Mark tests in integration directory
    if "integration" in str(item.fspath):
      item.add_marker(pytest.mark.integration)
    # Mark tests in unit directory  
    elif "unit" in str(item.fspath):
      item.add_marker(pytest.mark.unit)

#fin