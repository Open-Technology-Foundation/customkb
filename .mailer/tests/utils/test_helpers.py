#!/usr/bin/env python3
"""
Test helper utilities for email processor tests.

This module provides common utility functions and helpers used across
multiple test modules to reduce duplication and improve maintainability.
"""

import os
import tempfile
import time
from pathlib import Path
from typing import Dict, List


def create_test_email_file(directory: str, filename: str, content: str) -> str:
  """
  Create a test email file in Maildir format.
  
  Args:
    directory: Directory to create file in
    filename: Name of the email file
    content: Email content in RFC 2822 format
    
  Returns:
    str: Full path to created email file
  """
  filepath = os.path.join(directory, filename)
  with open(filepath, 'w', encoding='utf-8') as f:
    f.write(content)
  return filepath


def generate_maildir_filename(timestamp: int = None, unique_id: str = None, 
                             hostname: str = 'test', flags: str = 'S') -> str:
  """
  Generate a properly formatted Maildir filename.
  
  Args:
    timestamp: Unix timestamp (defaults to current time)
    unique_id: Unique identifier (defaults to generated)
    hostname: Hostname (defaults to 'test')
    flags: Maildir flags (defaults to 'S')
    
  Returns:
    str: Properly formatted Maildir filename
  """
  if timestamp is None:
    timestamp = int(time.time())
  
  if unique_id is None:
    unique_id = f"V{hex(timestamp)[2:]}M{os.getpid()}"
  
  return f"{timestamp}.{unique_id}.{hostname}:2,{flags}"


def create_sample_email_content(subject: str, sender: str, body: str, 
                               ticket_id: str = None) -> str:
  """
  Create RFC 2822 formatted email content for testing.
  
  Args:
    subject: Email subject line
    sender: Sender email address
    body: Email body content
    ticket_id: Optional ticket ID to include in subject
    
  Returns:
    str: Complete RFC 2822 email content
  """
  if ticket_id:
    subject = f"Re: [#:{ticket_id}] {subject}"
  
  return f"""From: {sender}
To: contact@okusi.id
Subject: {subject}
Date: Mon, 16 Jun 2025 12:00:00 +0700
Message-ID: <{int(time.time())}@{sender.split('@')[1]}>
MIME-Version: 1.0
Content-Type: text/plain; charset=utf-8

{body}
"""


def create_test_emails_with_scenarios(base_dir: str) -> Dict[str, str]:
  """
  Create a set of test emails covering various scenarios.
  
  Args:
    base_dir: Base directory to create emails in
    
  Returns:
    dict: Mapping of scenario names to email file paths
  """
  scenarios = {}
  cur_dir = os.path.join(base_dir, 'cur')
  
  # Legitimate business inquiry (company formation)
  company_content = create_sample_email_content(
    subject="Company formation inquiry",
    sender="business@example.com",
    body="I need help setting up a PMA company in Indonesia for my tech startup.",
    ticket_id="COMP001"
  )
  company_file = generate_maildir_filename(flags='S')
  scenarios['legitimate_company'] = create_test_email_file(
    cur_dir, company_file, company_content
  )
  
  # Tax inquiry
  tax_content = create_sample_email_content(
    subject="Tax consultation needed",
    sender="accountant@company.com", 
    body="We need assistance with Indonesian corporate tax compliance.",
    ticket_id="TAX001"
  )
  tax_file = generate_maildir_filename(flags='S')
  scenarios['legitimate_tax'] = create_test_email_file(
    cur_dir, tax_file, tax_content
  )
  
  # Visa inquiry
  visa_content = create_sample_email_content(
    subject="Work permit application",
    sender="hr@multinational.com",
    body="Our expat employee needs a work permit for Indonesia.",
    ticket_id="VISA001"
  )
  visa_file = generate_maildir_filename(flags='S')
  scenarios['legitimate_visa'] = create_test_email_file(
    cur_dir, visa_file, visa_content
  )
  
  # Spam email (no ticket pattern)
  spam_content = create_sample_email_content(
    subject="Amazing SEO services for your business!",
    sender="spam@marketing.com",
    body="Boost your business with our incredible SEO packages! Call now!"
  )
  spam_file = generate_maildir_filename(flags='S')
  scenarios['spam_no_ticket'] = create_test_email_file(
    cur_dir, spam_file, spam_content
  )
  
  # Already replied email
  replied_content = create_sample_email_content(
    subject="Director services inquiry",
    sender="client@business.com",
    body="We need a corporate secretary for our Indonesian company.",
    ticket_id="DIR001"
  )
  replied_file = generate_maildir_filename(flags='RS')
  scenarios['already_replied'] = create_test_email_file(
    cur_dir, replied_file, replied_content
  )
  
  # Malformed email (missing headers)
  malformed_content = "This is not a proper email format"
  malformed_file = generate_maildir_filename(flags='S')
  scenarios['malformed'] = create_test_email_file(
    cur_dir, malformed_file, malformed_content
  )
  
  return scenarios


def assert_maildir_file_exists(filepath: str, expected_flags: str = None):
  """
  Assert that a Maildir file exists and optionally check its flags.
  
  Args:
    filepath: Path to Maildir file
    expected_flags: Expected Maildir flags (optional)
    
  Raises:
    AssertionError: If file doesn't exist or flags don't match
  """
  assert os.path.exists(filepath), f"Maildir file does not exist: {filepath}"
  
  if expected_flags:
    filename = os.path.basename(filepath)
    if ':2,' in filename:
      actual_flags = filename.split(':2,')[-1]
      assert expected_flags in actual_flags, \
        f"Expected flags '{expected_flags}' not found in '{actual_flags}'"


def count_files_with_pattern(directory: str, pattern: str) -> int:
  """
  Count files in directory matching a pattern.
  
  Args:
    directory: Directory to search
    pattern: File pattern to match (shell glob)
    
  Returns:
    int: Number of matching files
  """
  import glob
  search_pattern = os.path.join(directory, pattern)
  return len(glob.glob(search_pattern))


def extract_maildir_flags(filename: str) -> str:
  """
  Extract Maildir flags from filename.
  
  Args:
    filename: Maildir filename
    
  Returns:
    str: Maildir flags (empty string if no flags)
  """
  if ':2,' in filename:
    return filename.split(':2,')[-1]
  return ''


def create_temp_timestamp_file(directory: str, timestamp: float = None) -> str:
  """
  Create a timestamp file for testing incremental processing.
  
  Args:
    directory: Directory to create timestamp file in
    timestamp: Timestamp to write (defaults to current time)
    
  Returns:
    str: Path to created timestamp file
  """
  if timestamp is None:
    timestamp = time.time()
  
  timestamp_file = os.path.join(directory, '.last_check')
  with open(timestamp_file, 'w') as f:
    f.write(str(timestamp))
  
  return timestamp_file


def mock_customkb_response(query_content: str = None) -> str:
  """
  Generate mock CustomKB response based on query content.
  
  Args:
    query_content: Content of the query (optional)
    
  Returns:
    str: Mock professional email response
  """
  if query_content and 'company' in query_content.lower():
    return """Thank you for your inquiry about company formation in Indonesia.

We can assist you with establishing a PMA (Foreign Investment Company) in Indonesia. Our services include:
- Legal structure consultation
- Documentation preparation  
- Government registration
- Ongoing compliance support

Please let us know your specific requirements and we'll provide a detailed proposal.

Best regards,
Okusi Associates Legal Team"""
  
  return """Thank you for contacting Okusi Associates.

We have received your inquiry and our team will review your requirements. We'll provide you with a comprehensive response within 24 hours.

If you have any urgent questions, please don't hesitate to contact us directly.

Best regards,
Okusi Associates Legal Team"""


class MockAIResponse:
  """Helper class for creating consistent AI API response mocks."""
  
  @staticmethod
  def openai_spam_response(is_legitimate: bool = True):
    """Create mock OpenAI response for spam detection."""
    from unittest.mock import Mock
    
    response = Mock()
    response.choices = [Mock()]
    response.choices[0].message.content = "LEGITIMATE" if is_legitimate else "SPAM"
    return response
  
  @staticmethod
  def anthropic_spam_response(is_legitimate: bool = True):
    """Create mock Anthropic response for spam detection."""
    from unittest.mock import Mock
    
    response = Mock()
    response.content = [Mock()]
    response.content[0].text = "LEGITIMATE" if is_legitimate else "SPAM"
    return response

#fin