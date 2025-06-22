#!/usr/bin/env python3

import sys
import os
import re
import json
import email
import time
import glob
import argparse
import tempfile
import subprocess
from datetime import datetime
from email.policy import default
from pathlib import Path
import logging
from config_loader import get_config

# Set up logging
def setup_logging(verbose=False, log_file=None):
  """Set up logging configuration."""
  log_level = logging.DEBUG if verbose else logging.INFO
  log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
  
  # Configure root logger
  logging.basicConfig(
    level=log_level,
    format=log_format,
    handlers=[]
  )
  
  logger = logging.getLogger(__name__)
  
  # Console handler
  console_handler = logging.StreamHandler()
  console_handler.setLevel(log_level)
  console_formatter = logging.Formatter(log_format)
  console_handler.setFormatter(console_formatter)
  logger.addHandler(console_handler)
  
  # File handler if specified
  if log_file:
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(log_format)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
  
  return logger

class EmailProcessor:
  """Unified email processing system combining evaluation and reply generation."""
  
  def __init__(self, logger=None, config_path=None):
    """Initialize the email processor."""
    self.logger = logger or logging.getLogger(__name__)
    self.config = get_config(config_path)
    self.base_dir = self.config.get_base_dir()
    self.timestamp_file = os.path.join(self.base_dir, self.config.get_timestamp_file())
    self.drafts_dir = self.config.get_drafts_dir()
    
    # Statistics
    self.stats = {
      'total_found': 0,
      'already_replied': 0,
      'no_ticket_pattern': 0,
      'spam_detected': 0,
      'reply_generated': 0,
      'errors': 0
    }
    
    # Initialize evaluation components (from email_evaluator.py)
    self._init_ai_clients()
    

  def _init_ai_clients(self):
    """Initialize AI clients for spam detection."""
    try:
      self.anthropic_key = os.environ.get('ANTHROPIC_API_KEY')
      self.openai_key = os.environ.get('OPENAI_API_KEY')
      
      # Try to initialize Anthropic client
      if self.anthropic_key:
        try:
          import anthropic
          self.anthropic_client = anthropic.Anthropic(api_key=self.anthropic_key)
          self.logger.info("Anthropic client initialized successfully")
        except ImportError as e:
          self.logger.warning(f"Anthropic client unavailable: {e}")
      
      # Try to initialize OpenAI client  
      if self.openai_key:
        try:
          from openai import OpenAI
          self.openai_client = OpenAI(api_key=self.openai_key)
          self.logger.info("OpenAI client initialized successfully")
        except ImportError as e:
          self.logger.warning(f"OpenAI client unavailable: {e}")
      
      # Verify at least one AI client is operational for spam detection
      if not hasattr(self, 'anthropic_client') and not hasattr(self, 'openai_client'):
        if not self.anthropic_key and not self.openai_key:
          self.logger.warning("No API keys found - AI spam detection will be disabled")
        else:
          self.logger.warning("API keys found but clients unavailable - AI spam detection will be disabled")
      
    except Exception as e:
      self.logger.error(f"Failed to initialize AI clients: {e}")
      self.logger.warning("Continuing without AI spam detection")

  def get_last_check_timestamp(self):
    """Get the timestamp of the last check."""
    try:
      timestamp_path = self.config.get_timestamp_file()
      if os.path.exists(timestamp_path):
        with open(timestamp_path, 'r') as f:
          return float(f.read().strip())
      return 0  # Start of epoch if no previous check
    except Exception as e:
      self.logger.warning(f"Could not read last check timestamp: {e}")
      return 0

  def update_last_check_timestamp(self, timestamp=None):
    """Update the last check timestamp."""
    try:
      timestamp = timestamp or time.time()
      timestamp_path = self.config.get_timestamp_file()
      with open(timestamp_path, 'w') as f:
        f.write(str(timestamp))
      self.logger.debug(f"Updated last check timestamp to {timestamp}")
    except Exception as e:
      self.logger.error(f"Could not update timestamp: {e}")

  def discover_emails(self, since_timestamp=None, full_scan=False):
    """Discover emails that need processing."""
    if full_scan:
      since_timestamp = 0
    elif since_timestamp is None:
      since_timestamp = self.get_last_check_timestamp()
    
    self.logger.info(f"Scanning for emails modified since {datetime.fromtimestamp(since_timestamp)}")
    
    email_files = []
    
    # Scan both 'cur' (read emails) and 'new' (unread emails) directories
    for subdir in ['cur', 'new']:
      try:
        # Use find command to get files without R flag
        file_pattern = self.config.get_unreplied_pattern()
        find_timeout = self.config.get_find_timeout()
        cmd = ['find', f"{self.base_dir}/{subdir}", '-type', 'f', '-name', file_pattern]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=find_timeout)
        
        if result.returncode != 0:
          self.logger.warning(f"Find command failed for {subdir}: {result.stderr}")
          continue
        
        for line in result.stdout.strip().split('\n'):
          if not line:
            continue
          
          filepath = line.strip()
          try:
            # Check modification time
            mtime = os.path.getmtime(filepath)
            if mtime <= since_timestamp:
              continue
            
            # Double-check for replied flag to avoid reprocessing
            filename = os.path.basename(filepath)
            if self._has_replied_flag(filename):
              self.stats['already_replied'] += 1
              continue
            
            email_files.append({
              'path': filepath,
              'mtime': mtime,
              'subdir': subdir
            })
            
          except Exception as e:
            self.logger.warning(f"Error checking file {filepath}: {e}")
      
      except Exception as e:
        self.logger.error(f"Error scanning {subdir} directory: {e}")
    
    # Sort by modification time (oldest first)
    email_files.sort(key=lambda x: x['mtime'])
    
    self.stats['total_found'] = len(email_files)
    self.logger.info(f"Found {len(email_files)} emails to process")
    
    return email_files

  def _has_replied_flag(self, filename):
    """Check if filename contains reply flag (R)."""
    if ':2,' in filename:
      flags = filename.split(':2,')[-1]
      return 'R' in flags
    return False

  def parse_email_file(self, filepath):
    """Parse email file and extract key information."""
    try:
      with open(filepath, 'rb') as f:
        msg = email.message_from_bytes(f.read(), policy=default)
      
      email_data = {
        'subject': msg.get('Subject', ''),
        'from': msg.get('From', ''),
        'to': msg.get('To', ''),
        'date': msg.get('Date', ''),
        'message_id': msg.get('Message-ID', ''),
        'body': self._extract_body(msg),
        # Additional metadata for context
        'country': msg.get('X-Country', ''),
        'ip_address': msg.get('X-IP', ''),
        'priority': msg.get('X-Priority', ''),
        'importance': msg.get('Importance', '')
      }
      
      return email_data
      
    except Exception as e:
      self.logger.error(f"Error parsing email file {filepath}: {e}")
      return None

  def _extract_body(self, msg):
    """Extract email body content."""
    try:
      body = ""
      
      if msg.is_multipart():
        for part in msg.walk():
          if part.get_content_type() == "text/plain":
            body = part.get_content()
            break
          elif part.get_content_type() == "text/html" and not body:
            body = part.get_content()
      else:
        body = msg.get_content()
      
      if body:
        body = re.sub(r'\n\s*\n\s*\n', '\n\n', body)
        body = body.strip()
      
      body_limit = self.config.get_body_limit()
      return body[:body_limit]  # Limit for efficiency
      
    except Exception as e:
      self.logger.error(f"Error extracting email body: {e}")
      return ""

  def has_ticket_pattern(self, subject):
    """Check if subject contains ticket pattern [#:alphanumeric] and is not a reply."""
    if not subject:
      return False
    
    # Skip emails that are already replies (contain "Re:" at start)
    if re.match(r'^Re:\s*', subject, re.IGNORECASE):
      return False
    
    # Pattern: [#:followed by alphanumeric characters] 
    pattern = self.config.get_ticket_pattern()
    return bool(re.search(pattern, subject))

  def is_legitimate_email(self, email_data):
    """Use GPT-4o-mini to determine if email is legitimate business communication."""
    try:
      # Prepare content for analysis
      content_limit = self.config.get_analysis_limit()
      content_for_analysis = f"""
Subject: {email_data['subject']}
From: {email_data['from']}
Content: {email_data['body'][:content_limit]}
"""
      
      # Create prompt for spam detection based on actual spam patterns
      prompt = self.config.get_spam_detection_prompt(content_for_analysis)

      # Try GPT-4o-mini first, fallback to Claude Haiku
      if hasattr(self, 'openai_client'):
        response = self.openai_client.chat.completions.create(
          model=self.config.get_openai_model(),
          messages=[
            {"role": "user", "content": prompt}
          ],
          max_tokens=self.config.get_openai_params()['max_tokens'],
          temperature=self.config.get_openai_params()['temperature']
        )
        result = response.choices[0].message.content.strip().upper()
        self.logger.debug(f"GPT-4o-mini spam detection result: {result}")
      elif hasattr(self, 'anthropic_client'):
        response = self.anthropic_client.messages.create(
          model=self.config.get_anthropic_model(),
          max_tokens=self.config.get_anthropic_params()['max_tokens'],
          temperature=self.config.get_anthropic_params()['temperature'],
          messages=[
            {"role": "user", "content": prompt}
          ]
        )
        result = response.content[0].text.strip().upper()
        self.logger.debug(f"Claude Haiku spam detection result: {result}")
      else:
        self.logger.warning("No AI client available, defaulting to LEGITIMATE")
        return True
      
      return result == "LEGITIMATE"
      
    except Exception as e:
      self.logger.error(f"Error in spam detection: {e}")
      # Default to legitimate if API fails (conservative approach)
      return True

  def evaluate_email(self, filepath):
    """Evaluate if email needs a reply based on all criteria."""
    try:
      filename = os.path.basename(filepath)
      
      # Check 1: Already replied?
      if self._has_replied_flag(filename):
        return {
          "needs_reply": False,
          "reason": "Email already has reply flag (R)",
          "email_data": None
        }
      
      # Check 2: Parse email content
      email_data = self.parse_email_file(filepath)
      if not email_data:
        return {
          "needs_reply": False,
          "reason": "Failed to parse email file",
          "email_data": None
        }
      
      # Check 3: Has ticket pattern in subject?
      if not self.has_ticket_pattern(email_data['subject']):
        return {
          "needs_reply": False,
          "reason": f"Subject does not contain ticket pattern [#:numbers]: '{email_data['subject']}'",
          "email_data": email_data
        }
      
      # Check 4: Is legitimate (not spam)?
      if not self.is_legitimate_email(email_data):
        return {
          "needs_reply": False,
          "reason": "Email detected as spam/promotional content",
          "email_data": email_data
        }
      
      # All checks passed
      return {
        "needs_reply": True,
        "reason": "Email meets all criteria: not replied, has ticket pattern, legitimate content",
        "email_data": email_data
      }
      
    except Exception as e:
      self.logger.error(f"Error evaluating email {filepath}: {e}")
      return {
        "needs_reply": False,
        "reason": f"Evaluation error: {e}",
        "email_data": None
      }

  def determine_consultant(self, email_data):
    """Determine appropriate consultant based on email content."""
    subject = email_data['subject']
    body = email_data['body']
    content = f"{subject} {body}"
    
    consultant_type = self.config.determine_consultant_type(content)
    return self.config.get_consultant(consultant_type)

  def construct_query(self, email_data):
    """Construct query for CustomKB."""
    # Build sender context
    sender_context = ""
    if email_data.get('country'):
      country_name = self.config.get_country_name(email_data['country'])
      sender_context += f"Sender Location: {country_name}\n"
    
    if email_data.get('ip_address'):
      sender_context += f"IP Address: {email_data['ip_address']}\n"
    
    if email_data.get('priority') or email_data.get('importance'):
      priority_info = email_data.get('importance', '') or f"Priority {email_data.get('priority', '')}"
      sender_context += f"Priority: {priority_info}\n"
    
    original_email = f"""
Original Email:
From: {email_data['from']}
Subject: {email_data['subject']}
Date: {email_data['date']}
{sender_context}
Message:
{email_data['body']}
"""
    
    sender_context_str = sender_context.strip() if sender_context else 'No additional sender context available.'
    query = self.config.get_customkb_query_prompt(sender_context_str, original_email)
    
    return query


  def call_customkb(self, query):
    """Call CustomKB with the constructed query."""
    try:
      settings = self.config.get_customkb_settings()
      # Call CustomKB query command directly with query text
      cmd = [
        'customkb', 
        'query', 
        settings['knowledge_base'],
        query,
        '-R', settings['role'],
        '--quiet'
      ]
      
      self.logger.debug(f"Calling CustomKB: customkb query {settings['knowledge_base']} [query] -R [role] --quiet")
      
      # Run with proper environment
      timeout = self.config.get_customkb_timeout()
      result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout
      )
      
      if result.returncode != 0:
        self.logger.error(f"CustomKB query failed: {result.stderr}")
        return None
      
      return result.stdout.strip()
      
    except Exception as e:
      self.logger.error(f"Error calling CustomKB: {e}")
      return None

  def generate_email_draft(self, email_data, reply_content, consultant):
    """Generate RFC 2822 formatted email draft."""
    try:
      # Extract original message ID for threading
      original_msg_id = email_data.get('message_id', '')
      
      # Generate new message ID
      timestamp = int(time.time())
      new_msg_id = f"<{timestamp}.reply.okusi.id>"
      
      # Prepare subject with Re: prefix (only if not already present)
      original_subject = email_data['subject']
      if not re.match(r'^Re:\s*', original_subject, re.IGNORECASE):
        reply_subject = f"Re: {original_subject}"
      else:
        reply_subject = original_subject
      
      # Create email headers
      headers = [
        f"From: {consultant['name']} <{consultant['email']}>",
        f"To: {email_data['from']}",
        f"Cc: contact@okusi.id",
        f"Subject: {reply_subject}",
        f"Date: {self._format_date()}",
        f"Message-ID: {new_msg_id}",
      ]
      
      # Add threading headers if original message ID exists
      if original_msg_id:
        headers.append(f"In-Reply-To: {original_msg_id}")
        headers.append(f"References: {original_msg_id}")
      
      # Add standard headers
      headers.extend([
        "MIME-Version: 1.0",
        "Content-Type: text/plain; charset=utf-8",
        "Content-Transfer-Encoding: 8bit"
      ])
      
      # Create email body with signature and quoted original message
      signature = self.config.get_email_signature(consultant)
      quoted_original = self._quote_original_message(email_data)
      
      body = f"{reply_content}\n\n{signature}\n\n{quoted_original}"
      
      # Combine headers and body
      email_draft = "\n".join(headers) + "\n\n" + body
      
      return email_draft
      
    except Exception as e:
      self.logger.error(f"Error generating email draft: {e}")
      return None

  def _format_date(self):
    """Format current date for email header."""
    from email.utils import formatdate
    return formatdate(localtime=True)

  def _quote_original_message(self, email_data):
    """Format original message with blockquote characters (>)."""
    try:
      # Format original message header
      quoted_lines = []
      quoted_lines.append(f"> From: {email_data['from']}")
      quoted_lines.append(f"> Subject: {email_data['subject']}")
      quoted_lines.append(f"> Date: {email_data['date']}")
      quoted_lines.append("> ")
      
      # Quote the original message body
      original_body = email_data.get('body', '').strip()
      if original_body:
        for line in original_body.split('\n'):
          quoted_lines.append(f"> {line}")
      else:
        quoted_lines.append("> [No message content]")
      
      return '\n'.join(quoted_lines)
      
    except Exception as e:
      self.logger.error(f"Error quoting original message: {e}")
      return "> [Error formatting original message]"

  def save_draft(self, email_draft, original_filepath):
    """Save email draft to .Drafts/cur directory."""
    try:
      # Generate unique filename for draft
      timestamp = int(time.time())
      unique_id = f"V{hex(timestamp)[2:]}M{os.getpid()}"
      hostname = self.config.get_hostname()
      
      draft_flag = self.config.get_maildir_flag('draft')
      draft_filename = f"{timestamp}.{unique_id}.{hostname}:2,{draft_flag}"
      draft_path = os.path.join(self.drafts_dir, draft_filename)
      
      # Write draft file to temp location first
      temp_dir = self.config.get_temp_dir()
      temp_path = os.path.join(temp_dir, draft_filename)
      with open(temp_path, 'w', encoding='utf-8') as f:
        f.write(email_draft)
      
      # Move to final location with proper permissions
      os.rename(temp_path, draft_path)
      os.chmod(draft_path, 0o644)
      
      self.logger.info(f"Draft saved to: {draft_path}")
      return draft_path
      
    except Exception as e:
      self.logger.error(f"Error saving draft: {e}")
      return None

  def mark_as_replied(self, original_filepath):
    """Mark original email as replied by updating filename."""
    try:
      # Add Reply flag to filename
      filename = os.path.basename(original_filepath)
      new_filename = self.config.add_maildir_flag(filename, 'replied')
      
      if new_filename != filename:
        directory = os.path.dirname(original_filepath)
        new_filepath = os.path.join(directory, new_filename)
        os.rename(original_filepath, new_filepath)
        self.logger.info(f"Marked as replied: {os.path.basename(new_filepath)}")
        return new_filepath
      else:
        self.logger.warning(f"Unexpected filename format: {original_filepath}")
        return original_filepath
        
    except Exception as e:
      self.logger.error(f"Error marking as replied: {e}")
      return original_filepath

  def generate_reply(self, email_filepath):
    """Generate reply for a single email."""
    try:
      # Parse original email
      email_data = self.parse_email_file(email_filepath)
      if not email_data:
        return {
          "success": False,
          "error": "Failed to parse email file"
        }
      
      # Determine appropriate consultant
      consultant = self.determine_consultant(email_data)
      
      # Construct query for CustomKB
      query = self.construct_query(email_data)
      
      # Get reply from CustomKB
      reply_content = self.call_customkb(query)
      if not reply_content:
        return {
          "success": False,
          "error": "Failed to generate reply from CustomKB"
        }
      
      # Generate email draft
      email_draft = self.generate_email_draft(email_data, reply_content, consultant)
      if not email_draft:
        return {
          "success": False,
          "error": "Failed to generate email draft"
        }
      
      # Save draft to .Drafts directory
      draft_path = self.save_draft(email_draft, email_filepath)
      if not draft_path:
        return {
          "success": False,
          "error": "Failed to save draft"
        }
      
      # Mark original email as replied
      updated_filepath = self.mark_as_replied(email_filepath)
      
      return {
        "success": True,
        "draft_path": draft_path,
        "consultant": consultant,
        "original_subject": email_data['subject'],
        "original_from": email_data['from'],
        "updated_original_path": updated_filepath
      }
      
    except Exception as e:
      self.logger.error(f"Error generating reply: {e}")
      return {
        "success": False,
        "error": f"Unexpected error: {e}"
      }

  def process_email(self, email_file_info, dry_run=False):
    """Process a single email through the complete pipeline."""
    filepath = email_file_info['path']
    self.logger.info(f"Processing: {os.path.basename(filepath)}")
    
    try:
      # Step 1: Evaluate email
      evaluation = self.evaluate_email(filepath)
      
      if not evaluation['needs_reply']:
        # Track why email was skipped
        reason = evaluation['reason']
        if 'already has reply flag' in reason:
          self.stats['already_replied'] += 1
        elif 'does not contain ticket pattern' in reason:
          self.stats['no_ticket_pattern'] += 1
        elif 'detected as spam' in reason:
          self.stats['spam_detected'] += 1
        
        self.logger.debug(f"Skipped: {reason}")
        return {
          "processed": False,
          "reason": reason,
          "filepath": filepath
        }
      
      # Step 2: Generate reply (if not dry run)
      if dry_run:
        self.logger.info(f"DRY RUN: Would generate reply for {evaluation['email_data']['subject']}")
        return {
          "processed": False,
          "reason": "Dry run mode - would generate reply",
          "filepath": filepath,
          "email_data": evaluation['email_data']
        }
      
      # Generate actual reply
      reply_result = self.generate_reply(filepath)
      
      if reply_result['success']:
        self.stats['reply_generated'] += 1
        self.logger.info(f"✅ Generated reply: {reply_result['draft_path']}")
        return {
          "processed": True,
          "reply_result": reply_result,
          "filepath": filepath
        }
      else:
        self.stats['errors'] += 1
        self.logger.error(f"❌ Failed to generate reply: {reply_result['error']}")
        return {
          "processed": False,
          "reason": f"Reply generation failed: {reply_result['error']}",
          "filepath": filepath
        }
    
    except Exception as e:
      self.stats['errors'] += 1
      self.logger.error(f"❌ Error processing {filepath}: {e}")
      return {
        "processed": False,
        "reason": f"Processing error: {e}",
        "filepath": filepath
      }

  def process_emails(self, since_timestamp=None, full_scan=False, dry_run=False, specific_file=None):
    """Main email processing function."""
    start_time = time.time()
    
    if specific_file:
      # Process single file
      self.logger.info(f"Processing specific file: {specific_file}")
      if not os.path.exists(specific_file):
        self.logger.error(f"File not found: {specific_file}")
        return False
      
      file_info = {'path': specific_file, 'mtime': os.path.getmtime(specific_file), 'subdir': 'manual'}
      result = self.process_email(file_info, dry_run)
      
      # Print result
      print(json.dumps({
        "single_file_result": result,
        "stats": self.stats
      }, indent=2))
      
      return result['processed']
    
    # Discover emails to process
    email_files = self.discover_emails(since_timestamp, full_scan)
    
    if not email_files:
      self.logger.info("No emails found to process")
      return True
    
    # Process each email
    results = []
    for i, email_file in enumerate(email_files, 1):
      self.logger.info(f"[{i}/{len(email_files)}] Processing...")
      result = self.process_email(email_file, dry_run)
      results.append(result)
      
      # Small delay to avoid overwhelming the system
      process_delay = self.config.get_process_delay()
      time.sleep(process_delay)
    
    # Update timestamp if not dry run and no specific timestamp was provided
    if not dry_run and since_timestamp is None:
      self.update_last_check_timestamp()
    
    # Report results
    duration = time.time() - start_time
    self.logger.info(f"Processing completed in {duration:.1f} seconds")
    self.print_summary()
    
    return True

  def print_summary(self):
    """Print processing summary."""
    self.logger.info("Processing Summary:")
    self.logger.info(f"  Total emails found: {self.stats['total_found']}")
    self.logger.info(f"  Already replied: {self.stats['already_replied']}")
    self.logger.info(f"  No ticket pattern: {self.stats['no_ticket_pattern']}")
    self.logger.info(f"  Spam detected: {self.stats['spam_detected']}")
    self.logger.info(f"  Replies generated: {self.stats['reply_generated']}")
    self.logger.info(f"  Errors: {self.stats['errors']}")

def main():
  """Command line interface."""
  parser = argparse.ArgumentParser(
    description="Unified Email Auto-Reply Processor",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
Examples:
  %(prog)s                          # Process new emails since last check
  %(prog)s --full-scan              # Process all unreplied emails
  %(prog)s --since 1749977000       # Process emails since timestamp
  %(prog)s --file /path/to/email    # Process specific email
  %(prog)s --dry-run                # Evaluate only, no replies
  %(prog)s --verbose                # Enable debug logging
"""
  )
  
  parser.add_argument('--since', type=int, metavar='TIMESTAMP',
                    help='Process emails modified since this timestamp')
  parser.add_argument('--full-scan', action='store_true',
                    help='Process all unreplied emails (ignore timestamp)')
  parser.add_argument('--file', metavar='PATH',
                    help='Process specific email file')
  parser.add_argument('--dry-run', action='store_true',
                    help='Evaluate emails but do not generate replies')
  parser.add_argument('--verbose', '-v', action='store_true',
                    help='Enable verbose logging')
  parser.add_argument('--log-file', metavar='PATH',
                    help='Log file path (optional)')
  
  args = parser.parse_args()
  
  # Set up logging
  logger = setup_logging(verbose=args.verbose, log_file=args.log_file)
  
  try:
    # Initialize processor
    processor = EmailProcessor(logger=logger)
    
    # Process emails
    success = processor.process_emails(
      since_timestamp=args.since,
      full_scan=args.full_scan,
      dry_run=args.dry_run,
      specific_file=args.file
    )
    
    sys.exit(0 if success else 1)
    
  except KeyboardInterrupt:
    logger.info("Processing interrupted by user")
    sys.exit(1)
  except Exception as e:
    logger.error(f"Fatal error: {e}")
    sys.exit(2)

if __name__ == "__main__":
  main()

#fin