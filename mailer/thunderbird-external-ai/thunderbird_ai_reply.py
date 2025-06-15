#!/usr/bin/env python3
"""
Main Thunderbird AI Reply System
Integrates all components: message detection, AI generation, and reply creation
"""

import sys
import os
import logging
import argparse
import tempfile
import subprocess
from pathlib import Path
from typing import Optional, Dict
import json
import time

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from thunderbird_message_detector import ThunderbirdMessageDetector
from imap_client import IMAPMessageClient
from ai_reply_generator import AIReplyGenerator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/tmp/thunderbird-ai-reply.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ThunderbirdAIReply:
    def __init__(self, config_file: str = None):
        self.config = self._load_config(config_file)
        self.detector = ThunderbirdMessageDetector()
        self.ai_generator = AIReplyGenerator()
        self.imap_client = None
        
        # Initialize IMAP client if credentials provided
        if self._has_imap_config():
            self.imap_client = IMAPMessageClient(
                self.config['imap']['server'],
                self.config['imap'].get('port', 993),
                self.config['imap'].get('use_ssl', True)
            )
    
    def _load_config(self, config_file: str = None) -> Dict:
        """Load configuration from file or create default"""
        default_config = {
            'ai': {
                'method': 'template',  # 'template', 'customkb', 'openai', 'anthropic'
                'customkb_name': 'default',
                'openai_api_key': '',
                'anthropic_api_key': ''
            },
            'imap': {
                'server': '',
                'username': '',
                'password': '',
                'port': 993,
                'use_ssl': True
            },
            'reply': {
                'auto_open': True,
                'save_drafts': True
            }
        }
        
        if config_file and Path(config_file).exists():
            try:
                with open(config_file) as f:
                    user_config = json.load(f)
                    # Merge with defaults
                    for section, values in user_config.items():
                        if section in default_config:
                            default_config[section].update(values)
                        else:
                            default_config[section] = values
            except Exception as e:
                logger.warning(f"Error loading config file: {e}")
        
        return default_config
    
    def _has_imap_config(self) -> bool:
        """Check if IMAP configuration is provided"""
        imap_config = self.config.get('imap', {})
        return bool(imap_config.get('server') and 
                   imap_config.get('username') and 
                   imap_config.get('password'))
    
    def generate_reply(self) -> bool:
        """Main method to generate AI reply for current message"""
        logger.info("Starting AI reply generation...")
        
        try:
            # Step 1: Detect current message
            message_headers = self.detector.detect_current_message()
            if not message_headers:
                logger.error("Could not detect current message")
                self._show_notification("Error", "Could not detect current message. Please select an email.")
                return False
            
            logger.info(f"Detected message: {message_headers.get('subject', 'Unknown')}")
            
            # Step 2: Get full message content if IMAP is available
            full_message_content = None
            if self.imap_client and self._connect_imap():
                full_message_content = self._get_message_via_imap(message_headers)
            
            # Step 3: Parse message data
            if full_message_content:
                email_data = self.ai_generator.parse_email_message(full_message_content)
                logger.info("Using full message content from IMAP")
            else:
                # Use detected headers as fallback
                email_data = message_headers
                logger.info("Using detected headers (IMAP not available)")
            
            # Step 4: Generate AI reply
            reply_content = self._generate_ai_reply(email_data)
            if not reply_content:
                logger.error("Failed to generate AI reply")
                self._show_notification("Error", "Failed to generate AI reply")
                return False
            
            # Step 5: Create and open reply
            success = self._create_and_open_reply(email_data, reply_content)
            if success:
                self._show_notification("Success", "AI reply generated! Review and send when ready.")
                return True
            else:
                logger.error("Failed to create reply")
                self._show_notification("Error", "Failed to create reply")
                return False
                
        except Exception as e:
            logger.error(f"Error in generate_reply: {e}")
            self._show_notification("Error", f"Unexpected error: {str(e)}")
            return False
    
    def _connect_imap(self) -> bool:
        """Connect to IMAP server"""
        try:
            imap_config = self.config['imap']
            return self.imap_client.connect(
                imap_config['username'],
                imap_config['password']
            )
        except Exception as e:
            logger.error(f"IMAP connection failed: {e}")
            return False
    
    def _get_message_via_imap(self, message_headers: Dict[str, str]) -> Optional[str]:
        """Get full message content via IMAP"""
        try:
            subject = message_headers.get('subject') or message_headers.get('detected_subject')
            sender = message_headers.get('from')
            message_id = message_headers.get('message-id')
            
            result = self.imap_client.find_message_by_details(
                subject=subject,
                sender=sender,
                message_id=message_id
            )
            
            if result:
                folder, imap_id, raw_message = result
                logger.info(f"Found message in folder {folder} with ID {imap_id}")
                return raw_message
            else:
                logger.warning("Message not found via IMAP")
                return None
                
        except Exception as e:
            logger.error(f"Error getting message via IMAP: {e}")
            return None
    
    def _generate_ai_reply(self, email_data: Dict[str, str]) -> Optional[str]:
        """Generate AI reply based on configured method"""
        ai_method = self.config['ai']['method']
        
        try:
            if ai_method == 'template':
                return self.ai_generator.generate_simple_template_reply(email_data)
            
            elif ai_method == 'customkb':
                kb_name = self.config['ai'].get('customkb_name', 'default')
                reply = self.ai_generator.generate_customkb_reply(email_data, kb_name)
                if reply:
                    return reply
                # Fallback to template if CustomKB fails
                logger.warning("CustomKB failed, falling back to template")
                return self.ai_generator.generate_simple_template_reply(email_data)
            
            elif ai_method == 'openai':
                api_key = self.config['ai'].get('openai_api_key')
                if not api_key:
                    logger.error("OpenAI API key not configured")
                    return self.ai_generator.generate_simple_template_reply(email_data)
                
                reply = self.ai_generator.generate_openai_reply(email_data, api_key)
                if reply:
                    return reply
                # Fallback to template
                logger.warning("OpenAI failed, falling back to template")
                return self.ai_generator.generate_simple_template_reply(email_data)
            
            elif ai_method == 'anthropic':
                api_key = self.config['ai'].get('anthropic_api_key')
                if not api_key:
                    logger.error("Anthropic API key not configured")
                    return self.ai_generator.generate_simple_template_reply(email_data)
                
                reply = self.ai_generator.generate_anthropic_reply(email_data, api_key)
                if reply:
                    return reply
                # Fallback to template
                logger.warning("Anthropic failed, falling back to template")
                return self.ai_generator.generate_simple_template_reply(email_data)
            
            else:
                logger.error(f"Unknown AI method: {ai_method}")
                return self.ai_generator.generate_simple_template_reply(email_data)
                
        except Exception as e:
            logger.error(f"Error generating AI reply: {e}")
            return self.ai_generator.generate_simple_template_reply(email_data)
    
    def _create_and_open_reply(self, email_data: Dict[str, str], reply_content: str) -> bool:
        """Create reply email and open in Thunderbird using multiple methods"""
        try:
            # Method 1: Use Thunderbird's compose functionality with proper formatting
            try:
                logger.info("Attempting Method 1: Thunderbird -compose with proper encoding")
                
                # Extract recipient and subject for compose command
                original_sender = email_data.get('from', '')
                original_subject = email_data.get('subject', '')
                
                # Create reply subject
                if not original_subject.startswith('Re:'):
                    reply_subject = f"Re: {original_subject}"
                else:
                    reply_subject = original_subject
                
                # Properly encode the body content with line breaks
                import urllib.parse
                # Replace newlines with proper CRLF encoding for mailto
                body_with_crlf = reply_content.replace('\n', '\r\n')
                encoded_body = urllib.parse.quote(body_with_crlf, safe='')
                # Ensure proper line break encoding
                encoded_body = encoded_body.replace('%0A', '%0D%0A')
                
                # Build compose parameters with proper quote nesting
                compose_parts = []
                
                # Use single quotes for values containing special characters
                if original_sender:
                    # Clean sender address
                    sender_clean = original_sender.strip()
                    compose_parts.append(f"to='{sender_clean}'")
                
                if reply_subject:
                    # Encode subject for safety
                    encoded_subject = urllib.parse.quote(reply_subject, safe='')
                    compose_parts.append(f"subject='{encoded_subject}'")
                
                if encoded_body:
                    compose_parts.append(f"body='{encoded_body}'")
                
                # Add threading information if available
                if email_data.get('message-id'):
                    in_reply_to = email_data.get('message-id')
                    compose_parts.append(f"in-reply-to='{in_reply_to}'")
                
                # Join all parts
                compose_params = ','.join(compose_parts)
                
                # Try both mailto and direct parameter format
                cmd_args = ['thunderbird', '-compose', compose_params]
                logger.debug(f"Running command: {' '.join(cmd_args[:2])} [params hidden for security]")
                
                result = subprocess.run(cmd_args, capture_output=True, text=True, timeout=5)
                
                if result.returncode == 0:
                    logger.info("Thunderbird compose window opened successfully")
                    time.sleep(2)
                    return True
                else:
                    logger.warning(f"Thunderbird returned non-zero exit code: {result.returncode}")
                    if result.stderr:
                        logger.warning(f"Stderr: {result.stderr}")
                
            except subprocess.TimeoutExpired:
                logger.warning("Thunderbird compose command timed out")
            except Exception as e:
                logger.warning(f"Compose method 1 failed: {e}")
            
            # Method 2: Try alternative compose syntax with message file
            try:
                logger.info("Attempting Method 2: Thunderbird -compose with message file")
                
                # Create HTML version of the reply
                html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <style>
        body {{ font-family: Arial, sans-serif; font-size: 14px; line-height: 1.6; }}
        p {{ margin: 0 0 10px 0; }}
    </style>
</head>
<body>
{reply_content.replace(chr(10), '<br>').replace('  ', '&nbsp;&nbsp;')}
</body>
</html>"""
                
                # Save to temporary HTML file
                with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
                    f.write(html_content)
                    html_file = f.name
                
                # Build compose parameters with message file
                compose_parts = []
                if email_data.get('from'):
                    compose_parts.append(f"to='{email_data['from'].strip()}'")
                
                subject = email_data.get('subject', '')
                if not subject.startswith('Re:'):
                    subject = f"Re: {subject}"
                compose_parts.append(f"subject='{subject}'")
                compose_parts.append(f"message='{html_file}'")
                
                compose_params = ','.join(compose_parts)
                
                result = subprocess.run(
                    ['thunderbird', '-compose', compose_params],
                    capture_output=True, text=True, timeout=5
                )
                
                if result.returncode == 0:
                    logger.info("Thunderbird compose window opened with HTML message")
                    time.sleep(2)
                    return True
                    
            except Exception as e:
                logger.warning(f"HTML message method failed: {e}")
            
            # Method 3: Create EML file with draft headers and open
            try:
                logger.info("Attempting Method 3: EML file with Mozilla draft headers")
                
                # Create EML with draft headers
                full_reply = self.ai_generator.create_reply_email(email_data, reply_content, as_draft=True)
                
                # Save to temporary file
                with tempfile.NamedTemporaryFile(mode='w', suffix='.eml', delete=False) as f:
                    f.write(full_reply)
                    reply_file = f.name
                
                # Try multiple methods to open the EML file
                # First try Thunderbird directly
                try:
                    result = subprocess.run(
                        ['thunderbird', reply_file],
                        capture_output=True, text=True, timeout=5
                    )
                    if result.returncode == 0:
                        logger.info("EML file opened in Thunderbird")
                        time.sleep(2)
                        return True
                except:
                    pass
                
                # Then try xdg-open
                try:
                    subprocess.run(['xdg-open', reply_file], check=False, timeout=5)
                    logger.info("EML file opened via xdg-open")
                    time.sleep(1)
                    return True
                except:
                    pass
                
            except Exception as e:
                logger.warning(f"EML draft method failed: {e}")
            
            # Method 4: Save to desktop as final fallback
            try:
                logger.info("Attempting Method 4: Save reply to desktop")
                
                # Create both text and EML versions
                timestamp = int(time.time())
                
                # Save plain text version
                txt_path = Path.home() / 'Desktop' / f'ai_reply_{timestamp}.txt'
                with open(txt_path, 'w') as f:
                    f.write(f"AI Reply for: {email_data.get('subject', 'Unknown')}\n")
                    f.write(f"To: {email_data.get('from', 'Unknown')}\n")
                    f.write("=" * 50 + "\n\n")
                    f.write(reply_content)
                    f.write("\n\n" + "=" * 50 + "\n")
                    f.write("Copy the content above and paste it into your email reply.\n")
                
                # Also save EML version for manual import
                eml_path = Path.home() / 'Desktop' / f'ai_reply_{timestamp}.eml'
                eml_content = self.ai_generator.create_reply_email(email_data, reply_content, as_draft=True)
                with open(eml_path, 'w') as f:
                    f.write(eml_content)
                
                logger.info(f"Reply saved to: {txt_path} and {eml_path}")
                self._show_notification(
                    "AI Reply Saved", 
                    f"Reply saved to Desktop as {txt_path.name}\nYou can also import {eml_path.name} into Thunderbird"
                )
                return True
                
            except Exception as e:
                logger.error(f"Fallback save failed: {e}")
                return False
            
        except Exception as e:
            logger.error(f"Error creating/opening reply: {e}")
            return False
    
    def _show_notification(self, title: str, message: str):
        """Show desktop notification"""
        try:
            subprocess.run([
                'notify-send', title, message
            ], check=False, timeout=5)
        except:
            # Fallback to console output
            print(f"{title}: {message}")

def create_default_config():
    """Create a default configuration file"""
    config_path = Path.home() / '.thunderbird-ai-reply.json'
    
    default_config = {
        "ai": {
            "method": "template",
            "customkb_name": "default",
            "openai_api_key": "",
            "anthropic_api_key": ""
        },
        "imap": {
            "server": "okusi0.okusi.co.id",
            "username": "contact@okusi.dev",
            "password": "",
            "port": 993,
            "use_ssl": True
        },
        "reply": {
            "auto_open": True,
            "save_drafts": True
        }
    }
    
    with open(config_path, 'w') as f:
        json.dump(default_config, f, indent=2)
    
    print(f"Default config created at: {config_path}")
    print("Please edit the file to add your credentials and preferences.")

def main():
    parser = argparse.ArgumentParser(description='Thunderbird AI Reply Generator')
    parser.add_argument('--config', '-c', help='Configuration file path')
    parser.add_argument('--create-config', action='store_true', 
                       help='Create default configuration file')
    parser.add_argument('--test', action='store_true', 
                       help='Test message detection without generating reply')
    
    args = parser.parse_args()
    
    if args.create_config:
        create_default_config()
        return
    
    # Use default config file if not specified
    config_file = args.config or str(Path.home() / '.thunderbird-ai-reply.json')
    
    reply_system = ThunderbirdAIReply(config_file)
    
    if args.test:
        print("Testing message detection...")
        message_headers = reply_system.detector.detect_current_message()
        if message_headers:
            print("✅ Message detected:")
            for key, value in message_headers.items():
                print(f"  {key}: {value[:100]}...")
        else:
            print("❌ No message detected")
        return
    
    # Generate reply
    success = reply_system.generate_reply()
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()