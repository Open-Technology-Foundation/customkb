#!/usr/bin/env python3
"""
AI Reply Generator - integrates with CustomKB and external AI services
"""

import json
import logging
import subprocess
import tempfile
import os
import sys
from pathlib import Path
from typing import Dict, Optional
import email
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import re

logger = logging.getLogger(__name__)

class AIReplyGenerator:
    def __init__(self):
        self.customkb_path = Path('/ai/scripts/customkb')
        self.venv_path = self.customkb_path / '.venv'
        
    def parse_email_message(self, raw_message: str) -> Dict[str, str]:
        """Parse raw email message into structured data"""
        try:
            msg = email.message_from_string(raw_message)
            
            # Extract basic headers
            headers = {
                'subject': msg.get('Subject', ''),
                'from': msg.get('From', ''),
                'to': msg.get('To', ''),
                'cc': msg.get('Cc', ''),
                'date': msg.get('Date', ''),
                'message_id': msg.get('Message-ID', ''),
                'in_reply_to': msg.get('In-Reply-To', ''),
                'references': msg.get('References', '')
            }
            
            # Extract body content
            body = self._extract_body_content(msg)
            headers['body'] = body
            
            return headers
            
        except Exception as e:
            logger.error(f"Error parsing email message: {e}")
            return {}
    
    def _extract_body_content(self, msg) -> str:
        """Extract plain text body from email message"""
        body = ""
        
        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                content_disposition = str(part.get('Content-Disposition'))
                
                # Skip attachments
                if 'attachment' in content_disposition:
                    continue
                
                if content_type == 'text/plain':
                    try:
                        body += part.get_payload(decode=True).decode('utf-8', errors='ignore')
                    except:
                        body += str(part.get_payload())
                elif content_type == 'text/html' and not body:
                    # Fallback to HTML if no plain text found
                    try:
                        html_content = part.get_payload(decode=True).decode('utf-8', errors='ignore')
                        # Simple HTML to text conversion
                        body = self._html_to_text(html_content)
                    except:
                        pass
        else:
            # Single part message
            try:
                body = msg.get_payload(decode=True).decode('utf-8', errors='ignore')
            except:
                body = str(msg.get_payload())
        
        return body.strip()
    
    def _html_to_text(self, html: str) -> str:
        """Simple HTML to text conversion"""
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', html)
        # Decode HTML entities
        text = text.replace('&nbsp;', ' ')
        text = text.replace('&lt;', '<')
        text = text.replace('&gt;', '>')
        text = text.replace('&amp;', '&')
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def generate_simple_template_reply(self, email_data: Dict[str, str]) -> str:
        """Generate a simple template-based reply"""
        sender_name = self._extract_sender_name(email_data.get('from', ''))
        subject = email_data.get('subject', '')
        body = email_data.get('body', '')
        
        # Clean and extract the actual body content if it's mixed with headers
        if 'Content-Type:' in body or 'X-Mozilla-Status:' in body:
            body = self._extract_body_from_raw_message(body)
        
        reply_lines = []
        
        # Greeting
        reply_lines.append(f"Hi {sender_name},")
        reply_lines.append("")
        
        # Acknowledgment
        reply_lines.append("Thank you for your email.")
        reply_lines.append("")
        
        # Context-specific response based on content analysis
        subject_lower = subject.lower()
        body_lower = body.lower()
        
        # Check for business partnership requests
        if any(word in subject_lower + body_lower for word in ['business partner', 'partnership', 'joint venture', 'collaboration']):
            reply_lines.append("Thank you for your interest in potential business collaboration.")
            reply_lines.append("I'd be happy to discuss partnership opportunities.")
            reply_lines.append("Please provide more details about your business and the specific collaboration you have in mind.")
        elif any(word in subject_lower + body_lower for word in ['pma', 'indonesia', 'investment']):
            reply_lines.append("Thank you for your inquiry about business opportunities in Indonesia.")
            reply_lines.append("We can certainly discuss potential collaboration and investment structures.")
            reply_lines.append("I'll need more information about your business background and specific requirements.")
        elif any(word in subject_lower for word in ['meeting', 'schedule', 'appointment', 'call']):
            reply_lines.append("I'll check my calendar and get back to you with my availability.")
        elif any(word in subject_lower for word in ['question', 'help', 'support', 'assistance']):
            reply_lines.append("I'll look into this and provide you with the information you need.")
        elif any(word in subject_lower + body_lower for word in ['urgent', 'asap', 'priority', 'immediate']):
            reply_lines.append("I understand this is urgent and will prioritize this accordingly.")
        elif any(word in subject_lower for word in ['proposal', 'offer', 'opportunity']):
            reply_lines.append("Thank you for the proposal. I'll review it carefully and respond with my thoughts.")
        elif any(word in subject_lower for word in ['invoice', 'payment', 'billing']):
            reply_lines.append("I've received your message regarding the financial matter and will address it promptly.")
        elif any(word in body_lower for word in ['thank', 'thanks']):
            reply_lines.append("You're welcome! Please don't hesitate to reach out if you need anything else.")
        else:
            reply_lines.append("I'll review this and get back to you soon.")
        
        reply_lines.append("")
        reply_lines.append("Best regards,")
        reply_lines.append("Gary Dean")
        reply_lines.append("Okusi Associates")
        
        return '\n'.join(reply_lines)
    
    def _extract_body_from_raw_message(self, raw_content: str) -> str:
        """Extract actual body content from raw email message"""
        try:
            lines = raw_content.split('\n')
            body_started = False
            body_lines = []
            
            for line in lines:
                # Look for the end of headers (empty line)
                if not body_started and line.strip() == '':
                    body_started = True
                    continue
                
                if body_started:
                    # Skip MIME boundaries and headers within multipart content
                    if line.startswith('--') and '=' in line:
                        continue
                    if line.startswith('Content-'):
                        continue
                    if line.startswith('MIME-Version:'):
                        continue
                    
                    # Extract actual content
                    if line.strip():
                        # Remove HTML encoding artifacts
                        clean_line = line.replace('=3D', '=').replace('=', '')
                        if not clean_line.startswith('<') and clean_line.strip():
                            body_lines.append(clean_line.strip())
            
            # Join and clean up
            body = ' '.join(body_lines)
            # Remove extra whitespace
            body = ' '.join(body.split())
            
            return body if body else "No message content available"
            
        except Exception as e:
            logger.error(f"Error extracting body from raw message: {e}")
            return "Unable to extract message content"
    
    def generate_customkb_reply(self, email_data: Dict[str, str], kb_name: str = None) -> Optional[str]:
        """Generate reply using CustomKB system"""
        try:
            # Use the existing CustomKB system
            body = email_data.get('body', '')
            subject = email_data.get('subject', '')
            
            # Combine subject and body for query
            query_text = f"{subject}\n\n{body}"
            
            # Create temporary query file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write(query_text)
                query_file = f.name
            
            try:
                # Activate virtual environment and run CustomKB query
                if kb_name:
                    cmd = [
                        str(self.venv_path / 'bin' / 'python3'),
                        str(self.customkb_path / 'customkb.py'),
                        'query',
                        kb_name,
                        query_text
                    ]
                else:
                    # Use default knowledge base if available
                    cmd = [
                        str(self.venv_path / 'bin' / 'python3'),
                        str(self.customkb_path / 'customkb.py'),
                        'query',
                        'default',  # Adjust this to your KB name
                        query_text
                    ]
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                
                if result.returncode == 0:
                    customkb_response = result.stdout.strip()
                    
                    # Format as email reply
                    sender_name = self._extract_sender_name(email_data.get('from', ''))
                    
                    reply = f"""Hi {sender_name},

Thank you for your email.

{customkb_response}

Best regards

---
Generated with CustomKB AI Assistant"""
                    
                    return reply
                else:
                    logger.error(f"CustomKB query failed: {result.stderr}")
                    return None
                    
            finally:
                # Clean up temp file
                try:
                    os.unlink(query_file)
                except:
                    pass
                    
        except Exception as e:
            logger.error(f"Error generating CustomKB reply: {e}")
            return None
    
    def generate_openai_reply(self, email_data: Dict[str, str], api_key: str) -> Optional[str]:
        """Generate reply using OpenAI API"""
        try:
            import openai
            
            client = openai.OpenAI(api_key=api_key)
            
            sender_name = self._extract_sender_name(email_data.get('from', ''))
            subject = email_data.get('subject', '')
            body = email_data.get('body', '')
            
            prompt = f"""Generate a professional email reply to the following email:

From: {email_data.get('from', '')}
Subject: {subject}

Message:
{body}

Please write a professional, helpful reply that addresses the sender's message appropriately. The reply should be:
- Professional and courteous
- Directly relevant to their message
- Appropriately detailed
- End with a professional closing"""
            
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that generates professional email replies."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.7
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error generating OpenAI reply: {e}")
            return None
    
    def generate_anthropic_reply(self, email_data: Dict[str, str], api_key: str) -> Optional[str]:
        """Generate reply using Anthropic Claude API"""
        try:
            import anthropic
            
            client = anthropic.Anthropic(api_key=api_key)
            
            sender_name = self._extract_sender_name(email_data.get('from', ''))
            subject = email_data.get('subject', '')
            body = email_data.get('body', '')
            
            prompt = f"""Generate a professional email reply to the following email:

From: {email_data.get('from', '')}
Subject: {subject}

Message:
{body}

Please write a professional, helpful reply that addresses the sender's message appropriately."""
            
            response = client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=500,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            return response.content[0].text.strip()
            
        except Exception as e:
            logger.error(f"Error generating Anthropic reply: {e}")
            return None
    
    def _extract_sender_name(self, from_field: str) -> str:
        """Extract sender name from From field"""
        if not from_field:
            return "there"
        
        # Try to extract name from "Name <email>" format
        match = re.search(r'^([^<]+)<', from_field)
        if match:
            name = match.group(1).strip().strip('"')
            # Get first name
            first_name = name.split()[0] if name else "there"
            return first_name
        
        # Try to extract from email address
        match = re.search(r'([^@\s]+)', from_field)
        if match:
            return match.group(1)
        
        return "there"
    
    def create_reply_email(self, original_email: Dict[str, str], reply_content: str, as_draft: bool = True) -> str:
        """Create a properly formatted reply email with optional draft headers"""
        # Create reply message
        reply = MIMEText(reply_content, 'plain', 'utf-8')
        
        # Set headers for reply
        original_subject = original_email.get('subject', '')
        if not original_subject.startswith('Re:'):
            reply_subject = f"Re: {original_subject}"
        else:
            reply_subject = original_subject
        
        reply['Subject'] = reply_subject
        reply['To'] = original_email.get('from', '')
        reply['From'] = 'Gary Dean <contact@okusi.dev>'  # Should be configurable
        reply['In-Reply-To'] = original_email.get('message_id', '')
        
        # Handle References header for threading
        references = original_email.get('references', '')
        original_message_id = original_email.get('message_id', '')
        if references and original_message_id:
            reply['References'] = f"{references} {original_message_id}"
        elif original_message_id:
            reply['References'] = original_message_id
        
        # Add Mozilla-specific draft headers if requested
        if as_draft:
            # This header helps Thunderbird recognize it as a draft
            reply['X-Mozilla-Draft-Info'] = 'internal/draft; vcard=0; receipt=0; DSN=0; uuencode=0'
            # X-Unsent header (works in Outlook, but we'll add it anyway)
            reply['X-Unsent'] = '1'
        
        return reply.as_string()

def main():
    """Test function"""
    generator = AIReplyGenerator()
    
    # Test with sample email data
    sample_email = {
        'from': 'John Doe <john@example.com>',
        'subject': 'Meeting Request',
        'body': 'Hi, I would like to schedule a meeting to discuss the project.',
        'message_id': '<12345@example.com>'
    }
    
    print("Testing AI Reply Generator...")
    print("=" * 50)
    
    # Test simple template
    print("1. Simple Template Reply:")
    template_reply = generator.generate_simple_template_reply(sample_email)
    print(template_reply)
    print("\n" + "=" * 50)
    
    # Test full email creation
    print("2. Full Reply Email:")
    full_reply = generator.create_reply_email(sample_email, template_reply)
    print(full_reply[:500] + "...")

if __name__ == '__main__':
    main()