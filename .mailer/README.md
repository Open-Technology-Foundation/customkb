# Okusi Associates Email Auto-Reply System

A production-ready AI-powered email processing system that automatically generates professional replies to business inquiries using CustomKB knowledge base integration.

## Overview

The Okusi Associates Email Auto-Reply System is a sophisticated email automation solution designed for Indonesian legal services firms. It intelligently processes incoming emails, filters spam, routes inquiries to appropriate consultants, and generates professional responses using AI technology.

### Key Features

- **Intelligent Email Processing**: Automated discovery and evaluation of emails in Maildir format
- **AI-Powered Spam Detection**: Uses OpenAI GPT-4o-mini or Anthropic Claude for content classification
- **Smart Consultant Routing**: Keyword-based assignment to appropriate specialists
- **Professional Reply Generation**: CustomKB integration for contextual, knowledge-based responses
- **Configuration-Driven Architecture**: All behavior controlled through comprehensive YAML settings
- **Maildir Compliance**: Standard email storage with proper flag handling and threading
- **Production-Ready**: Comprehensive logging, error handling, and monitoring capabilities

### Architecture

The system consists of four core components:

- **`email_processor`** - Bash launcher script that handles user switching and virtual environment activation
- **`email_processor.py`** - Main Python processing engine (~780 lines) with unified email pipeline
- **`config_loader.py`** - Configuration management system (~255 lines) providing type-safe YAML access
- **`email_config.yaml`** - Comprehensive configuration file (~200 lines) with all system settings

## Quick Start

### Prerequisites

- Python 3.8+ with virtual environment
- Access to email files as `vmail` user
- CustomKB installation with configured knowledge base
- OpenAI API key or Anthropic API key (for spam detection)

### Installation

1. **Set up environment variables:**
```bash
export OPENAI_API_KEY="your-openai-api-key"
export ANTHROPIC_API_KEY="your-anthropic-api-key"
```

2. **Install dependencies:**
```bash
source .venv/bin/activate
pip install PyYAML anthropic openai
```

3. **Test configuration:**
```bash
python3 -c "from config_loader import get_config; get_config(); print('✅ Config OK')"
```

### Basic Usage

```bash
# Process new emails since last run (recommended for automation)
./email_processor

# Process all unreplied emails
./email_processor --full-scan

# Test without generating replies
./email_processor --dry-run --verbose

# Process specific email file
./email_processor --file '/path/to/email/file'

# Process emails since specific timestamp
./email_processor --since 1750000000
```

**Note:** The launcher script automatically switches to the `vmail` user and activates the virtual environment.

## Email Processing Pipeline

### Automated Workflow

1. **Email Discovery** - Scans `cur/` and `new/` directories for emails matching pattern `*:2,S` (read but not replied)
2. **Ticket Pattern Detection** - Validates subject contains required ticket pattern `[#:alphanumeric]`
3. **Spam Classification** - AI-powered evaluation to filter legitimate business inquiries from spam
4. **Consultant Assignment** - Keyword-based routing to appropriate specialist (company, tax, visa, director, default)
5. **Reply Generation** - CustomKB integration generates professional responses using company knowledge base
6. **Draft Creation** - Saves formatted RFC 2822 email draft to `.Drafts/cur/` directory
7. **Flag Update** - Marks original email as replied (`:2,S` → `:2,RS`) to prevent reprocessing

### Processing Criteria

Emails are processed only if they meet **ALL** criteria:
- ✅ Not already replied (no 'R' flag in Maildir filename)
- ✅ Contains ticket pattern in subject (`[#:alphanumeric]`)
- ✅ Classified as legitimate business inquiry (not spam) by AI

### File Structure

```
Email Processing Flow:
/home/vmail/okusi.dev/contact/cur/1710754744.V811I.okusi0:2,S
   ↓ (evaluation and reply generation)
/home/vmail/okusi.dev/contact/.Drafts/cur/1710754800.V812I.okusi0:2,D
   ↓ (mark original as replied)
/home/vmail/okusi.dev/contact/cur/1710754744.V811I.okusi0:2,RS
```

## Configuration System

The system uses comprehensive YAML-based configuration that externalizes all hardcoded values:

### Key Configuration Sections

```yaml
# System directories and file paths
directories:
  base_dir: "/home/vmail/okusi.dev/contact"
  drafts_dir: ".Drafts/cur"
  temp_dir: "/tmp"
  timestamp_file: ".last_check"

# Email processing parameters
email:
  unreplied_pattern: "*:2,S"
  ticket_pattern: "\\[#:[a-zA-Z0-9]+\\]"
  body_limit: 2000
  analysis_limit: 1000
  hostname: "okusi0"
  process_delay: 0.5

# AI model configuration
models:
  openai:
    spam_detection: "gpt-4o-mini"
    max_tokens: 10
    temperature: 0.1
  customkb:
    knowledge_base: "okusimail"
    role: "You are a professional legal services consultant..."
    timeout: 120

# Consultant database with contact information
consultants:
  company:
    name: "Reni Debora S.Pd"
    email: "renidebora@okusi.id"
    title: "Head of Compliance and Permits"
    phone: "+6281366225007"
  # ... additional consultants
```

### Configuration API

```python
from config_loader import get_config

config = get_config()

# Directory access
base_dir = config.get_base_dir()
drafts_dir = config.get_drafts_dir()

# Model configuration
openai_model = config.get_openai_model()
customkb_settings = config.get_customkb_settings()

# Consultant routing
consultant = config.get_consultant('company')
consultant_type = config.determine_consultant_type('PMA setup inquiry')

# Maildir utilities
has_replied = config.has_maildir_flag(filename, 'replied')
new_filename = config.add_maildir_flag(filename, 'replied')
```

## Security Model

### User Isolation
- **Execution User**: System runs entirely as `vmail` user (owns email files)
- **No Privilege Escalation**: No sudo privileges or special group memberships required
- **File Access**: Natural access to email files through user ownership

### Input Validation
- **Email Content**: All headers and body content validated before processing
- **File Paths**: Proper path validation to prevent directory traversal
- **API Parameters**: Sanitized inputs to AI APIs

### API Key Management
- **Environment Variables**: API keys stored only in environment, never in config files
- **Runtime Only**: Keys loaded at startup, not persisted to disk
- **Separation**: Different keys for different services (OpenAI, Anthropic)

## Deployment

### Production Setup

1. **User Configuration:**
```bash
# Ensure vmail user exists and owns email directories
sudo chown -R vmail:vmail /home/vmail/okusi.dev/contact/
```

2. **Virtual Environment:**
```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate
pip install PyYAML anthropic openai
```

3. **Configuration Customization:**
```bash
# Edit email_config.yaml for your environment
# Update directory paths, consultant information, model settings
```

4. **Testing:**
```bash
# Test with dry run
./email_processor --dry-run --verbose

# Verify configuration
python3 -c "from config_loader import get_config; config = get_config(); print('✅ Config valid')"

# Test CustomKB connectivity
customkb query okusimail "test query" --quiet
```

### Automated Processing

#### Cron Job Setup
```bash
# Add to vmail user's crontab: crontab -u vmail -e
# Process emails every 5 minutes
*/5 * * * * cd /ai/scripts/customkb/mailer && ./email_processor >> /var/log/email_processor.log 2>&1
```

#### Systemd Service
```ini
# /etc/systemd/system/email-processor.service
[Unit]
Description=Email Auto-Reply Processor
After=network.target

[Service]
Type=oneshot
User=vmail
Group=vmail
WorkingDirectory=/ai/scripts/customkb/mailer
ExecStart=/ai/scripts/customkb/mailer/email_processor
Environment=PATH=/ai/scripts/customkb/mailer/.venv/bin:/usr/local/bin:/usr/bin:/bin
Environment=OPENAI_API_KEY=your-key-here
Environment=ANTHROPIC_API_KEY=your-key-here

[Install]
WantedBy=multi-user.target
```

#### Systemd Timer (Alternative to Cron)
```ini
# /etc/systemd/system/email-processor.timer
[Unit]
Description=Run Email Processor every 5 minutes

[Timer]
OnCalendar=*:0/5
Persistent=true

[Install]
WantedBy=timers.target
```

Enable with:
```bash
sudo systemctl enable email-processor.timer
sudo systemctl start email-processor.timer
```

## Monitoring and Maintenance

### Health Checks

```bash
# Configuration validation
python3 -c "from config_loader import get_config; get_config(); print('✅ Config OK')"

# Processor initialization test
python3 -c "from email_processor import EmailProcessor; EmailProcessor(); print('✅ Processor OK')"

# CustomKB connectivity test
customkb query okusimail "test query" --quiet
```

### Log Analysis

```bash
# Monitor processing in real-time
tail -f /var/log/email_processor.log

# View processing statistics
grep "Processing Summary" /var/log/email_processor.log

# Check for errors
grep "ERROR\|Failed\|Error" /var/log/email_processor.log

# Monitor draft generation
ls -la /home/vmail/okusi.dev/contact/.Drafts/cur/
```

### Performance Monitoring

```bash
# Check processing times
grep "Processing completed" /var/log/email_processor.log

# Monitor email discovery efficiency
grep "Found.*emails to process" /var/log/email_processor.log

# Track success rates
grep -E "(reply_generated|spam_detected|no_ticket_pattern)" /var/log/email_processor.log
```

## Customization

### Adding New Consultants

1. **Update configuration:**
```yaml
consultants:
  new_specialist:
    name: "New Consultant Name"
    email: "consultant@okusi.id"
    title: "Specialist Title"
    phone: "+628123456789"

keywords:
  new_specialist: ["keyword1", "keyword2", "keyword3"]
```

2. **No code changes required** - configuration is loaded dynamically

### Modifying AI Models

```yaml
models:
  openai:
    spam_detection: "gpt-4"  # Updated model
  anthropic:
    spam_detection: "claude-3-opus-20240229"  # Updated model
  customkb:
    knowledge_base: "new_knowledge_base"
    role: "Updated role description..."
```

### Custom Prompt Templates

```yaml
prompts:
  spam_detection: |
    Custom spam detection prompt with {content_for_analysis}
  
  customkb_query: |
    Custom reply generation prompt with:
    {sender_context}
    {original_email}
  
  email_signature: |
    Custom signature format:
    {consultant_name} - {consultant_title}
    {consultant_phone} | {consultant_email}
```

## Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| Permission denied | Not running as vmail user | Use `sudo -u vmail ./email_processor` |
| No emails found | Incorrect base directory | Check `directories.base_dir` in config |
| AI API errors | Invalid/missing API keys | Verify environment variables |
| CustomKB failures | Knowledge base unavailable | Test with `customkb query okusimail "test"` |
| Configuration errors | Invalid YAML syntax | Validate with `python3 -c "import yaml; yaml.safe_load(open('email_config.yaml'))"` |

### Debug Mode

```bash
# Enable verbose logging for troubleshooting
./email_processor --dry-run --verbose

# Check specific email processing
./email_processor --file '/path/to/email' --dry-run --verbose

# Test configuration loading
python3 -c "
from config_loader import get_config
import logging
logging.basicConfig(level=logging.DEBUG)
config = get_config()
print('Base dir:', config.get_base_dir())
print('Models:', config.get_openai_model())
"
```

### Performance Tuning

```yaml
# Adjust processing parameters in email_config.yaml
email:
  body_limit: 3000      # Increase for longer emails
  analysis_limit: 1500  # Increase for better spam detection
  process_delay: 0.2    # Decrease for faster processing

timeouts:
  customkb_query: 180   # Increase for complex queries
  find_command: 60      # Increase for large email directories
```

## Integration with CustomKB

### Knowledge Base Requirements

The system requires a CustomKB knowledge base named `okusimail` (configurable) containing:
- Indonesian corporate law information
- Business formation procedures
- Visa and immigration processes
- Tax and accounting regulations
- Standard response templates

### CustomKB Query Format

```bash
customkb query okusimail "
Please generate a professional email reply to this business inquiry...

SENDER CONTEXT: Sender Location: Australia

Original Email:
From: client@example.com
Subject: Re: [#:INQ001] Company formation inquiry
Date: Thu, 16 May 2024 14:30:00 +0700

Message:
I need help setting up a PMA company in Indonesia...
" -R "You are a professional legal services consultant..." --quiet
```

## Development

### Code Structure

```
/ai/scripts/customkb/mailer/
├── email_processor         # Bash launcher script
├── email_processor.py      # Main processing engine (780 lines)
├── config_loader.py        # Configuration manager (255 lines)
├── email_config.yaml       # System configuration (200 lines)
├── CLAUDE.md              # Development guide
└── README.md              # This file
```

### Development Setup

```bash
# Clone repository
cd /ai/scripts/customkb/mailer

# Create development environment
python3 -m venv .venv
source .venv/bin/activate
pip install PyYAML anthropic openai

# Run tests
python3 -c "from config_loader import get_config; get_config(); print('✅ Config test passed')"
python3 -c "from email_processor import EmailProcessor; EmailProcessor(); print('✅ Processor test passed')"

# Development run
./email_processor --dry-run --verbose
```

### Code Standards

- **Python Style**: 2-space indentation, comprehensive docstrings
- **Error Handling**: Graceful degradation with detailed logging
- **Configuration**: All hardcoded values externalized to YAML
- **Documentation**: Function/class docstrings with parameters and return values
- **Security**: Input validation, privilege separation, API key protection

## Support

### Documentation
- **Development Guide**: See `CLAUDE.md` for detailed coding guidelines
- **Configuration Reference**: All options documented in `email_config.yaml`
- **API Documentation**: Comprehensive docstrings in source code

### Logs and Diagnostics
- **Processing Logs**: `/var/log/email_processor.log` (if configured)
- **Debug Mode**: Use `--verbose` flag for detailed information
- **Health Checks**: Built-in configuration and connectivity validation

### Community
For technical support and development questions, consult the codebase documentation and configuration examples provided in this repository.

#fin