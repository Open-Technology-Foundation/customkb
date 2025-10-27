# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the **Okusi Associates Email Auto-Reply System** - a production-ready AI-powered email processing system that automatically generates professional replies to business inquiries using CustomKB knowledge base integration. The system processes emails in Maildir format, uses AI for spam detection and reply generation, and operates entirely as the `vmail` user for security.

## Core Architecture

### Key Components
- **`email_processor`** - Bash launcher script that auto-switches to vmail user and activates venv
- **`email_processor.py`** - Main Python processing engine (~780 lines)
- **`config_loader.py`** - Configuration management system (~470 lines) 
- **`email_config.yaml`** - Comprehensive YAML configuration (~200 lines)

### Configuration-Driven Design
The system uses a sophisticated YAML-based configuration that externalizes ALL hardcoded values:
- Directory paths and file patterns
- AI model selection (OpenAI/Anthropic/CustomKB)
- Consultant assignment database with contact details
- Email classification keywords
- Prompt templates with variable substitution
- Processing limits and security settings

## Development Commands

### Environment Setup
```bash
# Always activate the virtual environment first
source .venv/bin/activate

# Install dependencies
pip install PyYAML anthropic openai
```

### Running the System
```bash
# The launcher script automatically handles user switching and venv activation
./email_processor                    # Process new emails since last run
./email_processor --full-scan        # Process all unreplied emails
./email_processor --dry-run          # Evaluate only, no replies generated
./email_processor --verbose          # Enable debug logging
./email_processor --file '/path/to/email'  # Process specific email

# Manual execution (after switching to vmail user)
sudo -u vmail ./email_processor [options]
```

### Testing and Validation
```bash
# Test configuration loading
python3 -c "from config_loader import get_config; get_config(); print('✅ Config OK')"

# Test processor initialization  
python3 -c "from email_processor import EmailProcessor; EmailProcessor(); print('✅ Processor OK')"

# Validate email processing in dry-run mode
./email_processor --dry-run --verbose

# Check CustomKB connectivity
customkb query okusimail "test query" --quiet
```

## Critical Development Context

### Security Model
- **User Isolation**: System runs entirely as `vmail` user (owns email files)
- **No Privilege Escalation**: No sudo privileges or group memberships required  
- **API Key Management**: OpenAI/Anthropic keys stored in environment variables only
- **Input Validation**: All email headers and content validated before processing

### Email Processing Pipeline
1. **Discovery**: Scan Maildir for emails with `:2,S` pattern (read but not replied)
2. **Ticket Detection**: Check for ticket pattern `[#:alphanumeric]` in subject
3. **Spam Classification**: AI-powered LEGITIMATE/SPAM detection
4. **Consultant Assignment**: Keyword-based routing to appropriate consultant
5. **Reply Generation**: CustomKB integration for professional responses
6. **Draft Creation**: Save to `.Drafts/cur/` with proper Maildir flags
7. **Flag Update**: Mark original as replied (`:2,S` → `:2,RS`)

### Maildir Integration
The system works with standard Maildir format:
- `/home/vmail/okusi.dev/contact/cur/` - Read emails  
- `/home/vmail/okusi.dev/contact/new/` - Unread emails
- `/home/vmail/okusi.dev/contact/.Drafts/cur/` - Generated drafts

Filename format: `timestamp.unique_id.hostname:2,flags`
- Flags: `S`=Seen, `R`=Replied, `T`=Trashed, `F`=Flagged, `D`=Draft

## Code Style Requirements

### Python Conventions
- 2-space indentation (not 4 spaces)
- Shebang: `#!/usr/bin/env python3`
- Import order: standard library, third-party, local modules
- Always end files with `#fin`
- Comprehensive error handling with logging
- Descriptive function/variable names with docstrings

### Configuration Patterns
```python
# Load configuration
from config_loader import get_config
config = get_config()

# Access settings through methods
base_dir = config.get_base_dir()
consultant = config.get_consultant('company')
model = config.get_openai_model()
```

### Key Configuration Methods

#### Directory and Path Management
- `get_base_dir()` - Base email directory path
- `get_drafts_dir()` - Full path to drafts directory
- `get_temp_dir()` - Temporary directory for file operations
- `get_timestamp_file()` - Timestamp file for incremental processing

#### Email Processing Configuration
- `get_ticket_pattern()` - Regex pattern for ticket detection
- `get_unreplied_pattern()` - File pattern for unreplied emails
- `get_body_limit()`, `get_analysis_limit()` - Content processing limits
- `get_hostname()` - System hostname for Maildir files
- `get_process_delay()` - Delay between email processing

#### AI Model Configuration
- `get_openai_model()`, `get_anthropic_model()` - AI model names
- `get_openai_params()`, `get_anthropic_params()` - API parameters
- `get_customkb_settings()` - CustomKB configuration dict
- `get_customkb_timeout()`, `get_find_timeout()` - Timeout settings

#### Consultant Assignment System
- `get_consultant(type)` - Get consultant by type
- `get_all_consultants()` - Get complete consultant database
- `determine_consultant_type(content)` - Keyword-based routing
- `get_country_name(code)` - Country code to name conversion

#### Prompt Template System
- `get_spam_detection_prompt(content)` - Formatted spam detection prompt
- `get_customkb_query_prompt(context, email)` - CustomKB query formatting
- `get_email_signature(consultant)` - Consultant signature formatting

#### Maildir Flag Management
- `has_maildir_flag(filename, flag)` - Check for specific flags
- `add_maildir_flag(filename, flag)` - Add flag to filename
- `remove_maildir_flag(filename, flag)` - Remove flag from filename
- `get_maildir_flag(flag_name)` - Get flag character by name

## Development Notes

### Adding New Features

#### New Consultants
1. Add consultant information to `consultants` section:
```yaml
consultants:
  new_specialist:
    name: "Consultant Name"
    email: "consultant@okusi.id"
    title: "Professional Title"
    phone: "+628123456789"
```

2. Add classification keywords:
```yaml
keywords:
  new_specialist: ["keyword1", "keyword2", "domain-specific-term"]
```

3. No code changes required - configuration is loaded dynamically

#### New AI Models
Update model configuration in `models` section:
```yaml
models:
  openai:
    spam_detection: "gpt-4"  # New model
    max_tokens: 15           # Adjusted parameters
  anthropic:
    spam_detection: "claude-3-opus-20240229"  # New model
  customkb:
    knowledge_base: "new_kb_name"  # Different knowledge base
    role: "Updated role description..."
```

#### New Prompt Templates
Add custom prompts with variable substitution:
```yaml
prompts:
  custom_prompt: |
    Custom template with {variable1} and {variable2}
    Multi-line content supported.
    
    Use proper YAML literal block syntax.
```

#### New Processing Rules
Modify classification logic:
```yaml
keywords:
  existing_type: ["old", "keywords"]
  new_classification: ["new", "business", "domain", "keywords"]
```

### CustomKB Integration
The system integrates with the larger CustomKB ecosystem:
- Uses `customkb query okusimail` for reply generation
- Requires knowledge base "okusimail" to be available
- Configurable timeout and role settings
- Professional legal services context for Indonesian corporate law

### Performance Considerations
- **Incremental Processing**: Timestamp-based to avoid reprocessing
- **Rate Limiting**: Configurable delays between email processing
- **Memory Management**: Limited email content processing (2000 char limit)
- **Connection Reuse**: AI clients reused across requests

### Monitoring and Logging

#### Logging System
- **Configurable Verbosity**: `--verbose` flag enables DEBUG level logging
- **Structured Output**: Consistent log format with timestamps and context
- **Error Context**: All exceptions logged with full context and stack traces
- **Processing Statistics**: Detailed counts of processing outcomes

#### Statistics Tracking
The system tracks comprehensive processing metrics:
```python
stats = {
  'total_found': 0,         # Emails discovered for processing
  'already_replied': 0,     # Emails skipped due to reply flag
  'no_ticket_pattern': 0,   # Emails without required ticket pattern
  'spam_detected': 0,       # Emails classified as spam
  'reply_generated': 0,     # Successfully processed emails
  'errors': 0              # Processing failures
}
```

#### Health Check Utilities
```bash
# Configuration validation
python3 -c "from config_loader import get_config; get_config()"

# Email processor initialization
python3 -c "from email_processor import EmailProcessor; EmailProcessor()"

# AI client connectivity
# (Checks environment variables and import availability)

# CustomKB integration test
customkb query okusimail "test" --quiet

# File system permissions
ls -la /home/vmail/okusi.dev/contact/
```

#### Error Handling Patterns
- **Graceful Degradation**: Individual email failures don't stop batch processing
- **Detailed Error Context**: All errors include file paths, email subjects, and operation context
- **API Fallbacks**: OpenAI → Anthropic fallback for spam detection
- **Conservative Defaults**: Unknown emails treated as legitimate if AI unavailable

## Coding Principles
- K.I.S.S. (Keep It Simple, Stupid)
- "The best process is no process"
- "Everything should be made as simple as possible, but not simpler"
- Configuration over code changes
- Security through user isolation and input validation

## Development Best Practices

### Core Principles
- **K.I.S.S. (Keep It Simple, Stupid)**: Prefer simple, obvious solutions over clever complexity
- **"The best process is no process"**: Eliminate unnecessary steps and complexity
- **"Everything should be made as simple as possible, but not simpler"**: Balance simplicity with functionality
- **Configuration over Code**: Externalize behavior to YAML rather than hardcoding
- **Security through Isolation**: Use user separation and input validation, not complex permissions

### Code Quality Standards

#### Documentation Requirements
- **Module Docstrings**: Comprehensive description with usage examples
- **Function Docstrings**: Parameters, return values, exceptions, and examples
- **Inline Comments**: Explain non-obvious logic, business rules, and workarounds
- **Configuration Comments**: Document all YAML settings with purpose and examples

#### Error Handling Standards
```python
# Proper error handling pattern
try:
  result = risky_operation()
except SpecificException as e:
  logger.error(f"Operation failed with context: {e}", exc_info=True)
  # Graceful degradation or re-raise with context
  raise ProcessingError(f"Failed to process {item}: {e}") from e
```

#### Configuration Access Patterns
```python
# Always use configuration methods, never direct dict access
config = get_config()

# Good
base_dir = config.get_base_dir()
model = config.get_openai_model()

# Bad
base_dir = config.config['directories']['base_dir']
model = config.config['models']['openai']['spam_detection']
```

#### Security Practices
- **Input Validation**: Validate all email content and file paths
- **API Key Handling**: Environment variables only, never log or persist
- **File Operations**: Use atomic writes and proper permissions
- **User Context**: Maintain vmail user context throughout processing

### Testing and Validation

#### Development Testing
```bash
# Always test in dry-run mode first
./email_processor --dry-run --verbose

# Test with specific emails
./email_processor --file '/path/to/test/email' --dry-run

# Validate configuration changes
python3 -c "from config_loader import get_config; get_config()"
```

#### Production Deployment
```bash
# Comprehensive validation before deployment
./email_processor --dry-run --verbose
customkb query okusimail "test" --quiet
python3 -c "import yaml; yaml.safe_load(open('email_config.yaml'))"

# Monitor initial production runs
tail -f /var/log/email_processor.log
```

### Common Development Tasks

#### Debugging Email Processing Issues
1. **Enable verbose logging**: `./email_processor --dry-run --verbose`
2. **Check specific email**: `./email_processor --file '/path/to/email' --dry-run --verbose`
3. **Validate configuration**: Test each configuration method individually
4. **Test AI connectivity**: Verify API keys and client initialization
5. **Check file permissions**: Ensure vmail user has proper access

#### Performance Optimization
1. **Monitor processing times**: Track statistics in logs
2. **Adjust content limits**: Modify `body_limit` and `analysis_limit`
3. **Tune delays**: Adjust `process_delay` for system load
4. **Review timeout settings**: Update `customkb_timeout` and `find_timeout`

#### Configuration Management
1. **Version control**: Track all configuration changes in git
2. **Environment-specific configs**: Use different files per environment
3. **Validation**: Always test configuration changes in dry-run mode
4. **Backup**: Keep working configurations for rollback

### Integration Guidelines

#### CustomKB Integration
- **Knowledge Base**: Ensure "okusimail" knowledge base is available and current
- **Role Definition**: Keep role prompts specific to Indonesian legal services
- **Query Format**: Use structured prompts with clear sender context
- **Timeout Handling**: Set appropriate timeouts for complex queries

#### AI API Integration
- **Fallback Strategy**: OpenAI primary, Anthropic fallback for spam detection
- **Rate Limiting**: Respect API rate limits with configurable delays
- **Error Handling**: Graceful degradation when APIs unavailable
- **Cost Management**: Use minimal tokens for spam detection (max_tokens: 10)

## Troubleshooting Guide

### Common Development Issues

#### Configuration Problems
```bash
# YAML syntax errors
python3 -c "import yaml; yaml.safe_load(open('email_config.yaml'))"

# Missing configuration sections
python3 -c "from config_loader import get_config; config = get_config(); print('Config valid')"

# Path resolution issues
ls -la /home/vmail/okusi.dev/contact/
ls -la /home/vmail/okusi.dev/contact/.Drafts/cur/
```

#### Email Processing Issues
```bash
# No emails found
./email_processor --full-scan --dry-run --verbose
find /home/vmail/okusi.dev/contact/cur -name '*:2,S' -type f

# Permission problems
sudo -u vmail ls -la /home/vmail/okusi.dev/contact/
sudo -u vmail ./email_processor --dry-run

# AI client failures
export OPENAI_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"
python3 -c "from email_processor import EmailProcessor; p = EmailProcessor(); print('AI clients OK')"
```

#### CustomKB Integration Issues
```bash
# Knowledge base availability
customkb query okusimail "test query" --quiet

# Role and timeout configuration
cat email_config.yaml | grep -A 5 "customkb:"

# Query format validation
./email_processor --file '/path/to/email' --dry-run --verbose | grep -A 10 "CustomKB"
```

### Performance Tuning

#### Email Discovery Optimization
```yaml
# Adjust find command timeout for large directories
timeouts:
  find_command: 60  # Increase for large email volumes
```

#### Content Processing Limits
```yaml
# Balance processing speed vs. accuracy
email:
  body_limit: 3000      # Increase for longer emails
  analysis_limit: 1500  # More content for spam detection
  process_delay: 0.2    # Decrease for faster processing
```

#### API Timeout Settings
```yaml
# Adjust for network conditions and query complexity
models:
  customkb:
    timeout: 180  # Increase for complex queries

timeouts:
  customkb_query: 180  # Match model timeout
```

### Maintenance Tasks

#### Regular Health Checks
```bash
# Weekly configuration validation
python3 -c "from config_loader import get_config; get_config(); print('✅ Config OK')"

# Monthly processing statistics review
grep "Processing Summary" /var/log/email_processor.log | tail -30

# Quarterly performance analysis
grep "Processing completed" /var/log/email_processor.log | grep "$(date +'%Y-%m')"
```

#### Log Rotation and Cleanup
```bash
# Archive old logs
sudo logrotate -f /etc/logrotate.d/email-processor

# Clean old draft files (if needed)
find /home/vmail/okusi.dev/contact/.Drafts/cur -name '*:2,D' -mtime +30
```

#### Knowledge Base Updates
```bash
# Test knowledge base after updates
customkb query okusimail "test corporate law query" --quiet
customkb query okusimail "test visa inquiry" --quiet
```

#fin