# Thunderbird External AI Reply System

A complete external solution for generating AI-powered email replies in Thunderbird without extensions.

## 🚀 Features

- **No Extensions Required**: Works entirely outside Thunderbird - no XPI/extension issues
- **Multiple AI Methods**: Template, CustomKB, OpenAI, Anthropic Claude
- **Smart Message Detection**: Window title monitoring + keyboard automation
- **IMAP Integration**: Direct access to raw message content from mail server
- **Global Hotkey**: Alt+R triggers AI reply generation from anywhere
- **Snap Compatible**: Works with Snap Thunderbird packages
- **Multiple Compose Methods**: Robust fallback system for opening compose window
- **Proper URL Encoding**: Handles multiline content with CRLF encoding

## 📁 Project Structure

```
thunderbird-external-ai/
├── thunderbird_ai_reply.py      # Main application
├── thunderbird_message_detector.py  # Message detection logic
├── imap_client.py               # IMAP server access
├── ai_reply_generator.py        # AI reply generation
├── setup_hotkey.py              # Global hotkey setup
└── README.md                    # This file
```

## ⚡ Quick Start

### 1. **Test Message Detection**
```bash
cd /ai/scripts/customkb/mailer/thunderbird-external-ai

# Test if message detection works
python3 thunderbird_ai_reply.py --test
```

### 2. **Create Configuration**
```bash
# Create default config file
python3 thunderbird_ai_reply.py --create-config

# Edit the config file
nano ~/.thunderbird-ai-reply.json
```

### 3. **Basic Usage (Manual)**
```bash
# Generate reply for currently selected email
python3 thunderbird_ai_reply.py
```

### 4. **Setup Global Hotkey**
```bash
# Setup Alt+R hotkey
python3 setup_hotkey.py
```

## 🔧 Configuration

The configuration file `~/.thunderbird-ai-reply.json` contains:

```json
{
  "ai": {
    "method": "template",           # template|customkb|openai|anthropic
    "customkb_name": "default",
    "openai_api_key": "",
    "anthropic_api_key": ""
  },
  "imap": {
    "server": "okusi0.okusi.co.id",
    "username": "contact@okusi.dev",
    "password": "your_password",
    "port": 993,
    "use_ssl": true
  },
  "reply": {
    "auto_open": true,
    "save_drafts": true
  }
}
```

### AI Methods

1. **Template** (Default): Intelligent template-based replies
2. **CustomKB**: Uses your existing CustomKB knowledge base
3. **OpenAI**: GPT-powered replies (requires API key)
4. **Anthropic**: Claude-powered replies (requires API key)

## 🎯 How It Works

### Message Detection Process

1. **Window Title Analysis**: Extracts subject from Thunderbird window title
2. **Keyboard Automation**: Uses Ctrl+U to get message source
3. **Header Parsing**: Extracts Message-ID, sender, subject, etc.
4. **IMAP Lookup**: Fetches full message content from server (if configured)

### AI Reply Generation

1. **Content Analysis**: Analyzes subject and body for context
2. **AI Processing**: Generates reply using chosen AI method
3. **Reply Formatting**: Creates properly formatted email reply
4. **Thunderbird Integration**: Opens reply in compose window

## 📋 Usage Scenarios

### Scenario 1: Template Replies (No Setup Required)
- Works immediately with intelligent templates
- Analyzes email content for context-aware responses
- Perfect for common business communications

### Scenario 2: CustomKB Integration
- Uses your existing knowledge base
- Searches relevant information for informed replies
- Combines AI with your specific knowledge

### Scenario 3: Full AI Integration
- OpenAI or Anthropic for advanced responses
- Natural language generation
- Contextual and personalized replies

## 🔨 Installation & Setup

### Prerequisites
```bash
# Required system tools
sudo apt install xdotool xclip notify-send

# Optional: for advanced hotkey setup
sudo apt install xbindkeys autokey-gtk

# Python dependencies (if using external AI)
pip install openai anthropic imaplib email
```

### Step 1: Test Basic Functionality
```bash
# 1. Open Thunderbird and select an email
# 2. Run test
python3 thunderbird_ai_reply.py --test

# You should see message details printed
```

### Step 2: Configure IMAP (Optional but Recommended)
```bash
# 1. Create config
python3 thunderbird_ai_reply.py --create-config

# 2. Edit config file with your credentials
nano ~/.thunderbird-ai-reply.json

# 3. Test IMAP connection
python3 imap_client.py
```

### Step 3: Setup Global Hotkey
```bash
# Run hotkey setup wizard
python3 setup_hotkey.py

# Choose your preferred method:
# - xbindkeys (most reliable)
# - AutoKey (GUI-based)
# - Custom daemon (Python-based)
```

### Step 4: Test End-to-End
```bash
# 1. Select an email in Thunderbird
# 2. Press Alt+R (or run manually)
# 3. Reply window should open with AI-generated content
```

## 🐛 Troubleshooting

### Message Detection Issues

**Problem**: "Could not detect current message"
```bash
# Check if Thunderbird is running
ps aux | grep thunderbird

# Test window detection
xdotool search --class thunderbird

# Check window title
xdotool getactivewindow getwindowname
```

**Solution**: Make sure an email is selected in Thunderbird

### IMAP Connection Issues

**Problem**: IMAP authentication fails
```bash
# Test IMAP settings manually
python3 -c "
from imap_client import IMAPMessageClient
client = IMAPMessageClient('okusi0.okusi.co.id')
print(client.connect('your_username', 'your_password'))
"
```

**Solution**: Verify server settings and credentials

### Hotkey Not Working

**Problem**: Alt+R doesn't trigger
```bash
# Check if hotkey daemon is running
ps aux | grep -E "(xbindkeys|autokey|hotkey_daemon)"

# Test manual execution
python3 thunderbird_ai_reply.py
```

**Solution**: Re-run hotkey setup or use manual execution

### AI Generation Fails

**Problem**: Reply generation fails
```bash
# Check logs
tail -f /tmp/thunderbird-ai-reply.log

# Test with template method (always works)
# Edit config: "method": "template"
```

## 🔧 Advanced Usage

### Thunderbird Compose Methods

The system uses multiple methods to open Thunderbird's compose window:

1. **Command-line compose with URL encoding** (Primary)
   - Proper CRLF encoding (`%0D%0A`) for line breaks
   - Nested quotes for complex parameters
   - Threading support with In-Reply-To headers

2. **HTML message file** (Fallback 1)
   - Rich formatting support
   - No URL length limitations
   - Better for complex content

3. **EML file with draft headers** (Fallback 2)
   - Mozilla-specific draft headers
   - Compatible with manual import
   - Preserves all email metadata

4. **Desktop save** (Final fallback)
   - Saves both TXT and EML versions
   - Always works regardless of configuration

See `THUNDERBIRD_COMPOSE_METHODS.md` for detailed documentation.

### Testing Compose Methods

Test different compose methods:
```bash
# Run all tests
python3 test_compose_methods.py

# Test specific method
python3 test_compose_methods.py 1    # Basic compose
python3 test_compose_methods.py 1a   # Advanced with CRLF
python3 test_compose_methods.py 2    # HTML message
python3 test_compose_methods.py 3    # EML draft
python3 test_compose_methods.py mailto # Mailto format
```

### Custom AI Integration

Add your own AI service by extending `ai_reply_generator.py`:

```python
def generate_custom_ai_reply(self, email_data: Dict[str, str]) -> Optional[str]:
    # Your AI service integration here
    return generated_reply
```

### Custom Message Processing

Extend `thunderbird_message_detector.py` for specialized detection:

```python
def detect_specific_message_type(self) -> Optional[Dict[str, str]]:
    # Custom detection logic
    return message_info
```

### Batch Processing

Process multiple emails:

```bash
# Get recent emails and generate replies
python3 -c "
from imap_client import IMAPMessageClient
client = IMAPMessageClient('server')
client.connect('user', 'pass')
messages = client.get_recent_messages(count=10)
for msg in messages:
    print(f'Processing: {msg[\"subject\"]}')
"
```

## 📊 Performance Notes

- **Message Detection**: ~2-3 seconds (includes automation)
- **IMAP Lookup**: ~1-2 seconds (depends on server)
- **AI Generation**: 
  - Template: Instant
  - CustomKB: 5-15 seconds
  - OpenAI/Anthropic: 3-10 seconds
- **Total Time**: 5-20 seconds depending on configuration

## 🔒 Security Considerations

- **Credentials**: Stored in plain text config file (use file permissions)
- **IMAP Access**: Uses standard email credentials
- **AI APIs**: API keys transmitted over HTTPS
- **Message Content**: Temporarily stored in memory only
- **Log Files**: May contain email subjects (configure log level)

## 🤝 Integration with CustomKB

This system integrates seamlessly with your existing CustomKB:

```python
# Automatic integration if CustomKB is configured
"ai": {
    "method": "customkb",
    "customkb_name": "your_kb_name"
}
```

The system will:
1. Use CustomKB's query system
2. Search relevant knowledge for the email content
3. Generate informed replies based on your knowledge base
4. Fall back to templates if CustomKB is unavailable

## 📈 Future Enhancements

- **Email Threading**: Better reply-to handling
- **Multiple AI Models**: Support for more AI services
- **GUI Interface**: Desktop application for configuration
- **Signature Integration**: Automatic signature insertion
- **Template Library**: Pre-built reply templates
- **Learning System**: Learn from user corrections

## 🎉 Success! 

You now have a complete external AI reply system that works without any Thunderbird extensions!

**Next Steps:**
1. Test with `--test` flag
2. Configure your AI method
3. Setup global hotkey
4. Start generating AI replies with Alt+R!