# Thunderbird Compose Methods Documentation

This document details the various methods for programmatically invoking Thunderbird's compose window with pre-filled content, as implemented in the Thunderbird External AI Reply system.

## Overview

The system attempts multiple methods in sequence to maximize compatibility with different Thunderbird configurations and versions. All methods work with Thunderbird 128 and don't require extensions.

## Method 1: Command-Line Compose with Proper Encoding

**Primary method using `thunderbird -compose` with URL-encoded parameters.**

### Syntax
```bash
thunderbird -compose "to='recipient@example.com',subject='Subject',body='Body%20with%20spaces'"
```

### Key Features
- Proper CRLF encoding for line breaks (`%0D%0A`)
- URL encoding for special characters
- Nested quotes: outer double quotes, inner single quotes
- Support for multiple recipients (comma-separated)
- Threading support with `in-reply-to` parameter

### Example
```bash
thunderbird -compose "to='user@example.com',subject='Re%3A%20Meeting',body='Hi%20there%2C%0D%0AThank%20you%20for%20your%20email.%0D%0ABest%20regards'"
```

### Limitations
- Maximum URL length restrictions
- Complex formatting requires extensive encoding
- No automatic sending (user must click Send)

## Method 2: HTML Message File

**Alternative method using the `message=` parameter with an HTML file.**

### Syntax
```bash
thunderbird -compose "to='recipient@example.com',subject='Subject',message='/path/to/message.html'"
```

### Key Features
- Supports rich HTML formatting
- No URL length limitations
- Better handling of complex content
- Preserves formatting and spacing

### HTML Template
```html
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <style>
        body { font-family: Arial, sans-serif; font-size: 14px; line-height: 1.6; }
        p { margin: 0 0 10px 0; }
    </style>
</head>
<body>
    <!-- Email content here -->
</body>
</html>
```

## Method 3: EML File with Draft Headers

**Creates an EML file with Mozilla-specific draft headers.**

### Key Headers
- `X-Mozilla-Draft-Info: internal/draft; vcard=0; receipt=0; DSN=0; uuencode=0`
- `X-Unsent: 1` (for Outlook compatibility)

### EML Structure
```
From: sender@example.com
To: recipient@example.com
Subject: Re: Original Subject
In-Reply-To: <original-message-id>
References: <thread-references>
X-Mozilla-Draft-Info: internal/draft; vcard=0; receipt=0; DSN=0; uuencode=0
X-Unsent: 1
Content-Type: text/plain; charset=UTF-8

Email body content here...
```

### Opening Methods
1. Direct: `thunderbird /path/to/file.eml`
2. System default: `xdg-open /path/to/file.eml`

### Limitations
- Thunderbird opens EML files in read mode by default
- User must use "Edit as New" to modify the message
- Draft headers are not always honored

## Method 4: Desktop Fallback

**Final fallback saves both TXT and EML versions to the Desktop.**

### Files Created
1. `ai_reply_[timestamp].txt` - Plain text for copy/paste
2. `ai_reply_[timestamp].eml` - EML file for manual import

### Benefits
- Always works regardless of Thunderbird configuration
- Provides multiple options for the user
- Preserves all reply content and formatting

## Implementation Details

### URL Encoding Best Practices
```python
# Replace newlines with CRLF
body_with_crlf = reply_content.replace('\n', '\r\n')
# URL encode
encoded_body = urllib.parse.quote(body_with_crlf, safe='')
# Ensure proper line break encoding
encoded_body = encoded_body.replace('%0A', '%0D%0A')
```

### Quote Nesting Rules
- Outer quotes: Double quotes for the entire `-compose` argument
- Inner quotes: Single quotes for parameters containing special characters
- Example: `"to='user@example.com',subject='Re: Test'"`

### Error Handling
The system logs detailed information about each method's success or failure:
- Return codes from subprocess calls
- Stderr output for debugging
- Timeout handling for hung processes

## Troubleshooting

### Common Issues

1. **Thunderbird not opening**
   - Check if Thunderbird is in PATH
   - Verify Thunderbird is installed
   - Check for running instances

2. **Encoding problems**
   - Ensure proper URL encoding
   - Use `%0D%0A` for line breaks, not just `%0A`
   - Escape special characters in subjects

3. **EML files opening in read-only mode**
   - This is expected behavior
   - Users should right-click → "Edit as New"
   - Or use Message → Edit Message As New

### Debug Mode
Enable debug logging to see detailed command execution:
```python
logger.setLevel(logging.DEBUG)
```

## Future Enhancements

1. **Configuration Options**
   - User-selectable preferred method
   - Customizable From address
   - Template selection

2. **Additional Methods**
   - WebExtension API integration
   - D-Bus interface exploration
   - Platform-specific improvements

3. **Better Format Support**
   - Markdown to HTML conversion
   - Rich text formatting preservation
   - Attachment handling

## References

- [RFC 2368 - The mailto URL scheme](https://www.rfc-editor.org/rfc/rfc2368)
- [RFC 3676 - Text/Plain Format Parameter](https://www.rfc-editor.org/rfc/rfc3676)
- [Thunderbird Command Line Options](https://support.mozilla.org/kb/command-line-options)
- [Mozilla Bugzilla - Compose Arguments](https://bugzilla.mozilla.org/show_bug.cgi?id=488443)