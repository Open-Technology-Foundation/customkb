#!/usr/bin/env python3
"""
Test script for Thunderbird compose methods
Tests various ways to open Thunderbird compose window with pre-filled content
"""

import subprocess
import tempfile
import time
import sys
import urllib.parse
from pathlib import Path

def test_method_1_basic():
    """Test basic compose with URL encoding"""
    print("\n=== Testing Method 1: Basic Compose ===")
    
    # Simple test
    cmd = [
        'thunderbird', '-compose',
        "to='test@example.com',subject='Test Subject',body='Hello World'"
    ]
    
    print(f"Command: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        print(f"Return code: {result.returncode}")
        if result.stderr:
            print(f"Stderr: {result.stderr}")
        return result.returncode == 0
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_method_1_advanced():
    """Test compose with proper CRLF encoding and special characters"""
    print("\n=== Testing Method 1: Advanced Compose with CRLF ===")
    
    # Multi-line body with proper encoding
    body = "Hello there,\r\n\r\nThis is line 1.\r\nThis is line 2.\r\n\r\nBest regards,\r\nTest"
    encoded_body = urllib.parse.quote(body, safe='')
    # Ensure proper CRLF encoding
    encoded_body = encoded_body.replace('%0A', '%0D%0A')
    
    cmd = [
        'thunderbird', '-compose',
        f"to='test@example.com',subject='Test%20with%20spaces',body='{encoded_body}'"
    ]
    
    print(f"Command: thunderbird -compose [params with encoded body]")
    print(f"Body preview: {body[:50]}...")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        print(f"Return code: {result.returncode}")
        return result.returncode == 0
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_method_2_html():
    """Test compose with HTML message file"""
    print("\n=== Testing Method 2: HTML Message File ===")
    
    html_content = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <style>
        body { font-family: Arial, sans-serif; font-size: 14px; }
        .greeting { color: #333; font-weight: bold; }
        .signature { color: #666; font-style: italic; }
    </style>
</head>
<body>
    <p class="greeting">Hello there,</p>
    <p>This is a test email with <strong>HTML formatting</strong>.</p>
    <p>It includes:</p>
    <ul>
        <li>Bullet points</li>
        <li><em>Italic text</em></li>
        <li><strong>Bold text</strong></li>
    </ul>
    <p class="signature">Best regards,<br>Test System</p>
</body>
</html>"""
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
        f.write(html_content)
        html_file = f.name
    
    cmd = [
        'thunderbird', '-compose',
        f"to='test@example.com',subject='HTML Test',message='{html_file}'"
    ]
    
    print(f"HTML file: {html_file}")
    print(f"Command: thunderbird -compose [with message file]")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        print(f"Return code: {result.returncode}")
        return result.returncode == 0
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_method_3_eml():
    """Test opening EML file with draft headers"""
    print("\n=== Testing Method 3: EML with Draft Headers ===")
    
    eml_content = """From: Test Sender <sender@example.com>
To: test@example.com
Subject: Test EML Draft
Date: Thu, 15 Jun 2025 10:00:00 +0000
Message-ID: <test123@example.com>
X-Mozilla-Draft-Info: internal/draft; vcard=0; receipt=0; DSN=0; uuencode=0
X-Unsent: 1
MIME-Version: 1.0
Content-Type: text/plain; charset=UTF-8
Content-Transfer-Encoding: 8bit

Hello,

This is a test draft email created with Mozilla draft headers.

The X-Mozilla-Draft-Info header should help Thunderbird recognize this as a draft.
The X-Unsent header is for Outlook compatibility.

Best regards,
Test System
"""
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.eml', delete=False) as f:
        f.write(eml_content)
        eml_file = f.name
    
    print(f"EML file: {eml_file}")
    
    # Try opening directly with Thunderbird
    cmd = ['thunderbird', eml_file]
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        print(f"Return code: {result.returncode}")
        print("Note: EML files typically open in read-only mode. Use 'Edit as New' in Thunderbird.")
        return result.returncode == 0
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_mailto_format():
    """Test mailto: URL format"""
    print("\n=== Testing Mailto: URL Format ===")
    
    # Build mailto URL
    params = {
        'subject': 'Test Mailto Subject',
        'body': 'Line 1\r\nLine 2\r\n\r\nBest regards'
    }
    
    # Encode parameters
    encoded_params = urllib.parse.urlencode(params, quote_via=urllib.parse.quote)
    # Fix line break encoding
    encoded_params = encoded_params.replace('%0A', '%0D%0A')
    
    mailto_url = f"mailto:test@example.com?{encoded_params}"
    
    cmd = ['thunderbird', '-compose', mailto_url]
    print(f"Mailto URL: {mailto_url[:100]}...")
    print(f"Command: thunderbird -compose [mailto URL]")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        print(f"Return code: {result.returncode}")
        return result.returncode == 0
    except Exception as e:
        print(f"Error: {e}")
        return False

def main():
    """Run all tests"""
    print("Thunderbird Compose Methods Test Suite")
    print("=" * 50)
    
    if len(sys.argv) > 1:
        # Run specific test
        test_name = sys.argv[1]
        if test_name == "1":
            test_method_1_basic()
        elif test_name == "1a":
            test_method_1_advanced()
        elif test_name == "2":
            test_method_2_html()
        elif test_name == "3":
            test_method_3_eml()
        elif test_name == "mailto":
            test_mailto_format()
        else:
            print(f"Unknown test: {test_name}")
            print("Available tests: 1, 1a, 2, 3, mailto")
    else:
        # Run all tests
        print("\nRunning all tests...")
        print("Note: Each test will try to open Thunderbird. Close the compose window to continue.\n")
        
        tests = [
            ("Basic Compose", test_method_1_basic),
            ("Advanced Compose", test_method_1_advanced),
            ("HTML Message", test_method_2_html),
            ("EML Draft", test_method_3_eml),
            ("Mailto Format", test_mailto_format)
        ]
        
        results = []
        for name, test_func in tests:
            print(f"\nPress Enter to run: {name}")
            input()
            
            success = test_func()
            results.append((name, success))
            
            if success:
                print("✅ Test passed")
            else:
                print("❌ Test failed")
            
            time.sleep(2)
        
        # Summary
        print("\n" + "=" * 50)
        print("Test Summary:")
        for name, success in results:
            status = "✅ PASS" if success else "❌ FAIL"
            print(f"{name}: {status}")

if __name__ == "__main__":
    main()