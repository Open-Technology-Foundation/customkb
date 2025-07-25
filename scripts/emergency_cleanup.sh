#!/bin/bash
# Emergency cleanup script for CustomKB after system crashes

echo "=== CustomKB Emergency Cleanup ==="
echo "This script will clean up after test crashes"
echo

# Kill any remaining pytest processes
echo "Killing pytest processes..."
pkill -9 pytest 2>/dev/null
pkill -9 python.*pytest 2>/dev/null

# Kill any python processes that might be test-related
echo "Checking for runaway Python processes..."
ps aux | grep -E "python.*test|test.*\.py" | grep -v grep | awk '{print $2}' | xargs -r kill -9 2>/dev/null

# Clean pytest cache
echo "Cleaning pytest cache..."
find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null
find /tmp -name "pytest-of-*" -type d -mtime +1 -exec rm -rf {} + 2>/dev/null

# Clean temporary test files
echo "Cleaning temporary test files..."
rm -rf /tmp/test_* 2>/dev/null
rm -rf /tmp/tmp* 2>/dev/null
find /tmp -name "*.db" -mtime +1 -delete 2>/dev/null

# Clean any FAISS temporary files
echo "Cleaning FAISS temporary files..."
find /tmp -name "*.faiss" -delete 2>/dev/null
find /tmp -name "*.index" -delete 2>/dev/null

# Clear Python cache
echo "Cleaning Python cache..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -name "*.pyc" -delete 2>/dev/null

# Check memory usage
echo
echo "Current memory usage:"
free -h

# Suggest system cache clearing
echo
echo "To free more memory, you can run (requires sudo):"
echo "  sync && echo 3 | sudo tee /proc/sys/vm/drop_caches"
echo

# Check for zombie processes
zombies=$(ps aux | grep -E "\s+Z\s+" | grep -v grep | wc -l)
if [ $zombies -gt 0 ]; then
  echo "WARNING: Found $zombies zombie processes!"
  echo "You may need to reboot to clear them."
fi

echo "Cleanup complete."
echo
echo "Safe next steps:"
echo "1. Run diagnostics: python diagnose_crashes.py"
echo "2. Check system logs: dmesg | tail -50"
echo "3. Start with single test: pytest -v --timeout=10 tests/unit/test_config_manager.py::TestKnowledgeBase::test_init"

#fin