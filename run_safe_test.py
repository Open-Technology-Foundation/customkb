#!/usr/bin/env python
"""
Run a single safe test with monitoring.
"""

import subprocess
import sys
import time
import psutil
import os

# Activate virtual environment
venv_python = os.path.join(os.path.dirname(__file__), '.venv', 'bin', 'python')

print("=== Running Safe Test ===")
print(f"Memory before: {psutil.Process().memory_info().rss / 1024 / 1024:.1f}MB")

# Run a single simple test with timeout
cmd = [
    venv_python, '-m', 'pytest',
    '-v',
    '--timeout=30',
    '--timeout-method=thread',
    'tests/unit/test_config_manager.py::TestKnowledgeBase::test_init_with_kwargs'
]

print(f"Command: {' '.join(cmd)}")
print("-" * 60)

start_time = time.time()

try:
    # Run with real-time output
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                           text=True, bufsize=1, universal_newlines=True)
    
    # Stream output in real-time
    for line in proc.stdout:
        print(line, end='')
    
    proc.wait()
    duration = time.time() - start_time
    
    print("-" * 60)
    print(f"Exit code: {proc.returncode}")
    print(f"Duration: {duration:.1f}s")
    print(f"Memory after: {psutil.Process().memory_info().rss / 1024 / 1024:.1f}MB")
    
    if proc.returncode == 0:
        print("\n✅ Test passed successfully!")
    else:
        print("\n❌ Test failed!")
        
except KeyboardInterrupt:
    print("\n\nInterrupted by user")
    proc.terminate()
    sys.exit(1)
except Exception as e:
    print(f"\n\nError: {e}")
    sys.exit(1)

#fin