#!/usr/bin/env python3
"""Test script to verify PR dependency updates compatibility"""

import sys
import subprocess
import tempfile
import shutil
import os

def test_update(package, old_version, new_version):
  """Test if updating a package breaks the system"""
  print(f"\n{'='*60}")
  print(f"Testing {package}: {old_version} -> {new_version}")
  print('='*60)
  
  # Create a test virtual environment
  with tempfile.TemporaryDirectory() as tmpdir:
    venv_path = os.path.join(tmpdir, 'test_venv')
    print(f"Creating test environment in {tmpdir}...")
    
    # Copy current venv
    shutil.copytree('.venv', venv_path, symlinks=True)
    
    # Update the package
    pip_cmd = f"{venv_path}/bin/pip"
    python_cmd = f"{venv_path}/bin/python"
    
    print(f"Installing {package}=={new_version}...")
    result = subprocess.run(
      [pip_cmd, 'install', '--no-deps', f'{package}=={new_version}'],
      capture_output=True, text=True
    )
    
    if result.returncode != 0:
      print(f"❌ Failed to install: {result.stderr}")
      return False
    
    # Test basic imports
    print("Testing imports...")
    test_code = """
import sys
import numpy as np
import spacy
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tokenizers
from bs4 import BeautifulSoup

# Test numpy functionality
arr = np.array([1, 2, 3], dtype=np.float32)
print(f"NumPy version: {np.__version__}")
print(f"Array dtype: {arr.dtype}")

# Test tokenizers
print(f"Tokenizers version: {tokenizers.__version__}")

# Test langchain splitters
splitter = RecursiveCharacterTextSplitter(chunk_size=100)
print("LangChain text splitter initialized")

# Test beautifulsoup
html = "<p>Test</p>"
soup = BeautifulSoup(html, 'html.parser')
print("BeautifulSoup initialized")

# Test spacy
print(f"spaCy version: {spacy.__version__}")
"""
    
    result = subprocess.run(
      [python_cmd, '-c', test_code],
      capture_output=True, text=True
    )
    
    if result.returncode != 0:
      print(f"❌ Import test failed: {result.stderr}")
      return False
    
    print("✅ Basic imports successful")
    print(result.stdout)
    
    # Test CustomKB specific functionality
    print("\nTesting CustomKB compatibility...")
    customkb_test = """
import sys
sys.path.insert(0, '/ai/scripts/customkb')

# Test config manager
from config.config_manager import KnowledgeBase

# Test embedding manager
from embedding.embed_manager import EmbedManager

# Test query manager  
from query.query_manager import QueryManager

print("✅ CustomKB modules imported successfully")
"""
    
    result = subprocess.run(
      [python_cmd, '-c', customkb_test],
      capture_output=True, text=True
    )
    
    if result.returncode != 0:
      print(f"⚠️  CustomKB import test failed: {result.stderr}")
      return False
    
    print(result.stdout)
    return True

def main():
  updates = [
    ('langchain-text-splitters', '0.3.7', '0.3.11'),
    ('tokenizers', '0.21.1', '0.22.0'),
    # Skip numpy 2.x for now - major breaking changes
    # ('numpy', '1.26.4', '2.3.2'),
    # beautifulsoup4 already at 4.13.5
    # ('beautifulsoup4', '4.13.3', '4.13.5'),
    # Skip spacy for now - needs numpy 2.0
    # ('spacy', '3.7.5', '3.8.7'),
  ]
  
  results = []
  for package, old_ver, new_ver in updates:
    success = test_update(package, old_ver, new_ver)
    results.append((package, old_ver, new_ver, success))
  
  print(f"\n{'='*60}")
  print("SUMMARY")
  print('='*60)
  for package, old_ver, new_ver, success in results:
    status = "✅ SAFE" if success else "❌ FAILED"
    print(f"{status}: {package} {old_ver} -> {new_ver}")

if __name__ == '__main__':
  main()

#fin