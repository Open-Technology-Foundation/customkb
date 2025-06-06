#!/usr/bin/env python
"""
Basic test script to verify core CustomKB functionality without external dependencies.
"""

import sys
import os
import tempfile
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_config_manager():
  """Test basic config manager functionality."""
  print("Testing config_manager...")
  
  from config.config_manager import get_fq_cfg_filename, KnowledgeBase
  
  # Test empty filename
  result = get_fq_cfg_filename("")
  assert result is None, "Empty filename should return None"
  
  # Test with temporary config file
  with tempfile.NamedTemporaryFile(mode='w', suffix='.cfg', delete=False) as f:
    f.write("""[DEFAULT]
vector_model = text-embedding-3-small
vector_dimensions = 1536
query_model = gpt-4o
""")
    config_path = f.name
  
  try:
    # Test config file resolution
    result = get_fq_cfg_filename(config_path)
    assert result == config_path, f"Config file resolution failed: {result} != {config_path}"
    
    # Test KnowledgeBase initialization
    kb = KnowledgeBase(config_path)
    assert kb.vector_model == "text-embedding-3-small"
    assert kb.vector_dimensions == 1536
    assert kb.query_model == "gpt-4o"
    
    print("✅ config_manager tests passed")
    
  finally:
    os.unlink(config_path)

def test_text_utils():
  """Test basic text utils functionality."""
  print("Testing text_utils...")
  
  from utils.text_utils import clean_text, split_filepath, get_env
  
  # Test basic text cleaning
  text = "This is a SAMPLE Text with Mixed Case!"
  cleaned = clean_text(text)
  assert cleaned.lower() == cleaned, "Text should be lowercase"
  assert "sample" in cleaned, "Should contain 'sample'"
  
  # Test filepath splitting
  directory, basename, extension, fqfn = split_filepath("/path/to/file.txt", adddir=False, realpath=False)
  assert directory == "/path/to"
  assert basename == "file"
  assert extension == ".txt"
  
  # Test environment variable utility
  os.environ['TEST_VAR'] = 'test_value'
  result = get_env('TEST_VAR', 'default')
  assert result == 'test_value'
  
  result = get_env('NONEXISTENT_VAR', 'default')
  assert result == 'default'
  
  print("✅ text_utils tests passed")

def test_logging_utils():
  """Test basic logging utils functionality."""
  print("Testing logging_utils...")
  
  from utils.logging_utils import elapsed_time, get_kb_info_from_config, get_logger
  
  # Test elapsed time calculation
  result = elapsed_time(1000, 1065)  # 65 seconds
  assert result == "01m 05s"
  
  # Test KB info extraction
  directory, kb_name = get_kb_info_from_config("/path/to/mycompany.cfg")
  assert directory == "/path/to/"
  assert kb_name == "mycompany"
  
  # Test logger creation
  logger = get_logger("test_module")
  assert logger.name == "test_module"
  
  print("✅ logging_utils tests passed")

def test_model_manager():
  """Test basic model manager functionality."""
  print("Testing model_manager...")
  
  # Create temporary Models.json
  models_data = {
    "gpt-4": {
      "model": "gpt-4",
      "provider": "openai",
      "type": "chat"
    },
    "custom-model": {
      "model": "custom-model",
      "alias": "custom",
      "provider": "test"
    }
  }
  
  with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
    import json
    json.dump(models_data, f)
    models_file = f.name
  
  try:
    # Mock the models file path
    from unittest.mock import patch
    with patch('models.model_manager.models_file', models_file):
      from models.model_manager import get_canonical_model
      
      # Test direct lookup
      result = get_canonical_model("gpt-4")
      assert result["model"] == "gpt-4"
      assert result["provider"] == "openai"
      
      # Test alias lookup
      result = get_canonical_model("custom")
      assert result["model"] == "custom-model"
      assert result["alias"] == "custom"
      
      print("✅ model_manager tests passed")
      
  finally:
    os.unlink(models_file)

def main():
  """Run all basic tests."""
  print("Running CustomKB basic tests...")
  print("=" * 50)
  
  try:
    test_config_manager()
    test_text_utils()
    test_logging_utils()
    test_model_manager()
    
    print("=" * 50)
    print("✅ All basic tests passed!")
    print("\nTo run the full test suite:")
    print("  source .venv/bin/activate")
    print("  python run_tests.py")
    
    return 0
    
  except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()
    return 1

if __name__ == "__main__":
  sys.exit(main())

#fin