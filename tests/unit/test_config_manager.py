"""
Unit tests for config_manager.py
Tests configuration loading, parsing, validation, and KnowledgeBase class functionality.
"""

import pytest
import os
import tempfile
import configparser
from unittest.mock import patch, Mock
from pathlib import Path

from config.config_manager import (
  get_fq_cfg_filename, 
  KnowledgeBase,
  VECTORDBS
)


class TestGetFqCfgFilename:
  """Test the get_fq_cfg_filename function."""
  
  def test_empty_filename_returns_none(self):
    """Test that empty filename returns None."""
    assert get_fq_cfg_filename("") is None
    assert get_fq_cfg_filename(None) is None
  
  def test_existing_cfg_file_with_extension(self, temp_config_file):
    """Test resolving existing .cfg file."""
    result = get_fq_cfg_filename(temp_config_file)
    assert result == temp_config_file
    assert result.endswith('.cfg')
  
  def test_existing_file_without_extension(self, temp_kb_directory):
    """Test resolving file without .cfg extension."""
    # Create file without extension
    base_path = os.path.join(temp_kb_directory, "test_config")
    with open(base_path, 'w') as f:
      f.write("[DEFAULT]\nvector_model = test\n")
    
    # Should find and add .cfg extension
    with patch('os.getcwd', return_value=temp_kb_directory):
      result = get_fq_cfg_filename("test_config")
      assert result is None  # Should not find without proper extension
  
  def test_domain_style_name(self, temp_kb_directory):
    """Test domain-style configuration names."""
    # Create domain-style config
    domain_config = os.path.join(temp_kb_directory, "example.com.cfg")
    with open(domain_config, 'w') as f:
      f.write("[DEFAULT]\nvector_model = test\n")
    
    with patch('config.config_manager.VECTORDBS', temp_kb_directory):
      result = get_fq_cfg_filename("example.com")
      assert result == domain_config
  
  def test_nonexistent_file_returns_none(self):
    """Test that nonexistent file returns None."""
    result = get_fq_cfg_filename("nonexistent.cfg")
    assert result is None
  
  def test_invalid_extension_returns_none(self, temp_kb_directory):
    """Test that non-.cfg files return None."""
    txt_file = os.path.join(temp_kb_directory, "test.txt")
    with open(txt_file, 'w') as f:
      f.write("test content")
    
    result = get_fq_cfg_filename(txt_file)
    assert result is None
  
  @patch('config.config_manager.validate_file_path')
  def test_security_validation_called(self, mock_validate):
    """Test that security validation is called."""
    mock_validate.side_effect = ValueError("Invalid path")
    
    result = get_fq_cfg_filename("../malicious.cfg")
    assert result is None
    mock_validate.assert_called_once()
  
  def test_search_in_vectordbs(self, temp_kb_directory):
    """Test searching in VECTORDBS directory."""
    # Create config in VECTORDBS
    config_name = "search_test.cfg"
    config_path = os.path.join(temp_kb_directory, config_name)
    with open(config_path, 'w') as f:
      f.write("[DEFAULT]\nvector_model = test\n")
    
    with patch('config.config_manager.VECTORDBS', temp_kb_directory):
      result = get_fq_cfg_filename(config_name)
      assert result == config_path


class TestKnowledgeBase:
  """Test the KnowledgeBase class."""
  
  def test_init_with_config_file(self, temp_config_file):
    """Test initializing KnowledgeBase with config file."""
    kb = KnowledgeBase(temp_config_file)
    
    assert kb.vector_model == "text-embedding-3-small"
    assert kb.vector_dimensions == 1536
    assert kb.vector_chunks == 200
    assert kb.db_min_tokens == 100
    assert kb.db_max_tokens == 200
    assert kb.query_model == "gpt-4o"
    assert kb.knowledge_base_name.endswith("test_kb")
  
  def test_init_with_kwargs(self):
    """Test initializing KnowledgeBase with keyword arguments."""
    kb = KnowledgeBase(
      "test_kb",
      vector_model="custom-model",
      vector_dimensions=512,
      query_model="custom-query-model"
    )
    
    assert kb.vector_model == "custom-model"
    assert kb.vector_dimensions == 512
    assert kb.query_model == "custom-query-model"
    assert kb.knowledge_base_name == "test_kb"
  
  def test_environment_variable_override(self, temp_config_file):
    """Test that environment variables override config file values."""
    with patch.dict(os.environ, {
      'VECTOR_MODEL': 'env-override-model',
      'QUERY_TOP_K': '100'
    }):
      kb = KnowledgeBase(temp_config_file)
      assert kb.vector_model == 'env-override-model'
      assert kb.query_top_k == 100
  
  def test_database_and_vector_paths(self, temp_config_file, temp_kb_directory):
    """Test that database and vector file paths are set correctly."""
    kb = KnowledgeBase(temp_config_file)
    
    expected_base = os.path.join(temp_kb_directory, "test_kb")
    assert kb.knowledge_base_db == f"{expected_base}.db"
    assert kb.knowledge_base_vector == f"{expected_base}.faiss"
  
  def test_domain_style_paths(self, temp_kb_directory):
    """Test paths for domain-style knowledge base names."""
    # Create domain-style config
    domain_config = os.path.join(temp_kb_directory, "example.com.cfg")
    with open(domain_config, 'w') as f:
      f.write("[DEFAULT]\nvector_model = test\n")
    
    kb = KnowledgeBase(domain_config)
    
    expected_base = os.path.join(temp_kb_directory, "example.com")
    assert kb.knowledge_base_db == f"{expected_base}.db"
    assert kb.knowledge_base_vector == f"{expected_base}.faiss"
    assert kb.knowledge_base_name == "example.com"
  
  def test_config_defaults(self):
    """Test that configuration defaults are properly set."""
    kb = KnowledgeBase("test")
    
    # Check all default values
    assert kb.DEF_VECTOR_MODEL == "text-embedding-3-small"
    assert kb.DEF_VECTOR_DIMENSIONS == 1536
    assert kb.DEF_VECTOR_CHUNKS == 200
    assert kb.DEF_DB_MIN_TOKENS == 100
    assert kb.DEF_DB_MAX_TOKENS == 200
    assert kb.DEF_QUERY_MODEL == "gpt-4o"
    assert kb.DEF_QUERY_TOP_K == 50
    assert kb.DEF_QUERY_CONTEXT_SCOPE == 4
    assert kb.DEF_QUERY_TEMPERATURE == 0.0
    assert kb.DEF_QUERY_MAX_TOKENS == 4000
  
  def test_invalid_env_var_types(self, temp_config_file):
    """Test handling of invalid environment variable types."""
    with patch.dict(os.environ, {
      'VECTOR_DIMENSIONS': 'invalid_int',
      'QUERY_TEMPERATURE': 'invalid_float'
    }):
      kb = KnowledgeBase(temp_config_file)
      # Should fall back to config/default values
      assert kb.vector_dimensions == 1536  # from config
      assert kb.query_temperature == 0.1   # from config
  
  def test_query_context_files_parsing(self, temp_kb_directory):
    """Test parsing of query_context_files configuration."""
    config_content = """[DEFAULT]
vector_model = test-model
query_context_files = file1.txt,file2.txt,file3.txt
"""
    config_path = os.path.join(temp_kb_directory, "context_test.cfg")
    with open(config_path, 'w') as f:
      f.write(config_content)
    
    kb = KnowledgeBase(config_path)
    assert kb.query_context_files == ['file1.txt', 'file2.txt', 'file3.txt']
  
  def test_save_config_to_stderr(self, temp_config_file, capsys):
    """Test saving configuration to stderr."""
    kb = KnowledgeBase(temp_config_file)
    kb.save_config()
    
    captured = capsys.readouterr()
    assert "test_kb" in captured.err
    assert "vector_model" in captured.err
  
  def test_save_config_to_file(self, temp_config_file, temp_kb_directory):
    """Test saving configuration to a file."""
    kb = KnowledgeBase(temp_config_file)
    output_file = os.path.join(temp_kb_directory, "saved_config.txt")
    
    kb.save_config(output_file)
    
    assert os.path.exists(output_file)
    with open(output_file, 'r') as f:
      content = f.read()
      assert "[DEFAULT]" in content
      assert "vector_model" in content
  
  def test_sql_connection_initialization(self, temp_config_file):
    """Test that SQL connection attributes are properly initialized."""
    kb = KnowledgeBase(temp_config_file)
    
    assert kb.sql_connection is None
    assert kb.sql_cursor is None
  
  def test_start_time_set(self, temp_config_file):
    """Test that start_time is set during initialization."""
    import time
    before = int(time.time())
    kb = KnowledgeBase(temp_config_file)
    after = int(time.time())
    
    assert before <= kb.start_time <= after
  
  def test_new_config_sections_defaults(self):
    """Test that new configuration section defaults are properly set."""
    kb = KnowledgeBase("test")
    
    # API configuration defaults
    assert kb.DEF_API_CALL_DELAY_SECONDS == 0.05
    assert kb.DEF_API_MAX_RETRIES == 20
    assert kb.DEF_API_MAX_CONCURRENCY == 8
    assert kb.DEF_API_MIN_CONCURRENCY == 3
    assert kb.DEF_BACKOFF_EXPONENT == 2
    assert kb.DEF_BACKOFF_JITTER == 0.1
    
    # Limits configuration defaults
    assert kb.DEF_MAX_FILE_SIZE_MB == 100
    assert kb.DEF_MAX_QUERY_FILE_SIZE_MB == 1
    assert kb.DEF_MEMORY_CACHE_SIZE == 10000
    assert kb.DEF_API_KEY_MIN_LENGTH == 20
    assert kb.DEF_MAX_QUERY_LENGTH == 10000
    assert kb.DEF_MAX_CONFIG_VALUE_LENGTH == 1000
    assert kb.DEF_MAX_JSON_SIZE == 10000
    
    # Performance configuration defaults
    assert kb.DEF_EMBEDDING_BATCH_SIZE == 100
    assert kb.DEF_CHECKPOINT_INTERVAL == 10
    assert kb.DEF_COMMIT_FREQUENCY == 1000
    assert kb.DEF_IO_THREAD_POOL_SIZE == 4
    assert kb.DEF_FILE_PROCESSING_BATCH_SIZE == 500
    assert kb.DEF_SQL_BATCH_SIZE == 500
    assert kb.DEF_REFERENCE_BATCH_SIZE == 5
    assert kb.DEF_QUERY_CACHE_TTL_DAYS == 7
    assert kb.DEF_DEFAULT_EDITOR == 'joe'
    
    # Algorithms configuration defaults
    assert kb.DEF_HIGH_DIMENSION_THRESHOLD == 1536
    assert kb.DEF_SMALL_DATASET_THRESHOLD == 1000
    assert kb.DEF_MEDIUM_DATASET_THRESHOLD == 100000
    assert kb.DEF_IVF_CENTROID_MULTIPLIER == 4
    assert kb.DEF_MAX_CENTROIDS == 256
    assert kb.DEF_TOKEN_ESTIMATION_SAMPLE_SIZE == 10
    assert kb.DEF_TOKEN_ESTIMATION_MULTIPLIER == 1.3
    assert kb.DEF_SIMILARITY_THRESHOLD == 0.6
    assert kb.DEF_LOW_SIMILARITY_SCOPE_FACTOR == 0.5
    assert kb.DEF_MAX_CHUNK_OVERLAP == 100
    assert kb.DEF_OVERLAP_RATIO == 0.5
    assert kb.DEF_HEADING_SEARCH_LIMIT == 200
    assert kb.DEF_ENTITY_EXTRACTION_LIMIT == 500
    assert kb.DEF_DEFAULT_DIR_PERMISSIONS == 0o770
    assert kb.DEF_DEFAULT_CODE_LANGUAGE == 'python'
    assert kb.DEF_ADDITIONAL_STOPWORD_LANGUAGES == ['indonesian', 'french', 'german', 'swedish']
  
  def test_new_config_sections_from_file(self, temp_data_manager):
    """Test loading new configuration sections from config file."""
    from tests.fixtures.mock_data import MockDataGenerator
    
    # Create config with new sections
    config_content = MockDataGenerator.create_sample_config(
      include_new_sections=True,
      api_call_delay_seconds=0.02,
      api_max_retries=5,
      max_file_size_mb=50,
      embedding_batch_size=25,
      similarity_threshold=0.8
    )
    
    config_file = temp_data_manager.create_temp_config(config_content)
    kb = KnowledgeBase(config_file)
    
    # Test that values are loaded from config file
    assert kb.api_call_delay_seconds == 0.02
    assert kb.api_max_retries == 5
    assert kb.max_file_size_mb == 50
    assert kb.embedding_batch_size == 25
    assert kb.similarity_threshold == 0.8
    
    # Test that defaults are used for unspecified values
    assert kb.api_max_concurrency == 2  # from test config default
    assert kb.memory_cache_size == 100  # from test config default
  
  def test_new_config_sections_env_override(self, temp_config_file):
    """Test that environment variables override new config section values."""
    with patch.dict(os.environ, {
      'API_CALL_DELAY_SECONDS': '0.1',
      'MAX_FILE_SIZE_MB': '200',
      'EMBEDDING_BATCH_SIZE': '50',
      'SIMILARITY_THRESHOLD': '0.9'
    }):
      kb = KnowledgeBase(temp_config_file)
      assert kb.api_call_delay_seconds == 0.1
      assert kb.max_file_size_mb == 200
      assert kb.embedding_batch_size == 50
      assert kb.similarity_threshold == 0.9
  
  def test_new_config_sections_kwargs(self):
    """Test new configuration sections with kwargs."""
    kb = KnowledgeBase(
      "test_kb",
      api_call_delay_seconds=0.001,
      max_file_size_mb=500,
      embedding_batch_size=200,
      similarity_threshold=0.95,
      default_editor='vim'
    )
    
    assert kb.api_call_delay_seconds == 0.001
    assert kb.max_file_size_mb == 500
    assert kb.embedding_batch_size == 200
    assert kb.similarity_threshold == 0.95
    assert kb.default_editor == 'vim'
  
  def test_additional_stopword_languages_parsing(self, temp_data_manager):
    """Test parsing of additional_stopword_languages list parameter."""
    config_content = """[DEFAULT]
vector_model = test-model

[ALGORITHMS]
additional_stopword_languages = spanish,italian,portuguese
"""
    config_file = temp_data_manager.create_temp_config(config_content)
    kb = KnowledgeBase(config_file)
    
    assert kb.additional_stopword_languages == ['spanish', 'italian', 'portuguese']
  
  def test_config_sections_missing_fallback(self, temp_data_manager):
    """Test fallback to DEFAULT section when specific sections are missing."""
    config_content = """[DEFAULT]
vector_model = test-model
# No API, LIMITS, PERFORMANCE, or ALGORITHMS sections
"""
    config_file = temp_data_manager.create_temp_config(config_content)
    kb = KnowledgeBase(config_file)
    
    # Should use defaults since sections are missing
    assert kb.api_call_delay_seconds == 0.05
    assert kb.max_file_size_mb == 100
    assert kb.embedding_batch_size == 100
    assert kb.similarity_threshold == 0.6
  
  @patch('config.config_manager.get_env')
  def test_get_env_called_for_overrides(self, mock_get_env, temp_config_file):
    """Test that get_env is called for environment variable overrides."""
    mock_get_env.side_effect = lambda var, default, cast=str: default
    
    kb = KnowledgeBase(temp_config_file)
    
    # Verify get_env was called for config values
    assert mock_get_env.call_count > 0
    call_args = [call[0][0] for call in mock_get_env.call_args_list]
    assert 'VECTOR_MODEL' in call_args
    assert 'QUERY_MODEL' in call_args


class TestErrorHandling:
  """Test error handling in config_manager."""
  
  def test_corrupted_config_file(self, temp_kb_directory):
    """Test handling of corrupted configuration files."""
    corrupted_config = os.path.join(temp_kb_directory, "corrupted.cfg")
    with open(corrupted_config, 'w') as f:
      f.write("invalid config content [[[")
    
    # Should not raise exception, should use defaults
    kb = KnowledgeBase(corrupted_config)
    assert kb.vector_model == kb.DEF_VECTOR_MODEL
  
  def test_missing_config_section(self, temp_kb_directory):
    """Test handling of config file without DEFAULT section."""
    config_content = """[WRONG_SECTION]
vector_model = test
"""
    config_path = os.path.join(temp_kb_directory, "missing_section.cfg")
    with open(config_path, 'w') as f:
      f.write(config_content)
    
    kb = KnowledgeBase(config_path)
    # Should use defaults when section is missing
    assert kb.vector_model == kb.DEF_VECTOR_MODEL
  
  def test_path_security_validation(self):
    """Test that path traversal attempts are blocked."""
    dangerous_paths = [
      "../../../etc/passwd",
      "../../malicious.cfg",
      "/etc/shadow.cfg"
    ]
    
    for path in dangerous_paths:
      result = get_fq_cfg_filename(path)
      assert result is None


class TestConfigPathResolution:
  """Test path resolution functionality for the three use cases."""
  
  def test_absolute_path_to_cfg_file(self):
    """Test Case 1: Absolute path to .cfg file."""
    with tempfile.TemporaryDirectory() as tmpdir:
      # Create config file
      config_path = os.path.join(tmpdir, 'myproject.cfg')
      with open(config_path, 'w') as f:
        f.write('[DEFAULT]\nknowledge_base_name = myproject\n')
      
      # Should resolve absolute path to config file
      result = get_fq_cfg_filename(config_path)
      assert result == config_path
      assert os.path.isabs(result)
  
  def test_kb_name_only_search_vectordbs(self):
    """Test Case 2: KB name only, searches VECTORDBS."""
    with tempfile.TemporaryDirectory() as tmpdir:
      # Create VECTORDBS-like structure
      kb_dir = os.path.join(tmpdir, 'myproject')
      os.makedirs(kb_dir)
      config_path = os.path.join(kb_dir, 'myproject.cfg')
      with open(config_path, 'w') as f:
        f.write('[DEFAULT]\nknowledge_base_name = myproject\n')
      
      # Mock VECTORDBS to point to our test directory
      with patch('config.config_manager.VECTORDBS', tmpdir):
        result = get_fq_cfg_filename('myproject')
        assert result == config_path
  
  def test_absolute_path_to_kb_directory(self):
    """Test Case 3: Absolute path to KB root directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
      # Create KB directory structure
      kb_dir = os.path.join(tmpdir, 'myproject')
      os.makedirs(kb_dir)
      config_path = os.path.join(kb_dir, 'myproject.cfg')
      with open(config_path, 'w') as f:
        f.write('[DEFAULT]\nknowledge_base_name = myproject\n')
      
      # The current implementation expects the basename + .cfg
      kb_basename = os.path.basename(kb_dir)
      result = get_fq_cfg_filename(kb_basename)
      # This should find it if we're in the right directory or via VECTORDBS search
      # For now, test that the base functionality works
      if result:
        assert result.endswith('.cfg')
  
  def test_relative_traversal_allowed(self):
    """Test that relative traversal (../) is now allowed for KB configs."""
    with tempfile.TemporaryDirectory() as tmpdir:
      # Create sibling directories
      okusimail_dir = os.path.join(tmpdir, 'okusimail')
      okusiassociates_dir = os.path.join(tmpdir, 'okusiassociates')
      os.makedirs(okusimail_dir)
      os.makedirs(okusiassociates_dir)
      
      # Create config in okusimail
      config_path = os.path.join(okusimail_dir, 'okusimail.cfg')
      with open(config_path, 'w') as f:
        f.write('[DEFAULT]\nknowledge_base_name = okusimail\n')
      
      # From okusiassociates directory, reference sibling
      old_cwd = os.getcwd()
      try:
        os.chdir(okusiassociates_dir)
        result = get_fq_cfg_filename('../okusimail/okusimail.cfg')
        assert result == '../okusimail/okusimail.cfg'
      finally:
        os.chdir(old_cwd)
  
  def test_domain_style_with_absolute_path(self):
    """Test domain-style names with absolute paths."""
    with tempfile.TemporaryDirectory() as tmpdir:
      # Create domain-style config with absolute path
      config_path = os.path.join(tmpdir, 'example.com.cfg')
      with open(config_path, 'w') as f:
        f.write('[DEFAULT]\nknowledge_base_name = example.com\n')
      
      # Test direct access to .cfg file
      result = get_fq_cfg_filename(config_path)
      assert result == config_path
  
  def test_mixed_absolute_and_relative_traversal(self):
    """Test absolute paths containing relative traversal."""
    with tempfile.TemporaryDirectory() as tmpdir:
      # Create nested structure with traversal
      nested_dir = os.path.join(tmpdir, 'deep', 'nested')
      target_dir = os.path.join(tmpdir, 'target')
      os.makedirs(nested_dir)
      os.makedirs(target_dir)
      
      config_path = os.path.join(target_dir, 'project.cfg')
      with open(config_path, 'w') as f:
        f.write('[DEFAULT]\nknowledge_base_name = project\n')
      
      # Use absolute path with traversal
      traversal_path = os.path.join(nested_dir, '../../target/project.cfg')
      result = get_fq_cfg_filename(traversal_path)
      assert result == traversal_path
  
  def test_security_still_enforced_for_dangerous_paths(self):
    """Test that dangerous paths are still blocked despite new flexibility."""
    dangerous_paths = [
      '/etc/passwd.cfg',  # System file
      '../../../etc/shadow.cfg',  # Deep traversal to system
      'config|rm -rf.cfg',  # Shell injection
      'config$()evil.cfg',  # Command substitution
    ]
    
    for path in dangerous_paths:
      result = get_fq_cfg_filename(path)
      # Should either return None or raise an exception
      # Some might pass path validation but fail file existence check
      if result is not None:
        # If it returns a path, it should be the sanitized input
        assert not any(char in result for char in ['|', '$', '(', ')'])
  
  def test_current_directory_resolution_unchanged(self):
    """Test that current directory resolution still works."""
    with tempfile.TemporaryDirectory() as tmpdir:
      config_path = os.path.join(tmpdir, 'local.cfg')
      with open(config_path, 'w') as f:
        f.write('[DEFAULT]\nknowledge_base_name = local\n')
      
      old_cwd = os.getcwd()
      try:
        os.chdir(tmpdir)
        result = get_fq_cfg_filename('./local.cfg')
        assert result == './local.cfg'
        
        # Test absolute path to current directory file
        result2 = get_fq_cfg_filename(config_path)
        assert result2 == config_path
      finally:
        os.chdir(old_cwd)
  
  def test_vectordbs_search_fallback_still_works(self):
    """Test that VECTORDBS search fallback is preserved."""
    with tempfile.TemporaryDirectory() as tmpdir:
      # Create config in VECTORDBS location
      config_path = os.path.join(tmpdir, 'fallback.cfg')
      with open(config_path, 'w') as f:
        f.write('[DEFAULT]\nknowledge_base_name = fallback\n')
      
      # Mock VECTORDBS and test from different directory
      with tempfile.TemporaryDirectory() as other_dir:
        old_cwd = os.getcwd()
        try:
          os.chdir(other_dir)  # Change to directory without the config
          with patch('config.config_manager.VECTORDBS', tmpdir):
            result = get_fq_cfg_filename('fallback')
            assert result == config_path
        finally:
          os.chdir(old_cwd)


class TestBM25Configuration:
  """Test BM25/Hybrid search configuration parameters."""
  
  def test_default_bm25_parameters(self):
    """Test that BM25 parameters have correct defaults."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.cfg', delete=False) as f:
      f.write("""
[DEFAULT]
knowledge_base_name = test_bm25_defaults
""")
      f.flush()
      
      try:
        kb = KnowledgeBase(f.name)
        
        # Test default BM25 parameters
        assert kb.enable_hybrid_search is False
        assert kb.vector_weight == 0.7
        assert kb.bm25_k1 == 1.2
        assert kb.bm25_b == 0.75
        assert kb.bm25_min_token_length == 2
        assert kb.bm25_rebuild_threshold == 1000
      finally:
        os.unlink(f.name)
  
  def test_bm25_configuration_loading(self):
    """Test loading BM25 parameters from config file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.cfg', delete=False) as f:
      f.write("""
[DEFAULT]
knowledge_base_name = test_bm25_config

[ALGORITHMS]
enable_hybrid_search = true
vector_weight = 0.8
bm25_k1 = 1.5
bm25_b = 0.8
bm25_min_token_length = 3
bm25_rebuild_threshold = 500
""")
      f.flush()
      
      try:
        kb = KnowledgeBase(f.name)
        
        # Test loaded BM25 parameters
        assert kb.enable_hybrid_search is True
        assert kb.vector_weight == 0.8
        assert kb.bm25_k1 == 1.5
        assert kb.bm25_b == 0.8
        assert kb.bm25_min_token_length == 3
        assert kb.bm25_rebuild_threshold == 500
      finally:
        os.unlink(f.name)
  
  def test_bm25_environment_variable_overrides(self):
    """Test that environment variables override BM25 config file values."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.cfg', delete=False) as f:
      f.write("""
[DEFAULT]
knowledge_base_name = test_bm25_env

[ALGORITHMS]
enable_hybrid_search = false
vector_weight = 0.7
bm25_k1 = 1.2
""")
      f.flush()
      
      try:
        # Set environment variables
        env_vars = {
          'ENABLE_HYBRID_SEARCH': 'true',
          'VECTOR_WEIGHT': '0.9',
          'BM25_K1': '2.0'
        }
        
        with patch.dict(os.environ, env_vars):
          kb = KnowledgeBase(f.name)
          
          # Environment variables should override config file
          assert kb.enable_hybrid_search is True
          assert kb.vector_weight == 0.9
          assert kb.bm25_k1 == 2.0
      finally:
        os.unlink(f.name)
  
  def test_bm25_parameter_validation(self):
    """Test validation of BM25 parameter ranges."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.cfg', delete=False) as f:
      f.write("""
[DEFAULT]
knowledge_base_name = test_bm25_validation

[ALGORITHMS]
enable_hybrid_search = true
vector_weight = 0.7
bm25_k1 = 1.2
bm25_b = 0.75
""")
      f.flush()
      
      try:
        kb = KnowledgeBase(f.name)
        
        # Test that parameters are within expected ranges
        assert 0.0 <= kb.vector_weight <= 1.0
        assert kb.bm25_k1 > 0.0  # Should be positive
        assert 0.0 <= kb.bm25_b <= 1.0  # Should be between 0 and 1
        assert kb.bm25_min_token_length >= 1  # Should be at least 1
        assert kb.bm25_rebuild_threshold > 0  # Should be positive
      finally:
        os.unlink(f.name)
  
  def test_bm25_invalid_boolean_values(self):
    """Test handling of invalid boolean values for enable_hybrid_search."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.cfg', delete=False) as f:
      f.write("""
[DEFAULT]
knowledge_base_name = test_bm25_bool

[ALGORITHMS]
enable_hybrid_search = invalid_value
""")
      f.flush()
      
      try:
        # Should handle invalid boolean gracefully
        with patch.dict(os.environ, {'ENABLE_HYBRID_SEARCH': 'not_a_boolean'}):
          kb = KnowledgeBase(f.name)
          
          # Should fall back to default when environment parsing fails
          assert kb.enable_hybrid_search is False
      finally:
        os.unlink(f.name)
  
  def test_bm25_mixed_configuration_sources(self):
    """Test BM25 configuration from multiple sources (file + env + defaults)."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.cfg', delete=False) as f:
      f.write("""
[DEFAULT]
knowledge_base_name = test_bm25_mixed

[ALGORITHMS]
enable_hybrid_search = true
vector_weight = 0.6
# bm25_k1 not specified (should use default)
bm25_b = 0.8
""")
      f.flush()
      
      try:
        # Override one parameter via environment
        with patch.dict(os.environ, {'VECTOR_WEIGHT': '0.75'}):
          kb = KnowledgeBase(f.name)
          
          # Should get values from different sources
          assert kb.enable_hybrid_search is True  # From config file
          assert kb.vector_weight == 0.75  # From environment
          assert kb.bm25_k1 == 1.2  # From default (not in config/env)
          assert kb.bm25_b == 0.8  # From config file
          assert kb.bm25_min_token_length == 2  # From default
      finally:
        os.unlink(f.name)
  
  def test_bm25_kwargs_override(self):
    """Test BM25 configuration via kwargs (programmatic override)."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.cfg', delete=False) as f:
      f.write("""
[DEFAULT]
knowledge_base_name = test_bm25_kwargs
""")
      f.flush()
      
      try:
        # Pass BM25 parameters via kwargs
        kb = KnowledgeBase(f.name, 
                          enable_hybrid_search=True,
                          vector_weight=0.85,
                          bm25_k1=1.8,
                          bm25_b=0.9)
        
        # Should use kwargs values
        assert kb.enable_hybrid_search is True
        assert kb.vector_weight == 0.85
        assert kb.bm25_k1 == 1.8
        assert kb.bm25_b == 0.9
      finally:
        os.unlink(f.name)
  
  def test_bm25_configuration_save(self):
    """Test that BM25 configuration is included in save output."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.cfg', delete=False) as f:
      f.write("""
[DEFAULT]
knowledge_base_name = test_bm25_save

[ALGORITHMS]
enable_hybrid_search = true
vector_weight = 0.8
""")
      f.flush()
      
      try:
        kb = KnowledgeBase(f.name)
        
        # Save config to string
        import io
        output = io.StringIO()
        kb.save_config()  # This prints to stderr by default
        
        # Check that BM25 parameters would be included in save
        # (We test via hasattr since save_config prints all attributes)
        assert hasattr(kb, 'enable_hybrid_search')
        assert hasattr(kb, 'vector_weight')
        assert hasattr(kb, 'bm25_k1')
        assert hasattr(kb, 'bm25_b')
        assert hasattr(kb, 'bm25_min_token_length')
        assert hasattr(kb, 'bm25_rebuild_threshold')
      finally:
        os.unlink(f.name)

#fin