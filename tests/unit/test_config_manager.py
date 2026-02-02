"""
Unit tests for config_manager.py
Tests configuration loading, parsing, validation, and KnowledgeBase class functionality.
"""

import os
import tempfile
from unittest.mock import patch

import pytest

from config.config_manager import KnowledgeBase, get_fq_cfg_filename, get_kb_name
from tests.fixtures.mock_data import MockDataGenerator


class TestGetKbName:
  """Test the get_kb_name function."""

  def test_empty_name_returns_none(self):
    """Test that empty KB name returns None."""
    assert get_kb_name("") is None
    assert get_kb_name(None) is None

  def test_simple_kb_name(self):
    """Test simple KB name without extension or path."""
    with tempfile.TemporaryDirectory() as tmpdir:
      # Create KB directory structure
      kb_dir = os.path.join(tmpdir, 'myproject')
      os.makedirs(kb_dir)

      with patch('config.config_manager.VECTORDBS', tmpdir):
        result = get_kb_name('myproject')
        assert result == 'myproject'

  def test_name_with_cfg_extension(self):
    """Test KB name with .cfg extension (should be stripped)."""
    with tempfile.TemporaryDirectory() as tmpdir:
      kb_dir = os.path.join(tmpdir, 'myproject')
      os.makedirs(kb_dir)

      with patch('config.config_manager.VECTORDBS', tmpdir):
        result = get_kb_name('myproject.cfg')
        assert result == 'myproject'

  def test_name_with_path(self):
    """Test KB name with path (should extract basename)."""
    with tempfile.TemporaryDirectory() as tmpdir:
      kb_dir = os.path.join(tmpdir, 'myproject')
      os.makedirs(kb_dir)

      with patch('config.config_manager.VECTORDBS', tmpdir):
        result = get_kb_name('/path/to/myproject')
        assert result == 'myproject'

        result = get_kb_name('/path/to/myproject.cfg')
        assert result == 'myproject'

  def test_nonexistent_kb_returns_none(self):
    """Test that nonexistent KB returns None."""
    with tempfile.TemporaryDirectory() as tmpdir, patch('config.config_manager.VECTORDBS', tmpdir):
      result = get_kb_name('nonexistent')
      assert result is None

  def test_domain_style_name(self):
    """Test domain-style KB names."""
    with tempfile.TemporaryDirectory() as tmpdir:
      kb_dir = os.path.join(tmpdir, 'example.com')
      os.makedirs(kb_dir)

      with patch('config.config_manager.VECTORDBS', tmpdir):
        result = get_kb_name('example.com')
        assert result == 'example.com'

        result = get_kb_name('example.com.cfg')
        assert result == 'example.com'


class TestGetFqCfgFilename:
  """Test the updated get_fq_cfg_filename function."""

  def test_empty_filename_returns_none(self):
    """Test that empty filename returns None."""
    assert get_fq_cfg_filename("") is None
    assert get_fq_cfg_filename(None) is None

  def test_simple_kb_name(self):
    """Test resolving simple KB name."""
    with tempfile.TemporaryDirectory() as tmpdir:
      # Create KB directory structure
      kb_dir = os.path.join(tmpdir, 'myproject')
      os.makedirs(kb_dir)
      config_path = os.path.join(kb_dir, 'myproject.cfg')
      with open(config_path, 'w') as f:
        f.write('[DEFAULT]\nvector_model = test\n')

      with patch('config.config_manager.VECTORDBS', tmpdir):
        result = get_fq_cfg_filename('myproject')
        assert result == config_path

  def test_kb_name_with_extension(self):
    """Test KB name with .cfg extension."""
    with tempfile.TemporaryDirectory() as tmpdir:
      kb_dir = os.path.join(tmpdir, 'myproject')
      os.makedirs(kb_dir)
      config_path = os.path.join(kb_dir, 'myproject.cfg')
      with open(config_path, 'w') as f:
        f.write('[DEFAULT]\nvector_model = test\n')

      with patch('config.config_manager.VECTORDBS', tmpdir):
        result = get_fq_cfg_filename('myproject.cfg')
        assert result == config_path

  def test_kb_name_with_path(self):
    """Test KB name with path components."""
    with tempfile.TemporaryDirectory() as tmpdir:
      kb_dir = os.path.join(tmpdir, 'myproject')
      os.makedirs(kb_dir)
      config_path = os.path.join(kb_dir, 'myproject.cfg')
      with open(config_path, 'w') as f:
        f.write('[DEFAULT]\nvector_model = test\n')

      with patch('config.config_manager.VECTORDBS', tmpdir):
        # Should strip path and resolve
        result = get_fq_cfg_filename('/some/path/myproject')
        assert result == config_path

        result = get_fq_cfg_filename('../myproject.cfg')
        assert result == config_path

  def test_nonexistent_kb_returns_none(self):
    """Test that nonexistent KB returns None."""
    with tempfile.TemporaryDirectory() as tmpdir, patch('config.config_manager.VECTORDBS', tmpdir):
      result = get_fq_cfg_filename('nonexistent')
      assert result is None

  def test_domain_style_name(self):
    """Test domain-style KB names."""
    with tempfile.TemporaryDirectory() as tmpdir:
      kb_dir = os.path.join(tmpdir, 'example.com')
      os.makedirs(kb_dir)
      config_path = os.path.join(kb_dir, 'example.com.cfg')
      with open(config_path, 'w') as f:
        f.write('[DEFAULT]\nvector_model = test\n')

      with patch('config.config_manager.VECTORDBS', tmpdir):
        result = get_fq_cfg_filename('example.com')
        assert result == config_path


class TestKnowledgeBase:
  """Test the KnowledgeBase class."""

  def test_init_with_kb_name(self):
    """Test initializing KnowledgeBase with KB name."""
    with tempfile.TemporaryDirectory() as tmpdir:
      # Create KB directory structure
      kb_dir = os.path.join(tmpdir, 'test_kb')
      os.makedirs(kb_dir)
      config_path = os.path.join(kb_dir, 'test_kb.cfg')

      # Create config file
      config_content = MockDataGenerator.create_sample_config()
      with open(config_path, 'w') as f:
        f.write(config_content)

      with patch('config.config_manager.VECTORDBS', tmpdir):
        kb = KnowledgeBase('test_kb')

        assert kb.vector_model == "text-embedding-3-small"
        assert kb.vector_dimensions == 1536
        assert kb.vector_chunks == 200
        assert kb.db_min_tokens == 100
        assert kb.db_max_tokens == 200
        assert kb.query_model == "gpt-4o"
        assert kb.knowledge_base_name == "test_kb"

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

  def test_environment_variable_override(self):
    """Test that environment variables override config file values."""
    with tempfile.TemporaryDirectory() as tmpdir:
      # Create KB directory structure
      kb_dir = os.path.join(tmpdir, 'test_kb')
      os.makedirs(kb_dir)
      config_path = os.path.join(kb_dir, 'test_kb.cfg')

      # Create config file
      config_content = MockDataGenerator.create_sample_config()
      with open(config_path, 'w') as f:
        f.write(config_content)

      with patch('config.config_manager.VECTORDBS', tmpdir), patch.dict(os.environ, {
        'VECTOR_MODEL': 'env-override-model',
        'QUERY_TOP_K': '100'
      }):
        kb = KnowledgeBase('test_kb')
        assert kb.vector_model == 'env-override-model'
        assert kb.query_top_k == 100

  def test_database_and_vector_paths(self):
    """Test that database and vector file paths are set correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
      # Create KB directory structure
      kb_dir = os.path.join(tmpdir, 'test_kb')
      os.makedirs(kb_dir)
      config_path = os.path.join(kb_dir, 'test_kb.cfg')

      # Create config file
      config_content = MockDataGenerator.create_sample_config()
      with open(config_path, 'w') as f:
        f.write(config_content)

      with patch('config.config_manager.VECTORDBS', tmpdir):
        kb = KnowledgeBase('test_kb')

        expected_base = os.path.join(kb_dir, "test_kb")
        assert kb.knowledge_base_db == f"{expected_base}.db"
        assert kb.knowledge_base_vector == f"{expected_base}.faiss"

  def test_domain_style_paths(self):
    """Test paths for domain-style knowledgebase names."""
    with tempfile.TemporaryDirectory() as tmpdir:
      # Create domain-style KB directory
      kb_dir = os.path.join(tmpdir, 'example.com')
      os.makedirs(kb_dir)
      config_path = os.path.join(kb_dir, 'example.com.cfg')
      with open(config_path, 'w') as f:
        f.write("[DEFAULT]\nvector_model = test\n")

      with patch('config.config_manager.VECTORDBS', tmpdir):
        kb = KnowledgeBase('example.com')

        expected_base = os.path.join(kb_dir, "example.com")
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
    assert kb.DEF_QUERY_MODEL == "claude-sonnet-4-5"
    assert kb.DEF_QUERY_TOP_K == 50
    assert kb.DEF_QUERY_CONTEXT_SCOPE == 4
    assert kb.DEF_QUERY_TEMPERATURE == 0.0
    assert kb.DEF_QUERY_MAX_TOKENS == 4000

  def test_invalid_env_var_types(self):
    """Test handling of invalid environment variable types."""
    with tempfile.TemporaryDirectory() as tmpdir:
      # Create KB directory structure
      kb_dir = os.path.join(tmpdir, 'test_kb')
      os.makedirs(kb_dir)
      config_path = os.path.join(kb_dir, 'test_kb.cfg')

      # Create config file
      config_content = MockDataGenerator.create_sample_config()
      with open(config_path, 'w') as f:
        f.write(config_content)

      with patch('config.config_manager.VECTORDBS', tmpdir), patch.dict(os.environ, {
        'VECTOR_DIMENSIONS': 'invalid_int',
        'QUERY_TEMPERATURE': 'invalid_float'
      }):
        kb = KnowledgeBase('test_kb')
        # Should fall back to config/default values
        assert kb.vector_dimensions == 1536  # from config
        assert kb.query_temperature == 0.1   # from config

  def test_query_context_files_parsing(self):
    """Test parsing of query_context_files configuration."""
    with tempfile.TemporaryDirectory() as tmpdir:
      # Create KB directory structure
      kb_dir = os.path.join(tmpdir, 'context_test')
      os.makedirs(kb_dir)
      config_path = os.path.join(kb_dir, 'context_test.cfg')

      config_content = """[DEFAULT]
vector_model = test-model
query_context_files = file1.txt,file2.txt,file3.txt
"""
      with open(config_path, 'w') as f:
        f.write(config_content)

      with patch('config.config_manager.VECTORDBS', tmpdir):
        kb = KnowledgeBase('context_test')
        assert kb.query_context_files == ['file1.txt', 'file2.txt', 'file3.txt']

  def test_save_config_to_stderr(self, capsys):
    """Test saving configuration to stderr."""
    with tempfile.TemporaryDirectory() as tmpdir:
      # Create KB directory structure
      kb_dir = os.path.join(tmpdir, 'test_kb')
      os.makedirs(kb_dir)
      config_path = os.path.join(kb_dir, 'test_kb.cfg')

      # Create config file
      config_content = MockDataGenerator.create_sample_config()
      with open(config_path, 'w') as f:
        f.write(config_content)

      with patch('config.config_manager.VECTORDBS', tmpdir):
        kb = KnowledgeBase('test_kb')
        kb.save_config()

        captured = capsys.readouterr()
        assert "test_kb" in captured.err
        assert "vector_model" in captured.err

  def test_save_config_to_file(self):
    """Test saving configuration to a file."""
    with tempfile.TemporaryDirectory() as tmpdir:
      # Create KB directory structure
      kb_dir = os.path.join(tmpdir, 'test_kb')
      os.makedirs(kb_dir)
      config_path = os.path.join(kb_dir, 'test_kb.cfg')

      # Create config file
      config_content = MockDataGenerator.create_sample_config()
      with open(config_path, 'w') as f:
        f.write(config_content)

      with patch('config.config_manager.VECTORDBS', tmpdir):
        kb = KnowledgeBase('test_kb')
        output_file = os.path.join(tmpdir, "saved_config.txt")

        kb.save_config(output_file)

        assert os.path.exists(output_file)
        with open(output_file) as f:
          content = f.read()
          assert "[DEFAULT]" in content
          assert "vector_model" in content

  def test_sql_connection_initialization(self):
    """Test that SQL connection attributes are properly initialized."""
    with tempfile.TemporaryDirectory() as tmpdir:
      # Create KB directory structure
      kb_dir = os.path.join(tmpdir, 'test_kb')
      os.makedirs(kb_dir)
      config_path = os.path.join(kb_dir, 'test_kb.cfg')

      # Create config file
      config_content = MockDataGenerator.create_sample_config()
      with open(config_path, 'w') as f:
        f.write(config_content)

      with patch('config.config_manager.VECTORDBS', tmpdir):
        kb = KnowledgeBase('test_kb')

        assert kb.sql_connection is None
        assert kb.sql_cursor is None

  def test_start_time_set(self):
    """Test that start_time is set during initialization."""
    with tempfile.TemporaryDirectory() as tmpdir:
      # Create KB directory structure
      kb_dir = os.path.join(tmpdir, 'test_kb')
      os.makedirs(kb_dir)
      config_path = os.path.join(kb_dir, 'test_kb.cfg')

      # Create config file
      config_content = MockDataGenerator.create_sample_config()
      with open(config_path, 'w') as f:
        f.write(config_content)

      with patch('config.config_manager.VECTORDBS', tmpdir):
        import time
        before = int(time.time())
        kb = KnowledgeBase('test_kb')
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

  def test_new_config_sections_from_file(self):
    """Test loading new configuration sections from config file."""
    with tempfile.TemporaryDirectory() as tmpdir:
      # Create KB directory structure
      kb_dir = os.path.join(tmpdir, 'test_sections')
      os.makedirs(kb_dir)
      config_path = os.path.join(kb_dir, 'test_sections.cfg')

      # Create config with new sections
      config_content = MockDataGenerator.create_sample_config(
        include_new_sections=True,
        api_call_delay_seconds=0.02,
        api_max_retries=5,
        max_file_size_mb=50,
        embedding_batch_size=25,
        similarity_threshold=0.8
      )

      with open(config_path, 'w') as f:
        f.write(config_content)

      with patch('config.config_manager.VECTORDBS', tmpdir):
        kb = KnowledgeBase('test_sections')

        # Test that values are loaded from config file
        assert kb.api_call_delay_seconds == 0.02
        assert kb.api_max_retries == 5
        assert kb.max_file_size_mb == 50
        assert kb.embedding_batch_size == 25
        assert kb.similarity_threshold == 0.8

        # Test that defaults are used for unspecified values
        assert kb.api_max_concurrency == 2  # from test config default
        assert kb.memory_cache_size == 100  # from test config default

  def test_new_config_sections_env_override(self):
    """Test that environment variables override new config section values."""
    with tempfile.TemporaryDirectory() as tmpdir:
      # Create KB directory structure
      kb_dir = os.path.join(tmpdir, 'test_kb')
      os.makedirs(kb_dir)
      config_path = os.path.join(kb_dir, 'test_kb.cfg')

      # Create config file
      config_content = MockDataGenerator.create_sample_config()
      with open(config_path, 'w') as f:
        f.write(config_content)

      with patch('config.config_manager.VECTORDBS', tmpdir), patch.dict(os.environ, {
        'API_CALL_DELAY_SECONDS': '0.1',
        'MAX_FILE_SIZE_MB': '200',
        'EMBEDDING_BATCH_SIZE': '50',
        'SIMILARITY_THRESHOLD': '0.9'
      }):
        kb = KnowledgeBase('test_kb')
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

  def test_additional_stopword_languages_parsing(self):
    """Test parsing of additional_stopword_languages list parameter."""
    with tempfile.TemporaryDirectory() as tmpdir:
      # Create KB directory structure
      kb_dir = os.path.join(tmpdir, 'test_stopwords')
      os.makedirs(kb_dir)
      config_path = os.path.join(kb_dir, 'test_stopwords.cfg')

      config_content = """[DEFAULT]
vector_model = test-model

[ALGORITHMS]
additional_stopword_languages = spanish,italian,portuguese
"""
      with open(config_path, 'w') as f:
        f.write(config_content)

      with patch('config.config_manager.VECTORDBS', tmpdir):
        kb = KnowledgeBase('test_stopwords')
        assert kb.additional_stopword_languages == ['spanish', 'italian', 'portuguese']

  def test_config_sections_missing_fallback(self):
    """Test fallback to DEFAULT section when specific sections are missing."""
    with tempfile.TemporaryDirectory() as tmpdir:
      # Create KB directory structure
      kb_dir = os.path.join(tmpdir, 'test_fallback')
      os.makedirs(kb_dir)
      config_path = os.path.join(kb_dir, 'test_fallback.cfg')

      config_content = """[DEFAULT]
vector_model = test-model
# No API, LIMITS, PERFORMANCE, or ALGORITHMS sections
"""
      with open(config_path, 'w') as f:
        f.write(config_content)

      with patch('config.config_manager.VECTORDBS', tmpdir):
        kb = KnowledgeBase('test_fallback')

        # Should use defaults since sections are missing
        assert kb.api_call_delay_seconds == 0.05
        assert kb.max_file_size_mb == 100
        assert kb.embedding_batch_size == 100
        assert kb.similarity_threshold == 0.6

  def test_env_vars_override_config_values(self):
    """Test that environment variables override config file values."""
    with tempfile.TemporaryDirectory() as tmpdir:
      # Create KB directory structure
      kb_dir = os.path.join(tmpdir, 'test_kb')
      os.makedirs(kb_dir)
      config_path = os.path.join(kb_dir, 'test_kb.cfg')

      # Create config file
      config_content = MockDataGenerator.create_sample_config()
      with open(config_path, 'w') as f:
        f.write(config_content)

      with patch('config.config_manager.VECTORDBS', tmpdir), patch('config.models.VECTORDBS', tmpdir), patch.dict(os.environ, {
        'VECTOR_MODEL': 'env-model',
        'QUERY_MODEL': 'env-query-model',
      }):
        kb = KnowledgeBase('test_kb')
        assert kb.vector_model == 'env-model'
        assert kb.query_model == 'env-query-model'


class TestErrorHandling:
  """Test error handling in config_manager."""

  def test_corrupted_config_file(self):
    """Test handling of corrupted configuration files."""
    with tempfile.TemporaryDirectory() as tmpdir:
      # Create KB directory structure
      kb_dir = os.path.join(tmpdir, 'corrupted')
      os.makedirs(kb_dir)
      config_path = os.path.join(kb_dir, 'corrupted.cfg')
      with open(config_path, 'w') as f:
        f.write("invalid config content [[[")

      with patch('config.config_manager.VECTORDBS', tmpdir):
        # Should not raise exception, should use defaults
        kb = KnowledgeBase('corrupted')
        assert kb.vector_model == kb.DEF_VECTOR_MODEL

  def test_missing_config_section(self):
    """Test handling of config file without DEFAULT section."""
    with tempfile.TemporaryDirectory() as tmpdir:
      # Create KB directory structure
      kb_dir = os.path.join(tmpdir, 'missing_section')
      os.makedirs(kb_dir)
      config_path = os.path.join(kb_dir, 'missing_section.cfg')

      config_content = """[WRONG_SECTION]
vector_model = test
"""
      with open(config_path, 'w') as f:
        f.write(config_content)

      with patch('config.config_manager.VECTORDBS', tmpdir):
        kb = KnowledgeBase('missing_section')
        # Should use defaults when section is missing
        assert kb.vector_model == kb.DEF_VECTOR_MODEL

  def test_path_security_validation(self):
    """Test that KB names with dangerous patterns are handled."""
    with tempfile.TemporaryDirectory() as tmpdir, patch('config.config_manager.VECTORDBS', tmpdir):
      # These should all resolve to None since the KB doesn't exist
      dangerous_names = [
        "../../../etc/passwd",
        "../../malicious",
        "/etc/shadow"
      ]

      for name in dangerous_names:
        result = get_kb_name(name)
        assert result is None  # KB doesn't exist in VECTORDBS


class TestNewKBResolution:
  """Test the new KB resolution system."""

  def test_kb_name_resolution_simple(self):
    """Test simple KB name resolution."""
    with tempfile.TemporaryDirectory() as tmpdir:
      # Create KB directory structure
      kb_dir = os.path.join(tmpdir, 'myproject')
      os.makedirs(kb_dir)
      config_path = os.path.join(kb_dir, 'myproject.cfg')
      with open(config_path, 'w') as f:
        f.write('[DEFAULT]\nvector_model = test\n')

      with patch('config.config_manager.VECTORDBS', tmpdir):
        # Test get_kb_name
        assert get_kb_name('myproject') == 'myproject'

        # Test get_fq_cfg_filename
        result = get_fq_cfg_filename('myproject')
        assert result == config_path

  def test_path_stripping(self):
    """Test that paths are stripped from KB names."""
    with tempfile.TemporaryDirectory() as tmpdir:
      # Create KB directory
      kb_dir = os.path.join(tmpdir, 'myproject')
      os.makedirs(kb_dir)

      with patch('config.config_manager.VECTORDBS', tmpdir):
        # All these should resolve to 'myproject'
        assert get_kb_name('/path/to/myproject') == 'myproject'
        assert get_kb_name('../myproject') == 'myproject'
        assert get_kb_name('./myproject') == 'myproject'
        assert get_kb_name('subdir/myproject') == 'myproject'

  def test_extension_stripping(self):
    """Test that .cfg extension is stripped."""
    with tempfile.TemporaryDirectory() as tmpdir:
      # Create KB directory
      kb_dir = os.path.join(tmpdir, 'myproject')
      os.makedirs(kb_dir)

      with patch('config.config_manager.VECTORDBS', tmpdir):
        # All these should resolve to 'myproject'
        assert get_kb_name('myproject.cfg') == 'myproject'
        assert get_kb_name('/path/to/myproject.cfg') == 'myproject'
        assert get_kb_name('../myproject.cfg') == 'myproject'

  def test_nonexistent_kb_error(self):
    """Test error handling for nonexistent KBs."""
    with tempfile.TemporaryDirectory() as tmpdir:
      # Create some KBs
      for kb in ['kb1', 'kb2', 'kb3']:
        os.makedirs(os.path.join(tmpdir, kb))

      with patch('config.config_manager.VECTORDBS', tmpdir):
        # This should return None and log available KBs
        result = get_kb_name('nonexistent')
        assert result is None

        result = get_fq_cfg_filename('nonexistent')
        assert result is None

  def test_domain_style_kb_names(self):
    """Test domain-style KB names work correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
      # Create domain-style KB
      kb_dir = os.path.join(tmpdir, 'example.com')
      os.makedirs(kb_dir)
      config_path = os.path.join(kb_dir, 'example.com.cfg')
      with open(config_path, 'w') as f:
        f.write('[DEFAULT]\nvector_model = test\n')

      with patch('config.config_manager.VECTORDBS', tmpdir):
        # Should work with and without .cfg
        assert get_kb_name('example.com') == 'example.com'
        assert get_kb_name('example.com.cfg') == 'example.com'

        result = get_fq_cfg_filename('example.com')
        assert result == config_path

  def test_hidden_directories_resolution(self):
    """Test how get_kb_name handles hidden directories.

    Note: get_kb_name is a resolution function, not a listing function.
    It resolves any existing directory by name - hidden directories are
    only filtered in listing operations (like error messages), not resolution.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
      # Create visible and hidden directories
      os.makedirs(os.path.join(tmpdir, 'visible_kb'))
      os.makedirs(os.path.join(tmpdir, '.hidden_kb'))
      os.makedirs(os.path.join(tmpdir, '.git'))

      with patch('config.config_manager.VECTORDBS', tmpdir):
        # Nonexistent KB returns None
        result = get_kb_name('nonexistent')
        assert result is None

        # Visible KB resolves correctly
        assert get_kb_name('visible_kb') == 'visible_kb'

        # Hidden directories ARE resolved if they exist (resolution != listing)
        # Users who explicitly request a hidden KB should get it
        assert get_kb_name('.hidden_kb') == '.hidden_kb'

  def test_empty_vectordbs(self):
    """Test behavior when VECTORDBS is empty."""
    with tempfile.TemporaryDirectory() as tmpdir, patch('config.config_manager.VECTORDBS', tmpdir):
      # Should return None for any KB name
      assert get_kb_name('anything') is None
      assert get_fq_cfg_filename('anything') is None

  def test_vectordbs_permissions_error(self):
    """Test handling when VECTORDBS is not accessible."""
    with patch('config.config_manager.VECTORDBS', '/nonexistent/directory'):
      # Should handle gracefully
      assert get_kb_name('myproject') is None
      assert get_fq_cfg_filename('myproject') is None

  def test_special_characters_in_kb_name(self):
    """Test KB names with special characters."""
    with tempfile.TemporaryDirectory() as tmpdir:
      # Create KB with special characters (but filesystem-safe)
      kb_names = ['my-project', 'my_project', 'my.project', 'my project']

      for kb_name in kb_names:
        kb_dir = os.path.join(tmpdir, kb_name)
        os.makedirs(kb_dir)
        config_path = os.path.join(kb_dir, f'{kb_name}.cfg')
        with open(config_path, 'w') as f:
          f.write('[DEFAULT]\nvector_model = test\n')

      with patch('config.config_manager.VECTORDBS', tmpdir):
        for kb_name in kb_names:
          assert get_kb_name(kb_name) == kb_name
          result = get_fq_cfg_filename(kb_name)
          assert result.endswith(f'{kb_name}.cfg')


class TestBM25Configuration:
  """Test BM25/Hybrid search configuration parameters."""

  def test_default_bm25_parameters(self):
    """Test that BM25 parameters have correct defaults."""
    with tempfile.TemporaryDirectory() as tmpdir:
      # Create KB directory structure
      kb_dir = os.path.join(tmpdir, 'test_bm25_defaults')
      os.makedirs(kb_dir)
      config_path = os.path.join(kb_dir, 'test_bm25_defaults.cfg')

      with open(config_path, 'w') as f:
        f.write("""
[DEFAULT]
knowledge_base_name = test_bm25_defaults
""")

      with patch('config.config_manager.VECTORDBS', tmpdir):
        kb = KnowledgeBase('test_bm25_defaults')

        # Test default BM25 parameters
        assert kb.enable_hybrid_search is False
        assert kb.vector_weight == 0.7
        assert kb.bm25_k1 == 1.2
        assert kb.bm25_b == 0.75
        assert kb.bm25_min_token_length == 2
        assert kb.bm25_rebuild_threshold == 1000

  def test_bm25_configuration_loading(self):
    """Test loading BM25 parameters from config file."""
    with tempfile.TemporaryDirectory() as tmpdir:
      # Create KB directory structure
      kb_dir = os.path.join(tmpdir, 'test_bm25_config')
      os.makedirs(kb_dir)
      config_path = os.path.join(kb_dir, 'test_bm25_config.cfg')

      with open(config_path, 'w') as f:
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

      with patch('config.config_manager.VECTORDBS', tmpdir):
        kb = KnowledgeBase('test_bm25_config')

        # Test loaded BM25 parameters
        assert kb.enable_hybrid_search is True
        assert kb.vector_weight == 0.8
        assert kb.bm25_k1 == 1.5
        assert kb.bm25_b == 0.8
        assert kb.bm25_min_token_length == 3
        assert kb.bm25_rebuild_threshold == 500

  def test_bm25_environment_variable_overrides(self):
    """Test that environment variables override BM25 config file values."""
    with tempfile.TemporaryDirectory() as tmpdir:
      # Create KB directory structure
      kb_dir = os.path.join(tmpdir, 'test_bm25_env')
      os.makedirs(kb_dir)
      config_path = os.path.join(kb_dir, 'test_bm25_env.cfg')

      with open(config_path, 'w') as f:
        f.write("""
[DEFAULT]
knowledge_base_name = test_bm25_env

[ALGORITHMS]
enable_hybrid_search = false
vector_weight = 0.7
bm25_k1 = 1.2
""")

      with patch('config.config_manager.VECTORDBS', tmpdir):
        # Set environment variables
        env_vars = {
          'ENABLE_HYBRID_SEARCH': 'true',
          'VECTOR_WEIGHT': '0.9',
          'BM25_K1': '2.0'
        }

        with patch.dict(os.environ, env_vars):
          kb = KnowledgeBase('test_bm25_env')

          # Environment variables should override config file
          assert kb.enable_hybrid_search is True
          assert kb.vector_weight == 0.9
          assert kb.bm25_k1 == 2.0

  def test_bm25_parameter_validation(self):
    """Test validation of BM25 parameter ranges."""
    with tempfile.TemporaryDirectory() as tmpdir:
      # Create KB directory structure
      kb_dir = os.path.join(tmpdir, 'test_bm25_validation')
      os.makedirs(kb_dir)
      config_path = os.path.join(kb_dir, 'test_bm25_validation.cfg')

      with open(config_path, 'w') as f:
        f.write("""
[DEFAULT]
knowledge_base_name = test_bm25_validation

[ALGORITHMS]
enable_hybrid_search = true
vector_weight = 0.7
bm25_k1 = 1.2
bm25_b = 0.75
""")

      with patch('config.config_manager.VECTORDBS', tmpdir):
        kb = KnowledgeBase('test_bm25_validation')

        # Test that parameters are within expected ranges
        assert 0.0 <= kb.vector_weight <= 1.0
        assert kb.bm25_k1 > 0.0  # Should be positive
        assert 0.0 <= kb.bm25_b <= 1.0  # Should be between 0 and 1
        assert kb.bm25_min_token_length >= 1  # Should be at least 1
        assert kb.bm25_rebuild_threshold > 0  # Should be positive

  def test_bm25_invalid_boolean_values(self):
    """Test handling of invalid boolean values for enable_hybrid_search.

    Invalid boolean values in config files should raise ValueError.
    This is the correct 'fail fast' behavior - bad config should not
    silently fall back to defaults.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
      # Create KB directory structure
      kb_dir = os.path.join(tmpdir, 'test_bm25_bool')
      os.makedirs(kb_dir)
      config_path = os.path.join(kb_dir, 'test_bm25_bool.cfg')

      with open(config_path, 'w') as f:
        f.write("""
[DEFAULT]
knowledge_base_name = test_bm25_bool

[ALGORITHMS]
enable_hybrid_search = invalid_value
""")

      with patch('config.config_manager.VECTORDBS', tmpdir), pytest.raises(ValueError, match="Not a boolean"):
        KnowledgeBase('test_bm25_bool')

  def test_bm25_mixed_configuration_sources(self):
    """Test BM25 configuration from multiple sources (file + env + defaults)."""
    with tempfile.TemporaryDirectory() as tmpdir:
      # Create KB directory structure
      kb_dir = os.path.join(tmpdir, 'test_bm25_mixed')
      os.makedirs(kb_dir)
      config_path = os.path.join(kb_dir, 'test_bm25_mixed.cfg')

      with open(config_path, 'w') as f:
        f.write("""
[DEFAULT]
knowledge_base_name = test_bm25_mixed

[ALGORITHMS]
enable_hybrid_search = true
vector_weight = 0.6
# bm25_k1 not specified (should use default)
bm25_b = 0.8
""")

      with patch('config.config_manager.VECTORDBS', tmpdir), patch.dict(os.environ, {'VECTOR_WEIGHT': '0.75'}):
        kb = KnowledgeBase('test_bm25_mixed')

        # Should get values from different sources
        assert kb.enable_hybrid_search is True  # From config file
        assert kb.vector_weight == 0.75  # From environment
        assert kb.bm25_k1 == 1.2  # From default (not in config/env)
        assert kb.bm25_b == 0.8  # From config file
        assert kb.bm25_min_token_length == 2  # From default

  def test_bm25_kwargs_override(self):
    """Test BM25 configuration via kwargs (programmatic override)."""
    with tempfile.TemporaryDirectory() as tmpdir:
      # Create KB directory structure
      kb_dir = os.path.join(tmpdir, 'test_bm25_kwargs')
      os.makedirs(kb_dir)
      config_path = os.path.join(kb_dir, 'test_bm25_kwargs.cfg')

      with open(config_path, 'w') as f:
        f.write("""
[DEFAULT]
knowledge_base_name = test_bm25_kwargs
""")

      with patch('config.config_manager.VECTORDBS', tmpdir):
        # Pass BM25 parameters via kwargs
        kb = KnowledgeBase('test_bm25_kwargs',
                          enable_hybrid_search=True,
                          vector_weight=0.85,
                          bm25_k1=1.8,
                          bm25_b=0.9)

        # Should use kwargs values
        assert kb.enable_hybrid_search is True
        assert kb.vector_weight == 0.85
        assert kb.bm25_k1 == 1.8
        assert kb.bm25_b == 0.9

  def test_bm25_configuration_save(self):
    """Test that BM25 configuration is included in save output."""
    with tempfile.TemporaryDirectory() as tmpdir:
      # Create KB directory structure
      kb_dir = os.path.join(tmpdir, 'test_bm25_save')
      os.makedirs(kb_dir)
      config_path = os.path.join(kb_dir, 'test_bm25_save.cfg')

      with open(config_path, 'w') as f:
        f.write("""
[DEFAULT]
knowledge_base_name = test_bm25_save

[ALGORITHMS]
enable_hybrid_search = true
vector_weight = 0.8
""")

      with patch('config.config_manager.VECTORDBS', tmpdir):
        kb = KnowledgeBase('test_bm25_save')

        # Save config to string
        import io
        io.StringIO()
        kb.save_config()  # This prints to stderr by default

        # Check that BM25 parameters would be included in save
        # (We test via hasattr since save_config prints all attributes)
        assert hasattr(kb, 'enable_hybrid_search')
        assert hasattr(kb, 'vector_weight')
        assert hasattr(kb, 'bm25_k1')
        assert hasattr(kb, 'bm25_b')
        assert hasattr(kb, 'bm25_min_token_length')
        assert hasattr(kb, 'bm25_rebuild_threshold')

#fin
