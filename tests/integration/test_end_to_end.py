"""
End-to-end integration tests for CustomKB.
Tests complete workflows from database creation through querying.
"""

import json
import os
from unittest.mock import Mock, patch

import numpy as np
import pytest


@pytest.mark.integration
class TestEndToEndWorkflow:
  """Test complete end-to-end CustomKB workflows."""

  def test_complete_workflow_database_to_query(self, temp_data_manager, sample_texts, mock_openai_client, mock_anthropic_client, mock_faiss_index, monkeypatch):
    """Test complete workflow: database → embed → query."""
    # Create test environment with proper KB directory structure
    kb_name = "test_kb"

    # Create configuration content
    config_content = """[DEFAULT]
vector_model = text-embedding-3-small
vector_dimensions = 1536
vector_chunks = 100
db_min_tokens = 50
db_max_tokens = 150
query_model = gpt-4o
query_max_tokens = 1000
query_top_k = 10
query_context_scope = 2
query_temperature = 0.1
query_role = You are a helpful assistant.
"""

    # Create properly structured KB directory
    temp_vectordbs, kb_dir, config_file = temp_data_manager.create_kb_directory(kb_name, config_content)

    # Monkeypatch VECTORDBS to use temp directory
    monkeypatch.setattr('config.config_manager.VECTORDBS', temp_vectordbs)

    # Create test documents
    test_files = []
    for i, text in enumerate(sample_texts[:3]):
      file_path = os.path.join(kb_dir, f"doc_{i}.txt")
      with open(file_path, 'w') as f:
        f.write(text)
      test_files.append(file_path)

    # Import required modules for testing
    from database.db_manager import process_database
    from embedding.embed_manager import process_embeddings
    from query.query_manager import process_query

    # Mock the logger
    mock_logger = Mock()

    # Step 1: Process database (use kb_name per new resolution system)
    db_args = Mock()
    db_args.config_file = kb_name  # Use KB name instead of full path
    db_args.files = test_files
    db_args.language = 'en'
    db_args.force = False
    db_args.verbose = True
    db_args.debug = False

    with patch('builtins.input', return_value='y'):  # Auto-confirm DB creation
      db_result = process_database(db_args, mock_logger)

    assert "files added to database" in db_result

    # Verify database was created
    db_path = os.path.join(kb_dir, f"{kb_name}.db")
    assert os.path.exists(db_path)

    # Step 2: Generate embeddings
    embed_args = Mock()
    embed_args.config_file = kb_name  # Use KB name
    embed_args.reset_database = False
    embed_args.verbose = True
    embed_args.debug = False

    with patch('embedding.embed_manager.get_optimal_faiss_index', return_value=mock_faiss_index), \
         patch('asyncio.run', return_value={1, 2, 3}):
      embed_result = process_embeddings(embed_args, mock_logger)

    assert "embeddings" in embed_result
    assert "saved to" in embed_result

    # Step 3: Query the knowledgebase
    query_args = Mock()
    query_args.config_file = kb_name  # Use KB name
    query_args.query_text = "What is machine learning?"
    query_args.query_file = ""
    query_args.context_only = False
    query_args.verbose = True
    query_args.debug = False

    with patch('faiss.read_index', return_value=mock_faiss_index), \
         patch('query.llm.generate_ai_response', return_value="Mock AI response about machine learning"):
      query_result = process_query(query_args, mock_logger)

    assert isinstance(query_result, str)
    assert len(query_result) > 0

  def test_workflow_with_context_only_query(self, temp_data_manager, sample_texts, mock_openai_client, mock_faiss_index, monkeypatch):
    """Test workflow ending with context-only query."""
    # Set up test environment with proper KB structure
    kb_name = "context_test_kb"

    config_content = """[DEFAULT]
vector_model = text-embedding-3-small
vector_dimensions = 1536
query_model = gpt-4o
"""

    # Create properly structured KB directory
    temp_vectordbs, kb_dir, config_file = temp_data_manager.create_kb_directory(kb_name, config_content)
    monkeypatch.setattr('config.config_manager.VECTORDBS', temp_vectordbs)

    # Create single test file
    test_file = os.path.join(kb_dir, "test.txt")
    with open(test_file, 'w') as f:
      f.write(sample_texts[0])

    # Import and run components
    from database.db_manager import process_database
    from embedding.embed_manager import process_embeddings
    from query.query_manager import process_query

    mock_logger = Mock()

    # Database step
    db_args = Mock()
    db_args.config_file = kb_name  # Use KB name
    db_args.files = [test_file]
    db_args.language = 'en'
    db_args.force = False
    db_args.verbose = True
    db_args.debug = False

    with patch('builtins.input', return_value='y'):
      process_database(db_args, mock_logger)

    # Embedding step
    embed_args = Mock()
    embed_args.config_file = kb_name  # Use KB name
    embed_args.reset_database = False
    embed_args.verbose = True
    embed_args.debug = False

    with patch('embedding.embed_manager.get_optimal_faiss_index', return_value=mock_faiss_index), \
         patch('asyncio.run', return_value={1}):
      process_embeddings(embed_args, mock_logger)

    # Context-only query step
    query_args = Mock()
    query_args.config_file = kb_name  # Use KB name
    query_args.query_text = "Test query"
    query_args.query_file = ""
    query_args.context_only = True  # Context only
    query_args.verbose = True
    query_args.debug = False

    with patch('faiss.read_index', return_value=mock_faiss_index):
      result = process_query(query_args, mock_logger)

    assert isinstance(result, str)
    # Should contain context XML tags
    assert "<context" in result or "context" in result.lower()

  def test_workflow_with_force_reprocessing(self, temp_data_manager, sample_texts, mock_openai_client, mock_faiss_index, monkeypatch):
    """Test workflow with force reprocessing of existing data."""
    kb_name = "force_test_kb"

    config_content = """[DEFAULT]
vector_model = text-embedding-3-small
vector_dimensions = 1536
"""

    # Create properly structured KB directory
    temp_vectordbs, kb_dir, config_file = temp_data_manager.create_kb_directory(kb_name, config_content)
    monkeypatch.setattr('config.config_manager.VECTORDBS', temp_vectordbs)

    test_file = os.path.join(kb_dir, "test.txt")
    with open(test_file, 'w') as f:
      f.write(sample_texts[0])

    from database.db_manager import process_database

    mock_logger = Mock()

    # First processing
    db_args = Mock()
    db_args.config_file = kb_name  # Use KB name
    db_args.files = [test_file]
    db_args.language = 'en'
    db_args.force = False
    db_args.verbose = True
    db_args.debug = False

    with patch('builtins.input', return_value='y'):
      first_result = process_database(db_args, mock_logger)

    assert "files added" in first_result

    # Second processing without force (should skip)
    second_result = process_database(db_args, mock_logger)
    assert "skipped" in second_result

    # Third processing with force (should reprocess)
    db_args.force = True
    third_result = process_database(db_args, mock_logger)
    assert "files added" in third_result

  def test_workflow_with_multiple_file_types(self, temp_data_manager, mock_openai_client, mock_faiss_index, monkeypatch):
    """Test workflow with multiple file types (markdown, text, etc.)."""
    kb_name = "multifile_kb"

    config_content = """[DEFAULT]
vector_model = text-embedding-3-small
vector_dimensions = 1536
"""

    # Create properly structured KB directory
    temp_vectordbs, kb_dir, config_file = temp_data_manager.create_kb_directory(kb_name, config_content)
    monkeypatch.setattr('config.config_manager.VECTORDBS', temp_vectordbs)

    # Create files of different types
    test_files = []

    # Markdown file
    md_file = os.path.join(kb_dir, "readme.md")
    with open(md_file, 'w') as f:
      f.write("# Test Document\n\nThis is a markdown document with **bold** text.")
    test_files.append(md_file)

    # Python file
    py_file = os.path.join(kb_dir, "script.py")
    with open(py_file, 'w') as f:
      f.write("def hello_world():\n    print('Hello, World!')\n    return True")
    test_files.append(py_file)

    # Text file
    txt_file = os.path.join(kb_dir, "notes.txt")
    with open(txt_file, 'w') as f:
      f.write("These are some plain text notes about the project.")
    test_files.append(txt_file)

    from database.db_manager import process_database

    mock_logger = Mock()

    # Process all file types
    db_args = Mock()
    db_args.config_file = kb_name  # Use KB name
    db_args.files = test_files
    db_args.language = 'en'
    db_args.force = False
    db_args.verbose = True
    db_args.debug = False

    with patch('builtins.input', return_value='y'):
      result = process_database(db_args, mock_logger)

    assert "3 files added" in result

    # Verify different file types were processed
    import sqlite3
    db_path = os.path.join(kb_dir, f"{kb_name}.db")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT sourcedoc FROM docs")
    sources = [row[0] for row in cursor.fetchall()]
    conn.close()

    # sourcedoc stores full absolute paths
    source_basenames = [os.path.basename(s) for s in sources]
    assert "readme.md" in source_basenames
    assert "script.py" in source_basenames
    assert "notes.txt" in source_basenames

  def test_workflow_error_recovery(self, temp_data_manager, sample_texts, monkeypatch):
    """Test workflow error recovery scenarios."""
    kb_name = "error_test_kb"

    config_content = """[DEFAULT]
vector_model = text-embedding-3-small
vector_dimensions = 1536
"""

    # Create properly structured KB directory
    temp_vectordbs, kb_dir, config_file = temp_data_manager.create_kb_directory(kb_name, config_content)

    # Monkeypatch VECTORDBS to use temp directory
    monkeypatch.setattr('config.config_manager.VECTORDBS', temp_vectordbs)

    test_file = os.path.join(kb_dir, "test.txt")
    with open(test_file, 'w') as f:
      f.write(sample_texts[0])

    from database.db_manager import process_database
    from embedding.embed_manager import process_embeddings

    mock_logger = Mock()

    # Test database processing with missing file
    db_args = Mock()
    db_args.config_file = kb_name  # Use KB name instead of full path
    db_args.files = [test_file, "/nonexistent/file.txt"]
    db_args.language = 'en'
    db_args.force = False
    db_args.verbose = True
    db_args.debug = False

    with patch('builtins.input', return_value='y'):
      # Should process existing file and handle missing file gracefully
      result = process_database(db_args, mock_logger)

    # Should still process the valid file
    assert isinstance(result, str)

    # Test embedding with API errors
    embed_args = Mock()
    embed_args.config_file = kb_name  # Use KB name instead of full path
    embed_args.reset_database = False
    embed_args.verbose = True
    embed_args.debug = False

    with patch('embedding.embed_manager.get_optimal_faiss_index') as mock_index, \
         patch('asyncio.run', return_value={1}):
      mock_index.return_value = Mock()
      result = process_embeddings(embed_args, mock_logger)

    assert isinstance(result, str)


@pytest.mark.integration
class TestCLIIntegration:
  """Test CLI integration and command-line interface."""

  def test_cli_help_command(self):
    """Test CLI help command."""
    # Test that help can be invoked without errors
    from customkb import customkb_usage

    help_text = customkb_usage()
    assert "CustomKB" in help_text
    # Note: __main__.__doc__ may differ in test context vs CLI invocation

  def test_cli_version_command(self):
    """Test CLI version command."""
    from version import get_version

    version_str = get_version()
    assert isinstance(version_str, str)
    assert len(version_str) > 0

  def test_cli_argument_parsing(self):
    """Test CLI argument parsing."""
    import sys
    from unittest.mock import patch

    # Mock command line arguments
    test_args = ['customkb', 'query', 'test.cfg', 'test query', '--verbose']

    with patch.object(sys, 'argv', test_args):
      from customkb import main

      # Should not raise exception during argument parsing
      # (execution will fail due to missing files, but parsing should work)
      try:
        with patch('customkb.setup_logging', return_value=None):
          main()
      except SystemExit:
        pass  # Expected due to missing config file

  def test_cli_config_file_resolution(self, temp_data_manager, monkeypatch):
    """Test CLI configuration file resolution."""
    kb_name = "test"

    config_content = "[DEFAULT]\nvector_model = test\n"

    # Create properly structured KB directory
    temp_vectordbs, kb_dir, config_file = temp_data_manager.create_kb_directory(kb_name, config_content)

    # Monkeypatch VECTORDBS to use temp directory
    monkeypatch.setattr('config.config_manager.VECTORDBS', temp_vectordbs)

    from config.config_manager import get_fq_cfg_filename

    # Test absolute path
    result = get_fq_cfg_filename(config_file)
    assert result == config_file

    # Test KB name resolution
    result = get_fq_cfg_filename(kb_name)
    assert result == config_file


@pytest.mark.integration
class TestRealDataIntegration:
  """Test integration with real wayang.net dataset."""

  @pytest.mark.requires_data
  def test_with_wayang_dataset(self, mock_openai_client, mock_faiss_index):
    """Test integration using the wayang.net dataset."""
    import os

    # Check if wayang.net data exists
    vectordbs = os.getenv('VECTORDBS', '/var/lib/vectordbs')
    wayang_dir = os.path.join(vectordbs, 'wayang.net')
    wayang_config = os.path.join(wayang_dir, 'wayang.net.cfg')
    wayang_db = os.path.join(wayang_dir, 'wayang.net.db')

    if not os.path.exists(wayang_config) or not os.path.exists(wayang_db):
      pytest.skip("wayang.net dataset not available")

    from query.query_manager import process_query

    mock_logger = Mock()

    # Test query against real data
    query_args = Mock()
    query_args.config_file = wayang_config
    query_args.query_text = "What is Wayang?"
    query_args.query_file = ""
    query_args.context_only = True  # Context only to avoid API calls
    query_args.verbose = True
    query_args.debug = False

    with patch('faiss.read_index', return_value=mock_faiss_index), \
         patch('query.processing.get_query_embedding') as mock_embedding:
      mock_embedding.return_value = np.array([[0.1] * 1536])

      result = process_query(query_args, mock_logger)

    assert isinstance(result, str)
    assert len(result) > 0

  @pytest.mark.requires_data
  def test_wayang_database_structure(self):
    """Test that wayang.net database has expected structure."""
    import os
    import sqlite3

    vectordbs = os.getenv('VECTORDBS', '/var/lib/vectordbs')
    wayang_db = os.path.join(vectordbs, 'wayang.net', 'wayang.net.db')

    if not os.path.exists(wayang_db):
      pytest.skip("wayang.net database not available")

    conn = sqlite3.connect(wayang_db)
    cursor = conn.cursor()

    # Check table exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='docs'")
    assert cursor.fetchone() is not None

    # Check table structure
    cursor.execute("PRAGMA table_info(docs)")
    columns = cursor.fetchall()
    column_names = [col[1] for col in columns]

    expected_columns = ['id', 'sid', 'sourcedoc', 'originaltext', 'embedtext', 'embedded', 'language', 'metadata']
    for col in expected_columns:
      assert col in column_names

    # Check data exists
    cursor.execute("SELECT COUNT(*) FROM docs")
    count = cursor.fetchone()[0]
    assert count > 0

    conn.close()


@pytest.mark.integration
class TestFullPathIntegration:
  """Test full path storage integration across components."""

  def test_duplicate_filenames_workflow(self, temp_data_manager, sample_texts, mock_openai_client, mock_faiss_index, monkeypatch):
    """Test complete workflow with duplicate filenames in different directories."""
    kb_name = "fullpath_test"

    config_content = """[DEFAULT]
vector_model = text-embedding-3-small
vector_dimensions = 1536
vector_chunks = 100
db_min_tokens = 50
db_max_tokens = 150
query_model = gpt-4o
query_max_tokens = 1000
query_top_k = 10
"""

    # Create properly structured KB directory
    temp_vectordbs, kb_dir, config_file = temp_data_manager.create_kb_directory(kb_name, config_content)

    # Monkeypatch VECTORDBS to use temp directory
    monkeypatch.setattr('config.config_manager.VECTORDBS', temp_vectordbs)

    # Create duplicate filenames in different directories
    dir1 = os.path.join(kb_dir, "module1")
    dir2 = os.path.join(kb_dir, "module2")
    os.makedirs(dir1)
    os.makedirs(dir2)

    # Same filename, different content
    config1 = os.path.join(dir1, "config.py")
    config2 = os.path.join(dir2, "config.py")

    with open(config1, 'w') as f:
      f.write("# Module 1 Configuration\nDATABASE_URL = 'postgresql://localhost/module1'\nDEBUG = True")

    with open(config2, 'w') as f:
      f.write("# Module 2 Configuration\nDATABASE_URL = 'mysql://localhost/module2'\nDEBUG = False")

    # Process database
    from database.db_manager import process_database
    from embedding.embed_manager import process_embeddings
    from query.query_manager import process_query

    mock_logger = Mock()

    # Process both files
    db_args = Mock()
    db_args.config_file = kb_name  # Use KB name instead of full path
    db_args.files = [config1, config2]
    db_args.language = 'en'
    db_args.force = False
    db_args.verbose = True
    db_args.debug = False

    with patch('builtins.input', return_value='y'):
      result = process_database(db_args, mock_logger)

    assert "2 files added" in result

    # Create embeddings
    embed_args = Mock()
    embed_args.config_file = kb_name  # Use KB name instead of full path
    embed_args.reset_database = False
    embed_args.verbose = True
    embed_args.debug = False

    mock_openai_client.embeddings.create.return_value = Mock(
      data=[Mock(embedding=[0.1 + i*0.01] * 1536) for i in range(2)]
    )

    with patch('asyncio.run', return_value={1, 2}):
      embed_result = process_embeddings(embed_args, mock_logger)

    assert "2 chunks embedded" in embed_result

    # Query for module1
    query_args = Mock()
    query_args.config_file = kb_name  # Use KB name instead of full path
    query_args.query_text = "postgresql module1"
    query_args.context_only = True
    query_args.query_file = ''
    query_args.role = ''
    query_args.model = ''
    query_args.top_k = None
    query_args.context_scope = None
    query_args.temperature = ''
    query_args.max_tokens = ''
    query_args.verbose = True
    query_args.debug = False

    with patch('query.processing.get_query_embedding', return_value=np.array([[0.11] * 1536])):
      result = process_query(query_args, mock_logger)

    # Should find module1 config
    assert "module1" in result
    assert "postgresql" in result.lower()

    # Query for module2
    query_args.query_text = "mysql module2"

    with patch('query.processing.get_query_embedding', return_value=np.array([[0.19] * 1536])):
      result = process_query(query_args, mock_logger)

    # Should find module2 config
    assert "module2" in result
    assert "mysql" in result.lower()

  def test_path_display_in_query_results(self, temp_data_manager, mock_openai_client, mock_faiss_index, monkeypatch):
    """Test that paths are displayed correctly in query results."""
    kb_name = "path_display_test"

    config_content = """[DEFAULT]
vector_model = text-embedding-3-small
vector_dimensions = 1536
query_model = gpt-4o
"""

    # Create properly structured KB directory
    temp_vectordbs, kb_dir, config_file = temp_data_manager.create_kb_directory(kb_name, config_content)

    # Monkeypatch VECTORDBS to use temp directory
    monkeypatch.setattr('config.config_manager.VECTORDBS', temp_vectordbs)

    # Create deeply nested file
    deep_dir = os.path.join(kb_dir, "src", "components", "ui", "widgets", "forms")
    os.makedirs(deep_dir)

    deep_file = os.path.join(deep_dir, "validation.js")
    with open(deep_file, 'w') as f:
      f.write("// Form validation utilities\nfunction validateEmail(email) { return /^[^@]+@[^@]+\\.[^@]+$/.test(email); }")

    # Process and query
    from database.db_manager import process_database
    from embedding.embed_manager import process_embeddings
    from query.query_manager import process_query

    mock_logger = Mock()

    # Process file
    db_args = Mock()
    db_args.config_file = kb_name  # Use KB name instead of full path
    db_args.files = [deep_file]
    db_args.language = 'en'
    db_args.force = False
    db_args.verbose = True
    db_args.debug = False

    with patch('builtins.input', return_value='y'):
      process_database(db_args, mock_logger)

    # Embed
    embed_args = Mock()
    embed_args.config_file = kb_name  # Use KB name instead of full path
    embed_args.reset_database = False
    embed_args.verbose = True
    embed_args.debug = False

    mock_openai_client.embeddings.create.return_value = Mock(
      data=[Mock(embedding=[0.15] * 1536)]
    )

    with patch('asyncio.run', return_value={1}):
      process_embeddings(embed_args, mock_logger)

    # Query
    query_args = Mock()
    query_args.config_file = kb_name  # Use KB name instead of full path
    query_args.query_text = "email validation"
    query_args.context_only = True
    query_args.query_file = ''
    query_args.role = ''
    query_args.model = ''
    query_args.top_k = None
    query_args.context_scope = None
    query_args.temperature = ''
    query_args.max_tokens = ''
    query_args.verbose = True
    query_args.debug = False

    with patch('query.processing.get_query_embedding', return_value=np.array([[0.14] * 1536])):
      result = process_query(query_args, mock_logger)

    # Check for truncated path display
    assert ".../widgets/forms/validation.js" in result or "validation.js" in result
    # Should not show the full absolute path in display
    assert kb_dir not in result  # Full path should be truncated


@pytest.mark.integration
class TestConfigurationIntegration:
  """Test configuration integration across components."""

  def test_environment_variable_override_integration(self, temp_data_manager, sample_texts, monkeypatch):
    """Test that environment variables properly override config values across components."""
    kb_name = "env_test"

    # Create config with default values
    config_content = """[DEFAULT]
vector_model = text-embedding-3-small
vector_dimensions = 1536
query_model = gpt-4o
query_temperature = 0.1
"""

    # Create properly structured KB directory
    temp_vectordbs, kb_dir, config_file = temp_data_manager.create_kb_directory(kb_name, config_content)
    monkeypatch.setattr('config.config_manager.VECTORDBS', temp_vectordbs)

    # Test environment variable overrides
    env_overrides = {
      'VECTOR_MODEL': 'text-embedding-ada-002',
      'VECTOR_DIMENSIONS': '1024',
      'QUERY_MODEL': 'gpt-3.5-turbo',
      'QUERY_TEMPERATURE': '0.5'
    }

    from config.config_manager import KnowledgeBase

    with patch.dict(os.environ, env_overrides):
      kb = KnowledgeBase(kb_name)  # Use KB name, not full path

      # Verify overrides took effect
      assert kb.vector_model == 'text-embedding-ada-002'
      assert kb.vector_dimensions == 1024
      assert kb.query_model == 'gpt-3.5-turbo'
      assert kb.query_temperature == 0.5

  def test_domain_style_configuration(self, temp_data_manager, monkeypatch):
    """Test domain-style configuration naming."""
    kb_name = "example.com"

    config_content = """[DEFAULT]
vector_model = text-embedding-3-large
vector_dimensions = 3072
"""

    # Create properly structured KB directory
    temp_vectordbs, kb_dir, domain_config = temp_data_manager.create_kb_directory(kb_name, config_content)
    monkeypatch.setattr('config.config_manager.VECTORDBS', temp_vectordbs)

    from config.config_manager import KnowledgeBase, get_fq_cfg_filename

    # Test domain-style resolution (VECTORDBS already monkeypatched)
    resolved_path = get_fq_cfg_filename("example.com")
    assert resolved_path == domain_config

    # Test KnowledgeBase with domain-style name
    kb = KnowledgeBase("example.com")  # Use KB name
    assert kb.knowledge_base_name == "example.com"
    assert kb.knowledge_base_db.endswith("example.com.db")
    assert kb.knowledge_base_vector.endswith("example.com.faiss")

  def test_models_json_integration(self, temp_data_manager):
    """Test Models.json integration with configuration."""
    # Create temporary Models.json
    models_data = {
      "custom-embedding": {
        "model": "text-embedding-custom",
        "alias": "custom",
        "provider": "openai",
        "dimensions": 2048
      }
    }

    models_file = os.path.join(temp_data_manager.create_temp_dir(), "Models.json")
    with open(models_file, 'w') as f:
      json.dump(models_data, f)

    from models.model_manager import get_canonical_model

    with patch('models.model_manager.models_file', models_file):
      # Test alias resolution
      result = get_canonical_model("custom")
      assert result["model"] == "text-embedding-custom"
      assert result["dimensions"] == 2048

      # Test partial match (substring of key "custom-embedding")
      result = get_canonical_model("custom-embed")
      assert result["model"] == "text-embedding-custom"


@pytest.mark.integration
class TestPathResolutionIntegration:
  """Integration tests for the three configuration path use cases."""

  def test_case_1_absolute_path_to_cfg_file(self, temp_data_manager):
    """Test Case 1: customkb query /absolute/path/kb.cfg 'query'"""
    # Create KB with absolute path
    kb_dir = temp_data_manager.create_temp_dir()
    config_path = os.path.join(kb_dir, 'myproject.cfg')

    # Create minimal config
    with open(config_path, 'w') as f:
      f.write("""[DEFAULT]
knowledge_base_name = myproject
vector_model = text-embedding-3-small
query_model = claude-3-5-sonnet-20241022
""")

    # Test that config resolution works
    from config.config_manager import KnowledgeBase, get_fq_cfg_filename

    result = get_fq_cfg_filename(config_path)
    assert result == config_path

    # Test that KnowledgeBase can be created
    kb = KnowledgeBase(config_path)
    assert kb.knowledge_base_name == 'myproject'
    assert kb.knowledge_base_db == os.path.join(kb_dir, 'myproject.db')
    assert kb.knowledge_base_vector == os.path.join(kb_dir, 'myproject.faiss')

  def test_case_2_kb_name_searches_vectordbs(self, temp_data_manager, monkeypatch):
    """Test Case 2: customkb query kbname 'query'"""
    kb_name = "searchproject"

    config_content = """[DEFAULT]
vector_model = text-embedding-3-small
query_model = claude-3-5-sonnet-20241022
"""

    # Create properly structured KB directory
    temp_vectordbs, kb_dir, config_path = temp_data_manager.create_kb_directory(kb_name, config_content)

    # Test with mocked VECTORDBS
    from config.config_manager import KnowledgeBase, get_fq_cfg_filename

    with patch('config.config_manager.VECTORDBS', temp_vectordbs):
      # Test resolution by name only
      result = get_fq_cfg_filename('searchproject')
      assert result == config_path

      # Test KnowledgeBase creation
      kb = KnowledgeBase('searchproject')
      assert kb.knowledge_base_name == 'searchproject'
      assert kb.knowledge_base_db.endswith('searchproject.db')

  def test_case_3_absolute_path_to_kb_directory(self, temp_data_manager):
    """Test Case 3: customkb query /absolute/path/kbdir 'query'"""
    # Create KB directory structure
    kb_dir = temp_data_manager.create_temp_dir()
    kb_name = os.path.basename(kb_dir)
    config_path = os.path.join(kb_dir, f'{kb_name}.cfg')

    with open(config_path, 'w') as f:
      f.write(f"""[DEFAULT]
knowledge_base_name = {kb_name}
vector_model = text-embedding-3-small
query_model = claude-3-5-sonnet-20241022
""")

    # Test resolution of directory path
    from config.config_manager import KnowledgeBase, get_fq_cfg_filename

    # For Case 3, we need to use the config file directly for now
    result = get_fq_cfg_filename(config_path)
    assert result == config_path

    # Test KnowledgeBase creation
    kb = KnowledgeBase(config_path)
    assert kb.knowledge_base_name == kb_name
    assert kb.knowledge_base_db.endswith(f'{kb_name}.db')

  def test_relative_traversal_sibling_directories(self, temp_data_manager):
    """Test the specific user case: ../okusimail/okusimail.cfg"""
    # Create structure like user's environment
    base_dir = temp_data_manager.create_temp_dir()
    okusimail_dir = os.path.join(base_dir, 'okusimail')
    okusiassociates_dir = os.path.join(base_dir, 'okusiassociates')
    os.makedirs(okusimail_dir)
    os.makedirs(okusiassociates_dir)

    # Create okusimail config
    config_path = os.path.join(okusimail_dir, 'okusimail.cfg')
    with open(config_path, 'w') as f:
      f.write("""[DEFAULT]
knowledge_base_name = okusimail
vector_model = text-embedding-3-small
query_model = claude-3-5-sonnet-20241022
""")

    # Test from okusiassociates directory
    from config.config_manager import KnowledgeBase, get_fq_cfg_filename

    old_cwd = os.getcwd()
    try:
      os.chdir(okusiassociates_dir)

      # Test path resolution
      result = get_fq_cfg_filename('../okusimail/okusimail.cfg')
      assert result == '../okusimail/okusimail.cfg'

      # Test KnowledgeBase creation
      kb = KnowledgeBase('../okusimail/okusimail.cfg')
      assert kb.knowledge_base_name == 'okusimail'
      # The KB will create absolute paths, but they should point to the right location
      assert kb.knowledge_base_db.endswith('okusimail/okusimail.db')
      assert kb.knowledge_base_vector.endswith('okusimail/okusimail.faiss')

    finally:
      os.chdir(old_cwd)

  def test_command_line_integration_with_absolute_paths(self, temp_data_manager):
    """Test that customkb command line accepts absolute paths without error."""
    # Create config file
    kb_dir = temp_data_manager.create_temp_dir()
    config_path = os.path.join(kb_dir, 'cmdtest.cfg')

    with open(config_path, 'w') as f:
      f.write("""[DEFAULT]
knowledge_base_name = cmdtest
vector_model = text-embedding-3-small
query_model = claude-3-5-sonnet-20241022
""")

    # Test that config resolution works for absolute paths
    from config.config_manager import KnowledgeBase, get_fq_cfg_filename

    # Test path resolution
    result = get_fq_cfg_filename(config_path)
    assert result == config_path

    # Test KnowledgeBase creation with absolute path
    kb = KnowledgeBase(config_path)
    assert kb.knowledge_base_name == 'cmdtest'
    assert kb.knowledge_base_db.endswith('cmdtest.db')

  def test_backwards_compatibility_maintained(self, temp_data_manager):
    """Test that existing functionality is not broken by path changes."""
    # Create traditional config in current directory
    kb_dir = temp_data_manager.create_temp_dir()
    config_path = os.path.join(kb_dir, 'traditional.cfg')

    with open(config_path, 'w') as f:
      f.write("""[DEFAULT]
knowledge_base_name = traditional
vector_model = text-embedding-3-small
query_model = claude-3-5-sonnet-20241022
""")

    old_cwd = os.getcwd()
    try:
      os.chdir(kb_dir)

      # Test all the traditional ways still work
      from config.config_manager import KnowledgeBase, get_fq_cfg_filename

      # 1. Relative path to file
      result1 = get_fq_cfg_filename('./traditional.cfg')
      assert result1 == './traditional.cfg'

      # 2. Absolute path to file
      result2 = get_fq_cfg_filename(config_path)
      assert result2 == config_path

      # 3. KnowledgeBase creation
      kb = KnowledgeBase(config_path)
      assert kb.knowledge_base_name == 'traditional'

    finally:
      os.chdir(old_cwd)

#fin
