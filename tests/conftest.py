"""
Global test configuration and fixtures for CustomKB test suite.
"""

import pytest
import tempfile
import os
import sqlite3
import shutil
import sys
import gc
import psutil
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any, List

# Add project root to Python path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tests.fixtures.mock_data import MockDataGenerator, TestDataManager
from utils.resource_manager import ResourceGuard, cleanup_caches


@pytest.fixture(scope="session", autouse=True)
def session_resource_guard():
  """
  Session-level resource guard to monitor memory usage across all tests.
  """
  # Set conservative memory limit for tests (2GB)
  guard = ResourceGuard(memory_limit_gb=2.0)
  
  # Register cleanup handlers
  guard.register_cleanup("caches", cleanup_caches)
  guard.register_cleanup("gc", lambda: gc.collect())
  
  # Log initial state
  initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
  print(f"\nTest session starting. Initial memory: {initial_memory:.1f}MB")
  
  yield guard
  
  # Final cleanup
  guard.force_cleanup()
  gc.collect()
  
  final_memory = psutil.Process().memory_info().rss / 1024 / 1024
  print(f"\nTest session ending. Final memory: {final_memory:.1f}MB "
        f"(delta: {final_memory - initial_memory:+.1f}MB)")


@pytest.fixture(autouse=True)
def test_cleanup():
  """
  Automatic cleanup after each test to prevent memory accumulation.
  """
  yield
  
  # Force garbage collection after each test
  gc.collect()
  
  # Clear any matplotlib figures
  try:
    import matplotlib.pyplot as plt
    plt.close('all')
  except:
    pass


@pytest.fixture(scope="session")
def mock_env_vars():
  """Set up mock environment variables for testing."""
  with patch.dict(os.environ, {
    'OPENAI_API_KEY': 'sk-test' + 'x' * 40,  # Valid format but fake key
    'ANTHROPIC_API_KEY': 'sk-ant-test' + 'x' * 90,  # Valid format but fake key
    'VECTORDBS': tempfile.mkdtemp(prefix='test_vectordbs_'),
    'NLTK_DATA': '/usr/share/nltk_data'  # Use actual NLTK data path
  }):
    yield


@pytest.fixture
def temp_data_manager():
  """Provide a test data manager for temporary files with enhanced cleanup."""
  manager = TestDataManager()
  try:
    yield manager
  finally:
    # Ensure cleanup happens even if test fails
    manager.cleanup()
    # Extra cleanup for any missed temp files
    gc.collect()


@pytest.fixture
def mock_data_generator():
  """Provide mock data generator."""
  return MockDataGenerator()


@pytest.fixture
def sample_config_content(mock_data_generator):
  """Generate sample configuration content."""
  return mock_data_generator.create_sample_config()


@pytest.fixture
def sample_texts(mock_data_generator):
  """Generate sample text documents."""
  return mock_data_generator.create_sample_texts()


@pytest.fixture
def temp_kb_directory(temp_data_manager):
  """Create a temporary knowledge base directory."""
  kb_dir = temp_data_manager.create_temp_dir(prefix="kb_test_")
  
  # Create logs directory
  logs_dir = os.path.join(kb_dir, 'logs')
  os.makedirs(logs_dir, exist_ok=True)
  
  return kb_dir


@pytest.fixture
def temp_config_file(temp_data_manager, sample_config_content, temp_kb_directory):
  """Create a temporary configuration file."""
  config_path = os.path.join(temp_kb_directory, "test_kb.cfg")
  with open(config_path, 'w') as f:
    f.write(sample_config_content)
  
  temp_data_manager.temp_files.append(config_path)
  return config_path


@pytest.fixture
def temp_database(temp_kb_directory, sample_texts):
  """Create a temporary SQLite database with sample data (includes BM25 columns)."""
  db_path = os.path.join(temp_kb_directory, "test_kb.db")
  
  conn = None
  try:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create table with BM25 columns for backward compatibility
    cursor.execute('''
    CREATE TABLE docs (
      id INTEGER PRIMARY KEY,
      sid INTEGER,
      sourcedoc VARCHAR(255),
      originaltext TEXT,
      embedtext TEXT,
      embedded INTEGER DEFAULT 0,
      language TEXT default "en",
      metadata TEXT,
      keyphrase_processed INTEGER default 0,
      bm25_tokens TEXT,
      doc_length INTEGER DEFAULT 0
    )
    ''')
    
    # Insert sample data
    mock_gen = MockDataGenerator()
    rows = mock_gen.create_database_rows(sample_texts[:5])  # Use first 5 texts
    
    for row in rows:
      # Extend row to include BM25 columns (empty by default)
      # Row format: (id, sid, sourcedoc, originaltext, embedtext, embedded, language, metadata)
      # Need to add: keyphrase_processed, bm25_tokens, doc_length
      extended_row = row + (0, "", 0)  # keyphrase_processed, bm25_tokens, doc_length
      cursor.execute(
        "INSERT INTO docs (id, sid, sourcedoc, originaltext, embedtext, embedded, language, metadata, keyphrase_processed, bm25_tokens, doc_length) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        extended_row
      )
    
    conn.commit()
  finally:
    if conn:
      conn.close()
  
  return db_path


@pytest.fixture
def temp_legacy_database(temp_kb_directory, sample_texts):
  """Create a temporary SQLite database with legacy schema (no BM25 columns)."""
  db_path = os.path.join(temp_kb_directory, "legacy_test_kb.db")
  
  conn = None
  try:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
  
    # Create table with legacy schema
    cursor.execute('''
      CREATE TABLE docs (
        id INTEGER PRIMARY KEY,
        sid INTEGER,
        sourcedoc VARCHAR(255),
        originaltext TEXT,
        embedtext TEXT,
        embedded INTEGER DEFAULT 0,
        language TEXT default "en",
        metadata TEXT,
        keyphrase_processed INTEGER default 0
      )
    ''')
    
    # Insert sample data
    mock_gen = MockDataGenerator()
    rows = mock_gen.create_database_rows(sample_texts[:5])  # Use first 5 texts
    
    for row in rows:
      cursor.execute(
        "INSERT INTO docs (id, sid, sourcedoc, originaltext, embedtext, embedded, language, metadata) VALUES (?,?,?,?,?,?,?,?)",
        row
      )
    
    conn.commit()
  finally:
    if conn:
      conn.close()
  
  return db_path


@pytest.fixture
def mock_openai_client():
  """Mock OpenAI client for testing."""
  with patch('openai.OpenAI') as mock_client, \
       patch('openai.AsyncOpenAI') as mock_async_client:
    
    # Set up mock responses
    mock_embedding_response = Mock()
    mock_embedding_response.data = [Mock(embedding=[0.1] * 1536)]
    
    mock_chat_response = Mock()
    mock_chat_response.choices = [Mock(message=Mock(content="Test response"))]
    
    # Configure sync client
    mock_client_instance = Mock()
    mock_client_instance.embeddings.create.return_value = mock_embedding_response
    mock_client_instance.chat.completions.create.return_value = mock_chat_response
    mock_client.return_value = mock_client_instance
    
    # Configure async client  
    mock_async_client_instance = Mock()
    mock_async_client_instance.embeddings.create = AsyncMock(return_value=mock_embedding_response)
    mock_async_client_instance.chat.completions.create = AsyncMock(return_value=mock_chat_response)
    mock_async_client.return_value = mock_async_client_instance
    
    yield {
      'sync': mock_client_instance,
      'async': mock_async_client_instance
    }


@pytest.fixture
def mock_anthropic_client():
  """Mock Anthropic client for testing."""
  with patch('anthropic.Anthropic') as mock_client, \
       patch('anthropic.AsyncAnthropic') as mock_async_client:
    
    # Set up mock response
    mock_message = Mock()
    mock_message.content = [Mock(text="Test Anthropic response")]
    
    # Configure sync client
    mock_client_instance = Mock()
    mock_client_instance.messages.create.return_value = mock_message
    mock_client.return_value = mock_client_instance
    
    # Configure async client
    mock_async_client_instance = Mock()
    mock_async_client_instance.messages.create = AsyncMock(return_value=mock_message)
    mock_async_client.return_value = mock_async_client_instance
    
    yield {
      'sync': mock_client_instance,
      'async': mock_async_client_instance
    }


@pytest.fixture
def mock_faiss_index():
  """Mock FAISS index for testing."""
  with patch('faiss.IndexFlatIP') as mock_index_class, \
       patch('faiss.IndexIDMap') as mock_id_map, \
       patch('faiss.read_index') as mock_read, \
       patch('faiss.write_index') as mock_write:
    
    # Create mock index
    mock_index = Mock()
    mock_index.ntotal = 5
    mock_index.search.return_value = (
      [[0.9, 0.8, 0.7, 0.6, 0.5]],  # distances
      [[0, 1, 2, 3, 4]]               # indices
    )
    mock_index.add_with_ids = Mock()
    
    mock_index_class.return_value = mock_index
    mock_id_map.return_value = mock_index
    mock_read.return_value = mock_index
    
    yield mock_index


@pytest.fixture
def mock_nltk_data():
  """Mock NLTK data loading."""
  with patch('nltk.data.find'), \
       patch('nltk.download'):
    
    # Mock stopwords module at import time
    with patch.dict('sys.modules', {'nltk.corpus.stopwords': Mock()}):
      mock_stopwords_module = Mock()
      mock_stopwords_module.words.return_value = ['the', 'a', 'an', 'and', 'or', 'but']
      
      with patch('nltk.corpus.stopwords', mock_stopwords_module):
        yield


@pytest.fixture
def mock_spacy():
  """Mock spaCy NLP model."""
  with patch('spacy.load') as mock_load:
    mock_nlp = Mock()
    mock_doc = Mock()
    mock_doc.ents = []
    mock_nlp.return_value = mock_doc
    mock_load.return_value = mock_nlp
    yield mock_nlp


@pytest.fixture
def isolated_imports():
  """Ensure clean import state for each test."""
  # Store original sys.modules state
  original_modules = sys.modules.copy()
  
  yield
  
  # Remove any newly imported modules related to our project
  modules_to_remove = [
    name for name in sys.modules 
    if name.startswith(('config.', 'database.', 'embedding.', 'query.', 'models.', 'utils.'))
  ]
  
  for module_name in modules_to_remove:
    if module_name in sys.modules:
      del sys.modules[module_name]


@pytest.fixture(autouse=True)
def setup_test_environment(mock_env_vars):
  """Automatically set up test environment for all tests."""
  pass


# Test markers and configuration
def pytest_configure(config):
  """Configure pytest with custom markers."""
  config.addinivalue_line(
    "markers", "unit: mark test as a unit test"
  )
  config.addinivalue_line(
    "markers", "integration: mark test as an integration test"
  )
  config.addinivalue_line(
    "markers", "slow: mark test as slow (takes >5 seconds)"
  )
  config.addinivalue_line(
    "markers", "requires_api: mark test as requiring real API keys"
  )
  config.addinivalue_line(
    "markers", "requires_data: mark test as requiring external test data"
  )
  config.addinivalue_line(
    "markers", "performance: mark test as a performance test"
  )


def pytest_collection_modifyitems(config, items):
  """Modify test collection to add markers based on path."""
  for item in items:
    # Add markers based on test path
    if "unit" in item.nodeid:
      item.add_marker(pytest.mark.unit)
    elif "integration" in item.nodeid:
      item.add_marker(pytest.mark.integration)
    elif "performance" in item.nodeid:
      item.add_marker(pytest.mark.slow)
      item.add_marker(pytest.mark.performance)

#fin