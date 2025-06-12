"""
Backward compatibility tests for BM25 integration.
Ensures that existing functionality works with and without BM25 features.
"""

import pytest
import os
import sqlite3
import tempfile
from unittest.mock import patch, Mock

from config.config_manager import KnowledgeBase
from database.db_manager import process_text_file, migrate_for_bm25
from query.query_manager import process_query
from embedding.bm25_manager import ensure_bm25_index


class TestBM25BackwardCompatibility:
  """Test that BM25 changes don't break existing functionality."""
  
  def test_legacy_database_migration(self, temp_legacy_database, temp_kb_directory):
    """Test that legacy databases are properly migrated to support BM25."""
    # Create KnowledgeBase pointing to legacy database
    config_file = os.path.join(temp_kb_directory, "legacy_test.cfg")
    with open(config_file, 'w') as f:
      f.write("""[DEFAULT]
vector_model = text-embedding-3-small
vector_dimensions = 1536

[ALGORITHMS]
enable_hybrid_search = true
""")
    
    # Copy legacy database to expected location
    kb = KnowledgeBase(config_file)
    import shutil
    shutil.copy2(temp_legacy_database, kb.knowledge_base_db)
    
    # Setup KB with legacy database
    kb.sql_connection = sqlite3.connect(kb.knowledge_base_db)
    kb.sql_cursor = kb.sql_connection.cursor()
    
    # Verify legacy schema
    kb.sql_cursor.execute("PRAGMA table_info(docs)")
    columns_before = [col[1] for col in kb.sql_cursor.fetchall()]
    assert 'bm25_tokens' not in columns_before
    assert 'doc_length' not in columns_before
    
    # Test migration
    migrate_for_bm25(kb)
    
    # Verify migration worked
    kb.sql_cursor.execute("PRAGMA table_info(docs)")
    columns_after = [col[1] for col in kb.sql_cursor.fetchall()]
    assert 'bm25_tokens' in columns_after
    assert 'doc_length' in columns_after
    
    # Verify existing data is preserved
    kb.sql_cursor.execute("SELECT COUNT(*) FROM docs")
    count = kb.sql_cursor.fetchone()[0]
    assert count > 0  # Should have data from fixture
    
    # Verify existing data has null BM25 values (not processed yet)
    kb.sql_cursor.execute("SELECT bm25_tokens, doc_length FROM docs LIMIT 1")
    tokens, length = kb.sql_cursor.fetchone()
    assert tokens is None or tokens == ""
    assert length == 0
    
    kb.sql_connection.close()
  
  def test_database_operations_with_bm25_disabled(self, temp_data_manager, sample_texts):
    """Test that database operations work correctly when BM25 is disabled."""
    kb_dir = temp_data_manager.create_temp_dir()
    config_file = os.path.join(kb_dir, 'no_bm25.cfg')
    
    # Create config with BM25 disabled (default behavior)
    with open(config_file, 'w') as f:
      f.write("""[DEFAULT]
vector_model = text-embedding-3-small
vector_dimensions = 1536
db_min_tokens = 50
db_max_tokens = 150
""")
    
    # Create test file
    test_file = os.path.join(kb_dir, 'test.txt')
    with open(test_file, 'w') as f:
      f.write(sample_texts[0])
    
    # Change to kb directory to use relative paths
    old_cwd = os.getcwd()
    os.chdir(kb_dir)
    
    try:
      # Setup KnowledgeBase
      kb = KnowledgeBase(config_file)
      assert kb.enable_hybrid_search is False  # Should be disabled by default
      
      # Create database with new schema (migration should handle this)
      kb.sql_connection = sqlite3.connect(kb.knowledge_base_db)
      kb.sql_cursor = kb.sql_connection.cursor()
      
      # Create table (this would normally be done by process_database)
      kb.sql_cursor.execute('''
        CREATE TABLE docs (
          id INTEGER PRIMARY KEY,
          sid INTEGER,
          sourcedoc VARCHAR(255),
          originaltext TEXT,
          embedtext TEXT,
          embedded INTEGER DEFAULT 0,
          language TEXT DEFAULT "en",
          metadata TEXT,
          keyphrase_processed INTEGER DEFAULT 0,
          bm25_tokens TEXT,
          doc_length INTEGER DEFAULT 0
        )
      ''')
      kb.sql_connection.commit()
      
      # Test text processing without BM25
      from database.db_manager import init_text_splitter
      from utils.text_utils import enhanced_clean_text
      
      splitter = init_text_splitter(kb, 'text')
      stop_words = set(['the', 'a', 'an', 'and', 'or', 'but'])
      
      # Use relative path for file processing
      result = process_text_file(kb, 'test.txt', splitter, stop_words, 'english', 'text', force=False)
      
      assert result is True  # Should successfully process the file
      
      # Verify BM25 columns are not populated when disabled
      kb.sql_cursor.execute("SELECT bm25_tokens, keyphrase_processed FROM docs")
      results = kb.sql_cursor.fetchall()
      
      for tokens, processed in results:
        assert tokens is None or tokens == ""  # BM25 should not be processed
        assert processed == 0  # keyphrase_processed should be 0
      
      kb.sql_connection.close()
    
    finally:
      os.chdir(old_cwd)
  
  def test_database_operations_with_bm25_enabled(self, temp_data_manager, sample_texts):
    """Test that database operations work correctly when BM25 is enabled."""
    kb_dir = temp_data_manager.create_temp_dir()
    config_file = os.path.join(kb_dir, 'with_bm25.cfg')
    
    # Create config with BM25 enabled
    with open(config_file, 'w') as f:
      f.write("""[DEFAULT]
vector_model = text-embedding-3-small
vector_dimensions = 1536
db_min_tokens = 50
db_max_tokens = 150

[ALGORITHMS]
enable_hybrid_search = true
""")
    
    # Create test file
    test_file = os.path.join(kb_dir, 'test.txt')
    with open(test_file, 'w') as f:
      f.write(sample_texts[0])
    
    # Setup KnowledgeBase
    kb = KnowledgeBase(config_file)
    assert kb.enable_hybrid_search is True
    
    # Create database
    kb.sql_connection = sqlite3.connect(kb.knowledge_base_db)
    kb.sql_cursor = kb.sql_connection.cursor()
    
    kb.sql_cursor.execute('''
      CREATE TABLE docs (
        id INTEGER PRIMARY KEY,
        sid INTEGER,
        sourcedoc VARCHAR(255),
        originaltext TEXT,
        embedtext TEXT,
        embedded INTEGER DEFAULT 0,
        language TEXT DEFAULT "en",
        metadata TEXT,
        keyphrase_processed INTEGER DEFAULT 0,
        bm25_tokens TEXT,
        doc_length INTEGER DEFAULT 0
      )
    ''')
    kb.sql_connection.commit()
    
    # Test text processing with BM25
    from database.db_manager import init_text_splitter
    
    splitter = init_text_splitter(kb, 'text')
    stop_words = set(['the', 'a', 'an', 'and', 'or', 'but'])
    
    result = process_text_file(kb, test_file, splitter, stop_words, 'english', 'text', force=False)
    
    assert result is True  # Should successfully process the file
    
    # Verify BM25 columns are populated when enabled
    kb.sql_cursor.execute("SELECT bm25_tokens, keyphrase_processed FROM docs WHERE keyphrase_processed = 1")
    results = kb.sql_cursor.fetchall()
    
    assert len(results) > 0  # Should have processed some chunks
    for tokens, processed in results:
      assert tokens is not None and tokens != ""  # BM25 should be processed
      assert processed == 1  # keyphrase_processed should be 1
    
    kb.sql_connection.close()
  
  def test_configuration_backward_compatibility(self, temp_data_manager):
    """Test that old configuration files work without BM25 settings."""
    kb_dir = temp_data_manager.create_temp_dir()
    
    # Create old-style config without any BM25 settings
    old_config_content = """[DEFAULT]
vector_model = text-embedding-3-small
vector_dimensions = 1536
vector_chunks = 200
db_min_tokens = 100
db_max_tokens = 200
query_model = gpt-4o
query_top_k = 50
query_context_scope = 4
query_temperature = 0.0
query_max_tokens = 4000

[API]
api_call_delay_seconds = 0.05
api_max_retries = 20

[LIMITS]
max_file_size_mb = 100

[PERFORMANCE]
embedding_batch_size = 100

[ALGORITHMS]
similarity_threshold = 0.6
"""
    
    config_file = os.path.join(kb_dir, 'old_config.cfg')
    with open(config_file, 'w') as f:
      f.write(old_config_content)
    
    # Test that KnowledgeBase can load old config
    kb = KnowledgeBase(config_file)
    
    # Verify default BM25 values are set
    assert kb.enable_hybrid_search is False  # Should default to False
    assert kb.vector_weight == 0.7          # Should use default
    assert kb.bm25_k1 == 1.2               # Should use default
    assert kb.bm25_b == 0.75               # Should use default
    assert kb.bm25_min_token_length == 2   # Should use default
    assert kb.bm25_rebuild_threshold == 1000  # Should use default
    
    # Verify existing config values are preserved
    assert kb.vector_model == "text-embedding-3-small"
    assert kb.vector_dimensions == 1536
    assert kb.similarity_threshold == 0.6
  
  def test_query_operations_backward_compatibility(self, temp_database, temp_kb_directory):
    """Test that query operations work with both old and new database schemas."""
    config_file = os.path.join(temp_kb_directory, 'query_compat.cfg')
    
    # Test with BM25 disabled
    with open(config_file, 'w') as f:
      f.write("""[DEFAULT]
vector_model = text-embedding-3-small
vector_dimensions = 1536
query_model = gpt-4o

[ALGORITHMS]
enable_hybrid_search = false
""")
    
    kb = KnowledgeBase(config_file)
    
    # Copy test database to KB location
    import shutil
    shutil.copy2(temp_database, kb.knowledge_base_db)
    
    # Test ensure_bm25_index with disabled hybrid search
    result = ensure_bm25_index(kb)
    assert result is False  # Should return False when disabled
    
    # Test query processing without hybrid search
    mock_logger = Mock()
    query_args = Mock()
    query_args.config_file = config_file
    query_args.query_text = "test query"
    query_args.query_file = ""
    query_args.context_only = True
    query_args.verbose = True
    query_args.debug = False
    
    with patch('query.query_manager.faiss.read_index') as mock_read_index:
      mock_index = Mock()
      mock_index.search.return_value = ([[0.5, 0.7]], [[0, 1]])
      mock_read_index.return_value = mock_index
      
      with patch('query.query_manager.get_query_embedding') as mock_embedding:
        mock_embedding.return_value = [[0.1] * 1536]
        
        # Should work without BM25
        result = process_query(query_args, mock_logger)
        assert isinstance(result, str)
        assert len(result) > 0
  
  def test_bm25_graceful_degradation(self, temp_data_manager):
    """Test that BM25 features degrade gracefully when dependencies are missing."""
    kb_dir = temp_data_manager.create_temp_dir()
    config_file = os.path.join(kb_dir, 'degradation_test.cfg')
    
    with open(config_file, 'w') as f:
      f.write("""[DEFAULT]
vector_model = text-embedding-3-small

[ALGORITHMS]
enable_hybrid_search = true
""")
    
    kb = KnowledgeBase(config_file)
    
    # Test with missing BM25 dependencies
    with patch('embedding.bm25_manager.BM25Okapi', side_effect=ImportError("BM25 not available")):
      from embedding.bm25_manager import build_bm25_index
      
      # Should handle missing dependencies gracefully
      result = build_bm25_index(kb)
      assert result is None  # Should return None, not crash
      
      # ensure_bm25_index should also handle this gracefully
      index_result = ensure_bm25_index(kb)
      assert index_result is False  # Should return False when BM25 unavailable
  
  def test_existing_tests_still_pass(self, temp_config_file, temp_database):
    """Test that existing functionality still works as expected."""
    # Test KnowledgeBase initialization (should work as before)
    kb = KnowledgeBase(temp_config_file)
    
    # Verify core attributes are still available
    assert hasattr(kb, 'vector_model')
    assert hasattr(kb, 'vector_dimensions')
    assert hasattr(kb, 'query_model')
    assert hasattr(kb, 'knowledge_base_db')
    assert hasattr(kb, 'knowledge_base_vector')
    
    # Test database connection (should work as before)
    kb.sql_connection = sqlite3.connect(temp_database)
    kb.sql_cursor = kb.sql_connection.cursor()
    
    # Basic database query should work
    kb.sql_cursor.execute("SELECT COUNT(*) FROM docs")
    count = kb.sql_cursor.fetchone()[0]
    assert count > 0
    
    # Should have BM25 columns now (due to updated fixture)
    kb.sql_cursor.execute("PRAGMA table_info(docs)")
    columns = [col[1] for col in kb.sql_cursor.fetchall()]
    assert 'bm25_tokens' in columns
    assert 'doc_length' in columns
    
    kb.sql_connection.close()
  
  def test_file_processing_backward_compatibility(self, temp_data_manager, sample_texts):
    """Test that file processing works with both old and new configurations."""
    kb_dir = temp_data_manager.create_temp_dir()
    
    # Test scenarios
    test_configs = [
      # No BM25 section (old config)
      """[DEFAULT]
vector_model = text-embedding-3-small
db_min_tokens = 50
db_max_tokens = 100""",
      
      # BM25 disabled explicitly
      """[DEFAULT]
vector_model = text-embedding-3-small
db_min_tokens = 50
db_max_tokens = 100

[ALGORITHMS]
enable_hybrid_search = false""",
      
      # BM25 enabled
      """[DEFAULT]
vector_model = text-embedding-3-small
db_min_tokens = 50
db_max_tokens = 100

[ALGORITHMS]
enable_hybrid_search = true"""
    ]
    
    for i, config_content in enumerate(test_configs):
      config_file = os.path.join(kb_dir, f'test_config_{i}.cfg')
      with open(config_file, 'w') as f:
        f.write(config_content)
      
      test_file = os.path.join(kb_dir, f'test_{i}.txt')
      with open(test_file, 'w') as f:
        f.write(sample_texts[0])
      
      # Test that KnowledgeBase loads correctly
      kb = KnowledgeBase(config_file)
      
      # Create database
      kb.sql_connection = sqlite3.connect(kb.knowledge_base_db)
      kb.sql_cursor = kb.sql_connection.cursor()
      
      # Ensure BM25 columns exist (migration handles this)
      kb.sql_cursor.execute('''
        CREATE TABLE docs (
          id INTEGER PRIMARY KEY,
          sid INTEGER,
          sourcedoc VARCHAR(255),
          originaltext TEXT,
          embedtext TEXT,
          embedded INTEGER DEFAULT 0,
          language TEXT DEFAULT "en",
          metadata TEXT,
          keyphrase_processed INTEGER DEFAULT 0,
          bm25_tokens TEXT,
          doc_length INTEGER DEFAULT 0
        )
      ''')
      kb.sql_connection.commit()
      
      # Test file processing
      from database.db_manager import init_text_splitter
      
      splitter = init_text_splitter(kb, 'text')
      stop_words = set(['the', 'a', 'an', 'and', 'or', 'but'])
      
      result = process_text_file(kb, test_file, splitter, stop_words, 'english', 'text', force=False)
      assert result is True  # Should successfully process the file
      
      # Verify BM25 behavior based on config
      kb.sql_cursor.execute("SELECT keyphrase_processed FROM docs")
      processed_values = [row[0] for row in kb.sql_cursor.fetchall()]
      
      if i < 2:  # BM25 disabled or not configured
        # keyphrase_processed should be 0
        assert all(val == 0 for val in processed_values)
      else:  # BM25 enabled
        # Some should be processed (keyphrase_processed = 1)
        assert any(val == 1 for val in processed_values)
      
      kb.sql_connection.close()


class TestBM25MigrationEdgeCases:
  """Test edge cases in BM25 database migration."""
  
  def test_migration_idempotency(self, temp_legacy_database, temp_kb_directory):
    """Test that running migration multiple times is safe."""
    config_file = os.path.join(temp_kb_directory, "migration_test.cfg")
    with open(config_file, 'w') as f:
      f.write("""[DEFAULT]
vector_model = text-embedding-3-small

[ALGORITHMS]
enable_hybrid_search = true
""")
    
    kb = KnowledgeBase(config_file)
    import shutil
    shutil.copy2(temp_legacy_database, kb.knowledge_base_db)
    
    kb.sql_connection = sqlite3.connect(kb.knowledge_base_db)
    kb.sql_cursor = kb.sql_connection.cursor()
    
    # Run migration first time
    migrate_for_bm25(kb)
    
    # Get column count after first migration
    kb.sql_cursor.execute("PRAGMA table_info(docs)")
    columns_after_first = len(kb.sql_cursor.fetchall())
    
    # Run migration second time (should be safe)
    migrate_for_bm25(kb)
    
    # Get column count after second migration
    kb.sql_cursor.execute("PRAGMA table_info(docs)")
    columns_after_second = len(kb.sql_cursor.fetchall())
    
    # Should be the same (no duplicate columns)
    assert columns_after_first == columns_after_second
    
    # Verify BM25 columns exist
    kb.sql_cursor.execute("PRAGMA table_info(docs)")
    column_names = [col[1] for col in kb.sql_cursor.fetchall()]
    assert 'bm25_tokens' in column_names
    assert 'doc_length' in column_names
    
    kb.sql_connection.close()
  
  def test_migration_preserves_data_integrity(self, temp_legacy_database, temp_kb_directory):
    """Test that migration preserves all existing data."""
    config_file = os.path.join(temp_kb_directory, "integrity_test.cfg")
    with open(config_file, 'w') as f:
      f.write("""[DEFAULT]
vector_model = text-embedding-3-small

[ALGORITHMS]
enable_hybrid_search = true
""")
    
    kb = KnowledgeBase(config_file)
    import shutil
    shutil.copy2(temp_legacy_database, kb.knowledge_base_db)
    
    kb.sql_connection = sqlite3.connect(kb.knowledge_base_db)
    kb.sql_cursor = kb.sql_connection.cursor()
    
    # Get original data
    kb.sql_cursor.execute("SELECT id, sourcedoc, originaltext FROM docs ORDER BY id")
    original_data = kb.sql_cursor.fetchall()
    
    # Run migration
    migrate_for_bm25(kb)
    
    # Get data after migration
    kb.sql_cursor.execute("SELECT id, sourcedoc, originaltext FROM docs ORDER BY id")
    migrated_data = kb.sql_cursor.fetchall()
    
    # Should be identical
    assert original_data == migrated_data
    
    # Verify new columns have default values
    kb.sql_cursor.execute("SELECT bm25_tokens, doc_length FROM docs")
    bm25_data = kb.sql_cursor.fetchall()
    
    for tokens, length in bm25_data:
      assert tokens is None or tokens == ""
      assert length == 0
    
    kb.sql_connection.close()
  
  def test_migration_error_handling(self, temp_kb_directory):
    """Test migration error handling with corrupted database."""
    config_file = os.path.join(temp_kb_directory, "error_test.cfg")
    with open(config_file, 'w') as f:
      f.write("""[DEFAULT]
vector_model = text-embedding-3-small

[ALGORITHMS]
enable_hybrid_search = true
""")
    
    kb = KnowledgeBase(config_file)
    
    # Create corrupted database (table doesn't exist)
    kb.sql_connection = sqlite3.connect(kb.knowledge_base_db)
    kb.sql_cursor = kb.sql_connection.cursor()
    
    # Don't create the docs table
    
    # Migration should handle missing table gracefully
    try:
      migrate_for_bm25(kb)
      # Should not crash, might log error but continue
    except Exception as e:
      # If it does raise an exception, it should be a specific database error
      assert isinstance(e, sqlite3.Error)
    
    kb.sql_connection.close()

#fin