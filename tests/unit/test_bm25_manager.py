#!/usr/bin/env python
"""
Unit tests for BM25 manager functionality.
Tests BM25 index building, loading, and score calculation.
"""

import pytest
import tempfile
import os
import sqlite3
import pickle
from unittest.mock import Mock, patch, MagicMock
from embedding.bm25_manager import (
  build_bm25_index,
  load_bm25_index,
  get_bm25_index_path,
  rebuild_bm25_if_needed,
  get_bm25_scores,
  ensure_bm25_index
)

class TestBM25Manager:
  """Test suite for BM25 manager functions."""
  
  def setup_method(self):
    """Set up test fixtures."""
    # Create temporary files
    self.temp_dir = tempfile.mkdtemp()
    self.db_path = os.path.join(self.temp_dir, 'test.db')
    self.vector_path = os.path.join(self.temp_dir, 'test.faiss')
    self.bm25_path = os.path.join(self.temp_dir, 'test.bm25')
    
    # Create mock KB
    self.mock_kb = Mock()
    self.mock_kb.knowledge_base_vector = self.vector_path
    self.mock_kb.bm25_k1 = 1.2
    self.mock_kb.bm25_b = 0.75
    self.mock_kb.enable_hybrid_search = True
    self.mock_kb.bm25_rebuild_threshold = 100
    self.mock_kb.language = 'en'
    
    # Create test database with BM25 data
    self.create_test_database()
  
  def teardown_method(self):
    """Clean up test fixtures."""
    import shutil
    shutil.rmtree(self.temp_dir, ignore_errors=True)
  
  def create_test_database(self):
    """Create a test database with sample BM25 data."""
    conn = sqlite3.connect(self.db_path)
    cursor = conn.cursor()
    
    # Create table with BM25 columns
    cursor.execute('''
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
    
    # Insert test data
    test_data = [
      (1, 0, 'doc1.txt', 'Machine learning is great', 'machine learning great', 
       1, 'en', '{}', 1, 'machine learning great', 3),
      (2, 1, 'doc1.txt', 'Deep learning algorithms work well', 'deep learning algorithms work well',
       1, 'en', '{}', 1, 'deep learning algorithms work well', 5),
      (3, 0, 'doc2.txt', 'Natural language processing', 'natural language processing',
       1, 'en', '{}', 1, 'natural language processing', 3),
      (4, 1, 'doc2.txt', 'Text classification tasks', 'text classification tasks',
       1, 'en', '{}', 0, '', 0)  # Unprocessed entry
    ]
    
    cursor.executemany('''
      INSERT INTO docs (id, sid, sourcedoc, originaltext, embedtext, embedded, 
                       language, metadata, keyphrase_processed, bm25_tokens, doc_length)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', test_data)
    
    conn.commit()
    conn.close()
    
    # Set up mock cursor
    self.mock_kb.sql_cursor = Mock()
    self.mock_kb.sql_cursor.fetchall.return_value = [
      (1, 'machine learning great', 3),
      (2, 'deep learning algorithms work well', 5),
      (3, 'natural language processing', 3)
    ]
    self.mock_kb.sql_cursor.fetchone.return_value = [1]  # For rebuild check

  def test_get_bm25_index_path(self):
    """Test BM25 index path generation."""
    path = get_bm25_index_path(self.mock_kb)
    expected = self.vector_path.replace('.faiss', '.bm25')
    assert path == expected

  @patch('embedding.bm25_manager.BM25Okapi')
  @patch('builtins.open', create=True)
  @patch('pickle.dump')
  def test_build_bm25_index_success(self, mock_pickle_dump, mock_open, mock_bm25):
    """Test successful BM25 index building."""
    # Mock BM25Okapi
    mock_bm25_instance = Mock()
    mock_bm25.return_value = mock_bm25_instance
    
    # Mock file operations
    mock_file = MagicMock()
    mock_open.return_value.__enter__.return_value = mock_file
    
    # Execute
    result = build_bm25_index(self.mock_kb)
    
    # Verify
    assert result == mock_bm25_instance
    mock_bm25.assert_called_once_with(
      [['machine', 'learning', 'great'],
       ['deep', 'learning', 'algorithms', 'work', 'well'],
       ['natural', 'language', 'processing']],
      k1=1.2, b=0.75
    )
    mock_pickle_dump.assert_called_once()

  def test_build_bm25_index_no_data(self):
    """Test BM25 index building with no valid data."""
    # Mock empty result
    self.mock_kb.sql_cursor.fetchall.return_value = []
    
    result = build_bm25_index(self.mock_kb)
    assert result is None

  def test_build_bm25_index_database_error(self):
    """Test BM25 index building with database error."""
    # Mock database error
    self.mock_kb.sql_cursor.execute.side_effect = sqlite3.Error("DB error")
    
    result = build_bm25_index(self.mock_kb)
    assert result is None

  @patch('builtins.open', create=True)
  @patch('pickle.load')
  @patch('os.path.exists')
  def test_load_bm25_index_success(self, mock_exists, mock_pickle_load, mock_open):
    """Test successful BM25 index loading."""
    # Setup
    mock_exists.return_value = True
    mock_data = {
      'bm25': Mock(),
      'doc_ids': [1, 2, 3],
      'total_docs': 3
    }
    mock_pickle_load.return_value = mock_data
    mock_open.return_value.__enter__.return_value = Mock()
    
    # Execute
    result = load_bm25_index(self.mock_kb)
    
    # Verify
    assert result == mock_data
    mock_pickle_load.assert_called_once()

  @patch('os.path.exists')
  def test_load_bm25_index_file_not_found(self, mock_exists):
    """Test BM25 index loading when file doesn't exist."""
    mock_exists.return_value = False
    
    result = load_bm25_index(self.mock_kb)
    assert result is None

  @patch('builtins.open', create=True)
  @patch('pickle.load')
  @patch('os.path.exists')
  def test_load_bm25_index_invalid_data(self, mock_exists, mock_pickle_load, mock_open):
    """Test BM25 index loading with invalid data."""
    mock_exists.return_value = True
    mock_pickle_load.return_value = {'invalid': 'data'}  # Missing required keys
    mock_open.return_value.__enter__.return_value = Mock()
    
    result = load_bm25_index(self.mock_kb)
    assert result is None

  def test_rebuild_bm25_if_needed_no_rebuild(self):
    """Test rebuild check when no rebuild is needed."""
    # Mock fewer unprocessed documents than threshold
    self.mock_kb.sql_cursor.fetchone.return_value = [50]
    
    result = rebuild_bm25_if_needed(self.mock_kb)
    assert result is True

  @patch('embedding.bm25_manager.build_bm25_index')
  def test_rebuild_bm25_if_needed_rebuild_required(self, mock_build):
    """Test rebuild when threshold is exceeded."""
    # Mock more unprocessed documents than threshold
    self.mock_kb.sql_cursor.fetchone.return_value = [150]
    mock_build.return_value = Mock()  # Successful rebuild
    
    result = rebuild_bm25_if_needed(self.mock_kb)
    assert result is True
    mock_build.assert_called_once()

  @patch('embedding.bm25_manager.build_bm25_index')
  def test_rebuild_bm25_if_needed_rebuild_failed(self, mock_build):
    """Test rebuild when build fails."""
    self.mock_kb.sql_cursor.fetchone.return_value = [150]
    mock_build.return_value = None  # Failed rebuild
    
    result = rebuild_bm25_if_needed(self.mock_kb)
    assert result is False

  @patch('embedding.bm25_manager.tokenize_for_bm25')
  def test_get_bm25_scores_success(self, mock_tokenize):
    """Test successful BM25 score calculation."""
    # Setup
    mock_tokenize.return_value = ('machine learning', 2)
    mock_bm25 = Mock()
    mock_bm25.get_scores.return_value = [0.8, 0.6, 0.3, 0.1]
    
    bm25_data = {
      'bm25': mock_bm25,
      'doc_ids': [1, 2, 3, 4],
      'total_docs': 4
    }
    
    # Execute
    result = get_bm25_scores(self.mock_kb, 'machine learning algorithms', bm25_data)
    
    # Verify
    expected = [(1, 0.8), (2, 0.6), (3, 0.3), (4, 0.1)]
    assert result == expected
    mock_tokenize.assert_called_once_with('machine learning algorithms', 'en')

  @patch('embedding.bm25_manager.tokenize_for_bm25')
  def test_get_bm25_scores_empty_query(self, mock_tokenize):
    """Test BM25 scoring with empty query tokens."""
    mock_tokenize.return_value = ('', 0)
    
    bm25_data = {'bm25': Mock(), 'doc_ids': [1, 2], 'total_docs': 2}
    
    result = get_bm25_scores(self.mock_kb, '', bm25_data)
    assert result == []

  @patch('embedding.bm25_manager.tokenize_for_bm25')
  def test_get_bm25_scores_only_positive_scores(self, mock_tokenize):
    """Test that only positive BM25 scores are returned."""
    mock_tokenize.return_value = ('test query', 2)
    mock_bm25 = Mock()
    mock_bm25.get_scores.return_value = [0.5, 0.0, -0.1, 0.3]
    
    bm25_data = {
      'bm25': mock_bm25,
      'doc_ids': [1, 2, 3, 4],
      'total_docs': 4
    }
    
    result = get_bm25_scores(self.mock_kb, 'test query', bm25_data)
    expected = [(1, 0.5), (4, 0.3)]  # Only positive scores
    assert result == expected

  def test_ensure_bm25_index_disabled(self):
    """Test ensure_bm25_index when hybrid search is disabled."""
    self.mock_kb.enable_hybrid_search = False
    
    result = ensure_bm25_index(self.mock_kb)
    assert result is False

  @patch('embedding.bm25_manager.load_bm25_index')
  @patch('embedding.bm25_manager.rebuild_bm25_if_needed')
  def test_ensure_bm25_index_exists(self, mock_rebuild, mock_load):
    """Test ensure_bm25_index when index exists."""
    mock_load.return_value = {'bm25': Mock()}  # Index exists
    mock_rebuild.return_value = True
    
    result = ensure_bm25_index(self.mock_kb)
    assert result is True
    mock_rebuild.assert_called_once()

  @patch('embedding.bm25_manager.load_bm25_index')
  @patch('embedding.bm25_manager.build_bm25_index')
  def test_ensure_bm25_index_build_new(self, mock_build, mock_load):
    """Test ensure_bm25_index when building new index."""
    mock_load.return_value = None  # No existing index
    mock_build.return_value = Mock()  # Successful build
    
    result = ensure_bm25_index(self.mock_kb)
    assert result is True
    mock_build.assert_called_once()

  @patch('embedding.bm25_manager.load_bm25_index')
  @patch('embedding.bm25_manager.build_bm25_index')
  def test_ensure_bm25_index_build_failed(self, mock_build, mock_load):
    """Test ensure_bm25_index when build fails."""
    mock_load.return_value = None
    mock_build.return_value = None  # Failed build
    
    result = ensure_bm25_index(self.mock_kb)
    assert result is False

class TestBM25Integration:
  """Integration tests for BM25 with real data."""
  
  def setup_method(self):
    """Set up integration test fixtures."""
    self.temp_dir = tempfile.mkdtemp()
    
  def teardown_method(self):
    """Clean up integration test fixtures."""
    import shutil
    shutil.rmtree(self.temp_dir, ignore_errors=True)
  
  @patch('embedding.bm25_manager.BM25Okapi')
  def test_end_to_end_bm25_workflow(self, mock_bm25_class):
    """Test complete BM25 workflow from build to search."""
    # Setup mock BM25
    mock_bm25 = Mock()
    mock_bm25.get_scores.return_value = [0.9, 0.7, 0.3]
    mock_bm25_class.return_value = mock_bm25
    
    # Setup mock KB
    kb = Mock()
    kb.knowledge_base_vector = os.path.join(self.temp_dir, 'test.faiss')
    kb.bm25_k1 = 1.2
    kb.bm25_b = 0.75
    kb.enable_hybrid_search = True
    kb.language = 'en'
    
    # Mock database results
    kb.sql_cursor.fetchall.return_value = [
      (1, 'machine learning algorithms', 3),
      (2, 'deep learning networks', 3),
      (3, 'natural language processing', 3)
    ]
    
    # Build index
    with patch('builtins.open', create=True), \
         patch('pickle.dump'):
      bm25_index = build_bm25_index(kb)
      assert bm25_index is not None
    
    # Load index
    mock_data = {
      'bm25': mock_bm25,
      'doc_ids': [1, 2, 3],
      'total_docs': 3
    }
    
    with patch('os.path.exists', return_value=True), \
         patch('builtins.open', create=True), \
         patch('pickle.load', return_value=mock_data):
      loaded_data = load_bm25_index(kb)
      assert loaded_data == mock_data
    
    # Test scoring
    with patch('embedding.bm25_manager.tokenize_for_bm25', 
               return_value=('machine learning', 2)):
      scores = get_bm25_scores(kb, 'machine learning', loaded_data)
      expected = [(1, 0.9), (2, 0.7), (3, 0.3)]
      assert scores == expected

#fin