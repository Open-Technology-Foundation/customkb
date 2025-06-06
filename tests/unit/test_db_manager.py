"""
Unit tests for db_manager.py
Tests database operations, text processing, chunking, and metadata extraction.
"""

import pytest
import os
import sqlite3
import tempfile
import json
from unittest.mock import patch, Mock, MagicMock
from pathlib import Path

from database.db_manager import (
  detect_file_type,
  init_text_splitter,
  extract_metadata,
  connect_to_database,
  close_database,
  process_text_file,
  process_database
)
from config.config_manager import KnowledgeBase


class TestDetectFileType:
  """Test file type detection functionality."""
  
  def test_markdown_files(self):
    """Test detection of markdown files."""
    assert detect_file_type("document.md") == "markdown"
    assert detect_file_type("README.markdown") == "markdown"
    assert detect_file_type("/path/to/file.MD") == "markdown"
  
  def test_code_files(self):
    """Test detection of code files."""
    code_extensions = ['.py', '.js', '.java', '.c', '.cpp', '.go', '.rs', '.php', '.rb', '.ts', '.swift']
    for ext in code_extensions:
      assert detect_file_type(f"test{ext}") == "code"
  
  def test_html_files(self):
    """Test detection of HTML files."""
    assert detect_file_type("page.html") == "html"
    assert detect_file_type("page.htm") == "html"
    assert detect_file_type("data.xml") == "html"
  
  def test_text_files_default(self):
    """Test that unknown files default to text."""
    assert detect_file_type("document.txt") == "text"
    assert detect_file_type("unknown.xyz") == "text"
    assert detect_file_type("no_extension") == "text"


class TestInitTextSplitter:
  """Test text splitter initialization."""
  
  def test_markdown_splitter(self, temp_config_file):
    """Test markdown text splitter initialization."""
    kb = KnowledgeBase(temp_config_file)
    splitter = init_text_splitter(kb, 'markdown')
    
    # Should return MarkdownTextSplitter
    assert hasattr(splitter, 'chunk_size')
    assert splitter.chunk_size == kb.db_max_tokens
  
  def test_code_splitter(self, temp_config_file):
    """Test code text splitter initialization."""
    kb = KnowledgeBase(temp_config_file)
    splitter = init_text_splitter(kb, 'code')
    
    # Should return RecursiveCharacterTextSplitter for code
    assert hasattr(splitter, 'chunk_size')
    assert splitter.chunk_size == kb.db_max_tokens
  
  def test_html_splitter(self, temp_config_file):
    """Test HTML text splitter initialization."""
    kb = KnowledgeBase(temp_config_file)
    splitter = init_text_splitter(kb, 'html')
    
    # Should return a callable function for HTML processing
    assert callable(splitter)
  
  def test_default_text_splitter(self, temp_config_file):
    """Test default text splitter initialization."""
    kb = KnowledgeBase(temp_config_file)
    splitter = init_text_splitter(kb, 'text')
    
    # Should return RecursiveCharacterTextSplitter
    assert hasattr(splitter, 'chunk_size')
    assert splitter.chunk_size == kb.db_max_tokens
  
  def test_chunk_overlap_calculation(self, temp_config_file):
    """Test that chunk overlap is calculated correctly."""
    kb = KnowledgeBase(temp_config_file)
    splitter = init_text_splitter(kb, 'text')
    
    expected_overlap = min(100, kb.db_min_tokens // 2)
    assert splitter.chunk_overlap == expected_overlap


class TestExtractMetadata:
  """Test metadata extraction functionality."""
  
  def test_basic_metadata(self):
    """Test extraction of basic metadata."""
    text = "This is a sample text for testing metadata extraction."
    metadata = extract_metadata(text, "/path/to/test.txt")
    
    assert metadata["source"] == "/path/to/test.txt"
    assert metadata["char_length"] == len(text)
    assert metadata["word_count"] == len(text.split())
    assert metadata["file_type"] == "txt"
  
  def test_heading_extraction(self):
    """Test extraction of headings from text."""
    markdown_text = "# Main Heading\nThis is some content under the heading."
    metadata = extract_metadata(markdown_text, "test.md")
    
    assert metadata.get("heading") == "Main Heading"
    assert metadata.get("section_type") == "heading"
  
  def test_code_block_detection(self):
    """Test detection of code blocks."""
    code_text = "Here's some code:\n```python\nprint('hello')\n```"
    metadata = extract_metadata(code_text, "test.md")
    
    assert metadata.get("section_type") == "code_block"
  
  def test_list_detection(self):
    """Test detection of different list types."""
    bullet_text = "Items:\n- First item\n- Second item"
    metadata = extract_metadata(bullet_text, "test.md")
    assert metadata.get("section_type") == "bullet_list"
    
    numbered_text = "Steps:\n1. First step\n2. Second step"
    metadata = extract_metadata(numbered_text, "test.md")
    assert metadata.get("section_type") == "numbered_list"
  
  def test_document_section_detection(self):
    """Test detection of document sections."""
    intro_text = "Introduction\nThis document provides an overview of the system."
    metadata = extract_metadata(intro_text, "test.md")
    
    assert metadata.get("document_section") == "introduction"
  
  @patch('database.db_manager.nlp')
  def test_entity_extraction_with_spacy(self, mock_nlp):
    """Test named entity extraction when spaCy is available."""
    # Mock spaCy NLP pipeline
    mock_entity = Mock()
    mock_entity.text = "OpenAI"
    mock_entity.label_ = "ORG"
    
    mock_doc = Mock()
    mock_doc.ents = [mock_entity]
    mock_nlp.return_value = mock_doc
    
    text = "OpenAI is a company working on AI."
    metadata = extract_metadata(text, "test.txt")
    
    assert "entities" in metadata
    assert "ORG" in metadata["entities"]
    assert "OpenAI" in metadata["entities"]["ORG"]
  
  @patch('database.db_manager.nlp', None)
  def test_no_entity_extraction_without_spacy(self):
    """Test that entity extraction is skipped when spaCy is not available."""
    text = "OpenAI is a company working on AI."
    metadata = extract_metadata(text, "test.txt")
    
    assert "entities" not in metadata
  
  def test_table_detection(self):
    """Test detection of HTML tables."""
    table_text = "<table><tr><td>Cell 1</td><td>Cell 2</td></tr></table>"
    metadata = extract_metadata(table_text, "test.html")
    
    assert metadata.get("section_type") == "table"
  
  def test_metadata_with_no_extension(self):
    """Test metadata extraction for files without extensions."""
    text = "Sample text content"
    metadata = extract_metadata(text, "/path/to/filename")
    
    assert "file_type" not in metadata


class TestDatabaseOperations:
  """Test database connection and operations."""
  
  def test_connect_to_database_new(self, temp_config_file, temp_kb_directory):
    """Test connecting to a new database."""
    kb = KnowledgeBase(temp_config_file)
    
    # Mock user input to create database
    with patch('builtins.input', return_value='y'):
      connect_to_database(kb)
    
    assert kb.sql_connection is not None
    assert kb.sql_cursor is not None
    assert os.path.exists(kb.knowledge_base_db)
    
    # Verify table structure
    kb.sql_cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = kb.sql_cursor.fetchall()
    assert ('docs',) in tables
    
    close_database(kb)
  
  def test_connect_to_existing_database(self, temp_database, temp_config_file):
    """Test connecting to existing database."""
    kb = KnowledgeBase(temp_config_file)
    kb.knowledge_base_db = temp_database
    
    connect_to_database(kb)
    
    assert kb.sql_connection is not None
    assert kb.sql_cursor is not None
    
    # Verify data exists
    kb.sql_cursor.execute("SELECT COUNT(*) FROM docs")
    count = kb.sql_cursor.fetchone()[0]
    assert count > 0
    
    close_database(kb)
  
  def test_connect_database_user_refuses_creation(self, temp_config_file):
    """Test behavior when user refuses database creation."""
    kb = KnowledgeBase(temp_config_file)
    
    with patch('builtins.input', return_value='n'):
      with pytest.raises(Exception, match="does not exist. Process aborted"):
        connect_to_database(kb)
  
  def test_close_database(self, temp_database, temp_config_file):
    """Test database closing functionality."""
    kb = KnowledgeBase(temp_config_file)
    kb.knowledge_base_db = temp_database
    
    connect_to_database(kb)
    assert kb.sql_connection is not None
    
    close_database(kb)
    assert kb.sql_connection is None
    assert kb.sql_cursor is None
  
  def test_database_indexes_created(self, temp_config_file, temp_kb_directory):
    """Test that necessary indexes are created."""
    kb = KnowledgeBase(temp_config_file)
    
    with patch('builtins.input', return_value='y'):
      connect_to_database(kb)
    
    # Check that indexes exist
    kb.sql_cursor.execute("SELECT name FROM sqlite_master WHERE type='index'")
    indexes = [row[0] for row in kb.sql_cursor.fetchall()]
    
    expected_indexes = ['idx_sourcedoc', 'idx_embedded', 'idx_sourcedoc_sid']
    for expected in expected_indexes:
      assert expected in indexes
    
    close_database(kb)


class TestProcessTextFile:
  """Test text file processing functionality."""
  
  def test_process_new_text_file(self, temp_config_file, temp_kb_directory, sample_texts):
    """Test processing a new text file."""
    kb = KnowledgeBase(temp_config_file)
    
    # Set up database
    with patch('builtins.input', return_value='y'):
      connect_to_database(kb)
    
    # Create test file
    test_file = os.path.join(temp_kb_directory, "test.txt")
    with open(test_file, 'w') as f:
      f.write(sample_texts[0])
    
    # Mock text splitter
    mock_splitter = Mock()
    mock_splitter.split_text.return_value = [sample_texts[0]]
    
    # Mock stopwords
    stop_words = set(['the', 'a', 'an'])
    
    result = process_text_file(kb, test_file, mock_splitter, stop_words, 'english', 'text')
    
    assert result is True
    
    # Verify data was inserted
    kb.sql_cursor.execute("SELECT COUNT(*) FROM docs WHERE sourcedoc = 'test.txt'")
    count = kb.sql_cursor.fetchone()[0]
    assert count == 1
    
    close_database(kb)
  
  def test_process_existing_file_without_force(self, temp_database, temp_config_file):
    """Test processing a file that already exists without force flag."""
    kb = KnowledgeBase(temp_config_file)
    kb.knowledge_base_db = temp_database
    connect_to_database(kb)
    
    # Try to process existing file (should skip)
    mock_splitter = Mock()
    stop_words = set()
    
    result = process_text_file(kb, "existing.txt", mock_splitter, stop_words, 'english', 'text')
    
    assert result is False  # Should be skipped
    
    close_database(kb)
  
  def test_process_existing_file_with_force(self, temp_database, temp_config_file, sample_texts):
    """Test processing existing file with force flag."""
    kb = KnowledgeBase(temp_config_file)
    kb.knowledge_base_db = temp_database
    connect_to_database(kb)
    
    # Create test file
    test_file = os.path.join(os.path.dirname(temp_database), "force_test.txt")
    with open(test_file, 'w') as f:
      f.write(sample_texts[0])
    
    # Insert existing record
    kb.sql_cursor.execute(
      "INSERT INTO docs (sid, sourcedoc, originaltext, embedtext) VALUES (?, ?, ?, ?)",
      (0, "force_test.txt", "old content", "old content")
    )
    kb.sql_connection.commit()
    
    # Mock splitter
    mock_splitter = Mock()
    mock_splitter.split_text.return_value = [sample_texts[0]]
    
    stop_words = set()
    
    result = process_text_file(kb, test_file, mock_splitter, stop_words, 'english', 'text', force=True)
    
    assert result is True
    
    # Verify old record was replaced
    kb.sql_cursor.execute("SELECT originaltext FROM docs WHERE sourcedoc = 'force_test.txt'")
    content = kb.sql_cursor.fetchone()[0]
    assert content == sample_texts[0]
    
    close_database(kb)
  
  def test_invalid_file_path_handling(self, temp_config_file, temp_kb_directory):
    """Test handling of invalid file paths."""
    kb = KnowledgeBase(temp_config_file)
    
    with patch('builtins.input', return_value='y'):
      connect_to_database(kb)
    
    # Test with dangerous path
    mock_splitter = Mock()
    stop_words = set()
    
    with patch('database.db_manager.validate_file_path') as mock_validate:
      mock_validate.side_effect = ValueError("Invalid path")
      
      result = process_text_file(kb, "../../../etc/passwd", mock_splitter, stop_words, 'english', 'text')
      
      assert result is False
    
    close_database(kb)
  
  def test_large_file_rejection(self, temp_config_file, temp_kb_directory):
    """Test rejection of files that are too large."""
    kb = KnowledgeBase(temp_config_file)
    
    with patch('builtins.input', return_value='y'):
      connect_to_database(kb)
    
    # Create large test file
    large_file = os.path.join(temp_kb_directory, "large.txt")
    with open(large_file, 'w') as f:
      f.write("x" * 1000)  # Small file for testing
    
    # Mock file size to be too large
    with patch('os.path.getsize', return_value=200 * 1024 * 1024):  # 200MB
      mock_splitter = Mock()
      stop_words = set()
      
      result = process_text_file(kb, large_file, mock_splitter, stop_words, 'english', 'text')
      
      assert result is False
    
    close_database(kb)
  
  def test_metadata_storage(self, temp_config_file, temp_kb_directory, sample_texts):
    """Test that metadata is properly stored in database."""
    kb = KnowledgeBase(temp_config_file)
    
    with patch('builtins.input', return_value='y'):
      connect_to_database(kb)
    
    # Create test file
    test_file = os.path.join(temp_kb_directory, "metadata_test.md")
    test_content = "# Test Heading\nThis is test content."
    with open(test_file, 'w') as f:
      f.write(test_content)
    
    # Mock splitter
    mock_splitter = Mock()
    mock_splitter.split_text.return_value = [test_content]
    
    stop_words = set()
    
    result = process_text_file(kb, test_file, mock_splitter, stop_words, 'english', 'markdown')
    
    assert result is True
    
    # Verify metadata was stored
    kb.sql_cursor.execute("SELECT metadata FROM docs WHERE sourcedoc = 'metadata_test.md'")
    metadata_str = kb.sql_cursor.fetchone()[0]
    metadata = json.loads(metadata_str)
    
    assert metadata["char_length"] > 0
    assert metadata["word_count"] > 0
    assert metadata["source"] == test_file
    
    close_database(kb)


class TestProcessDatabase:
  """Test the main process_database function."""
  
  def test_process_database_success(self, temp_config_file, temp_kb_directory, sample_texts):
    """Test successful database processing."""
    # Create test files
    test_files = []
    for i, text in enumerate(sample_texts[:3]):
      file_path = os.path.join(temp_kb_directory, f"test_{i}.txt")
      with open(file_path, 'w') as f:
        f.write(text)
      test_files.append(file_path)
    
    # Mock command line arguments
    args = Mock()
    args.config_file = temp_config_file
    args.files = test_files
    args.language = 'english'
    args.force = False
    args.verbose = True
    args.debug = False
    
    # Mock logger
    mock_logger = Mock()
    
    with patch('builtins.input', return_value='y'):
      result = process_database(args, mock_logger)
    
    assert "3 files added to database" in result
  
  def test_process_database_no_files(self, temp_config_file):
    """Test processing with no input files."""
    args = Mock()
    args.config_file = temp_config_file
    args.files = []
    args.language = 'english'
    args.force = False
    args.verbose = True
    args.debug = False
    
    mock_logger = Mock()
    
    result = process_database(args, mock_logger)
    
    assert "No input files provided" in result
  
  def test_process_database_invalid_config(self):
    """Test processing with invalid config file."""
    args = Mock()
    args.config_file = "nonexistent.cfg"
    args.files = ["test.txt"]
    args.language = 'english'
    
    mock_logger = Mock()
    
    result = process_database(args, mock_logger)
    
    assert "Configuration file not found" in result
  
  def test_process_database_with_force(self, temp_database, temp_config_file, temp_kb_directory, sample_texts):
    """Test processing with force flag to reprocess existing files."""
    kb = KnowledgeBase(temp_config_file)
    kb.knowledge_base_db = temp_database
    
    # Create test file that "exists" in database
    test_file = os.path.join(temp_kb_directory, "force_test.txt")
    with open(test_file, 'w') as f:
      f.write(sample_texts[0])
    
    args = Mock()
    args.config_file = temp_config_file
    args.files = [test_file]
    args.language = 'english'
    args.force = True
    args.verbose = True
    args.debug = False
    
    mock_logger = Mock()
    
    result = process_database(args, mock_logger)
    
    assert "files processed" in result
  
  @patch('database.db_manager.get_files')
  def test_glob_pattern_expansion(self, mock_get_files, temp_config_file):
    """Test that glob patterns are properly expanded."""
    mock_get_files.return_value = ["file1.txt", "file2.txt"]
    
    args = Mock()
    args.config_file = temp_config_file
    args.files = ["*.txt"]
    args.language = 'english'
    args.force = False
    args.verbose = True
    args.debug = False
    
    mock_logger = Mock()
    
    with patch('builtins.input', return_value='y'):
      process_database(args, mock_logger)
    
    mock_get_files.assert_called_with("*.txt")

#fin