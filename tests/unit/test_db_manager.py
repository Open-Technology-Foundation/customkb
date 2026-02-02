"""
Unit tests for db_manager.py
Tests database operations, text processing, chunking, and metadata extraction.
"""

import json
import os
import tempfile
from unittest.mock import Mock, patch

import pytest

from config.config_manager import KnowledgeBase
from database.chunking import detect_file_type, init_text_splitter
from database.connection import close_database, connect_to_database
from database.db_manager import extract_metadata, process_database, process_text_file

# Initialize logger for tests
from utils.logging_config import get_logger

logger = get_logger(__name__)


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
    """Test detection of HTML and XML files."""
    assert detect_file_type("page.html") == "html"
    assert detect_file_type("page.htm") == "html"
    # XML files are now properly detected as "xml" not "html"
    assert detect_file_type("data.xml") == "xml"

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
    assert hasattr(splitter, '_chunk_size')
    assert splitter._chunk_size == kb.db_max_tokens

  def test_code_splitter(self, temp_config_file):
    """Test code text splitter initialization."""
    kb = KnowledgeBase(temp_config_file)
    splitter = init_text_splitter(kb, 'code')

    # Should return RecursiveCharacterTextSplitter for code
    assert hasattr(splitter, '_chunk_size')
    assert splitter._chunk_size == kb.db_max_tokens

  def test_html_splitter(self, temp_config_file):
    """Test HTML text splitter initialization."""
    kb = KnowledgeBase(temp_config_file)
    splitter = init_text_splitter(kb, 'html')

    # Should return RecursiveCharacterTextSplitter
    assert hasattr(splitter, '_chunk_size')
    assert splitter._chunk_size == kb.db_max_tokens

  def test_default_text_splitter(self, temp_config_file):
    """Test default text splitter initialization."""
    kb = KnowledgeBase(temp_config_file)
    splitter = init_text_splitter(kb, 'text')

    # Should return RecursiveCharacterTextSplitter
    assert hasattr(splitter, '_chunk_size')
    assert splitter._chunk_size == kb.db_max_tokens

  def test_chunk_overlap_calculation(self, temp_config_file):
    """Test that chunk overlap is calculated correctly."""
    kb = KnowledgeBase(temp_config_file)
    splitter = init_text_splitter(kb, 'text')

    expected_overlap = min(100, kb.db_min_tokens // 2)
    assert splitter._chunk_overlap == expected_overlap


class TestExtractMetadata:
  """Test metadata extraction functionality."""

  def test_basic_metadata(self, mock_kb):
    """Test extraction of basic metadata."""
    text = "This is a sample text for testing metadata extraction."
    metadata = extract_metadata(text, "/path/to/test.txt", mock_kb)

    assert metadata["source"] == "/path/to/test.txt"
    assert metadata["char_length"] == len(text)
    assert metadata["word_count"] == len(text.split())
    assert metadata["file_type"] == "txt"

  def test_heading_extraction(self, mock_kb):
    """Test extraction of headings from text."""
    markdown_text = "# Main Heading\nThis is some content under the heading."
    metadata = extract_metadata(markdown_text, "test.md", mock_kb)

    assert metadata.get("heading") == "Main Heading"
    assert metadata.get("section_type") == "heading"

  def test_code_block_detection(self, mock_kb):
    """Test detection of code blocks."""
    code_text = "Here's some code:\n```python\nprint('hello')\n```"
    metadata = extract_metadata(code_text, "test.md", mock_kb)

    assert metadata.get("section_type") == "code_block"

  def test_list_detection(self, mock_kb):
    """Test detection of different list types."""
    bullet_text = "Items:\n- First item\n- Second item"
    metadata = extract_metadata(bullet_text, "test.md", mock_kb)
    assert metadata.get("section_type") == "bullet_list"

    numbered_text = "Steps:\n1. First step\n2. Second step"
    metadata = extract_metadata(numbered_text, "test.md", mock_kb)
    assert metadata.get("section_type") == "numbered_list"

  def test_document_section_detection(self, mock_kb):
    """Test detection of document sections."""
    intro_text = "Introduction\nThis document provides an overview of the system."
    metadata = extract_metadata(intro_text, "test.md", mock_kb)

    assert metadata.get("document_section") == "introduction"

  @patch('database.db_manager.logger')
  @patch('database.db_manager.nlp')
  def test_entity_extraction_with_spacy(self, mock_nlp, mock_logger, mock_kb):
    """Test named entity extraction when spaCy is available."""
    # Mock spaCy NLP pipeline
    mock_entity = Mock()
    mock_entity.text = "OpenAI"
    mock_entity.label_ = "ORG"

    mock_doc = Mock()
    mock_doc.ents = [mock_entity]
    mock_nlp.return_value = mock_doc

    text = "OpenAI is a company working on AI."
    metadata = extract_metadata(text, "test.txt", mock_kb)

    assert "entities" in metadata
    assert "ORG" in metadata["entities"]
    assert "OpenAI" in metadata["entities"]["ORG"]

  @patch('database.db_manager.nlp', None)
  def test_no_entity_extraction_without_spacy(self, mock_kb):
    """Test that entity extraction is skipped when spaCy is not available."""
    text = "OpenAI is a company working on AI."
    metadata = extract_metadata(text, "test.txt", mock_kb)

    assert "entities" not in metadata

  def test_table_detection(self):
    """Test detection of HTML tables."""
    table_text = "<table><tr><td>Cell 1</td><td>Cell 2</td></tr></table>"
    mock_kb = Mock()
    mock_kb.heading_search_limit = 200
    mock_kb.entity_extraction_limit = 500
    metadata = extract_metadata(table_text, "test.html", mock_kb)

    assert metadata.get("section_type") == "table"

  def test_metadata_with_no_extension(self):
    """Test metadata extraction for files without extensions."""
    text = "Sample text content"
    mock_kb = Mock()
    mock_kb.heading_search_limit = 200
    mock_kb.entity_extraction_limit = 500
    metadata = extract_metadata(text, "/path/to/filename", mock_kb)

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

  @pytest.mark.skip(reason="User confirmation removed - SQLite auto-creates databases, connect_to_database no longer prompts")
  def test_connect_database_user_refuses_creation(self, temp_config_file):
    """Test behavior when user refuses database creation.

    OBSOLETE: The new connect_to_database implementation (in database/connection.py)
    doesn't prompt users - SQLite auto-creates databases on connect. This test
    checked old behavior from the legacy db_manager implementation.
    """
    pass

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
    stop_words = {'the', 'a', 'an'}

    result = process_text_file(kb, test_file, mock_splitter, stop_words, 'english', 'text')

    assert result is True

    # Verify data was inserted (sourcedoc now stores full absolute path)
    expected_path = os.path.abspath(test_file)
    kb.sql_cursor.execute("SELECT COUNT(*) FROM docs WHERE sourcedoc = ?", [expected_path])
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

    # Insert existing record with full path
    expected_path = os.path.abspath(test_file)
    kb.sql_cursor.execute(
      "INSERT INTO docs (sid, sourcedoc, originaltext, embedtext) VALUES (?, ?, ?, ?)",
      (0, expected_path, "old content", "old content")
    )
    kb.sql_connection.commit()

    # Mock splitter
    mock_splitter = Mock()
    mock_splitter.split_text.return_value = [sample_texts[0]]

    stop_words = set()

    result = process_text_file(kb, test_file, mock_splitter, stop_words, 'english', 'text', force=True)

    assert result is True

    # Verify old record was replaced
    kb.sql_cursor.execute("SELECT originaltext FROM docs WHERE sourcedoc = ?", [expected_path])
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

    with patch('utils.security_utils.validate_file_path') as mock_validate:
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
    expected_path = os.path.abspath(test_file)
    kb.sql_cursor.execute("SELECT metadata FROM docs WHERE sourcedoc = ?", [expected_path])
    metadata_str = kb.sql_cursor.fetchone()[0]
    metadata = json.loads(metadata_str)

    assert metadata["char_length"] > 0
    assert metadata["word_count"] > 0
    assert metadata["source"] == test_file

    close_database(kb)


class TestProcessDatabase:
  """Test the main process_database function."""

  def test_process_database_success(self, temp_config_file, sample_texts):
    """Test successful database processing."""
    # Create test files in independent temp directory (not tied to KB structure)
    # Input files can be anywhere - only the KB config resolution needs VECTORDBS
    with tempfile.TemporaryDirectory() as files_dir:
      test_files = []
      for i, text in enumerate(sample_texts[:3]):
        file_path = os.path.join(files_dir, f"test_{i}.txt")
        with open(file_path, 'w') as f:
          f.write(text)
        test_files.append(file_path)

      # Mock command line arguments
      args = Mock()
      args.config_file = temp_config_file
      args.files = test_files
      args.language = 'en'
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
    args.language = 'en'
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
    args.language = 'en'

    mock_logger = Mock()

    result = process_database(args, mock_logger)

    # Error message format: "Knowledgebase 'name' not found" (new) or
    # "Configuration file not found" (old style - logged but not returned)
    assert "not found" in result.lower()

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
    args.language = 'en'
    args.force = True
    args.verbose = True
    args.debug = False

    mock_logger = Mock()

    result = process_database(args, mock_logger)

    assert "files added to database" in result or "files processed" in result

  @patch('database.db_manager.get_files')
  def test_glob_pattern_expansion(self, mock_get_files, temp_config_file):
    """Test that glob patterns are properly expanded."""
    mock_get_files.return_value = ["file1.txt", "file2.txt"]

    args = Mock()
    args.config_file = temp_config_file
    args.files = ["*.txt"]
    args.language = 'en'
    args.force = False
    args.verbose = True
    args.debug = False

    mock_logger = Mock()

    with patch('builtins.input', return_value='y'):
      process_database(args, mock_logger)

    mock_get_files.assert_called_with("*.txt")


class TestFullPathStorage:
  """Test full path storage functionality."""

  @patch('nltk.tokenize.word_tokenize')
  @patch('nltk.tokenize.sent_tokenize')
  @patch('nltk.corpus.stopwords.words')
  def test_full_path_stored_in_sourcedoc(self, mock_stopwords, mock_sent_tokenize, mock_word_tokenize, temp_config_file, temp_kb_directory, sample_texts):
    """Test that full canonical paths are stored in sourcedoc field."""
    # Mock NLTK functions
    mock_stopwords.return_value = ['the', 'a', 'an', 'and', 'or']
    mock_sent_tokenize.return_value = [sample_texts[0]]
    mock_word_tokenize.return_value = sample_texts[0].split()

    kb = KnowledgeBase(temp_config_file)
    with patch('builtins.input', return_value='y'):
      connect_to_database(kb)

    # Create test file
    test_file = os.path.join(temp_kb_directory, "subdir", "test_file.txt")
    os.makedirs(os.path.dirname(test_file), exist_ok=True)
    with open(test_file, 'w') as f:
      f.write(sample_texts[0])

    # Mock splitter
    mock_splitter = Mock()
    mock_splitter.split_text.return_value = [sample_texts[0]]

    stop_words = set()

    result = process_text_file(kb, test_file, mock_splitter, stop_words, 'english', 'text')
    assert result is True

    # Verify full path was stored
    expected_path = os.path.abspath(test_file)
    kb.sql_cursor.execute("SELECT sourcedoc FROM docs WHERE sourcedoc = ?", [expected_path])
    result = kb.sql_cursor.fetchone()

    assert result is not None
    assert result[0] == expected_path

    close_database(kb)

  @patch('nltk.tokenize.word_tokenize')
  @patch('nltk.tokenize.sent_tokenize')
  @patch('nltk.corpus.stopwords.words')
  def test_duplicate_filenames_different_directories(self, mock_stopwords, mock_sent_tokenize, mock_word_tokenize, temp_config_file, temp_kb_directory, sample_texts):
    """Test that files with same name in different directories are handled correctly."""
    # Mock NLTK functions
    mock_stopwords.return_value = ['the', 'a', 'an', 'and', 'or']
    mock_sent_tokenize.side_effect = lambda text, lang=None: [text]
    mock_word_tokenize.side_effect = lambda text: text.split()

    kb = KnowledgeBase(temp_config_file)
    with patch('builtins.input', return_value='y'):
      connect_to_database(kb)

    # Create two files with same name in different directories
    file1 = os.path.join(temp_kb_directory, "dir1", "config.py")
    file2 = os.path.join(temp_kb_directory, "dir2", "config.py")

    os.makedirs(os.path.dirname(file1), exist_ok=True)
    os.makedirs(os.path.dirname(file2), exist_ok=True)

    with open(file1, 'w') as f:
      f.write("# Config 1\nSETTING = 1")
    with open(file2, 'w') as f:
      f.write("# Config 2\nSETTING = 2")

    # Mock splitter
    mock_splitter = Mock()
    mock_splitter.split_text.side_effect = [["# Config 1\nSETTING = 1"], ["# Config 2\nSETTING = 2"]]

    stop_words = set()

    # Process both files
    result1 = process_text_file(kb, file1, mock_splitter, stop_words, 'english', 'code')
    result2 = process_text_file(kb, file2, mock_splitter, stop_words, 'english', 'code')

    assert result1 is True
    assert result2 is True

    # Verify both files were stored with their full paths
    path1 = os.path.abspath(file1)
    path2 = os.path.abspath(file2)

    kb.sql_cursor.execute("SELECT COUNT(*) FROM docs WHERE sourcedoc = ?", [path1])
    count1 = kb.sql_cursor.fetchone()[0]

    kb.sql_cursor.execute("SELECT COUNT(*) FROM docs WHERE sourcedoc = ?", [path2])
    count2 = kb.sql_cursor.fetchone()[0]

    assert count1 > 0
    assert count2 > 0

    # Verify content is different
    kb.sql_cursor.execute("SELECT originaltext FROM docs WHERE sourcedoc = ?", [path1])
    content1 = kb.sql_cursor.fetchone()[0]

    kb.sql_cursor.execute("SELECT originaltext FROM docs WHERE sourcedoc = ?", [path2])
    content2 = kb.sql_cursor.fetchone()[0]

    assert "Config 1" in content1
    assert "Config 2" in content2

    close_database(kb)

  @patch('nltk.tokenize.word_tokenize')
  @patch('nltk.tokenize.sent_tokenize')
  @patch('nltk.corpus.stopwords.words')
  def test_path_normalization(self, mock_stopwords, mock_sent_tokenize, mock_word_tokenize, temp_config_file, temp_kb_directory, sample_texts):
    """Test that paths are normalized (resolved symlinks, etc)."""
    # Mock NLTK functions
    mock_stopwords.return_value = ['the', 'a', 'an', 'and', 'or']
    mock_sent_tokenize.side_effect = lambda text, lang=None: [text]
    mock_word_tokenize.side_effect = lambda text: text.split()

    kb = KnowledgeBase(temp_config_file)
    with patch('builtins.input', return_value='y'):
      connect_to_database(kb)

    # Create test file with relative path
    test_file = os.path.join(temp_kb_directory, "test.txt")
    with open(test_file, 'w') as f:
      f.write(sample_texts[0])

    # Use relative path for processing
    relative_path = os.path.relpath(test_file)

    # Mock splitter
    mock_splitter = Mock()
    mock_splitter.split_text.return_value = [sample_texts[0]]

    stop_words = set()

    result = process_text_file(kb, relative_path, mock_splitter, stop_words, 'english', 'text')
    assert result is True

    # Verify absolute path was stored
    expected_path = os.path.abspath(test_file)
    kb.sql_cursor.execute("SELECT sourcedoc FROM docs WHERE sourcedoc = ?", [expected_path])
    result = kb.sql_cursor.fetchone()

    assert result is not None
    assert result[0] == expected_path
    assert not result[0].startswith(".")  # Not relative

    close_database(kb)

  @patch('nltk.tokenize.word_tokenize')
  @patch('nltk.tokenize.sent_tokenize')
  @patch('nltk.corpus.stopwords.words')
  def test_existing_file_check_with_full_paths(self, mock_stopwords, mock_sent_tokenize, mock_word_tokenize, temp_config_file, temp_kb_directory, sample_texts):
    """Test that existing file detection works with full paths."""
    # Mock NLTK functions
    mock_stopwords.return_value = ['the', 'a', 'an', 'and', 'or']
    mock_sent_tokenize.side_effect = lambda text, lang=None: [text]
    mock_word_tokenize.side_effect = lambda text: text.split()

    kb = KnowledgeBase(temp_config_file)
    with patch('builtins.input', return_value='y'):
      connect_to_database(kb)

    # Create test file
    test_file = os.path.join(temp_kb_directory, "existing.txt")
    with open(test_file, 'w') as f:
      f.write(sample_texts[0])

    # Mock splitter
    mock_splitter = Mock()
    mock_splitter.split_text.return_value = [sample_texts[0]]

    stop_words = set()

    # Process file first time
    result1 = process_text_file(kb, test_file, mock_splitter, stop_words, 'english', 'text', force=False)
    assert result1 is True

    # Try to process again without force
    result2 = process_text_file(kb, test_file, mock_splitter, stop_words, 'english', 'text', force=False)
    assert result2 is False  # Should skip

    # Process with force
    result3 = process_text_file(kb, test_file, mock_splitter, stop_words, 'english', 'text', force=True)
    assert result3 is True

    close_database(kb)

#fin
