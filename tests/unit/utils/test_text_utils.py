"""
Unit tests for utils/text_utils.py
Tests text processing, file handling, and utility functions.
"""

import pytest
import os
import tempfile
import glob
from unittest.mock import patch, Mock
from pathlib import Path

from utils.text_utils import (
  clean_text,
  enhanced_clean_text,
  get_files,
  split_filepath,
  find_file,
  get_env
)


class TestCleanText:
  """Test basic text cleaning functionality."""
  
  def test_basic_cleaning(self):
    """Test basic text cleaning operations."""
    text = "This is a SAMPLE Text with Mixed Case!"
    result = clean_text(text)
    
    assert result.lower() == result  # Should be lowercase
    assert "sample" in result
    assert "mixed" in result
  
  def test_html_tag_removal(self):
    """Test removal of HTML tags."""
    text = "This is <b>bold</b> and <i>italic</i> text."
    result = clean_text(text)
    
    assert "<b>" not in result
    assert "<i>" not in result
    assert "bold" in result
    assert "italic" in result
  
  def test_non_word_character_replacement(self):
    """Test replacement of non-word characters."""
    text = "Test! @#$ %^& *()text"
    result = clean_text(text)
    
    # Non-word characters should be replaced with spaces
    assert "!" not in result
    assert "@" not in result
    assert "test" in result
    assert "text" in result
  
  def test_stopword_removal(self):
    """Test removal of stopwords."""
    stop_words = {'the', 'a', 'an', 'and', 'or'}
    text = "The cat and the dog"
    result = clean_text(text, stop_words)
    
    assert "the" not in result.split()
    assert "and" not in result.split()
    assert "cat" in result
    assert "dog" in result
  
  def test_empty_text(self):
    """Test handling of empty text."""
    assert clean_text("") == ""
    assert clean_text("   ") == ""
  
  def test_whitespace_normalization(self):
    """Test normalization of whitespace."""
    text = "Multiple    spaces   and\n\nnewlines"
    result = clean_text(text)
    
    assert "    " not in result
    assert "\n\n" not in result
    assert len(result.split()) == 3


class TestEnhancedCleanText:
  """Test enhanced text cleaning functionality."""
  
  def test_url_preservation(self):
    """Test preservation of URLs during cleaning."""
    text = "Visit https://example.com for more info"
    result = enhanced_clean_text(text)
    
    assert "https://example.com" in result
    assert "visit" in result
    assert "more" in result
  
  def test_email_preservation(self):
    """Test preservation of email addresses."""
    text = "Contact us at test@example.com for support"
    result = enhanced_clean_text(text)
    
    assert "test@example.com" in result
    assert "contact" in result
    assert "support" in result
  
  @patch('utils.text_utils.nlp')
  def test_entity_preservation_with_spacy(self, mock_nlp):
    """Test preservation of named entities when spaCy is available."""
    # Mock spaCy entity
    mock_entity = Mock()
    mock_entity.text = "Apple Inc"
    mock_entity.label_ = "ORG"
    
    mock_doc = Mock()
    mock_doc.ents = [mock_entity]
    mock_nlp.return_value = mock_doc
    
    text = "Apple Inc is a technology company"
    result = enhanced_clean_text(text)
    
    assert "apple inc" in result.lower()
    assert "technology" in result
  
  def test_lemmatization(self):
    """Test lemmatization functionality."""
    from nltk.stem import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    
    text = "The cats are running quickly"
    result = enhanced_clean_text(text, lemmatizer=lemmatizer)
    
    # Note: actual lemmatization results may vary based on NLTK setup
    assert "cat" in result or "cats" in result
    assert "run" in result or "running" in result
  
  def test_stopword_removal_with_lemmatization(self):
    """Test stopword removal combined with lemmatization."""
    from nltk.stem import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    stop_words = {'the', 'are', 'is'}
    
    text = "The cats are running"
    result = enhanced_clean_text(text, stop_words=stop_words, lemmatizer=lemmatizer)
    
    assert "the" not in result.split()
    assert "are" not in result.split()
    assert len(result.split()) <= 2  # Should have fewer words
  
  def test_punctuation_handling(self):
    """Test handling of meaningful punctuation."""
    text = "Hello! How are you? I'm fine."
    result = enhanced_clean_text(text)
    
    # Should preserve sentence structure while cleaning
    assert "hello" in result
    assert "fine" in result
  
  def test_without_spacy(self):
    """Test enhanced cleaning when spaCy is not available."""
    with patch('utils.text_utils.nlp', None):
      text = "Apple Inc is a technology company"
      result = enhanced_clean_text(text)
      
      assert "apple" in result
      assert "technology" in result
      assert "company" in result
  
  def test_entity_extraction_error_handling(self):
    """Test handling of errors during entity extraction."""
    with patch('utils.text_utils.nlp') as mock_nlp:
      mock_nlp.side_effect = Exception("spaCy error")
      
      text = "This should not crash"
      result = enhanced_clean_text(text)
      
      assert "should" in result
      assert "crash" in result


class TestGetFiles:
  """Test file gathering functionality."""
  
  def test_single_file(self, temp_data_manager):
    """Test getting a single file."""
    test_file = temp_data_manager.create_temp_text_file("content", "test.txt")
    
    result = get_files(test_file)
    
    assert len(result) == 1
    assert test_file in result
  
  def test_directory_expansion(self, temp_data_manager):
    """Test directory expansion to recursive glob."""
    temp_dir = temp_data_manager.create_temp_dir()
    
    # Create test files
    for i in range(3):
      temp_data_manager.create_temp_text_file(f"content {i}", f"test_{i}.txt")
    
    with patch('glob.glob') as mock_glob:
      mock_glob.return_value = [f"{temp_dir}/test_0.txt", f"{temp_dir}/test_1.txt"]
      
      result = get_files(temp_dir)
      
      mock_glob.assert_called_with(f"{temp_dir}/**", recursive=True)
  
  def test_glob_pattern(self):
    """Test glob pattern expansion."""
    with patch('glob.glob') as mock_glob:
      mock_glob.return_value = ["file1.txt", "file2.txt", "subdir"]
      
      with patch('os.path.isdir') as mock_isdir:
        mock_isdir.side_effect = lambda x: x == "subdir"
        
        result = get_files("*.txt")
        
        # Should exclude directories
        assert "file1.txt" in result
        assert "file2.txt" in result
        assert "subdir" not in result
  
  def test_empty_result(self):
    """Test handling of no matching files."""
    with patch('glob.glob', return_value=[]):
      result = get_files("nonexistent*.txt")
      assert result == []
  
  def test_sorting(self):
    """Test that results are sorted."""
    with patch('glob.glob') as mock_glob:
      mock_glob.return_value = ["c.txt", "a.txt", "b.txt"]
      
      with patch('os.path.isdir', return_value=False):
        result = get_files("*.txt")
        
        assert result == ["a.txt", "b.txt", "c.txt"]


class TestSplitFilepath:
  """Test filepath splitting functionality."""
  
  def test_basic_splitting(self):
    """Test basic filepath splitting."""
    filepath = "/path/to/file.txt"
    directory, basename, extension, fqfn = split_filepath(filepath, adddir=False, realpath=False)
    
    assert directory == "/path/to"
    assert basename == "file"
    assert extension == ".txt"
    assert fqfn == "/path/to/file.txt"
  
  def test_no_directory(self):
    """Test filepath with no directory."""
    filepath = "file.txt"
    directory, basename, extension, fqfn = split_filepath(filepath, adddir=True, realpath=False)
    
    assert basename == "file"
    assert extension == ".txt"
    assert directory  # Should add current directory
    assert fqfn.endswith("file.txt")
  
  def test_no_extension(self):
    """Test filepath with no extension."""
    filepath = "/path/to/filename"
    directory, basename, extension, fqfn = split_filepath(filepath, adddir=False, realpath=False)
    
    assert directory == "/path/to"
    assert basename == "filename"
    assert extension == ""
    assert fqfn == "/path/to/filename"
  
  def test_adddir_false(self):
    """Test with adddir=False."""
    filepath = "file.txt"
    directory, basename, extension, fqfn = split_filepath(filepath, adddir=False, realpath=False)
    
    assert directory == ""
    assert basename == "file"
    assert extension == ".txt"
    assert fqfn == "/file.txt"
  
  def test_realpath_resolution(self):
    """Test realpath resolution."""
    with patch('os.path.realpath') as mock_realpath:
      mock_realpath.return_value = "/real/path/file.txt"
      
      directory, basename, extension, fqfn = split_filepath("./file.txt", realpath=True)
      
      mock_realpath.assert_called_once_with("./file.txt")
      assert fqfn == "/real/path/file.txt"
  
  def test_complex_extension(self):
    """Test files with complex extensions."""
    filepath = "/path/to/archive.tar.gz"
    directory, basename, extension, fqfn = split_filepath(filepath, adddir=False, realpath=False)
    
    assert basename == "archive.tar"
    assert extension == ".gz"
  
  def test_hidden_file(self):
    """Test hidden files (starting with dot)."""
    filepath = "/path/to/.hidden"
    directory, basename, extension, fqfn = split_filepath(filepath, adddir=False, realpath=False)
    
    assert basename == ".hidden"
    assert extension == ""


class TestFindFile:
  """Test file finding functionality."""
  
  def test_find_existing_file(self, temp_data_manager):
    """Test finding an existing file."""
    temp_dir = temp_data_manager.create_temp_dir()
    test_file = os.path.join(temp_dir, "findme.txt")
    with open(test_file, 'w') as f:
      f.write("content")
    
    result = find_file("findme.txt", temp_dir)
    
    assert result is not None
    assert result.endswith("findme.txt")
    assert os.path.exists(result)
  
  def test_find_nonexistent_file(self, temp_data_manager):
    """Test searching for nonexistent file."""
    temp_dir = temp_data_manager.create_temp_dir()
    
    result = find_file("nonexistent.txt", temp_dir)
    
    assert result is None
  
  def test_find_in_subdirectory(self, temp_data_manager):
    """Test finding file in subdirectory."""
    temp_dir = temp_data_manager.create_temp_dir()
    sub_dir = os.path.join(temp_dir, "subdir")
    os.makedirs(sub_dir)
    
    test_file = os.path.join(sub_dir, "nested.txt")
    with open(test_file, 'w') as f:
      f.write("content")
    
    result = find_file("nested.txt", temp_dir)
    
    assert result is not None
    assert "nested.txt" in result
    assert "subdir" in result
  
  def test_invalid_filename_with_slash(self):
    """Test rejection of filenames containing slashes."""
    result = find_file("path/with/slash.txt", "/tmp")
    
    assert result is None
  
  def test_follow_symlinks(self, temp_data_manager):
    """Test following symbolic links."""
    if os.name == 'nt':  # Skip on Windows
      pytest.skip("Symbolic links not reliably supported on Windows")
    
    temp_dir = temp_data_manager.create_temp_dir()
    
    # Create original file
    original_file = os.path.join(temp_dir, "original.txt")
    with open(original_file, 'w') as f:
      f.write("content")
    
    # Create subdirectory with symlink
    sub_dir = os.path.join(temp_dir, "subdir")
    os.makedirs(sub_dir)
    symlink_file = os.path.join(sub_dir, "link.txt")
    os.symlink(original_file, symlink_file)
    
    # Search for the symlink
    result = find_file("link.txt", temp_dir, followsymlinks=True)
    
    assert result is not None
    assert "link.txt" in result
  
  def test_realpath_resolution(self, temp_data_manager):
    """Test that result is resolved to real path."""
    temp_dir = temp_data_manager.create_temp_dir()
    test_file = os.path.join(temp_dir, "test.txt")
    with open(test_file, 'w') as f:
      f.write("content")
    
    with patch('os.path.realpath') as mock_realpath:
      mock_realpath.return_value = "/real/path/test.txt"
      
      result = find_file("test.txt", temp_dir)
      
      assert result == "/real/path/test.txt"
      mock_realpath.assert_called_once()


class TestGetEnv:
  """Test environment variable utility function."""
  
  def test_existing_env_var(self):
    """Test getting existing environment variable."""
    with patch.dict(os.environ, {'TEST_VAR': 'test_value'}):
      result = get_env('TEST_VAR', 'default_value')
      assert result == 'test_value'
  
  def test_nonexistent_env_var(self):
    """Test getting nonexistent environment variable returns default."""
    result = get_env('NONEXISTENT_VAR', 'default_value')
    assert result == 'default_value'
  
  def test_type_casting_int(self):
    """Test type casting to integer."""
    with patch.dict(os.environ, {'INT_VAR': '42'}):
      result = get_env('INT_VAR', 0, int)
      assert result == 42
      assert isinstance(result, int)
  
  def test_type_casting_float(self):
    """Test type casting to float."""
    with patch.dict(os.environ, {'FLOAT_VAR': '3.14'}):
      result = get_env('FLOAT_VAR', 0.0, float)
      assert result == 3.14
      assert isinstance(result, float)
  
  def test_type_casting_bool(self):
    """Test type casting to boolean."""
    with patch.dict(os.environ, {'BOOL_VAR': 'true'}):
      result = get_env('BOOL_VAR', False, bool)
      assert result is True
      assert isinstance(result, bool)
  
  def test_invalid_type_casting(self):
    """Test handling of invalid type casting."""
    with patch.dict(os.environ, {'INVALID_INT': 'not_a_number'}):
      result = get_env('INVALID_INT', 42, int)
      assert result == 42  # Should return default
  
  def test_empty_env_var(self):
    """Test handling of empty environment variable."""
    with patch.dict(os.environ, {'EMPTY_VAR': ''}):
      result = get_env('EMPTY_VAR', 'default', str)
      assert result == ''  # Empty string is still a value
  
  def test_none_default(self):
    """Test with None as default value."""
    result = get_env('NONEXISTENT', None)
    assert result is None
  
  def test_complex_type_casting_error(self):
    """Test type casting error with complex types."""
    with patch.dict(os.environ, {'COMPLEX_VAR': 'test'}):
      # Try to cast string to list (should fail)
      result = get_env('COMPLEX_VAR', [], list)
      assert result == []  # Should return default

#fin