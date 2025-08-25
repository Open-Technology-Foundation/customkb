#!/usr/bin/env python
"""
Unit tests for database.chunking module.

Tests text chunking, file type detection, and chunk optimization.
"""

import os
import unittest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from database.chunking import (
  detect_file_type,
  init_text_splitter,
  get_language_specific_splitter,
  split_text,
  calculate_chunk_statistics,
  optimize_chunk_size,
  merge_small_chunks,
  validate_chunks
)
from utils.exceptions import ChunkingError, ProcessingError


class TestFileTypeDetection(unittest.TestCase):
  """Test file type detection functionality."""
  
  def test_detect_markdown_files(self):
    """Test detection of Markdown files."""
    self.assertEqual(detect_file_type('document.md'), 'markdown')
    self.assertEqual(detect_file_type('README.markdown'), 'markdown')
    self.assertEqual(detect_file_type('notes.mdown'), 'markdown')
    self.assertEqual(detect_file_type('guide.mkd'), 'markdown')
  
  def test_detect_html_files(self):
    """Test detection of HTML files."""
    self.assertEqual(detect_file_type('index.html'), 'html')
    self.assertEqual(detect_file_type('page.htm'), 'html')
    self.assertEqual(detect_file_type('doc.xhtml'), 'html')
  
  def test_detect_code_files(self):
    """Test detection of code files."""
    self.assertEqual(detect_file_type('script.py'), 'code')
    self.assertEqual(detect_file_type('app.js'), 'code')
    self.assertEqual(detect_file_type('main.java'), 'code')
    self.assertEqual(detect_file_type('program.cpp'), 'code')
    self.assertEqual(detect_file_type('lib.rs'), 'code')
    self.assertEqual(detect_file_type('test.go'), 'code')
  
  def test_detect_json_files(self):
    """Test detection of JSON files."""
    self.assertEqual(detect_file_type('config.json'), 'json')
    self.assertEqual(detect_file_type('data.jsonl'), 'json')
  
  def test_detect_yaml_files(self):
    """Test detection of YAML files."""
    self.assertEqual(detect_file_type('config.yaml'), 'yaml')
    self.assertEqual(detect_file_type('settings.yml'), 'yaml')
  
  def test_detect_xml_files(self):
    """Test detection of XML files."""
    self.assertEqual(detect_file_type('data.xml'), 'xml')
    self.assertEqual(detect_file_type('image.svg'), 'xml')
  
  def test_detect_config_files(self):
    """Test detection of config files."""
    self.assertEqual(detect_file_type('settings.ini'), 'config')
    self.assertEqual(detect_file_type('app.cfg'), 'config')
    self.assertEqual(detect_file_type('system.conf'), 'config')
    self.assertEqual(detect_file_type('pyproject.toml'), 'config')
  
  def test_detect_text_files(self):
    """Test detection of text files."""
    self.assertEqual(detect_file_type('document.txt'), 'text')
    self.assertEqual(detect_file_type('notes.text'), 'text')
    self.assertEqual(detect_file_type('app.log'), 'text')
    self.assertEqual(detect_file_type('data.csv'), 'text')
    self.assertEqual(detect_file_type('data.tsv'), 'text')
  
  def test_detect_unknown_files(self):
    """Test default to text for unknown extensions."""
    self.assertEqual(detect_file_type('file.xyz'), 'text')
    self.assertEqual(detect_file_type('noextension'), 'text')
    self.assertEqual(detect_file_type('file.unknown'), 'text')
  
  def test_case_insensitive_detection(self):
    """Test that detection is case-insensitive."""
    self.assertEqual(detect_file_type('Document.MD'), 'markdown')
    self.assertEqual(detect_file_type('Script.PY'), 'code')
    self.assertEqual(detect_file_type('Page.HTML'), 'html')


class TestTextSplitter(unittest.TestCase):
  """Test text splitter initialization."""
  
  def setUp(self):
    """Set up test fixtures."""
    self.kb = Mock()
    self.kb.chunk_size = 500
    self.kb.chunk_overlap = 50
  
  def test_init_markdown_splitter(self):
    """Test Markdown splitter initialization."""
    splitter = init_text_splitter(self.kb, 'markdown')
    self.assertIsNotNone(splitter)
    # Verify it's a MarkdownTextSplitter
    self.assertEqual(splitter.__class__.__name__, 'MarkdownTextSplitter')
  
  def test_init_code_splitter(self):
    """Test code splitter initialization."""
    splitter = init_text_splitter(self.kb, 'code')
    self.assertIsNotNone(splitter)
    # Verify it's a RecursiveCharacterTextSplitter
    self.assertEqual(splitter.__class__.__name__, 'RecursiveCharacterTextSplitter')
  
  def test_init_html_splitter(self):
    """Test HTML splitter initialization."""
    splitter = init_text_splitter(self.kb, 'html')
    self.assertIsNotNone(splitter)
    self.assertEqual(splitter.__class__.__name__, 'RecursiveCharacterTextSplitter')
  
  def test_init_json_splitter(self):
    """Test JSON splitter initialization."""
    splitter = init_text_splitter(self.kb, 'json')
    self.assertIsNotNone(splitter)
    # Should use specific separators for structured data
    self.assertIn(',', splitter._separators)
  
  def test_init_default_splitter(self):
    """Test default text splitter initialization."""
    splitter = init_text_splitter(self.kb, 'text')
    self.assertIsNotNone(splitter)
    # Should have sentence-aware separators
    self.assertIn('. ', splitter._separators)
  
  def test_init_splitter_error(self):
    """Test splitter initialization error handling."""
    self.kb.chunk_size = None  # Invalid configuration
    
    with self.assertRaises(ChunkingError) as cm:
      init_text_splitter(self.kb, 'text')
    
    self.assertIn("Text splitter initialization failed", str(cm.exception))


class TestLanguageSpecificSplitter(unittest.TestCase):
  """Test language-specific code splitting."""
  
  def setUp(self):
    """Set up test fixtures."""
    self.kb = Mock()
    self.kb.chunk_size = 500
    self.kb.chunk_overlap = 50
  
  def test_python_splitter(self):
    """Test Python-specific splitter."""
    splitter = get_language_specific_splitter('script.py', self.kb)
    self.assertIsNotNone(splitter)
  
  def test_javascript_splitter(self):
    """Test JavaScript-specific splitter."""
    splitter = get_language_specific_splitter('app.js', self.kb)
    self.assertIsNotNone(splitter)
  
  def test_unsupported_language(self):
    """Test unsupported language returns None."""
    splitter = get_language_specific_splitter('file.xyz', self.kb)
    self.assertIsNone(splitter)
  
  @patch('database.chunking.logger')
  def test_splitter_creation_error(self, mock_logger):
    """Test error handling in splitter creation."""
    self.kb.chunk_size = -1  # Invalid
    
    splitter = get_language_specific_splitter('script.py', self.kb)
    self.assertIsNone(splitter)
    mock_logger.warning.assert_called()


class TestTextSplitting(unittest.TestCase):
  """Test text splitting functionality."""
  
  def setUp(self):
    """Set up test fixtures."""
    self.kb = Mock()
    self.kb.chunk_size = 100
    self.kb.chunk_overlap = 20
    self.splitter = Mock()
  
  def test_split_text_basic(self):
    """Test basic text splitting."""
    text = "This is a test document that needs to be split into chunks."
    self.splitter.split_text.return_value = [
      "This is a test document",
      "that needs to be split",
      "into chunks."
    ]
    
    chunks = split_text(text, self.splitter)
    
    self.assertEqual(len(chunks), 3)
    self.assertEqual(chunks[0]['text'], "This is a test document")
    self.assertEqual(chunks[0]['chunk_index'], 0)
    self.assertEqual(chunks[0]['total_chunks'], 3)
    self.assertIn('char_count', chunks[0])
  
  def test_split_text_with_metadata(self):
    """Test text splitting with metadata."""
    text = "Test content"
    metadata = {'source': 'test.txt', 'author': 'Test Author'}
    self.splitter.split_text.return_value = ["Test content"]
    
    chunks = split_text(text, self.splitter, metadata)
    
    self.assertEqual(len(chunks), 1)
    self.assertEqual(chunks[0]['source'], 'test.txt')
    self.assertEqual(chunks[0]['author'], 'Test Author')
  
  def test_split_text_empty(self):
    """Test splitting empty text."""
    self.splitter.split_text.return_value = []
    
    chunks = split_text("", self.splitter)
    
    self.assertEqual(chunks, [])
  
  def test_split_text_error(self):
    """Test error handling in text splitting."""
    self.splitter.split_text.side_effect = Exception("Split failed")
    
    with self.assertRaises(ChunkingError) as cm:
      split_text("Test text", self.splitter)
    
    self.assertIn("Failed to split text", str(cm.exception))


class TestChunkStatistics(unittest.TestCase):
  """Test chunk statistics calculation."""
  
  def test_calculate_statistics(self):
    """Test statistics calculation for chunks."""
    chunks = [
      {'text': 'Short'},
      {'text': 'A medium chunk'},
      {'text': 'This is a longer chunk of text'}
    ]
    
    stats = calculate_chunk_statistics(chunks)
    
    self.assertEqual(stats['total_chunks'], 3)
    self.assertEqual(stats['total_chars'], 49)  # 5 + 14 + 30
    self.assertAlmostEqual(stats['avg_chunk_size'], 16.33, places=1)
    self.assertEqual(stats['min_chunk_size'], 5)
    self.assertEqual(stats['max_chunk_size'], 30)
  
  def test_calculate_statistics_empty(self):
    """Test statistics for empty chunk list."""
    stats = calculate_chunk_statistics([])
    
    self.assertEqual(stats['total_chunks'], 0)
    self.assertEqual(stats['total_chars'], 0)
    self.assertEqual(stats['avg_chunk_size'], 0)
    self.assertEqual(stats['min_chunk_size'], 0)
    self.assertEqual(stats['max_chunk_size'], 0)


class TestChunkOptimization(unittest.TestCase):
  """Test chunk size optimization."""
  
  def test_optimize_chunk_size_normal(self):
    """Test normal chunk size optimization."""
    size = optimize_chunk_size(10000, target_chunks=10)
    
    # Should be around 1000, rounded to nearest 50
    self.assertEqual(size, 1000)
  
  def test_optimize_chunk_size_min_constraint(self):
    """Test minimum size constraint."""
    size = optimize_chunk_size(500, target_chunks=10)
    
    # Should be at minimum (100)
    self.assertEqual(size, 100)
  
  def test_optimize_chunk_size_max_constraint(self):
    """Test maximum size constraint."""
    size = optimize_chunk_size(50000, target_chunks=10)
    
    # Should be at maximum (2000)
    self.assertEqual(size, 2000)
  
  def test_optimize_chunk_size_zero_length(self):
    """Test optimization with zero length."""
    size = optimize_chunk_size(0)
    
    # Should return default
    self.assertEqual(size, 500)
  
  def test_optimize_chunk_size_rounding(self):
    """Test that sizes are rounded to nearest 50."""
    size = optimize_chunk_size(1234, target_chunks=10)
    
    # 1234/10 = 123.4, should round to 100
    self.assertEqual(size, 100)


class TestChunkMerging(unittest.TestCase):
  """Test small chunk merging."""
  
  def test_merge_small_chunks(self):
    """Test merging of small chunks."""
    chunks = [
      {'text': 'Small', 'chunk_index': 0},
      {'text': 'Tiny', 'chunk_index': 1},
      {'text': 'This is a normal sized chunk that should not be merged', 'chunk_index': 2}
    ]
    
    merged = merge_small_chunks(chunks, min_size=20)
    
    # First two should be merged
    self.assertEqual(len(merged), 2)
    self.assertEqual(merged[0]['text'], 'Small\nTiny')
    self.assertEqual(merged[0]['chunk_index'], 0)
    self.assertEqual(merged[1]['chunk_index'], 1)
  
  def test_merge_all_small_chunks(self):
    """Test merging when all chunks are small."""
    chunks = [
      {'text': 'A'},
      {'text': 'B'},
      {'text': 'C'}
    ]
    
    merged = merge_small_chunks(chunks, min_size=10)
    
    # All should be merged into one
    self.assertEqual(len(merged), 1)
    self.assertEqual(merged[0]['text'], 'A\nB\nC')
  
  def test_merge_empty_chunks(self):
    """Test merging with empty chunk list."""
    merged = merge_small_chunks([])
    
    self.assertEqual(merged, [])
  
  def test_merge_updates_indices(self):
    """Test that chunk indices are updated after merging."""
    chunks = [
      {'text': 'Small', 'chunk_index': 0, 'total_chunks': 3},
      {'text': 'Tiny', 'chunk_index': 1, 'total_chunks': 3},
      {'text': 'Normal chunk', 'chunk_index': 2, 'total_chunks': 3}
    ]
    
    merged = merge_small_chunks(chunks, min_size=10)
    
    # Verify indices and totals updated
    for i, chunk in enumerate(merged):
      self.assertEqual(chunk['chunk_index'], i)
      self.assertEqual(chunk['total_chunks'], len(merged))


class TestChunkValidation(unittest.TestCase):
  """Test chunk validation."""
  
  def setUp(self):
    """Set up test fixtures."""
    self.kb = Mock()
    self.kb.max_chunk_size = 1000
    self.kb.min_chunk_size = 10
  
  def test_validate_valid_chunks(self):
    """Test validation of valid chunks."""
    chunks = [
      {'text': 'This is a valid chunk'},
      {'text': 'Another valid chunk of text'},
      {'text': 'A third chunk'}
    ]
    
    result = validate_chunks(chunks, self.kb)
    self.assertTrue(result)
  
  def test_validate_empty_chunks(self):
    """Test validation fails for empty chunk list."""
    with self.assertRaises(ProcessingError) as cm:
      validate_chunks([], self.kb)
    
    self.assertIn("No chunks to validate", str(cm.exception))
  
  def test_validate_missing_text(self):
    """Test validation fails for chunk without text."""
    chunks = [
      {'text': 'Valid chunk'},
      {'no_text': 'Missing text key'}
    ]
    
    with self.assertRaises(ProcessingError) as cm:
      validate_chunks(chunks, self.kb)
    
    self.assertIn("Chunk 1 has no text", str(cm.exception))
  
  def test_validate_empty_text(self):
    """Test validation fails for empty text."""
    chunks = [{'text': ''}]
    
    with self.assertRaises(ProcessingError) as cm:
      validate_chunks(chunks, self.kb)
    
    self.assertIn("Chunk 0 has no text", str(cm.exception))
  
  def test_validate_exceeds_max_size(self):
    """Test validation fails for oversized chunks."""
    chunks = [{'text': 'x' * 1001}]  # Exceeds max_chunk_size of 1000
    
    with self.assertRaises(ProcessingError) as cm:
      validate_chunks(chunks, self.kb)
    
    self.assertIn("exceeds maximum size", str(cm.exception))
  
  @patch('database.chunking.logger')
  def test_validate_below_min_size_warning(self, mock_logger):
    """Test validation warns for undersized non-final chunks."""
    chunks = [
      {'text': 'tiny'},  # Below min_chunk_size of 10
      {'text': 'This is a normal chunk'}
    ]
    
    result = validate_chunks(chunks, self.kb)
    
    # Should pass but with warning
    self.assertTrue(result)
    mock_logger.warning.assert_called()
  
  def test_validate_last_chunk_can_be_small(self):
    """Test that last chunk is allowed to be small."""
    chunks = [
      {'text': 'This is a normal chunk'},
      {'text': 'tiny'}  # Below min but it's the last chunk
    ]
    
    # Should not raise
    result = validate_chunks(chunks, self.kb)
    self.assertTrue(result)
  
  def test_validate_custom_limits(self):
    """Test validation with custom size limits."""
    self.kb.max_chunk_size = 50
    self.kb.min_chunk_size = 5
    
    chunks = [
      {'text': 'Valid'},  # Exactly at min
      {'text': 'x' * 50}  # Exactly at max
    ]
    
    result = validate_chunks(chunks, self.kb)
    self.assertTrue(result)


if __name__ == '__main__':
  unittest.main()

#fin