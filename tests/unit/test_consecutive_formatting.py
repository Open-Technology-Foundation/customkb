"""
Unit tests for consecutive document formatting in different formats.
"""

import unittest
from unittest.mock import Mock

from query.formatters import MarkdownFormatter
from query.query_manager import build_reference_string


class TestConsecutiveDocumentFormatting(unittest.TestCase):
  """Test cases for consecutive document formatting."""

  def setUp(self):
    """Set up test data."""
    # Create consecutive documents from same source
    self.consecutive_reference = [
      ('chunk1', 'doc1.txt', 1, 'Content 1', 0.95, '{"heading": "Section 1"}'),
      ('chunk2', 'doc1.txt', 2, 'Content 2', 0.94, '{"heading": "Section 2"}'),  # Consecutive
      ('chunk3', 'doc1.txt', 3, 'Content 3', 0.93, '{"heading": "Section 3"}'),  # Consecutive
      ('chunk4', 'doc2.txt', 1, 'Different doc', 0.92, '{"heading": "Other"}'),  # Not consecutive
    ]

    # Mock KB object
    self.mock_kb = Mock()
    self.mock_kb.reference_format = 'markdown'

  def test_markdown_consecutive_sections(self):
    """Test that consecutive documents get section headers in Markdown."""
    formatter = MarkdownFormatter()

    # Test the formatter methods directly
    result = ""

    # First document
    result += formatter.format_document_start("doc1.txt", 1)
    result += formatter.format_document_content("Content 1")

    # Consecutive documents should get section headers
    section_header = formatter.format_section_header("doc1.txt", 2)
    self.assertIn("Section 2", section_header)
    result += section_header
    result += formatter.format_document_content("Content 2")

    section_header = formatter.format_section_header("doc1.txt", 3)
    self.assertIn("Section 3", section_header)
    result += section_header
    result += formatter.format_document_content("Content 3")

    result += formatter.format_document_end()

    # Verify structure
    self.assertEqual(result.count("## References"), 1)  # Only one main header
    self.assertEqual(result.count("### doc1.txt:1"), 1)  # First doc header
    self.assertEqual(result.count("#### Section"), 2)  # Two section headers

  def test_format_references_with_consecutive_docs(self):
    """Test build_reference_string function with consecutive documents.

    Note: The implementation merges consecutive sections from the same document,
    showing only the first chunk's heading but including all content. This is
    intentional UX - consecutive references don't need redundant section headers.
    """
    result = build_reference_string(self.mock_kb, self.consecutive_reference, [], format_type='markdown')

    # Should have one References header
    self.assertEqual(result.count("## References"), 1)

    # Should have headers for different source documents
    self.assertIn("doc1.txt", result)
    self.assertIn("doc2.txt", result)

    # First section heading should be present (consecutive merging uses first heading)
    self.assertIn("Section 1", result)

    # All content should be present (consecutive sections are merged)
    self.assertIn("Content 1", result)
    self.assertIn("Content 2", result)
    self.assertIn("Content 3", result)
    self.assertIn("Different doc", result)

  def test_xml_consecutive_grouping(self):
    """Test XML formatter groups consecutive documents correctly."""
    result = build_reference_string(self.mock_kb, self.consecutive_reference, [], format_type='xml')

    # Should have two context blocks (one for doc1, one for doc2)
    self.assertEqual(result.count('<context src="doc1.txt:1">'), 1)
    self.assertEqual(result.count('<context src="doc2.txt:1">'), 1)
    self.assertEqual(result.count('</context>'), 2)

    # All content should be within the contexts
    self.assertIn("Content 1", result)
    self.assertIn("Content 2", result)
    self.assertIn("Content 3", result)

  def test_json_formatter_no_grouping(self):
    """Test JSON formatter doesn't group documents."""
    result = build_reference_string(self.mock_kb, self.consecutive_reference, [], format_type='json')

    import json
    data = json.loads(result)

    # Should have 4 separate references
    self.assertEqual(len(data['references']), 4)

    # Each should have its own metadata
    for i, ref in enumerate(data['references']):
      self.assertEqual(ref['content'], f"Content {i+1}" if i < 3 else "Different doc")
      self.assertIn('metadata', ref)

  def test_plain_text_consecutive(self):
    """Test plain text formatter with consecutive documents."""
    result = build_reference_string(self.mock_kb, self.consecutive_reference, [], format_type='plain')

    # Should have headers for each source/section change
    self.assertIn("--- doc1.txt:1 ---", result)
    self.assertIn("--- doc2.txt:1 ---", result)

    # All content should be present
    self.assertIn("Content 1", result)
    self.assertIn("Content 2", result)
    self.assertIn("Content 3", result)
    self.assertIn("Different doc", result)

  def test_non_consecutive_sections(self):
    """Test formatting when sections are not consecutive."""
    non_consecutive = [
      ('chunk1', 'doc1.txt', 1, 'Content 1', 0.95, None),
      ('chunk2', 'doc1.txt', 3, 'Content 3', 0.94, None),  # Skip section 2
      ('chunk3', 'doc1.txt', 5, 'Content 5', 0.93, None),  # Skip section 4
    ]

    result = build_reference_string(self.mock_kb, non_consecutive, [], format_type='markdown')

    # Each non-consecutive section should get its own header
    self.assertIn("### doc1.txt:1", result)
    self.assertIn("### doc1.txt:3", result)
    self.assertIn("### doc1.txt:5", result)

    # Should have two separators between three documents (no trailing separator)
    self.assertEqual(result.count("---"), 2)


if __name__ == '__main__':
  unittest.main()

#fin
