"""
Unit tests for reference formatters.
"""

import json
import unittest

from query.formatters import JSONFormatter, MarkdownFormatter, PlainTextFormatter, XMLFormatter, get_formatter


class TestFormatters(unittest.TestCase):
  """Test cases for reference formatters."""

  def setUp(self):
    """Set up test data."""
    self.test_content = "This is test content"
    self.test_filename = "test_file.txt"
    self.test_source = "path/to/document.txt"
    self.test_sid = 42
    self.test_display_source = ".../document.txt"
    self.test_metadata_elems = [
      '<meta name="heading">Test Section</meta>',
      '<meta name="section_type">paragraph</meta>'
    ]
    self.test_similarity = '<meta name="similarity">0.9500</meta>'

  def test_get_formatter(self):
    """Test formatter factory function."""
    # Test valid formatters
    self.assertIsInstance(get_formatter('xml'), XMLFormatter)
    self.assertIsInstance(get_formatter('json'), JSONFormatter)
    self.assertIsInstance(get_formatter('markdown'), MarkdownFormatter)
    self.assertIsInstance(get_formatter('md'), MarkdownFormatter)  # Alias
    self.assertIsInstance(get_formatter('plain'), PlainTextFormatter)
    self.assertIsInstance(get_formatter('text'), PlainTextFormatter)  # Alias

    # Test case insensitivity
    self.assertIsInstance(get_formatter('XML'), XMLFormatter)
    self.assertIsInstance(get_formatter('Json'), JSONFormatter)

    # Test invalid formatter
    with self.assertRaises(ValueError):
      get_formatter('invalid')

  def test_xml_formatter(self):
    """Test XML formatter output."""
    formatter = XMLFormatter()

    # Test context file formatting
    result = formatter.format_context_file(self.test_content, self.test_filename)
    self.assertIn('<reference src="test_file.txt">', result)
    self.assertIn(self.test_content, result)
    self.assertIn('</reference>', result)

    # Test document formatting
    start = formatter.format_document_start(self.test_source, self.test_sid, self.test_display_source)
    self.assertIn(f'<context src=".../document.txt:{self.test_sid}">', start)

    # Test metadata formatting
    metadata = formatter.format_metadata(self.test_metadata_elems, self.test_similarity, debug=True)
    self.assertIn('<metadata>', metadata)
    self.assertIn('Test Section', metadata)
    self.assertIn('0.9500', metadata)

    # Test without debug
    metadata_no_debug = formatter.format_metadata(self.test_metadata_elems, self.test_similarity, debug=False)
    self.assertNotIn('0.9500', metadata_no_debug)

    # Test document end
    end = formatter.format_document_end()
    self.assertEqual(end, "</context>\n\n")

  def test_json_formatter(self):
    """Test JSON formatter output."""
    formatter = JSONFormatter()

    # Format a complete document
    formatter.format_context_file(self.test_content, self.test_filename)
    formatter.format_document_start(self.test_source, self.test_sid, self.test_display_source)
    formatter.format_metadata(self.test_metadata_elems, self.test_similarity, debug=True)
    formatter.format_document_content("Document content here")
    formatter.format_document_end()

    # Get final JSON
    result = formatter.finalize("")
    data = json.loads(result)

    # Verify structure
    self.assertIn('context_files', data)
    self.assertIn('references', data)
    self.assertEqual(len(data['context_files']), 1)
    self.assertEqual(len(data['references']), 1)

    # Verify context file
    self.assertEqual(data['context_files'][0]['source'], self.test_filename)
    self.assertEqual(data['context_files'][0]['content'], self.test_content)

    # Verify reference
    ref = data['references'][0]
    self.assertEqual(ref['source'], self.test_source)
    self.assertEqual(ref['sid'], self.test_sid)
    self.assertEqual(ref['content'], "Document content here")
    self.assertEqual(ref['metadata']['heading'], "Test Section")
    self.assertEqual(ref['similarity'], 0.95)

  def test_markdown_formatter(self):
    """Test Markdown formatter output."""
    formatter = MarkdownFormatter()

    # Test context file
    result = formatter.format_context_file(self.test_content, self.test_filename)
    self.assertIn('## Context Files', result)
    self.assertIn(f'### {self.test_filename}', result)
    self.assertIn(self.test_content, result)

    # Test document formatting
    start = formatter.format_document_start(self.test_source, self.test_sid, self.test_display_source)
    self.assertIn('## References', start)
    self.assertIn(f'### .../document.txt:{self.test_sid}', start)

    # Test metadata
    metadata = formatter.format_metadata(self.test_metadata_elems, self.test_similarity, debug=True)
    self.assertIn('**Heading:** Test Section', metadata)
    self.assertIn('**Section Type:** paragraph', metadata)
    self.assertIn('**Similarity:** 0.9500', metadata)

    # Test document end
    end = formatter.format_document_end()
    self.assertEqual(end, "---\n\n")

    # Test finalize removes trailing separator
    content = "Some content\n---\n\n"
    finalized = formatter.finalize(content)
    self.assertEqual(finalized, "Some content\n")

  def test_plain_formatter(self):
    """Test plain text formatter output."""
    formatter = PlainTextFormatter()

    # Test context file
    result = formatter.format_context_file(self.test_content, self.test_filename)
    self.assertIn(f'=== Context File: {self.test_filename} ===', result)
    self.assertIn(self.test_content, result)

    # Test document formatting
    start = formatter.format_document_start(self.test_source, self.test_sid, self.test_display_source)
    self.assertIn(f'--- .../document.txt:{self.test_sid} ---', start)

    # Test metadata (only in debug mode)
    metadata_debug = formatter.format_metadata(self.test_metadata_elems, self.test_similarity, debug=True)
    self.assertIn('[heading: Test Section]', metadata_debug)
    self.assertIn('[similarity: 0.9500]', metadata_debug)

    # No metadata in normal mode
    metadata_normal = formatter.format_metadata(self.test_metadata_elems, self.test_similarity, debug=False)
    self.assertEqual(metadata_normal, "")

    # Test document end (empty for plain text)
    end = formatter.format_document_end()
    self.assertEqual(end, "")

  def test_formatter_escaping(self):
    """Test that formatters properly handle special characters."""
    # Test content with special characters
    special_content = 'Test with <tag> & "quotes" and \'apostrophes\''

    # XML should escape
    xml_formatter = XMLFormatter()
    xml_result = xml_formatter.format_context_file(special_content, "test.txt")
    self.assertIn('&lt;tag&gt;', xml_result)
    self.assertIn('&amp;', xml_result)
    # Note: xml.sax.saxutils.escape doesn't escape quotes by default, which is fine for XML content

    # JSON should handle via json.dumps
    json_formatter = JSONFormatter()
    json_formatter.format_context_file(special_content, "test.txt")
    json_result = json_formatter.finalize("")
    data = json.loads(json_result)
    self.assertEqual(data['context_files'][0]['content'], special_content)

    # Markdown and plain should preserve original
    md_formatter = MarkdownFormatter()
    md_result = md_formatter.format_context_file(special_content, "test.txt")
    self.assertIn(special_content, md_result)

    plain_formatter = PlainTextFormatter()
    plain_result = plain_formatter.format_context_file(special_content, "test.txt")
    self.assertIn(special_content, plain_result)


if __name__ == '__main__':
  unittest.main()

#fin
