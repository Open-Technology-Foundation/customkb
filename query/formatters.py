"""
Reference formatters for different output formats.

This module provides formatters to convert search results into various formats
like XML, JSON, Markdown, and plain text for feeding to LLMs.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
import json
import xml.sax.saxutils
from utils.logging_config import get_logger

logger = get_logger(__name__)


class ReferenceFormatter(ABC):
  """Abstract base class for reference formatters."""
  
  @abstractmethod
  def format_context_file(self, content: str, filename: str) -> str:
    """Format a context file reference."""
    pass
  
  @abstractmethod
  def format_document_start(self, source: str, sid: int, display_source: str = None) -> str:
    """Format the start of a document context."""
    pass
  
  @abstractmethod
  def format_metadata(self, metadata_elems: List[str], similarity_score: str, debug: bool) -> str:
    """Format metadata section."""
    pass
  
  @abstractmethod
  def format_document_content(self, content: str) -> str:
    """Format document content."""
    pass
  
  @abstractmethod
  def format_document_end(self) -> str:
    """Format the end of a document context."""
    pass
  
  def format_section_header(self, source: str, sid: int, display_source: str = None) -> str:
    """Format a section header within a grouped context (optional override)."""
    return ""  # Default: no additional header for grouped sections
  
  @abstractmethod
  def finalize(self, content: str) -> str:
    """Apply any final formatting to the complete reference string."""
    pass
  
  def needs_document_grouping(self) -> bool:
    """Whether this formatter needs consecutive docs from same source grouped."""
    return True


class XMLFormatter(ReferenceFormatter):
  """Formatter for XML output (current default format)."""
  
  def format_context_file(self, content: str, filename: str) -> str:
    """Format a context file as XML."""
    safe_filename = xml.sax.saxutils.escape(str(filename))
    safe_content = xml.sax.saxutils.escape(str(content))
    return f'<reference src="{safe_filename}">\n{safe_content}\n</reference>\n\n'
  
  def format_document_start(self, source: str, sid: int, display_source: str = None) -> str:
    """Format the start of a document context."""
    src = display_source or source
    # Ensure src is a string (in case it's an int or other type)
    safe_src = xml.sax.saxutils.escape(str(src))
    return f'<context src="{safe_src}:{sid}">\n'
  
  def format_metadata(self, metadata_elems: List[str], similarity_score: str, debug: bool) -> str:
    """Format metadata section as XML."""
    if not metadata_elems and not (similarity_score and debug):
      return ""
    
    parts = []
    if metadata_elems:
      parts.extend(metadata_elems)
    if similarity_score and debug:
      parts.append(similarity_score)
    
    if parts:
      return f'<metadata>\n{chr(10).join(parts)}\n</metadata>\n'
    return ""
  
  def format_document_content(self, content: str) -> str:
    """Format document content - escape XML entities."""
    # Ensure content is a string and escape XML entities
    safe_content = xml.sax.saxutils.escape(str(content))
    return safe_content + "\n"
  
  def format_document_end(self) -> str:
    """Format the end of a document context."""
    return "</context>\n\n"
  
  def finalize(self, content: str) -> str:
    """Wrap content in root element for valid XML."""
    if content:
      return f"<results>\n{content}</results>\n"
    return "<results/>\n"


class JSONFormatter(ReferenceFormatter):
  """Formatter for JSON output."""
  
  def __init__(self):
    self.context_files = []
    self.references = []
    self.current_ref = None
  
  def format_context_file(self, content: str, filename: str) -> str:
    """Store context file for JSON output."""
    self.context_files.append({
      "source": filename,
      "content": content
    })
    return ""  # Build JSON at the end
  
  def format_document_start(self, source: str, sid: int, display_source: str = None) -> str:
    """Start a new reference document."""
    self.current_ref = {
      "source": source,
      "display_source": display_source or source,
      "sid": sid,
      "content": "",
      "metadata": {}
    }
    return ""
  
  def format_metadata(self, metadata_elems: List[str], similarity_score: str, debug: bool) -> str:
    """Store metadata for current reference."""
    if not self.current_ref:
      return ""
    
    # Parse XML meta tags back to dict
    for elem in metadata_elems:
      if '<meta name="' in elem:
        start = elem.find('name="') + 6
        end = elem.find('"', start)
        key = elem[start:end]
        
        start = elem.find('>', end) + 1
        end = elem.find('</meta>', start)
        value = elem[start:end]
        
        # Unescape XML entities
        value = xml.sax.saxutils.unescape(value)
        self.current_ref["metadata"][key] = value
    
    # Add similarity if in debug mode
    if similarity_score and debug:
      if '<meta name="similarity">' in similarity_score:
        start = similarity_score.find('>') + 1
        end = similarity_score.find('</meta>')
        self.current_ref["similarity"] = float(similarity_score[start:end])
    
    return ""
  
  def format_document_content(self, content: str) -> str:
    """Add content to current reference."""
    if self.current_ref:
      # Unescape XML entities
      content = xml.sax.saxutils.unescape(content.strip())
      if self.current_ref["content"]:
        self.current_ref["content"] += "\n" + content
      else:
        self.current_ref["content"] = content
    return ""
  
  def format_document_end(self) -> str:
    """Complete current reference."""
    if self.current_ref:
      self.references.append(self.current_ref)
      self.current_ref = None
    return ""
  
  def finalize(self, content: str) -> str:
    """Build final JSON output."""
    output = {
      "context_files": self.context_files,
      "references": self.references
    }
    return json.dumps(output, indent=2, ensure_ascii=False)
  
  def needs_document_grouping(self) -> bool:
    """JSON formatter handles its own grouping."""
    return False


class MarkdownFormatter(ReferenceFormatter):
  """Formatter for Markdown output."""
  
  def __init__(self):
    self.has_context_files = False
    self.has_references = False
    self.in_document_group = False
  
  def format_context_file(self, content: str, filename: str) -> str:
    """Format a context file as Markdown."""
    header = ""
    if not self.has_context_files:
      header = "## Context Files\n\n"
      self.has_context_files = True
    
    return f"{header}### {filename}\n\n{content}\n\n"
  
  def format_document_start(self, source: str, sid: int, display_source: str = None) -> str:
    """Format the start of a document context."""
    header = ""
    if not self.has_references:
      header = "## References\n\n"
      self.has_references = True
    
    src = display_source or source
    self.in_document_group = True
    return f"{header}### {src}:{sid}\n\n"
  
  def format_metadata(self, metadata_elems: List[str], similarity_score: str, debug: bool) -> str:
    """Format metadata section as Markdown."""
    if not metadata_elems and not (similarity_score and debug):
      return ""
    
    lines = []
    
    # Parse XML meta tags and convert to Markdown
    for elem in metadata_elems:
      if '<meta name="' in elem:
        start = elem.find('name="') + 6
        end = elem.find('"', start)
        key = elem[start:end]
        
        start = elem.find('>', end) + 1
        end = elem.find('</meta>', start)
        value = elem[start:end]
        
        # Unescape XML entities
        value = xml.sax.saxutils.unescape(value)
        
        # Format key nicely
        nice_key = key.replace('_', ' ').title()
        lines.append(f"**{nice_key}:** {value}")
    
    # Add similarity if in debug mode
    if similarity_score and debug:
      if '<meta name="similarity">' in similarity_score:
        start = similarity_score.find('>') + 1
        end = similarity_score.find('</meta>')
        sim_value = similarity_score[start:end]
        lines.append(f"**Similarity:** {sim_value}")
    
    if lines:
      return '\n'.join(lines) + "\n\n"
    return ""
  
  def format_document_content(self, content: str) -> str:
    """Format document content - unescape XML."""
    unescaped = xml.sax.saxutils.unescape(content.strip())
    return unescaped + "\n\n"
  
  def format_document_end(self) -> str:
    """Add separator between documents."""
    self.in_document_group = False
    return "---\n\n"
  
  def format_section_header(self, source: str, sid: int, display_source: str = None) -> str:
    """Format a section header within a grouped context."""
    if self.in_document_group:
      src = display_source or source
      return f"#### Section {sid}\n\n"
    return ""
  
  def finalize(self, content: str) -> str:
    """Remove trailing separator."""
    if content.endswith("---\n\n"):
      content = content[:-5]
    return content


class PlainTextFormatter(ReferenceFormatter):
  """Formatter for plain text output (most compact)."""
  
  def format_context_file(self, content: str, filename: str) -> str:
    """Format a context file as plain text."""
    return f"=== Context File: {filename} ===\n{content}\n\n"
  
  def format_document_start(self, source: str, sid: int, display_source: str = None) -> str:
    """Format the start of a document context."""
    src = display_source or source
    return f"--- {src}:{sid} ---\n"
  
  def format_metadata(self, metadata_elems: List[str], similarity_score: str, debug: bool) -> str:
    """Format metadata inline for plain text."""
    if not debug:
      return ""  # Skip metadata in plain text unless debugging
    
    lines = []
    
    # Parse XML meta tags
    for elem in metadata_elems:
      if '<meta name="' in elem:
        start = elem.find('name="') + 6
        end = elem.find('"', start)
        key = elem[start:end]
        
        start = elem.find('>', end) + 1
        end = elem.find('</meta>', start)
        value = elem[start:end]
        
        # Unescape XML entities
        value = xml.sax.saxutils.unescape(value)
        lines.append(f"[{key}: {value}]")
    
    # Add similarity if in debug mode
    if similarity_score and debug:
      if '<meta name="similarity">' in similarity_score:
        start = similarity_score.find('>') + 1
        end = similarity_score.find('</meta>')
        sim_value = similarity_score[start:end]
        lines.append(f"[similarity: {sim_value}]")
    
    if lines:
      return ' '.join(lines) + "\n"
    return ""
  
  def format_document_content(self, content: str) -> str:
    """Format document content - unescape XML."""
    unescaped = xml.sax.saxutils.unescape(content.strip())
    return unescaped + "\n\n"
  
  def format_document_end(self) -> str:
    """Plain text doesn't need end markers."""
    return ""
  
  def finalize(self, content: str) -> str:
    """Clean up any extra whitespace."""
    return content.rstrip() + "\n"


def get_formatter(format_type: str) -> ReferenceFormatter:
  """
  Factory function to get appropriate formatter.
  
  Args:
      format_type: One of 'xml', 'json', 'markdown', 'plain'
      
  Returns:
      ReferenceFormatter instance
      
  Raises:
      ValueError: If format_type is not supported
  """
  formatters = {
    'xml': XMLFormatter,
    'json': JSONFormatter,
    'markdown': MarkdownFormatter,
    'md': MarkdownFormatter,  # Alias
    'plain': PlainTextFormatter,
    'text': PlainTextFormatter  # Alias
  }
  
  format_lower = format_type.lower()
  if format_lower not in formatters:
    raise ValueError(f"Unsupported format type: {format_type}. Supported: {list(formatters.keys())}")
  
  return formatters[format_lower]()


def format_references(references: List[Tuple], format_type: str = 'xml',
                     context_files: Optional[List[str]] = None,
                     debug: bool = False) -> str:
  """
  Format references using the appropriate formatter.
  
  This is a convenience function that creates a formatter and formats references.
  
  Args:
      references: List of reference tuples (typically containing source, sid, content, metadata, etc.)
      format_type: Output format ('xml', 'json', 'markdown', 'plain')
      context_files: Optional list of context file contents
      debug: Whether to include debug information
      
  Returns:
      Formatted reference string
  """
  formatter = get_formatter(format_type)
  content_parts = []
  
  # Add context files if provided
  if context_files:
    for file_data in context_files:
      # Handle both tuple format (content, filename) and plain string
      if isinstance(file_data, tuple):
        content, filename = file_data
      else:
        content = file_data
        filename = f"context_file.txt"
      content_parts.append(formatter.format_context_file(content, filename))
  
  # Track document grouping
  old_src = ''
  old_sid = 0
  end_context = False
  
  # Format each reference with intelligent grouping
  for ref in references:
    # Unpack reference tuple from process_reference_batch
    # Structure: [rid, rsrc, rsid, originaltext, distance, metadata]
    if len(ref) >= 4:
      doc_id = ref[0]
      source = ref[1]  # source document path
      sid = int(ref[2]) if ref[2] is not None else 0
      content = ref[3]  # originaltext - the actual content!
      
      # Get distance (index 4) and metadata (index 5)
      distance = ref[4] if len(ref) > 4 else 0.0
      metadata_str = ref[5] if len(ref) > 5 else None
      
      # Parse metadata
      metadata_elems = []
      if metadata_str:
        # Try to parse as JSON
        try:
          import json
          metadata_dict = json.loads(metadata_str)
          # Format metadata as XML meta tags (filtered by debug mode)
          for key, value in metadata_dict.items():
            if value:
              # In debug mode, show all metadata; otherwise just heading and section_type
              if debug or key in ['heading', 'section_type']:
                safe_value = xml.sax.saxutils.escape(str(value))
                metadata_elems.append(f'<meta name="{key}">{safe_value}</meta>')
        except (json.JSONDecodeError, TypeError):
          # If not JSON, treat as plain text
          if metadata_str:
            metadata_elems = [str(metadata_str)]
      
      # Calculate similarity from distance (1 - normalized distance)
      similarity_str = ''
      if distance is not None and debug:
        try:
          # Convert distance to similarity score for display
          similarity = 1.0 / (1.0 + float(distance))
          similarity_str = f'<meta name="similarity">{similarity:.4f}</meta>'
        except (ValueError, TypeError):
          pass
      
      # Handle document grouping if formatter needs it
      if formatter.needs_document_grouping():
        # Check if this is a new document or non-consecutive section
        if source != old_src or sid != (old_sid + 1):
          # Close previous context if needed
          if end_context:
            content_parts.append(formatter.format_document_end())
          
          # Start new context
          content_parts.append(formatter.format_document_start(source, sid))
          
          # Add metadata for the first section
          if metadata_elems or similarity_str:
            content_parts.append(formatter.format_metadata(metadata_elems, similarity_str, debug))
          
          end_context = True
        else:
          # Consecutive section in same document - don't add metadata inline
          # The metadata was already added for the first section of this group
          pass
      else:
        # For formatters that don't need grouping (like JSON)
        content_parts.append(formatter.format_document_start(source, sid))
        if metadata_elems or similarity_str:
          content_parts.append(formatter.format_metadata(metadata_elems, similarity_str, debug))
      
      # Update tracking variables
      old_src = source
      old_sid = sid
      
      # Add content
      content_parts.append(formatter.format_document_content(content))
      
      # For non-grouping formatters, close each document immediately
      if not formatter.needs_document_grouping():
        content_parts.append(formatter.format_document_end())
  
  # Close final context if needed for grouping formatters
  if end_context and formatter.needs_document_grouping():
    content_parts.append(formatter.format_document_end())
  
  # Join all parts and finalize
  full_content = ''.join(content_parts)
  return formatter.finalize(full_content)


#fin