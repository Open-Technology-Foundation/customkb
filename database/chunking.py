#!/usr/bin/env python
"""
Text chunking and splitting functionality for CustomKB.

This module handles breaking down documents into manageable chunks
for processing and embedding.
"""

from typing import List, Dict, Any, Optional
from pathlib import Path

from langchain_text_splitters import (
  RecursiveCharacterTextSplitter,
  MarkdownTextSplitter,
  Language
)

from utils.logging_config import get_logger
from utils.exceptions import ChunkingError, ProcessingError

logger = get_logger(__name__)


# File type detection patterns
FILE_TYPE_PATTERNS = {
  'markdown': ['.md', '.markdown', '.mdown', '.mkd'],
  'html': ['.html', '.htm', '.xhtml'],
  'code': ['.py', '.js', '.ts', '.java', '.cpp', '.c', '.cs', '.go', '.rs', 
           '.rb', '.php', '.swift', '.kt', '.scala', '.r', '.m', '.sh'],
  'json': ['.json', '.jsonl'],
  'yaml': ['.yaml', '.yml'],
  'xml': ['.xml', '.svg'],
  'config': ['.ini', '.cfg', '.conf', '.config', '.toml'],
  'text': ['.txt', '.text', '.log', '.csv', '.tsv']
}

# Language-specific splitters
LANGUAGE_MAP = {
  '.py': Language.PYTHON,
  '.js': Language.JS,
  '.ts': Language.TS,
  '.java': Language.JAVA,
  '.cpp': Language.CPP,
  '.c': Language.C,
  '.cs': Language.CSHARP,
  '.go': Language.GO,
  '.rs': Language.RUST,
  '.rb': Language.RUBY,
  '.php': Language.PHP,
  '.swift': Language.SWIFT,
  '.kt': Language.KOTLIN,
  '.scala': Language.SCALA,
  '.html': Language.HTML,
  '.md': Language.MARKDOWN,
  '.xml': Language.HTML,  # Use HTML splitter for XML
}


def detect_file_type(filename: str) -> str:
  """
  Detect file type based on extension.
  
  Args:
      filename: Name or path of the file
      
  Returns:
      File type string ('markdown', 'code', 'html', 'text', etc.)
  """
  file_path = Path(filename)
  extension = file_path.suffix.lower()
  
  # Check each file type category
  for file_type, extensions in FILE_TYPE_PATTERNS.items():
    if extension in extensions:
      logger.debug(f"Detected file type '{file_type}' for {filename}")
      return file_type
  
  # Default to text for unknown extensions
  logger.debug(f"Unknown extension {extension}, defaulting to 'text'")
  return 'text'


def init_text_splitter(kb: Any, file_type: str = 'text') -> Any:
  """
  Initialize appropriate text splitter based on file type.
  
  Args:
      kb: KnowledgeBase configuration
      file_type: Type of file being processed
      
  Returns:
      Configured text splitter instance
      
  Raises:
      ChunkingError: If splitter initialization fails
  """
  try:
    chunk_size = getattr(kb, 'chunk_size', 500)
    chunk_overlap = getattr(kb, 'chunk_overlap', 50)
    
    if file_type == 'markdown':
      logger.debug(f"Initializing Markdown splitter (chunk_size={chunk_size})")
      return MarkdownTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
      )
    
    elif file_type == 'code':
      # For code files, use language-specific splitter if available
      logger.debug(f"Initializing code splitter (chunk_size={chunk_size})")
      return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
        length_function=len
      )
    
    elif file_type == 'html':
      logger.debug(f"Initializing HTML splitter (chunk_size={chunk_size})")
      return RecursiveCharacterTextSplitter.from_language(
        language=Language.HTML,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
      )
    
    elif file_type == 'json' or file_type == 'yaml':
      # For structured data, preserve structure where possible
      logger.debug(f"Initializing structured data splitter (chunk_size={chunk_size})")
      return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ",", " ", ""],
        length_function=len
      )
    
    else:
      # Default text splitter
      logger.debug(f"Initializing default text splitter (chunk_size={chunk_size})")
      return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len
      )
  
  except Exception as e:
    logger.error(f"Failed to initialize text splitter: {e}")
    raise ChunkingError(f"Text splitter initialization failed: {e}") from e


def get_language_specific_splitter(file_path: str, kb: Any) -> Optional[Any]:
  """
  Get a language-specific code splitter if available.
  
  Args:
      file_path: Path to the file
      kb: KnowledgeBase configuration
      
  Returns:
      Language-specific splitter or None
  """
  extension = Path(file_path).suffix.lower()
  
  if extension in LANGUAGE_MAP:
    try:
      language = LANGUAGE_MAP[extension]
      chunk_size = getattr(kb, 'chunk_size', 500)
      chunk_overlap = getattr(kb, 'chunk_overlap', 50)
      
      logger.debug(f"Creating {language} splitter for {extension}")
      
      return RecursiveCharacterTextSplitter.from_language(
        language=language,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
      )
    except Exception as e:
      logger.warning(f"Failed to create language-specific splitter: {e}")
      return None
  
  return None


def split_text(text: str, splitter: Any, metadata: Optional[Dict] = None) -> List[Dict]:
  """
  Split text into chunks using the provided splitter.
  
  Args:
      text: Text to split
      splitter: Text splitter instance
      metadata: Optional metadata to attach to chunks
      
  Returns:
      List of chunk dictionaries with text and metadata
      
  Raises:
      ChunkingError: If splitting fails
  """
  try:
    # Split the text
    chunks = splitter.split_text(text)
    
    if not chunks:
      logger.warning("Text splitter returned no chunks")
      return []
    
    # Create chunk dictionaries with metadata
    result = []
    for i, chunk_text in enumerate(chunks):
      chunk_dict = {
        'text': chunk_text,
        'chunk_index': i,
        'total_chunks': len(chunks),
        'char_count': len(chunk_text)
      }
      
      # Add provided metadata
      if metadata:
        chunk_dict.update(metadata)
      
      result.append(chunk_dict)
    
    logger.debug(f"Split text into {len(chunks)} chunks")
    return result
    
  except Exception as e:
    logger.error(f"Text splitting failed: {e}")
    raise ChunkingError(f"Failed to split text: {e}") from e


def calculate_chunk_statistics(chunks: List[Dict]) -> Dict[str, Any]:
  """
  Calculate statistics about chunks.
  
  Args:
      chunks: List of chunk dictionaries
      
  Returns:
      Dictionary with statistics
  """
  if not chunks:
    return {
      'total_chunks': 0,
      'total_chars': 0,
      'avg_chunk_size': 0,
      'min_chunk_size': 0,
      'max_chunk_size': 0
    }
  
  sizes = [len(chunk.get('text', '')) for chunk in chunks]
  
  return {
    'total_chunks': len(chunks),
    'total_chars': sum(sizes),
    'avg_chunk_size': sum(sizes) / len(sizes) if sizes else 0,
    'min_chunk_size': min(sizes) if sizes else 0,
    'max_chunk_size': max(sizes) if sizes else 0
  }


def optimize_chunk_size(text_length: int, target_chunks: int = 10) -> int:
  """
  Calculate optimal chunk size based on text length.
  
  Args:
      text_length: Total length of text
      target_chunks: Desired number of chunks
      
  Returns:
      Recommended chunk size
  """
  if text_length <= 0:
    return 500  # Default
  
  # Calculate base chunk size
  base_size = text_length // target_chunks
  
  # Apply constraints
  min_size = 100
  max_size = 2000
  
  # Round to nearest 50
  chunk_size = max(min_size, min(max_size, base_size))
  chunk_size = (chunk_size // 50) * 50
  
  logger.debug(f"Optimized chunk size: {chunk_size} for text length {text_length}")
  return chunk_size


def merge_small_chunks(chunks: List[Dict], min_size: int = 100) -> List[Dict]:
  """
  Merge chunks that are too small.
  
  Args:
      chunks: List of chunk dictionaries
      min_size: Minimum chunk size
      
  Returns:
      List of merged chunks
  """
  if not chunks:
    return chunks
  
  merged = []
  current = None
  
  for chunk in chunks:
    chunk_text = chunk.get('text', '')
    chunk_size = len(chunk_text)
    
    if current is None:
      # Start with first chunk
      current = chunk.copy()
    elif len(current.get('text', '')) < min_size and chunk_size < min_size:
      # Both current and next are small, merge them
      current['text'] += '\n' + chunk_text
      current['char_count'] = len(current['text'])
    else:
      # Current is large enough or next chunk is large, save current
      merged.append(current)
      current = chunk.copy()
  
  # Add last chunk
  if current:
    merged.append(current)
  
  # Update chunk indices
  for i, chunk in enumerate(merged):
    chunk['chunk_index'] = i
    chunk['total_chunks'] = len(merged)
  
  if len(merged) < len(chunks):
    logger.debug(f"Merged {len(chunks)} chunks into {len(merged)}")
  
  return merged


def validate_chunks(chunks: List[Dict], kb: Any) -> bool:
  """
  Validate that chunks meet requirements.
  
  Args:
      chunks: List of chunk dictionaries
      kb: KnowledgeBase configuration
      
  Returns:
      True if chunks are valid
      
  Raises:
      ProcessingError: If validation fails
  """
  if not chunks:
    raise ProcessingError("No chunks to validate")
  
  max_chunk_size = getattr(kb, 'max_chunk_size', 2000)
  min_chunk_size = getattr(kb, 'min_chunk_size', 50)
  
  for i, chunk in enumerate(chunks):
    text = chunk.get('text', '')
    
    if not text:
      raise ProcessingError(f"Chunk {i} has no text")
    
    if len(text) > max_chunk_size:
      raise ProcessingError(
        f"Chunk {i} exceeds maximum size: {len(text)} > {max_chunk_size}"
      )
    
    if len(text) < min_chunk_size and i < len(chunks) - 1:
      # Allow last chunk to be smaller
      logger.warning(f"Chunk {i} below minimum size: {len(text)} < {min_chunk_size}")
  
  return True


#fin