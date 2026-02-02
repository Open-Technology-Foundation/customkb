#!/usr/bin/env python
"""
Text chunking and splitting functionality for CustomKB.

This module handles breaking down documents into manageable chunks
for processing and embedding.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

from langchain_text_splitters import Language, MarkdownTextSplitter, RecursiveCharacterTextSplitter

from utils.exceptions import ChunkingError, ProcessingError
from utils.logging_config import get_logger

logger = get_logger(__name__)

# Cache for tiktoken encodings
_tiktoken_encodings: dict[str, Any] = {}


def get_token_counter(encoding_name: str = "cl100k_base") -> Callable[[str], int]:
  """
  Get a token counting function using tiktoken.

  Uses cl100k_base encoding by default, which is used by:
  - text-embedding-ada-002
  - text-embedding-3-small
  - text-embedding-3-large
  - gpt-4, gpt-3.5-turbo

  Args:
      encoding_name: Tiktoken encoding name (default: cl100k_base)

  Returns:
      Function that counts tokens in a string
  """
  global _tiktoken_encodings

  # Return cached encoding if available
  if encoding_name in _tiktoken_encodings:
    encoding = _tiktoken_encodings[encoding_name]
    return lambda text: len(encoding.encode(text))

  try:
    import tiktoken
    encoding = tiktoken.get_encoding(encoding_name)
    _tiktoken_encodings[encoding_name] = encoding
    logger.debug(f"Initialized tiktoken encoding: {encoding_name}")
    return lambda text: len(encoding.encode(text))
  except ImportError:
    logger.warning("tiktoken not installed, falling back to word-based estimation")
    # Fallback: ~1.3 tokens per word is a reasonable approximation
    return lambda text: int(len(text.split()) * 1.3)
  except (KeyError, ValueError) as e:
    logger.warning(f"Failed to load tiktoken encoding {encoding_name}: {e}")
    return lambda text: int(len(text.split()) * 1.3)


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

  Uses tiktoken for accurate token counting instead of character length.

  Args:
      kb: KnowledgeBase configuration
      file_type: Type of file being processed

  Returns:
      Configured text splitter instance

  Raises:
      ChunkingError: If splitter initialization fails
  """
  try:
    # Use canonical configuration parameters
    chunk_size = getattr(kb, 'db_max_tokens', 200)
    max_overlap = getattr(kb, 'max_chunk_overlap', 100)
    min_tokens = getattr(kb, 'db_min_tokens', 100)
    chunk_overlap = min(max_overlap, min_tokens // 2)

    # Get token counter (uses tiktoken with fallback)
    token_counter = get_token_counter()

    if file_type == 'markdown':
      logger.debug(f"Initializing Markdown splitter (chunk_size={chunk_size} tokens)")
      return MarkdownTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=token_counter
      )

    elif file_type == 'code':
      # For code files, use language-specific splitter if available
      logger.debug(f"Initializing code splitter (chunk_size={chunk_size} tokens)")
      return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
        length_function=token_counter
      )

    elif file_type == 'html':
      logger.debug(f"Initializing HTML splitter (chunk_size={chunk_size} tokens)")
      return RecursiveCharacterTextSplitter.from_language(
        language=Language.HTML,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=token_counter
      )

    elif file_type == 'json' or file_type == 'yaml':
      # For structured data, preserve structure where possible
      logger.debug(f"Initializing structured data splitter (chunk_size={chunk_size} tokens)")
      return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ",", " ", ""],
        length_function=token_counter
      )

    else:
      # Default text splitter
      logger.debug(f"Initializing default text splitter (chunk_size={chunk_size} tokens)")
      return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=token_counter
      )

  except (ValueError, KeyError, ImportError, AttributeError, TypeError) as e:
    logger.error(f"Failed to initialize text splitter: {e}")
    raise ChunkingError(f"Text splitter initialization failed: {e}") from e


def get_language_specific_splitter(file_path: str, kb: Any) -> Any | None:
  """
  Get a language-specific code splitter if available.

  Uses tiktoken for accurate token counting.

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
      # Use canonical configuration parameters
      chunk_size = getattr(kb, 'db_max_tokens', 200)
      max_overlap = getattr(kb, 'max_chunk_overlap', 100)
      min_tokens = getattr(kb, 'db_min_tokens', 100)
      chunk_overlap = min(max_overlap, min_tokens // 2)

      # Get token counter (uses tiktoken with fallback)
      token_counter = get_token_counter()

      logger.debug(f"Creating {language} splitter for {extension} (chunk_size={chunk_size} tokens)")

      return RecursiveCharacterTextSplitter.from_language(
        language=language,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=token_counter
      )
    except (ValueError, KeyError, AttributeError) as e:
      logger.warning(f"Failed to create language-specific splitter: {e}")
      return None

  return None


def split_text(text: str, splitter: Any, metadata: dict | None = None) -> list[dict]:
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

  except (ValueError, TypeError, AttributeError, RuntimeError) as e:
    logger.error(f"Text splitting failed: {e}")
    raise ChunkingError(f"Failed to split text: {e}") from e


def calculate_chunk_statistics(chunks: list[dict]) -> dict[str, Any]:
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


def merge_small_chunks(chunks: list[dict], min_size: int = 100) -> list[dict]:
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


def validate_chunks(chunks: list[dict], kb: Any) -> bool:
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
