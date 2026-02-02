#!/usr/bin/env python
"""
Text processing utilities for CustomKB.
Provides text cleaning, tokenization, file handling, and environment variable management.
"""

import glob
import os
import re
from typing import Any

from utils.logging_config import get_logger

# This will be initialized in db_manager.py when importing spacy
nlp = None

logger = get_logger(__name__)

def clean_text(text: str, stop_words: set[str] | None = None) -> str:
  """
  Basic text cleaning for embeddings.

  Args:
      text: The text to clean.
      stop_words: Optional set of stop words to remove.

  Returns:
      The cleaned text with lowercase conversion, HTML removal, and optional stopword filtering.
  """
  html_tag_re = re.compile(r'<[^>]+>')
  non_word_re = re.compile(r'\W+')

  text = text.lower()
  text = html_tag_re.sub('', text)
  text = non_word_re.sub(' ', text).strip()

  if stop_words:
    # Lazy import to avoid loading NLTK unless needed
    from nltk.tokenize import word_tokenize
    filtered_text = ' '.join(
      w for w in word_tokenize(text) if w not in stop_words
    )
    return filtered_text.strip()

  return text.strip()

def enhanced_clean_text(text: str, stop_words: set[str] | None = None,
                       lemmatizer: Any | None = None) -> str:
  """
  Advanced text cleaning that preserves semantic structure and named entities.

  Args:
      text: The text to clean.
      stop_words: Optional set of stop words to remove.
      lemmatizer: Optional lemmatizer for word normalization.

  Returns:
      Cleaned text optimized for embeddings while preserving meaningful elements.
  """
  global nlp

  # Preserve URLs and email addresses
  url_pattern = re.compile(r'https?://\S+|www\.\S+')
  email_pattern = re.compile(r'\S+@\S+\.\S+')

  urls = url_pattern.findall(text)
  emails = email_pattern.findall(text)

  for i, url in enumerate(urls):
    text = text.replace(url, f"__URL_{i}__")

  for i, email in enumerate(emails):
    text = text.replace(email, f"__EMAIL_{i}__")

  # Preserve named entities if spaCy is available
  entities = []
  if nlp is not None:
    try:
      doc = nlp(text)
      for i, ent in enumerate(doc.ents):
        if ent.label_ in ["PERSON", "ORG", "GPE", "LOC", "PRODUCT", "WORK_OF_ART"]:
          placeholder = f"__ENTITY_{i}_{ent.label_}__"
          text = text.replace(ent.text, placeholder)
          entities.append((placeholder, ent.text))
    except (AttributeError, RuntimeError, ValueError):
      pass

  # Clean text while preserving sentence structure
  text = text.lower()

  # Remove HTML tags
  html_tag_re = re.compile(r'<[^>]+>')
  text = html_tag_re.sub('', text)

  # Keep meaningful punctuation
  text = re.sub(r'[^\w\s\.\!\?\:\;\-]', ' ', text)

  # Normalize whitespace
  text = re.sub(r'\s+', ' ', text).strip()

  # Apply lemmatization and stopword removal
  if lemmatizer:
    # Lazy import to avoid loading NLTK unless needed
    from nltk.tokenize import word_tokenize
    tokens = word_tokenize(text)
    if stop_words:
      tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words
                and not all(c in '.,!?:;-' for c in w)]
    else:
      tokens = [lemmatizer.lemmatize(w) for w in tokens
                if not all(c in '.,!?:;-' for c in w)]

    text = ' '.join(tokens)
  elif stop_words:
    # Lazy import to avoid loading NLTK unless needed
    from nltk.tokenize import word_tokenize
    text = ' '.join(w for w in word_tokenize(text) if w not in stop_words
                   and not all(c in '.,!?:;-' for c in w))

  # Restore entities, URLs and emails (placeholders are now lowercase)
  for placeholder, original in entities:
    text = text.replace(placeholder.lower(), original.lower())

  for i, url in enumerate(urls):
    text = text.replace(f"__url_{i}__", url)

  for i, email in enumerate(emails):
    text = text.replace(f"__email_{i}__", email)

  return text.strip()

def get_files(pathspec: str) -> list[str]:
  """
  Get file paths matching a specification, with directory recursion support.

  Args:
      pathspec: A file path, directory path, or glob pattern.

  Returns:
      A sorted list of matching file paths, excluding directories.
  """
  if os.path.isdir(pathspec):
    pathspec = f"{pathspec}/**"

  files = sorted(glob.glob(pathspec, recursive=True))
  return [f for f in files if not os.path.isdir(f)]

def split_filepath(filepath: str, *, adddir: bool = True, realpath: bool = True) -> tuple[str, str, str, str]:
  """
  Split a file path into directory, basename, extension, and full path.

  Args:
      filepath: The file path to split.
      adddir: Whether to add the current directory if no directory is specified.
      realpath: Whether to convert to the real path.

  Returns:
      (directory, basename, extension, fully_qualified_path)
  """
  if realpath:
    filepath = os.path.realpath(filepath)

  directory, filename = os.path.split(filepath)
  basename, extension = os.path.splitext(filename)

  if adddir and not directory:
    directory = os.getcwd()

  fqfn = f'{directory}/{basename}{extension}'
  return directory, basename, extension, fqfn

def find_file(filename: str, search_path: str = './', followsymlinks: bool = True) -> str | None:
  """
  Search for a file in a directory tree.

  Args:
      filename: The name of the file to find.
      search_path: The path to search in.
      followsymlinks: Whether to follow symbolic links.

  Returns:
      Full path to the file if found, None otherwise.
  """
  if '/' in filename:
    logger.error(f"Warning: Invalid {filename=}. The filename cannot contain '/' characters.")
    return None

  for root, _dirs, files in os.walk(search_path, followlinks=followsymlinks):
    if filename in files:
      file_path = os.path.join(root, filename)
      return os.path.realpath(file_path)

  return None

def tokenize_for_bm25(text: str, language: str = 'en') -> tuple[str, int]:
  """
  Tokenize text specifically for BM25 indexing.
  Uses different processing than vector embeddings to optimize for keyword matching.

  Args:
      text: Input text to tokenize.
      language: ISO 639-1 language code (e.g., 'en', 'fr') or full name.

  Returns:
      Tuple of (space-separated tokens string, document length).
  """
  # Import here to avoid circular imports
  from database.db_manager import get_full_language_name, get_iso_code

  # Convert to lowercase but preserve acronyms and important terms
  text = text.lower()

  # Keep alphanumeric, hyphens (for compound words), periods (for decimals/domains)
  # More conservative cleaning than vector processing
  text = re.sub(r'[^\w\s\-\.]', ' ', text)

  # Convert language to appropriate format for NLTK
  try:
    iso_code = get_iso_code(language)
    full_language = get_full_language_name(iso_code)
  except ValueError:
    # If language not recognized, default to English
    iso_code = 'en'
    full_language = 'english'

  # Tokenize using NLTK (lazy import to avoid loading unless needed)
  try:
    from nltk.tokenize import word_tokenize
    tokens = word_tokenize(text, language=full_language)
  except (LookupError, FileNotFoundError, OSError, AttributeError, ImportError):
    # Fallback to basic split when NLTK fails (test environments or missing data)
    tokens = text.lower().split()

  # Filter tokens: remove single chars but keep numbers
  tokens = [t for t in tokens if len(t) > 1 or t.isdigit()]

  # Light stopword removal (less aggressive than vector processing)
  # Keep important terms that might be relevant for exact matching
  if iso_code == 'en':
    try:
      # Only remove very common words, preserve domain-specific terms
      essential_stops = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
      tokens = [t for t in tokens if t not in essential_stops]
    except LookupError:
      # If stopwords not available, continue without filtering
      pass

  # Remove empty tokens and duplicates while preserving order
  seen = set()
  filtered_tokens = []
  for token in tokens:
    if token and token not in seen:
      filtered_tokens.append(token)
      seen.add(token)

  return ' '.join(filtered_tokens), len(filtered_tokens)

def get_env(var_name: str, default: Any, cast_type: Any = str) -> Any:
  """
  Get environment variable with type casting and default value.

  Args:
      var_name: The name of the environment variable.
      default: The default value to use if the variable is not set.
      cast_type: The type to cast the value to.

  Returns:
      The environment variable value cast to the specified type, or the default value.
  """
  value = os.getenv(var_name)
  if value is None:
    return default

  try:
    return cast_type(value)
  except (ValueError, TypeError):
    return default

def detect_file_encoding(file_path: str, sample_size: int = 65536) -> str:
  """
  Detect the encoding of a text file using charset-normalizer.

  Args:
      file_path: Path to the file to analyze.
      sample_size: Number of bytes to read for detection (default 64KB).

  Returns:
      Detected encoding name (e.g., 'utf-8', 'windows-1252'), or 'utf-8' if detection fails.
  """
  try:
    from charset_normalizer import from_path

    # Read sample from file for detection
    result = from_path(file_path, steps=1)

    if result and result.best():
      encoding = result.best().encoding
      if encoding:
        return encoding.lower()

    # Fallback to utf-8 if detection fails
    return 'utf-8'

  except (FileNotFoundError, OSError, ImportError) as e:
    # Log error but only once to avoid spam
    logger.error(f"Encoding detection failed for {file_path}: {e}")
    return 'utf-8'

def read_text_file(file_path: str, config: dict | None = None) -> str:
  """
  Read a text file with automatic encoding detection.

  Args:
      file_path: Path to the text file to read.
      config: Optional configuration dict with encoding settings:
          - auto_detect_encoding: Enable/disable auto-detection (default: True)
          - default_encoding: Fallback encoding (default: 'utf-8')
          - encoding_fallbacks: List of encodings to try (default: ['utf-8', 'windows-1252', 'latin-1', 'cp1252'])

  Returns:
      The file contents as a string.

  Raises:
      FileNotFoundError: If file doesn't exist.
      Exception: If file cannot be read with any encoding.
  """
  # Get config settings
  if config is None:
    config = {}

  auto_detect = config.get('auto_detect_encoding', True)
  default_encoding = config.get('default_encoding', 'utf-8')
  fallback_encodings = config.get('encoding_fallbacks', ['utf-8', 'windows-1252', 'latin-1', 'cp1252'])

  # Try auto-detection first if enabled
  if auto_detect:
    try:
      detected_encoding = detect_file_encoding(file_path)
      with open(file_path, encoding=detected_encoding) as f:
        return f.read()
    except (UnicodeDecodeError, OSError, FileNotFoundError) as e:
      # Only log if detection was explicitly requested
      logger.error(f"Failed to read {file_path} with detected encoding: {e}")

  # Try fallback encodings
  for encoding in fallback_encodings:
    try:
      with open(file_path, encoding=encoding) as f:
        return f.read()
    except (UnicodeDecodeError, LookupError):
      continue

  # Last resort: read with default encoding and error handling
  try:
    with open(file_path, encoding=default_encoding, errors='replace') as f:
      content = f.read()
      logger.error(f"File {file_path} read with errors='replace' using {default_encoding}")
      return content
  except (FileNotFoundError, OSError, PermissionError) as e:
    logger.error(f"Failed to read {file_path}: {e}")
    raise

#fin
