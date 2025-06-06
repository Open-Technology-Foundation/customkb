#!/usr/bin/env python
"""
Text processing utilities for CustomKB.
Provides text cleaning, tokenization, file handling, and environment variable management.
"""

import os
import re
import glob
from typing import List, Tuple, Optional, Any, Set

from nltk.tokenize import word_tokenize
from utils.logging_utils import get_logger

# This will be initialized in db_manager.py when importing spacy
nlp = None

logger = get_logger(__name__)

def clean_text(text: str, stop_words: Optional[Set[str]] = None) -> str:
  """
  Basic text cleaning for embeddings.
  
  Args:
      text: The text to clean.
      stop_words: Optional set of stop words to remove.

  Returns:
      The cleaned text with lowercase conversion, HTML removal, and optional stopword filtering.
  """
  HTML_TAG_RE = re.compile(r'<[^>]+>')
  NON_WORD_RE = re.compile(r'\W+')

  text = text.lower()
  text = HTML_TAG_RE.sub('', text)
  text = NON_WORD_RE.sub(' ', text).strip()

  if stop_words:
    filtered_text = ' '.join(
      w for w in word_tokenize(text) if w not in stop_words
    )
    return filtered_text.strip()

  return text.strip()

def enhanced_clean_text(text: str, stop_words: Optional[Set[str]] = None, 
                       lemmatizer: Optional[Any] = None) -> str:
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
          placeholder = f"__ENTITY_{i}_{ent.label__}__"
          text = text.replace(ent.text, placeholder)
          entities.append((placeholder, ent.text))
    except Exception:
      pass
      
  # Clean text while preserving sentence structure
  text = text.lower()
  
  # Remove HTML tags
  HTML_TAG_RE = re.compile(r'<[^>]+>')
  text = HTML_TAG_RE.sub('', text)
  
  # Keep meaningful punctuation
  text = re.sub(r'[^\w\s\.\!\?\:\;\-]', ' ', text)
  
  # Normalize whitespace
  text = re.sub(r'\s+', ' ', text).strip()
  
  # Apply lemmatization and stopword removal
  if lemmatizer:
    tokens = word_tokenize(text)
    if stop_words:
      tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words
                and not all(c in '.,!?:;-' for c in w)]
    else:
      tokens = [lemmatizer.lemmatize(w) for w in tokens 
                if not all(c in '.,!?:;-' for c in w)]
    
    text = ' '.join(tokens)
  elif stop_words:
    text = ' '.join(w for w in word_tokenize(text) if w not in stop_words
                   and not all(c in '.,!?:;-' for c in w))
  
  # Restore entities, URLs and emails
  for placeholder, original in entities:
    text = text.replace(placeholder, original.lower())
  
  for i, url in enumerate(urls):
    text = text.replace(f"__URL_{i}__", url)
  
  for i, email in enumerate(emails):
    text = text.replace(f"__EMAIL_{i}__", email)
    
  return text.strip()

def get_files(pathspec: str) -> List[str]:
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

def split_filepath(filepath: str, *, adddir: bool = True, realpath: bool = True) -> Tuple[str, str, str, str]:
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

def find_file(filename: str, search_path: str = './', followsymlinks: bool = True) -> Optional[str]:
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

  for root, dirs, files in os.walk(search_path, followlinks=followsymlinks):
    if filename in files:
      file_path = os.path.join(root, filename)
      return os.path.realpath(file_path)

  return None

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

#fin
