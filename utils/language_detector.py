#!/usr/bin/env python
"""
Language detection utilities for CustomKB.

Provides automatic language detection for documents with caching
and fallback mechanisms.
"""

import logging
import os
from pathlib import Path

try:
  from langdetect import LangDetectException, detect_langs
  HAS_LANGDETECT = True
except ImportError:
  HAS_LANGDETECT = False

from database.db_manager import get_iso_code
from utils.text_utils import read_text_file

logger = logging.getLogger(__name__)

# Language detection cache to avoid re-detection
_detection_cache: dict[str, str] = {}

# Map langdetect language codes to our ISO codes
# langdetect uses some different codes than ISO 639-1
LANGDETECT_TO_ISO = {
  'zh-cn': 'zh',
  'zh-tw': 'zh',
  # Most others match ISO 639-1
}

def detect_file_language(
    file_path: str,
    sample_size: int = 3072,
    min_confidence: float = 0.95,
    fallback_language: str = 'en'
) -> tuple[str, float]:
  """
  Detect the language of a file by sampling its content.

  Args:
      file_path: Path to the file to analyze.
      sample_size: Number of bytes to sample from file start.
      min_confidence: Minimum confidence threshold (0-1).
      fallback_language: Language to use if detection fails or confidence is low.

  Returns:
      Tuple of (ISO language code, confidence score).
  """
  if not HAS_LANGDETECT:
    logger.warning("langdetect not installed, using fallback language")
    return fallback_language, 0.0

  # Check cache first
  cache_key = os.path.abspath(file_path)
  if cache_key in _detection_cache:
    return _detection_cache[cache_key], 1.0

  try:
    # Read sample from file with automatic encoding detection
    full_text = read_text_file(file_path)
    sample_text = full_text[:sample_size]

    if len(sample_text.strip()) < 20:  # Too short for reliable detection
      logger.debug(f"File {file_path} too short for language detection")
      return fallback_language, 0.0

    # Detect language with confidence scores
    detected_langs = detect_langs(sample_text)

    if not detected_langs:
      return fallback_language, 0.0

    # Get top detection
    top_lang = detected_langs[0]
    lang_code = top_lang.lang
    confidence = top_lang.prob

    # Map langdetect code to our ISO code
    if lang_code in LANGDETECT_TO_ISO:
      lang_code = LANGDETECT_TO_ISO[lang_code]

    # Check if we support this language
    try:
      iso_code = get_iso_code(lang_code)
    except ValueError:
      logger.info(f"Detected unsupported language '{lang_code}' for {os.path.basename(file_path)}, using fallback")
      return fallback_language, 0.0

    # Check confidence threshold
    if confidence < min_confidence:
      logger.info(f"Low confidence {confidence:.2f} for language '{iso_code}' in {os.path.basename(file_path)}, using fallback")
      return fallback_language, confidence

    logger.info(f"Detected language '{iso_code}' (confidence: {confidence:.2f}) for {os.path.basename(file_path)}")

    # Cache the result
    _detection_cache[cache_key] = iso_code

    return iso_code, confidence

  except LangDetectException as e:
    logger.debug(f"Language detection failed for {file_path}: {e}")
    return fallback_language, 0.0
  except (FileNotFoundError, OSError, UnicodeDecodeError) as e:
    logger.warning(f"Error detecting language for {file_path}: {e}")
    return fallback_language, 0.0

def should_skip_detection(file_path: str) -> str | None:
  """
  Check if language detection should be skipped based on file type.

  Args:
      file_path: Path to check.

  Returns:
      ISO language code if detection should be skipped, None otherwise.
  """
  ext = Path(file_path).suffix.lower()

  # Code files are typically in English
  code_extensions = {
    '.py', '.js', '.java', '.c', '.cpp', '.h', '.hpp',
    '.go', '.rs', '.php', '.rb', '.ts', '.swift', '.kt',
    '.scala', '.r', '.m', '.sh', '.bash', '.zsh'
  }

  if ext in code_extensions:
    return 'en'

  # Some extensions hint at specific languages
  language_hints = {
    '.zh': 'zh',
    '.cn': 'zh',
    '.de': 'de',
    '.fr': 'fr',
    '.es': 'es',
    '.it': 'it',
    '.pt': 'pt',
    '.nl': 'nl',
    '.sv': 'sv',
    '.da': 'da',
    '.fi': 'fi',
    '.id': 'id',
  }

  if ext in language_hints:
    return language_hints[ext]

  return None

def clear_detection_cache():
  """Clear the language detection cache."""
  _detection_cache.clear()
  logger.debug("Language detection cache cleared")

def get_cache_stats() -> dict[str, int]:
  """Get statistics about the detection cache."""
  return {
    'cache_size': len(_detection_cache),
    'cached_files': list(_detection_cache.keys())
  }

#fin
