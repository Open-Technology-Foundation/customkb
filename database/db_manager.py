#!/usr/bin/env python
"""
Database management module for CustomKB knowledgebase system.

This module handles document processing operations including:
- Text file ingestion and processing with multiple format support
- Intelligent text chunking with configurable token limits
- Multi-language stopword filtering and text normalization
- Named entity recognition and metadata extraction
- Batch processing with performance optimization
- Transaction management and error recovery

For database connections, use database.connection.
For text chunking utilities, use database.chunking.
For schema migrations, use database.migrations.
"""

import argparse
import json
import os
import re
import sqlite3
from typing import Any

# Import NLTK for text processing
import nltk
import spacy
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from post_slug import post_slug

from config.config_manager import KnowledgeBase, get_fq_cfg_filename
from utils.logging_config import get_logger, time_to_finish
from utils.text_utils import enhanced_clean_text, get_files, nlp, read_text_file, split_filepath

from .chunking import detect_file_type, init_text_splitter
from .connection import close_database, connect_to_database

# Set up NLTK data path
NLTK_DATA = os.getenv('NLTK_DATA')
if not NLTK_DATA:
  raise OSError("NLTK_DATA environment variable not set.")

nltk.data.path = [NLTK_DATA]
try:
  nltk.data.find('tokenizers/punkt_tab')
  nltk.data.find('tokenizers/punkt')
  nltk.data.find('corpora/stopwords')
  nltk.data.find('corpora/wordnet')
except LookupError:
  os.makedirs(NLTK_DATA, exist_ok=True)
  nltk.download('punkt_tab', download_dir=NLTK_DATA, quiet=True)
  nltk.download('punkt', download_dir=NLTK_DATA, quiet=True)
  nltk.download('stopwords', download_dir=NLTK_DATA, quiet=True)
  nltk.download('wordnet', download_dir=NLTK_DATA, quiet=True)

# Ensure required languages are available for stopwords
required_languages = ['english', 'indonesian', 'french', 'german', 'swedish']
for lang in required_languages:
  try:
    # Test if stopwords for this language can be loaded
    stopwords.words(lang)
  except LookupError:
    # If not available, download stopwords package again
    nltk.download('stopwords', download_dir=NLTK_DATA, quiet=True)

# Load spaCy model for entity recognition
try:
  nlp = spacy.load("en_core_web_sm")
except (OSError, ImportError) as e:
  # Fall back if spacy model isn't installed
  logger = get_logger(__name__)
  logger.debug(f"spaCy model 'en_core_web_sm' not available: {e}")
  nlp = None

# Language codes mapping - Limited to languages with NLTK support that we use
language_codes = {
  'zh': 'chinese',
  'da': 'danish',
  'nl': 'dutch',
  'en': 'english',
  'fi': 'finnish',
  'fr': 'french',
  'de': 'german',
  'id': 'indonesian',
  'it': 'italian',
  'pt': 'portuguese',
  'es': 'spanish',
  'sv': 'swedish'
}

# Reverse mapping: full name to ISO code
language_names_to_codes = {v: k for k, v in language_codes.items()}

def sanitize_filename(file_path: str, logger=None) -> tuple[bool, str, bool, str]:
  """
  Sanitize filename by replacing dangerous characters using post_slug.

  Only sanitizes the basename (filename), leaving directory path unchanged.
  If the sanitized filename already exists, returns False (skip file).

  Args:
      file_path: Full path to the file
      logger: Optional logger instance

  Returns:
      Tuple of (success, new_path, was_renamed, message):
        - success: True if sanitization succeeded or not needed, False if collision
        - new_path: The sanitized file path (may be same as input)
        - was_renamed: True if file was actually renamed
        - message: Description of what happened
  """
  # Define dangerous characters (same as in security_utils.py)
  dangerous_chars = ['<', '>', '|', '&', ';', '`', '$']

  # Split path into components
  directory = os.path.dirname(file_path)
  full_basename = os.path.basename(file_path)

  # Split basename into name and extension
  # Handle multiple extensions like .pdf.md
  parts = full_basename.split('.')
  if len(parts) > 1:
    basename = '.'.join(parts[:-1])  # Everything except last extension
    extension = '.' + parts[-1]        # Last extension including dot
  else:
    basename = full_basename
    extension = ''

  # Check if basename contains dangerous characters
  has_dangerous = any(char in basename for char in dangerous_chars)

  if not has_dangerous:
    # No dangerous characters, return original path
    return True, file_path, False, 'No dangerous characters found'

  # Sanitize the basename using post_slug with period separator and preserve case
  sanitized_basename = post_slug(basename, '.', True)

  if not sanitized_basename:
    # post_slug returned empty string (error case)
    return False, file_path, False, 'Failed to sanitize filename'

  # Reconstruct the full filename
  sanitized_filename = sanitized_basename + extension

  # Construct new path
  new_path = os.path.join(directory, sanitized_filename) if directory else sanitized_filename

  # Check if target path already exists
  if os.path.exists(new_path) and new_path != file_path:
    message = f'Sanitized filename already exists: {sanitized_filename}'
    return False, file_path, False, message

  # Path is same (shouldn't happen but check anyway)
  if new_path == file_path:
    return True, file_path, False, 'Filename already clean'

  message = f'Will rename: {full_basename} → {sanitized_filename}'
  return True, new_path, True, message

def get_iso_code(language: str) -> str:
  """
  Convert language to ISO code.

  Args:
      language: Either ISO code (e.g., 'en') or full name (e.g., 'english').

  Returns:
      ISO code.

  Raises:
      ValueError: If language is not recognized.
  """
  # If already an ISO code, return it
  if language in language_codes:
    return language
  # If it's a full name, convert to ISO code
  if language in language_names_to_codes:
    return language_names_to_codes[language]
  # Not recognized
  raise ValueError(f"Unrecognized language: '{language}'. Use ISO 639-1 code (e.g., 'en') or full name (e.g., 'english').")

def get_full_language_name(iso_code: str) -> str:
  """
  Convert ISO code to full language name for NLTK.

  Args:
      iso_code: ISO 639-1 language code (e.g., 'en').

  Returns:
      Full language name (e.g., 'english').

  Raises:
      ValueError: If ISO code is not recognized.
  """
  if iso_code in language_codes:
    return language_codes[iso_code]
  raise ValueError(f"Unrecognized ISO code: '{iso_code}'")

logger = get_logger(__name__)

def extract_metadata(text: str, file_path: str, kb) -> dict[str, Any]:
  """
  Extract and track metadata about a text chunk.

  Args:
      text: The text chunk.
      file_path: Path to the source file.
      kb: KnowledgeBase instance for configuration.

  Returns:
      Dictionary containing metadata.
  """
  metadata = {
    "source": file_path,
    "char_length": len(text),
    "word_count": len(text.split()),
  }

  # Extract file extension
  _, _, ext, _ = split_filepath(file_path)
  if ext:
    metadata["file_type"] = ext.lstrip('.')

  # Extract headings if possible (expanded pattern for different heading formats)
  # Get configurable heading search limit
  heading_search_limit = getattr(kb, 'heading_search_limit', 200)
  heading_match = re.search(r'^(#+|=+|[-]+)\s*(.+?)(?:\s*[=|-]+)?$', text[:heading_search_limit], re.MULTILINE)
  if heading_match:
    metadata["heading"] = heading_match.group(2).strip()

  # Try to identify document section type
  if text.startswith('#'):
    metadata["section_type"] = "heading"
  elif re.search(r'```\w*\n[\s\S]*?```', text):
    metadata["section_type"] = "code_block"
  elif re.search(r'<table[\s>].*?</table>', text, re.DOTALL | re.IGNORECASE):
    metadata["section_type"] = "table"
  elif re.search(r'<(ul|ol)[\s>].*?</(ul|ol)>', text, re.DOTALL | re.IGNORECASE):
    metadata["section_type"] = "list"
  elif re.search(r'^\s*[-*•]\s', text, re.MULTILINE):
    metadata["section_type"] = "bullet_list"
  elif re.search(r'^\s*\d+[.)]\s', text, re.MULTILINE):
    metadata["section_type"] = "numbered_list"

  # Try to identify common document sections
  if re.search(r'\b(summary|overview|introduction|conclusion|abstract)\b', text[:200], re.IGNORECASE):
    section_match = re.search(r'\b(summary|overview|introduction|conclusion|abstract)\b', text[:200], re.IGNORECASE)
    metadata["document_section"] = section_match.group(1).lower()

  # Extract named entities if spaCy is available
  if nlp:
    try:
      # Get configurable entity extraction limit
      entity_limit = getattr(kb, 'entity_extraction_limit', 500)
      doc = nlp(text[:entity_limit])  # Process configurable amount for efficiency
      entities = {}
      for ent in doc.ents:
        if ent.label_ not in entities:
          entities[ent.label_] = []
        if ent.text not in entities[ent.label_]:
          entities[ent.label_].append(ent.text)

      # Add only if entities were found
      if entities:
        metadata["entities"] = entities
    except (AttributeError, RuntimeError, TypeError) as e:
      if logger:
        logger.warning(f"Error extracting entities: {e}")

  return metadata

def process_database(args: argparse.Namespace, logger) -> str:
  """
  Process and store text files into the CustomKB knowledgebase.

  Takes input text files, processes them with appropriate text splitters based on file type,
  and stores them as chunks in the SQLite database. Supports various file formats and
  implements multilingual stopword filtering to improve text quality.

  Args:
      args: Command-line arguments containing:
          config_file: Path to knowledgebase configuration
          files: List of file paths or patterns to process
          language: Language for stopwords (default: english)
          force: Whether to reprocess files already in the database
          verbose: Enable verbose output
          debug: Enable debug output
      logger: Initialized logger instance

  Returns:
      A status message indicating the number of files processed and skipped.
  """
  # Convert language to ISO code for storage
  try:
    iso_language = get_iso_code(args.language)
  except ValueError as e:
    return f"Error: {e}"

  logger.info(f"Input language: {args.language}, ISO code: {iso_language}")

  # Check if language detection is enabled
  detect_language = getattr(args, 'detect_language', False)
  if detect_language:
    logger.info("Language detection enabled for individual files")

  # Get configuration file
  config_file = get_fq_cfg_filename(args.config_file)
  if not config_file:
    return f"Error: Knowledgebase '{args.config_file}' not found."

  logger.info(f"{config_file=}")

  # Initialize knowledgebase
  kb = KnowledgeBase(config_file)
  if args.verbose:
    kb.save_config()

  logger.info(f"Config file: {args.config_file}")
  if args.files:
    logger.info(f"Processing {len(args.files)} files")
  else:
    logger.info("No input files provided")
    return "No input files provided. Nothing to do."

  # Connect to database
  connect_to_database(kb)

  # Initialize stopwords with multiple languages
  stop_words = set()
  # Convert ISO code to full name for NLTK stopwords
  try:
    full_language_name = get_full_language_name(iso_language)
    stop_words.update(stopwords.words(full_language_name))
  except LookupError:
    error_msg = (f"NLTK stopwords not available for language '{full_language_name}' (ISO: {iso_language}). "
                 f"Please install NLTK stopwords data:\n"
                 f"  python -m nltk.downloader stopwords\n"
                 f"Then verify '{full_language_name}' is included in the stopwords corpus.")
    logger.error(error_msg)
    return f"Error: {error_msg}"

  # Add additional language stopwords (configurable)
  additional_languages = getattr(kb, 'additional_stopword_languages', ['indonesian', 'french', 'german', 'swedish'])
  for lang in additional_languages:
    # Convert to full name if it's an ISO code
    try:
      full_lang = language_codes.get(lang, lang)

      if full_lang != full_language_name:  # Skip if same as primary language
        try:
          stop_words.update(stopwords.words(full_lang))
        except LookupError:
          if logger:
            logger.warning(f"Failed to load stopwords for {full_lang}")
    except (KeyError, LookupError, ValueError) as e:
      if logger:
        logger.warning(f"Error processing language {lang}: {e}")

  # Pre-scan to get actual total file count
  all_files = []
  for arg in args.files:
    all_files.extend(get_files(arg))

  numfiles = len(all_files)
  logger.info(f"Found {numfiles} total files to process")

  # Process files in optimized batches
  filecount = 0
  processed_count = 0
  # Get configurable file processing batch size
  batch_size = getattr(kb, 'file_processing_batch_size', 500)  # Process files per batch for better performance

  # Pre-compute batch splitting and file types to reduce overhead
  for i in range(0, len(all_files), batch_size):
    batch = all_files[i:i+batch_size]
    logger.info(f"Processing batch of {len(batch)} files ({i+1}-{min(i+batch_size, numfiles)} of {numfiles})")

    # If not forcing reprocessing, check which files already exist in the database
    existing_paths = set()
    if not args.force and batch:
      try:
        # Get canonical paths for efficient lookup
        # Using full paths prevents collisions between files with same name
        canonical_paths = [os.path.abspath(f) for f in batch]

        # Efficiently query which file paths already exist
        if canonical_paths:
          # Convert to list of strings (safe for IN query)
          safe_paths = [str(path) for path in canonical_paths]
          query_template = "SELECT DISTINCT sourcedoc FROM docs WHERE sourcedoc IN ({placeholders})"
          kb.sql_cursor.execute(query_template.format(placeholders=','.join(['?'] * len(safe_paths))), safe_paths)
          existing_paths = {row[0] for row in kb.sql_cursor.fetchall()}
          if logger:
            logger.debug(f"Found {len(existing_paths)} existing files in database")
      except sqlite3.Error as e:
        if logger:
          logger.warning(f"Error checking existing files: {e}, will process all files in batch")

    # Process each file in the batch
    for pfile in batch:
      canonical_path = os.path.abspath(pfile)
      # Skip if file exists and not forcing
      if canonical_path in existing_paths and not args.force:
        logger.info(f"Skipping {pfile} (already in database)")
        filecount += 1
        continue

      file_type = detect_file_type(pfile)
      splitter = init_text_splitter(kb, file_type)

      # Track if file was actually processed or skipped
      if process_text_file(kb, pfile, splitter, stop_words, iso_language, file_type, args.force, detect_language):
        processed_count += 1

      filecount += 1
      if (filecount % 5) == 0:
        logger.info(f"{filecount}/{numfiles} ~{time_to_finish(kb.start_time, filecount, numfiles)}")

  # Build BM25 index if hybrid search is enabled and we processed files
  if processed_count > 0 and getattr(kb, 'enable_hybrid_search', False):
    try:
      from embedding.bm25_manager import build_bm25_index
      logger.info("Building BM25 index for hybrid search...")
      bm25 = build_bm25_index(kb)
      if bm25:
        logger.info("BM25 index built successfully")
      else:
        logger.warning("BM25 index build failed")
    except (ImportError, RuntimeError, OSError) as e:
      logger.warning(f"Failed to build BM25 index: {e}")

  # Close database connection
  close_database(kb)

  # Update return message to show both examined and processed counts
  if filecount == processed_count:
    return f'{filecount} files added to database {kb.knowledge_base_db}'
  else:
    return f'{processed_count} files processed ({filecount - processed_count} skipped) in database {kb.knowledge_base_db}'

def process_text_file(kb: KnowledgeBase, sourcefile: str, splitter: Any,
                     stop_words: set, language: str, file_type: str = 'text',
                     force: bool = False, detect_language: bool = False) -> bool:
  """
  Process a text file, split it into chunks, and store in the database.

  Args:
      kb: The KnowledgeBase instance.
      sourcefile: Path to the source file.
      splitter: The text splitter instance.
      stop_words: Set of stopwords for the current language.
      language: Current language code.
      file_type: Type of file being processed.
      force: Whether to force reprocessing of files already in the database.
      detect_language: Whether to auto-detect language for this file.

  Returns:
      True if file was processed, False if skipped.
  """
  # Store the original language for comparison
  original_language = language

  from utils.logging_config import log_file_operation, log_operation_error

  # Sanitize filename before any processing
  success, new_path, was_renamed, message = sanitize_filename(sourcefile, logger)

  if not success:
    # Collision or sanitization error
    logger.warning(f"Skipping file due to sanitization issue: {message}")
    return False

  if was_renamed:
    # Rename file on disk
    try:
      os.rename(sourcefile, new_path)
      logger.warning(f"Renamed file due to dangerous characters: {os.path.basename(sourcefile)} → {os.path.basename(new_path)}")
      # Update sourcefile to use new path
      sourcefile = new_path
    except OSError as e:
      logger.error(f"Failed to rename file {sourcefile}: {e}")
      return False

  # Store full canonical absolute path instead of basename
  # This allows proper handling of files with same name in different directories
  sourcedoc_value = os.path.abspath(sourcefile)

  # Check if file already exists in the database using full path
  kb.sql_cursor.execute("SELECT COUNT(*) FROM docs WHERE sourcedoc = ?", [sourcedoc_value])
  count = kb.sql_cursor.fetchone()[0]

  if count > 0 and not force:
    logger.info(f"Skipping {sourcefile} (already in database, use --force to reprocess)")
    return False

  # Log file processing start with enhanced context
  log_file_operation(logger, "processing_start", sourcefile,
                    file_type=file_type, language=language, force=force)

  # Read file content with validation
  try:
    from utils.security_utils import validate_file_path

    # Validate the source file path
    try:
      validated_sourcefile = validate_file_path(sourcefile)
    except ValueError as e:
      log_operation_error(logger, "file_validation", e,
                         file_path=sourcefile, file_type=file_type)
      return False

    # Check file size (prevent extremely large files)
    try:
      file_size = os.path.getsize(validated_sourcefile)
      # Get configurable max file size
      max_file_size_mb = getattr(kb, 'max_file_size_mb', 100)
      max_file_size = max_file_size_mb * 1024 * 1024  # Convert MB to bytes
      if file_size > max_file_size:
        log_operation_error(logger, "file_size_check",
                           ValueError(f"File too large: {file_size} bytes"),
                           file_path=validated_sourcefile,
                           file_size=file_size,
                           max_size=max_file_size)
        return False
    except OSError as e:
      log_operation_error(logger, "file_access", e,
                         file_path=validated_sourcefile)
      return False

    # Prepare encoding config from knowledgebase settings
    encoding_config = {
      'auto_detect_encoding': getattr(kb, 'auto_detect_encoding', True),
      'default_encoding': getattr(kb, 'default_encoding', 'utf-8'),
      'encoding_fallbacks': getattr(kb, 'encoding_fallbacks', ['utf-8', 'windows-1252', 'latin-1', 'cp1252'])
    }
    content = read_text_file(validated_sourcefile, encoding_config)
  except OSError as e:
    log_operation_error(logger, "file_read", e,
                       file_path=validated_sourcefile, file_size=file_size)
    return False

  # Split content into chunks
  try:
    # Different handling based on splitter type
    if file_type == 'html' and callable(splitter):
      # For HTML, we call the function directly
      chunks = splitter(content)
    elif hasattr(splitter, 'split_text'):
      # For LangChain splitters
      chunks = splitter.split_text(content)
    elif hasattr(splitter, 'chunks'):
      # For semantic_text_splitter
      chunks = splitter.chunks(content)
    else:
      # Fallback to basic chunking
      chunks = [content[i:i+kb.db_max_tokens] for i in range(0, len(content), kb.db_max_tokens)]

    if logger:
      logger.debug(f"Split {sourcefile} into {len(chunks)} chunks")
  except (AttributeError, ValueError, RuntimeError, TypeError) as e:
    logger.error(f"Failed to split chunks for {sourcefile}: {e}")
    return False

  # Detect language if enabled
  if detect_language:
    from utils.language_detector import detect_file_language, should_skip_detection

    # Check if we should skip detection for this file type
    skip_lang = should_skip_detection(sourcefile)
    if skip_lang:
      language = skip_lang
      logger.info(f"Using language '{language}' based on file type for {sourcefile}")
    else:
      # Get configuration parameters
      sample_size = getattr(kb, 'language_detection_sample_size', 3072)
      min_confidence = getattr(kb, 'language_detection_confidence', 0.95)

      # Detect language
      detected_lang, confidence = detect_file_language(
        sourcefile,
        sample_size=sample_size,
        min_confidence=min_confidence,
        fallback_language=language
      )

      if detected_lang != language:
        language = detected_lang
        logger.info(f"Detected language '{language}' (confidence: {confidence:.2f}) for {os.path.basename(sourcefile)}")

  # Check if language needs to be updated based on directory name
  # Configuration controls whether directory name overrides detected language
  directory_override_enabled = getattr(kb, 'directory_language_override', True)

  dirname = os.path.dirname(sourcefile)
  _, langkey, _, _ = split_filepath(dirname)
  if langkey in language_codes and langkey != language:
    # langkey is already an ISO code
    if directory_override_enabled or not detect_language:
      # Override if: (1) directory override is enabled, or (2) not using content detection
      language = langkey  # Update to new ISO code
      logger.info(f"Using language from directory: {language=}")
    else:
      # Directory override disabled and content detection active - keep detected language
      logger.debug(f"Directory suggests language '{langkey}' but using detected '{language}' (directory_language_override=False)")


  # If language changed, re-initialize stopwords
  if language != original_language:
    stop_words = set()
    # Convert ISO code to full name for NLTK
    try:
      full_lang_name = get_full_language_name(language)
      stop_words.update(stopwords.words(full_lang_name))
    except LookupError:
      logger.warning(f"NLTK stopwords not available for language '{full_lang_name}' (ISO: {language})")
      # Continue without stopwords for this language

    # Add additional language stopwords (use same config as before)
    additional_languages = getattr(kb, 'additional_stopword_languages', ['indonesian', 'french', 'german', 'swedish'])
    for lang in additional_languages:
      # Convert to full name if it's an ISO code
      try:
        full_lang = language_codes.get(lang, lang)

        if full_lang != full_lang_name:  # Skip if same as primary language
          try:
            stop_words.update(stopwords.words(full_lang))
          except LookupError:
            if logger:
              logger.warning(f"Failed to load stopwords for {full_lang}")
      except (KeyError, LookupError, ValueError) as e:
        if logger:
          logger.warning(f"Error processing language {lang}: {e}")

  # Delete any existing entries with this file path
  kb.sql_cursor.execute(
    "DELETE FROM docs WHERE sourcedoc = ?",
    [sourcedoc_value]
  )
  kb.sql_connection.commit()

  # Set up lemmatizer for better cleaning
  lemmatizer = WordNetLemmatizer()

  # Insert chunks into database
  sid = 0
  for _i, chunk in enumerate(chunks):
    # Extract metadata about the chunk
    metadata = extract_metadata(chunk, sourcefile, kb)

    # Enhanced text cleaning with entity preservation
    clean_chunk = enhanced_clean_text(chunk, stop_words, lemmatizer)
    if not clean_chunk:
      continue

    # BM25 tokenization (if hybrid search is enabled)
    bm25_tokens = ""
    doc_length = 0
    keyphrase_processed = 0

    if getattr(kb, 'enable_hybrid_search', False):
      from utils.text_utils import tokenize_for_bm25
      try:
        bm25_tokens, doc_length = tokenize_for_bm25(chunk, language)
        keyphrase_processed = 1 if bm25_tokens else 0
      except (ValueError, TypeError, AttributeError) as e:
        if logger:
          logger.warning(f"BM25 tokenization failed for chunk: {e}")
        bm25_tokens = ""
        doc_length = 0
        keyphrase_processed = 0

    # Store metadata as JSON string in the database (safer than Python dict string)
    try:
      metadata_str = json.dumps(metadata)
    except (TypeError, ValueError) as e:
      if logger:
        logger.warning(f"Could not serialize metadata to JSON: {e}")
      metadata_str = "{}"

    # Add chunk with metadata and BM25 data
    kb.sql_cursor.execute(
      "INSERT INTO docs (sid, sourcedoc, originaltext, embedtext, embedded, language, metadata, bm25_tokens, doc_length, keyphrase_processed) VALUES (?,?,?,?,?,?,?,?,?,?)",
      (sid, sourcedoc_value, chunk, clean_chunk, 0, language, metadata_str, bm25_tokens, doc_length, keyphrase_processed))

    # Commit periodically (configurable frequency)
    commit_frequency = getattr(kb, 'commit_frequency', 1000)
    if sid % commit_frequency == 0:
      kb.sql_connection.commit()

    sid += 1

  kb.sql_connection.commit()
  return True

#fin
