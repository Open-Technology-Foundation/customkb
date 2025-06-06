#!/usr/bin/env python
"""
Configuration management for CustomKB.
Handles loading, parsing, and validating configuration files for knowledge bases.
"""

import os
import configparser
import time
from typing import Dict, Any, Optional, Tuple

from utils.logging_utils import get_logger
from utils.text_utils import split_filepath, find_file, get_env

# Initialize vector database directory
VECTORDBS = os.getenv('VECTORDBS', '/var/lib/vectordbs')
if not os.path.exists(VECTORDBS):
  try:
    os.makedirs(VECTORDBS, mode=0o770, exist_ok=True)
  except Exception as e:
    raise EnvironmentError(f"Failed to create directory {VECTORDBS}: {e}")

logger = get_logger(__name__)

def get_fq_cfg_filename(cfgfile: str) -> Optional[str]:
  """
  Resolve a configuration filename to its fully-qualified path.
  
  Handles domain-style names (e.g., 'example.com') and regular paths,
  adding the .cfg extension if needed, and searching in VECTORDBS
  directory if the file is not in the current path.

  Args:
      cfgfile: The configuration file path or name.

  Returns:
      The fully-qualified path to the configuration file, or None if not found.
  """
  from utils.security_utils import validate_file_path, validate_safe_path
  
  # Validate input
  if not cfgfile:
    logger.error("Configuration file name cannot be empty")
    return None
  
  try:
    # Basic validation for dangerous characters and path traversal
    validated_cfgfile = validate_file_path(cfgfile, ['.cfg', ''])
  except ValueError as e:
    logger.error(f"Invalid configuration file path: {e}")
    return None
  # Handle domain-style names
  if '.' in validated_cfgfile and not validated_cfgfile.endswith('.cfg'):
    candidate_cfg = f"{validated_cfgfile}.cfg"
    if os.path.exists(candidate_cfg):
      # Validate the final path is safe
      if validate_safe_path(candidate_cfg, VECTORDBS) or validate_safe_path(candidate_cfg, os.getcwd()):
        return candidate_cfg
    
    domain_path = find_file(candidate_cfg, VECTORDBS)
    if domain_path and validate_safe_path(domain_path, VECTORDBS):
      return domain_path
  
  # Handle regular paths
  _dir, _file, _ext, _fqfn = split_filepath(validated_cfgfile, adddir=False, realpath=False)
  if not _ext:
    logger.info('adding ext .cfg')
    _ext = '.cfg'
    _fqfn += _ext
  if _ext != '.cfg':
    logger.error('Not a .cfg file!')
    return None
    
  if not os.path.exists(_fqfn):
    if logger:
      logger.warning(f'{_fqfn} does not exist, searching in {VECTORDBS}')
    if _dir:
      if logger:
        logger.error(f"File '{_fqfn}' does not exist.")
      return None
    _fqfn = find_file(f"{_file}{_ext}", VECTORDBS)
    if not _fqfn:
      if logger:
        logger.error(f"File '{_file}{_ext}' could not be found.")
      return None
    
    # Validate the found file is in a safe location
    if not validate_safe_path(_fqfn, VECTORDBS):
      if logger:
        logger.error(f"Found file '{_fqfn}' is outside allowed directory")
      return None
    
    return _fqfn
  
  # Validate the existing file path is safe
  if not (validate_safe_path(_fqfn, VECTORDBS) or validate_safe_path(_fqfn, os.getcwd())):
    if logger:
      logger.error(f"File '{_fqfn}' is outside allowed directories")
    return None
    
  return _fqfn

class KnowledgeBase:
  """
  Manages configuration and resources for a knowledge base.
  
  Handles loading settings from config files, environment variables, or defaults.
  Maintains paths to database and vector files, and provides configuration for
  embedding models, database parameters, and query settings.
  """

  def __init__(self, kb_base: str, **kwargs):
    """
    Initialize a KnowledgeBase with configuration from file or defaults.

    Args:
      kb_base: Base name or path of the knowledge base.
      **kwargs: Additional parameters to override configuration values.
    """
    self.CONFIG = {
      'DEF_VECTOR_MODEL': (str, 'text-embedding-3-small'),
      'DEF_VECTOR_DIMENSIONS': (int, 1536),
      'DEF_VECTOR_CHUNKS': (int, 200),
      'DEF_DB_MIN_TOKENS': (int, 100),
      'DEF_DB_MAX_TOKENS': (int, 200),
      'DEF_QUERY_MODEL': (str, 'gpt-4o'),
      'DEF_QUERY_ROLE': (str, 'You are a helpful assistant.'),
      'DEF_QUERY_TOP_K': (int, 50),
      'DEF_QUERY_CONTEXT_SCOPE': (int, 4),
      'DEF_QUERY_TEMPERATURE': (float, 0.0),
      'DEF_QUERY_MAX_TOKENS': (int, 4000)
    }

    # Set default values from environment or defaults
    for var_name, (var_type, default_value) in self.CONFIG.items():
      try:
        env_value = os.getenv(var_name)
        if env_value is not None:
          setattr(self, var_name, var_type(env_value))
        else:
          setattr(self, var_name, default_value)
      except ValueError as e:
        if logger:
          logger.warning(f"Invalid value for {var_name}, using default. Error: {e}")
        setattr(self, var_name, default_value)
      except Exception as e:
        if logger:
          logger.error(f"Error initializing {var_name}: {e}")
        raise

    self.start_time = int(time.time())
    self.load_config(kb_base, **kwargs)

    # Set up database and vector file paths
    directory, basename, extension, fqfn = split_filepath(kb_base)
    self.knowledge_base_name = basename
    
    # Properly handle domain-style names with multiple extensions
    if extension == '.cfg':
      full_basename = basename
      if full_basename.endswith('.cfg'):
        full_basename = full_basename[:-4]
    else:
      full_basename = basename
        
    self.knowledge_base_db = f'{directory}/{full_basename}.db'
    self.knowledge_base_vector = f'{directory}/{full_basename}.faiss'
    self.sql_connection = None
    self.sql_cursor = None

  def load_config(self, kb_base: str, **kwargs):
    """
    Load configuration from .cfg file or use provided values.

    Args:
        kb_base: Base name or path of the knowledge base.
        **kwargs: Additional parameters to override configuration values.
    """
    if kb_base.endswith('.cfg'):
      config = configparser.ConfigParser()
      config.read(kb_base)
      df = config['DEFAULT']
      self.vector_model = get_env('VECTOR_MODEL',
        df.get('vector_model', fallback=self.DEF_VECTOR_MODEL))
      self.vector_dimensions = get_env('VECTOR_DIMENSIONS',
        df.getint('vector_dimensions', fallback=self.DEF_VECTOR_DIMENSIONS), int)
      self.vector_chunks = get_env('VECTOR_CHUNKS',
        df.getint('vector_chunks', fallback=self.DEF_VECTOR_CHUNKS), int)
      self.db_min_tokens = get_env('DB_MIN_TOKENS',
        df.getint('db_min_tokens', fallback=self.DEF_DB_MIN_TOKENS), int)
      self.db_max_tokens = get_env('DB_MAX_TOKENS',
        df.getint('db_max_tokens', fallback=self.DEF_DB_MAX_TOKENS), int)
      self.query_model = get_env('QUERY_MODEL',
        df.get('query_model', fallback=self.DEF_QUERY_MODEL))
      self.query_top_k = get_env('QUERY_TOP_K',
        df.getint('query_top_k', fallback=self.DEF_QUERY_TOP_K), int)
      self.query_temperature = get_env('QUERY_TEMPERATURE',
        df.getfloat('query_temperature', fallback=self.DEF_QUERY_TEMPERATURE), float)
      self.query_max_tokens = get_env('QUERY_MAX_TOKENS',
        df.getint('query_max_tokens', fallback=self.DEF_QUERY_MAX_TOKENS), int)
      self.query_role = get_env('QUERY_ROLE',
        df.get('query_role', fallback=self.DEF_QUERY_ROLE))
      self.query_context_scope = get_env('QUERY_CONTEXT_SCOPE',
        df.getint('query_context_scope', fallback=self.DEF_QUERY_CONTEXT_SCOPE), int)
      self.query_context_files = get_env('QUERY_CONTEXT_FILES',
        df.get('query_context_files', fallback='').split(','))
    else:
      self.vector_model = kwargs.get('vector_model', self.DEF_VECTOR_MODEL)
      self.vector_dimensions = kwargs.get('vector_dimensions', self.DEF_VECTOR_DIMENSIONS)
      self.vector_chunks = kwargs.get('vector_chunks', self.DEF_VECTOR_CHUNKS)
      self.db_min_tokens = kwargs.get('db_min_tokens', self.DEF_DB_MIN_TOKENS)
      self.db_max_tokens = kwargs.get('db_max_tokens', self.DEF_DB_MAX_TOKENS)
      self.query_model = kwargs.get('query_model', self.DEF_QUERY_MODEL)
      self.query_top_k = kwargs.get('query_top_k', self.DEF_QUERY_TOP_K)
      self.query_temperature = kwargs.get('query_temperature', self.DEF_QUERY_TEMPERATURE)
      self.query_max_tokens = kwargs.get('query_max_tokens', self.DEF_QUERY_MAX_TOKENS)
      self.query_role = kwargs.get('query_role', self.DEF_QUERY_ROLE)
      self.query_context_scope = kwargs.get('query_context_scope', self.DEF_QUERY_CONTEXT_SCOPE)
      self.query_context_files = kwargs.get('query_context_files', [])

  def save_config(self, output_to: Optional[str] = None) -> None:
    """
    Save current configuration to a file or print to stderr.

    Args:
        output_to: Path to save config to. If None, prints to stderr.
    """
    if output_to:
      filehandle = open(output_to, 'w')
    else:
      import sys
      filehandle = sys.stderr

    print(f"# {self.knowledge_base_name}", file=filehandle)
    if output_to:
      print("[DEFAULT]", file=filehandle)

    attrs = vars(self)
    for key, value in attrs.items():
      print(f"{key} = {value}", file=filehandle)

    if output_to:
      filehandle.close()

#fin
