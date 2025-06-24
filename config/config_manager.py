#!/usr/bin/env python
"""
Configuration management for CustomKB knowledge base system.

This module provides the core configuration infrastructure for CustomKB, handling:
- Configuration file loading and validation with security checks
- Environment variable overrides with type conversion
- Domain-style knowledge base naming (e.g., 'example.com.cfg')
- Path resolution and validation for configuration files
- Comprehensive configuration parameters across 5 categories:
  * DEFAULT: Core model and processing settings
  * API: External API interaction parameters
  * LIMITS: Resource and security constraints
  * PERFORMANCE: Optimization and tuning settings
  * ALGORITHMS: Algorithm-specific thresholds and parameters

The configuration resolution hierarchy is:
1. Environment variables (highest priority)
2. Configuration file values
3. Built-in defaults (lowest priority)
"""

import os
import configparser
import time
from typing import Optional

from utils.logging_utils import get_logger
from utils.text_utils import split_filepath, find_file, get_env

# Initialize vector database directory
VECTORDBS = os.getenv('VECTORDBS', '/var/lib/vectordbs')
if not os.path.exists(VECTORDBS):
  try:
    os.makedirs(VECTORDBS, mode=0o770, exist_ok=True)  # Will be made configurable per-instance
  except Exception as e:
    raise EnvironmentError(f"Failed to create directory {VECTORDBS}: {e}")

logger = get_logger(__name__)


def get_fq_cfg_filename(cfgfile: str) -> Optional[str]:
  """
  Resolve a configuration filename to its fully-qualified, validated path.
  
  This function implements a sophisticated path resolution system that:
  1. Validates the input path for security (no path traversal, etc.)
  2. Handles domain-style names (e.g., 'example.com' → 'example.com.cfg')
  3. Automatically appends '.cfg' extension if missing
  4. Searches in VECTORDBS directory if file not found locally
  5. Ensures the resolved path is within allowed directories

  Args:
      cfgfile: Configuration file path or name. Can be:
        - Full path: '/path/to/config.cfg'
        - Relative path: 'configs/myproject.cfg'
        - Domain-style: 'example.com' (becomes 'example.com.cfg')
        - Base name: 'myproject' (becomes 'myproject.cfg')

  Returns:
      The fully-qualified, validated path to the configuration file,
      or None if the file cannot be found or fails validation.

  Examples:
      >>> get_fq_cfg_filename('example.com')
      '/var/lib/vectordbs/example.com.cfg'
      >>> get_fq_cfg_filename('myproject')
      '/current/dir/myproject.cfg'
  """
  from utils.security_utils import validate_file_path, validate_safe_path
  
  if not cfgfile:
    logger.error("Configuration file name cannot be empty")
    return None
  
  try:
    # Security validation: prevent path traversal and dangerous characters
    # Allow absolute paths and relative traversal for KB config files (trusted user input)
    validated_cfgfile = validate_file_path(cfgfile, ['.cfg', ''], allow_absolute=True, allow_relative_traversal=True)
  except ValueError as e:
    logger.error(f"Invalid configuration file path: {e}")
    return None
  # Handle domain-style names (e.g., 'example.com' → 'example.com.cfg')
  if '.' in validated_cfgfile and not validated_cfgfile.endswith('.cfg'):
    candidate_cfg = f"{validated_cfgfile}.cfg"
    if os.path.exists(candidate_cfg):
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
  
  return _fqfn

class KnowledgeBase:
  """
  Central configuration and resource manager for a CustomKB knowledge base instance.
  
  This class serves as the primary configuration hub, managing all settings and
  resources needed for a knowledge base to function. It implements a three-tier
  configuration hierarchy: environment variables (highest priority), config file
  values, and built-in defaults.
  
  Configuration Categories:
  - DEFAULT: Core settings (models, dimensions, chunking parameters)
  - API: External API interaction (rate limiting, concurrency, retries)
  - LIMITS: Resource constraints and security limits
  - PERFORMANCE: Optimization parameters (batch sizes, caching, threading)
  - ALGORITHMS: Algorithm-specific thresholds and tuning parameters
  
  Attributes:
      knowledge_base_name: The base name of this knowledge base
      knowledge_base_db: Path to the SQLite database file
      knowledge_base_vector: Path to the FAISS vector index file
      vector_model: Name of the embedding model to use
      query_model: Name of the LLM model for response generation
      
  Example:
      >>> kb = KnowledgeBase('myproject.cfg')
      >>> print(kb.vector_model)
      'text-embedding-3-small'
      >>> print(kb.knowledge_base_db)
      '/var/lib/vectordbs/myproject.db'
  """

  def __init__(self, kb_base: str, **kwargs):
    """
    Initialize a KnowledgeBase with configuration from file or defaults.

    Args:
        kb_base: Path to configuration file (e.g., 'myproject.cfg') or 
            base name of the knowledge base. If a .cfg file is provided,
            configuration will be loaded from it. Otherwise, defaults or
            kwargs will be used.
        **kwargs: Additional parameters to override configuration values.
            Any configuration parameter can be overridden by passing it
            as a keyword argument.

    Raises:
        EnvironmentError: If required directories cannot be created.
        ValueError: If configuration values fail type conversion.
        
    Example:
        >>> # Load from config file
        >>> kb = KnowledgeBase('myproject.cfg')
        >>> 
        >>> # Create with overrides
        >>> kb = KnowledgeBase('myproject', vector_model='text-embedding-ada-002')
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

    # New configuration sections for performance tuning
    self.API_CONFIG = {
      'DEF_API_CALL_DELAY_SECONDS': (float, 0.05),
      'DEF_API_MAX_RETRIES': (int, 20),
      'DEF_API_MAX_CONCURRENCY': (int, 8),
      'DEF_API_MIN_CONCURRENCY': (int, 3),
      'DEF_BACKOFF_EXPONENT': (int, 2),
      'DEF_BACKOFF_JITTER': (float, 0.1)
    }

    self.LIMITS_CONFIG = {
      'DEF_MAX_FILE_SIZE_MB': (int, 100),
      'DEF_MAX_QUERY_FILE_SIZE_MB': (int, 1),
      'DEF_MEMORY_CACHE_SIZE': (int, 10000),
      'DEF_API_KEY_MIN_LENGTH': (int, 20),
      'DEF_MAX_QUERY_LENGTH': (int, 10000),
      'DEF_MAX_CONFIG_VALUE_LENGTH': (int, 1000),
      'DEF_MAX_JSON_SIZE': (int, 10000)
    }

    self.PERFORMANCE_CONFIG = {
      'DEF_EMBEDDING_BATCH_SIZE': (int, 100),
      'DEF_CHECKPOINT_INTERVAL': (int, 10),
      'DEF_COMMIT_FREQUENCY': (int, 1000),
      'DEF_IO_THREAD_POOL_SIZE': (int, 4),
      'DEF_CACHE_THREAD_POOL_SIZE': (int, 4),  # Dedicated thread pool for cache operations
      'DEF_FILE_PROCESSING_BATCH_SIZE': (int, 500),
      'DEF_SQL_BATCH_SIZE': (int, 500),
      'DEF_REFERENCE_BATCH_SIZE': (int, 5),
      'DEF_QUERY_CACHE_TTL_DAYS': (int, 7),
      'DEF_DEFAULT_EDITOR': (str, 'joe')
    }

    self.ALGORITHMS_CONFIG = {
      'DEF_HIGH_DIMENSION_THRESHOLD': (int, 1536),
      'DEF_SMALL_DATASET_THRESHOLD': (int, 1000),
      'DEF_MEDIUM_DATASET_THRESHOLD': (int, 100000),
      'DEF_IVF_CENTROID_MULTIPLIER': (int, 4),
      'DEF_MAX_CENTROIDS': (int, 256),
      'DEF_TOKEN_ESTIMATION_SAMPLE_SIZE': (int, 10),
      'DEF_TOKEN_ESTIMATION_MULTIPLIER': (float, 1.3),
      'DEF_SIMILARITY_THRESHOLD': (float, 0.6),
      'DEF_LOW_SIMILARITY_SCOPE_FACTOR': (float, 0.5),
      'DEF_MAX_CHUNK_OVERLAP': (int, 100),
      'DEF_OVERLAP_RATIO': (float, 0.5),
      'DEF_HEADING_SEARCH_LIMIT': (int, 200),
      'DEF_ENTITY_EXTRACTION_LIMIT': (int, 500),
      'DEF_DEFAULT_DIR_PERMISSIONS': (int, 0o770),
      'DEF_DEFAULT_CODE_LANGUAGE': (str, 'python'),
      'DEF_ADDITIONAL_STOPWORD_LANGUAGES': (list, ['indonesian', 'french', 'german', 'swedish']),
      # BM25/Hybrid search configuration
      'DEF_ENABLE_HYBRID_SEARCH': (bool, False),
      'DEF_VECTOR_WEIGHT': (float, 0.7),
      'DEF_BM25_K1': (float, 1.2),
      'DEF_BM25_B': (float, 0.75),
      'DEF_BM25_MIN_TOKEN_LENGTH': (int, 2),
      'DEF_BM25_REBUILD_THRESHOLD': (int, 1000),
      # Query enhancement configuration
      'DEF_ENABLE_QUERY_ENHANCEMENT': (bool, True),
      'DEF_QUERY_ENHANCEMENT_SYNONYMS': (bool, True),
      'DEF_QUERY_ENHANCEMENT_SPELLING': (bool, True),
      'DEF_MAX_SYNONYMS_PER_WORD': (int, 2),
      'DEF_QUERY_ENHANCEMENT_CACHE_TTL_DAYS': (int, 30),
      'DEF_SPELLING_CORRECTION_THRESHOLD': (float, 0.8),
      'DEF_SYNONYM_RELEVANCE_THRESHOLD': (float, 0.6),
      # Reranking configuration
      'DEF_ENABLE_RERANKING': (bool, True),
      'DEF_RERANKING_MODEL': (str, 'cross-encoder/ms-marco-MiniLM-L-6-v2'),
      'DEF_RERANKING_TOP_K': (int, 20),
      'DEF_RERANKING_BATCH_SIZE': (int, 32),
      'DEF_RERANKING_DEVICE': (str, 'cpu'),
      'DEF_RERANKING_CACHE_SIZE': (int, 1000)
    }

    # Set default values from environment or defaults
    all_configs = [self.CONFIG, self.API_CONFIG, self.LIMITS_CONFIG, 
                   self.PERFORMANCE_CONFIG, self.ALGORITHMS_CONFIG]
    
    for config_dict in all_configs:
      for var_name, (var_type, default_value) in config_dict.items():
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
      
      # Load original config parameters
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
      # Handle QUERY_CONTEXT_FILES environment variable (comma-separated string)
      query_context_env = get_env('QUERY_CONTEXT_FILES', None)
      if query_context_env is not None:
        self.query_context_files = [f.strip() for f in query_context_env.split(',') if f.strip()]
      else:
        self.query_context_files = [f.strip() for f in df.get('query_context_files', fallback='').split(',') if f.strip()]

      # Load new configuration sections
      api_section = config['API'] if 'API' in config else df
      limits_section = config['LIMITS'] if 'LIMITS' in config else df
      performance_section = config['PERFORMANCE'] if 'PERFORMANCE' in config else df
      algorithms_section = config['ALGORITHMS'] if 'ALGORITHMS' in config else df

      # API configuration
      self.api_call_delay_seconds = get_env('API_CALL_DELAY_SECONDS',
        api_section.getfloat('api_call_delay_seconds', fallback=self.DEF_API_CALL_DELAY_SECONDS), float)
      self.api_max_retries = get_env('API_MAX_RETRIES',
        api_section.getint('api_max_retries', fallback=self.DEF_API_MAX_RETRIES), int)
      self.api_max_concurrency = get_env('API_MAX_CONCURRENCY',
        api_section.getint('api_max_concurrency', fallback=self.DEF_API_MAX_CONCURRENCY), int)
      self.api_min_concurrency = get_env('API_MIN_CONCURRENCY',
        api_section.getint('api_min_concurrency', fallback=self.DEF_API_MIN_CONCURRENCY), int)
      self.backoff_exponent = get_env('BACKOFF_EXPONENT',
        api_section.getint('backoff_exponent', fallback=self.DEF_BACKOFF_EXPONENT), int)
      self.backoff_jitter = get_env('BACKOFF_JITTER',
        api_section.getfloat('backoff_jitter', fallback=self.DEF_BACKOFF_JITTER), float)

      # Limits configuration
      self.max_file_size_mb = get_env('MAX_FILE_SIZE_MB',
        limits_section.getint('max_file_size_mb', fallback=self.DEF_MAX_FILE_SIZE_MB), int)
      self.max_query_file_size_mb = get_env('MAX_QUERY_FILE_SIZE_MB',
        limits_section.getint('max_query_file_size_mb', fallback=self.DEF_MAX_QUERY_FILE_SIZE_MB), int)
      self.memory_cache_size = get_env('MEMORY_CACHE_SIZE',
        limits_section.getint('memory_cache_size', fallback=self.DEF_MEMORY_CACHE_SIZE), int)
      self.api_key_min_length = get_env('API_KEY_MIN_LENGTH',
        limits_section.getint('api_key_min_length', fallback=self.DEF_API_KEY_MIN_LENGTH), int)
      self.max_query_length = get_env('MAX_QUERY_LENGTH',
        limits_section.getint('max_query_length', fallback=self.DEF_MAX_QUERY_LENGTH), int)
      self.max_config_value_length = get_env('MAX_CONFIG_VALUE_LENGTH',
        limits_section.getint('max_config_value_length', fallback=self.DEF_MAX_CONFIG_VALUE_LENGTH), int)
      self.max_json_size = get_env('MAX_JSON_SIZE',
        limits_section.getint('max_json_size', fallback=self.DEF_MAX_JSON_SIZE), int)

      # Performance configuration
      self.embedding_batch_size = get_env('EMBEDDING_BATCH_SIZE',
        performance_section.getint('embedding_batch_size', fallback=self.DEF_EMBEDDING_BATCH_SIZE), int)
      self.checkpoint_interval = get_env('CHECKPOINT_INTERVAL',
        performance_section.getint('checkpoint_interval', fallback=self.DEF_CHECKPOINT_INTERVAL), int)
      self.commit_frequency = get_env('COMMIT_FREQUENCY',
        performance_section.getint('commit_frequency', fallback=self.DEF_COMMIT_FREQUENCY), int)
      self.io_thread_pool_size = get_env('IO_THREAD_POOL_SIZE',
        performance_section.getint('io_thread_pool_size', fallback=self.DEF_IO_THREAD_POOL_SIZE), int)
      self.cache_thread_pool_size = get_env('CACHE_THREAD_POOL_SIZE',
        performance_section.getint('cache_thread_pool_size', fallback=self.DEF_CACHE_THREAD_POOL_SIZE), int)
      self.file_processing_batch_size = get_env('FILE_PROCESSING_BATCH_SIZE',
        performance_section.getint('file_processing_batch_size', fallback=self.DEF_FILE_PROCESSING_BATCH_SIZE), int)
      self.sql_batch_size = get_env('SQL_BATCH_SIZE',
        performance_section.getint('sql_batch_size', fallback=self.DEF_SQL_BATCH_SIZE), int)
      self.reference_batch_size = get_env('REFERENCE_BATCH_SIZE',
        performance_section.getint('reference_batch_size', fallback=self.DEF_REFERENCE_BATCH_SIZE), int)
      self.query_cache_ttl_days = get_env('QUERY_CACHE_TTL_DAYS',
        performance_section.getint('query_cache_ttl_days', fallback=self.DEF_QUERY_CACHE_TTL_DAYS), int)
      self.default_editor = get_env('DEFAULT_EDITOR',
        performance_section.get('default_editor', fallback=self.DEF_DEFAULT_EDITOR))

      # Algorithms configuration
      self.high_dimension_threshold = get_env('HIGH_DIMENSION_THRESHOLD',
        algorithms_section.getint('high_dimension_threshold', fallback=self.DEF_HIGH_DIMENSION_THRESHOLD), int)
      self.small_dataset_threshold = get_env('SMALL_DATASET_THRESHOLD',
        algorithms_section.getint('small_dataset_threshold', fallback=self.DEF_SMALL_DATASET_THRESHOLD), int)
      self.medium_dataset_threshold = get_env('MEDIUM_DATASET_THRESHOLD',
        algorithms_section.getint('medium_dataset_threshold', fallback=self.DEF_MEDIUM_DATASET_THRESHOLD), int)
      self.ivf_centroid_multiplier = get_env('IVF_CENTROID_MULTIPLIER',
        algorithms_section.getint('ivf_centroid_multiplier', fallback=self.DEF_IVF_CENTROID_MULTIPLIER), int)
      self.max_centroids = get_env('MAX_CENTROIDS',
        algorithms_section.getint('max_centroids', fallback=self.DEF_MAX_CENTROIDS), int)
      self.token_estimation_sample_size = get_env('TOKEN_ESTIMATION_SAMPLE_SIZE',
        algorithms_section.getint('token_estimation_sample_size', fallback=self.DEF_TOKEN_ESTIMATION_SAMPLE_SIZE), int)
      self.token_estimation_multiplier = get_env('TOKEN_ESTIMATION_MULTIPLIER',
        algorithms_section.getfloat('token_estimation_multiplier', fallback=self.DEF_TOKEN_ESTIMATION_MULTIPLIER), float)
      self.similarity_threshold = get_env('SIMILARITY_THRESHOLD',
        algorithms_section.getfloat('similarity_threshold', fallback=self.DEF_SIMILARITY_THRESHOLD), float)
      self.low_similarity_scope_factor = get_env('LOW_SIMILARITY_SCOPE_FACTOR',
        algorithms_section.getfloat('low_similarity_scope_factor', fallback=self.DEF_LOW_SIMILARITY_SCOPE_FACTOR), float)
      self.max_chunk_overlap = get_env('MAX_CHUNK_OVERLAP',
        algorithms_section.getint('max_chunk_overlap', fallback=self.DEF_MAX_CHUNK_OVERLAP), int)
      self.overlap_ratio = get_env('OVERLAP_RATIO',
        algorithms_section.getfloat('overlap_ratio', fallback=self.DEF_OVERLAP_RATIO), float)
      self.heading_search_limit = get_env('HEADING_SEARCH_LIMIT',
        algorithms_section.getint('heading_search_limit', fallback=self.DEF_HEADING_SEARCH_LIMIT), int)
      self.entity_extraction_limit = get_env('ENTITY_EXTRACTION_LIMIT',
        algorithms_section.getint('entity_extraction_limit', fallback=self.DEF_ENTITY_EXTRACTION_LIMIT), int)
      self.default_dir_permissions = get_env('DEFAULT_DIR_PERMISSIONS',
        algorithms_section.getint('default_dir_permissions', fallback=self.DEF_DEFAULT_DIR_PERMISSIONS), int)
      self.default_code_language = get_env('DEFAULT_CODE_LANGUAGE',
        algorithms_section.get('default_code_language', fallback=self.DEF_DEFAULT_CODE_LANGUAGE))
      
      # Handle list parameter specially
      stopword_langs_str = algorithms_section.get('additional_stopword_languages', fallback=','.join(self.DEF_ADDITIONAL_STOPWORD_LANGUAGES))
      self.additional_stopword_languages = [lang.strip() for lang in stopword_langs_str.split(',') if lang.strip()]
      
      # BM25/Hybrid search configuration
      self.enable_hybrid_search = get_env('ENABLE_HYBRID_SEARCH',
        algorithms_section.getboolean('enable_hybrid_search', fallback=self.DEF_ENABLE_HYBRID_SEARCH), bool)
      self.vector_weight = get_env('VECTOR_WEIGHT',
        algorithms_section.getfloat('vector_weight', fallback=self.DEF_VECTOR_WEIGHT), float)
      self.bm25_k1 = get_env('BM25_K1',
        algorithms_section.getfloat('bm25_k1', fallback=self.DEF_BM25_K1), float)
      self.bm25_b = get_env('BM25_B',
        algorithms_section.getfloat('bm25_b', fallback=self.DEF_BM25_B), float)
      self.bm25_min_token_length = get_env('BM25_MIN_TOKEN_LENGTH',
        algorithms_section.getint('bm25_min_token_length', fallback=self.DEF_BM25_MIN_TOKEN_LENGTH), int)
      self.bm25_rebuild_threshold = get_env('BM25_REBUILD_THRESHOLD',
        algorithms_section.getint('bm25_rebuild_threshold', fallback=self.DEF_BM25_REBUILD_THRESHOLD), int)
      
      # Query enhancement configuration
      self.enable_query_enhancement = get_env('ENABLE_QUERY_ENHANCEMENT',
        algorithms_section.getboolean('enable_query_enhancement', fallback=self.DEF_ENABLE_QUERY_ENHANCEMENT), bool)
      self.query_enhancement_synonyms = get_env('QUERY_ENHANCEMENT_SYNONYMS',
        algorithms_section.getboolean('query_enhancement_synonyms', fallback=self.DEF_QUERY_ENHANCEMENT_SYNONYMS), bool)
      self.query_enhancement_spelling = get_env('QUERY_ENHANCEMENT_SPELLING',
        algorithms_section.getboolean('query_enhancement_spelling', fallback=self.DEF_QUERY_ENHANCEMENT_SPELLING), bool)
      self.max_synonyms_per_word = get_env('MAX_SYNONYMS_PER_WORD',
        algorithms_section.getint('max_synonyms_per_word', fallback=self.DEF_MAX_SYNONYMS_PER_WORD), int)
      self.query_enhancement_cache_ttl_days = get_env('QUERY_ENHANCEMENT_CACHE_TTL_DAYS',
        algorithms_section.getint('query_enhancement_cache_ttl_days', fallback=self.DEF_QUERY_ENHANCEMENT_CACHE_TTL_DAYS), int)
      self.spelling_correction_threshold = get_env('SPELLING_CORRECTION_THRESHOLD',
        algorithms_section.getfloat('spelling_correction_threshold', fallback=self.DEF_SPELLING_CORRECTION_THRESHOLD), float)
      self.synonym_relevance_threshold = get_env('SYNONYM_RELEVANCE_THRESHOLD',
        algorithms_section.getfloat('synonym_relevance_threshold', fallback=self.DEF_SYNONYM_RELEVANCE_THRESHOLD), float)
      
      # Reranking configuration
      self.enable_reranking = get_env('ENABLE_RERANKING',
        algorithms_section.getboolean('enable_reranking', fallback=self.DEF_ENABLE_RERANKING), bool)
      self.reranking_model = get_env('RERANKING_MODEL',
        algorithms_section.get('reranking_model', fallback=self.DEF_RERANKING_MODEL), str)
      self.reranking_top_k = get_env('RERANKING_TOP_K',
        algorithms_section.getint('reranking_top_k', fallback=self.DEF_RERANKING_TOP_K), int)
      self.reranking_batch_size = get_env('RERANKING_BATCH_SIZE',
        algorithms_section.getint('reranking_batch_size', fallback=self.DEF_RERANKING_BATCH_SIZE), int)
      self.reranking_device = get_env('RERANKING_DEVICE',
        algorithms_section.get('reranking_device', fallback=self.DEF_RERANKING_DEVICE), str)
      self.reranking_cache_size = get_env('RERANKING_CACHE_SIZE',
        algorithms_section.getint('reranking_cache_size', fallback=self.DEF_RERANKING_CACHE_SIZE), int)
    else:
      # Original configuration
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

      # New configuration parameters (for kwargs support)
      # API parameters
      self.api_call_delay_seconds = kwargs.get('api_call_delay_seconds', self.DEF_API_CALL_DELAY_SECONDS)
      self.api_max_retries = kwargs.get('api_max_retries', self.DEF_API_MAX_RETRIES)
      self.api_max_concurrency = kwargs.get('api_max_concurrency', self.DEF_API_MAX_CONCURRENCY)
      self.api_min_concurrency = kwargs.get('api_min_concurrency', self.DEF_API_MIN_CONCURRENCY)
      self.backoff_exponent = kwargs.get('backoff_exponent', self.DEF_BACKOFF_EXPONENT)
      self.backoff_jitter = kwargs.get('backoff_jitter', self.DEF_BACKOFF_JITTER)

      # Limits parameters
      self.max_file_size_mb = kwargs.get('max_file_size_mb', self.DEF_MAX_FILE_SIZE_MB)
      self.max_query_file_size_mb = kwargs.get('max_query_file_size_mb', self.DEF_MAX_QUERY_FILE_SIZE_MB)
      self.memory_cache_size = kwargs.get('memory_cache_size', self.DEF_MEMORY_CACHE_SIZE)
      self.api_key_min_length = kwargs.get('api_key_min_length', self.DEF_API_KEY_MIN_LENGTH)
      self.max_query_length = kwargs.get('max_query_length', self.DEF_MAX_QUERY_LENGTH)
      self.max_config_value_length = kwargs.get('max_config_value_length', self.DEF_MAX_CONFIG_VALUE_LENGTH)
      self.max_json_size = kwargs.get('max_json_size', self.DEF_MAX_JSON_SIZE)

      # Performance parameters
      self.embedding_batch_size = kwargs.get('embedding_batch_size', self.DEF_EMBEDDING_BATCH_SIZE)
      self.checkpoint_interval = kwargs.get('checkpoint_interval', self.DEF_CHECKPOINT_INTERVAL)
      self.commit_frequency = kwargs.get('commit_frequency', self.DEF_COMMIT_FREQUENCY)
      self.io_thread_pool_size = kwargs.get('io_thread_pool_size', self.DEF_IO_THREAD_POOL_SIZE)
      self.file_processing_batch_size = kwargs.get('file_processing_batch_size', self.DEF_FILE_PROCESSING_BATCH_SIZE)
      self.sql_batch_size = kwargs.get('sql_batch_size', self.DEF_SQL_BATCH_SIZE)
      self.reference_batch_size = kwargs.get('reference_batch_size', self.DEF_REFERENCE_BATCH_SIZE)
      self.query_cache_ttl_days = kwargs.get('query_cache_ttl_days', self.DEF_QUERY_CACHE_TTL_DAYS)
      self.default_editor = kwargs.get('default_editor', self.DEF_DEFAULT_EDITOR)

      # Algorithms parameters
      self.high_dimension_threshold = kwargs.get('high_dimension_threshold', self.DEF_HIGH_DIMENSION_THRESHOLD)
      self.small_dataset_threshold = kwargs.get('small_dataset_threshold', self.DEF_SMALL_DATASET_THRESHOLD)
      self.medium_dataset_threshold = kwargs.get('medium_dataset_threshold', self.DEF_MEDIUM_DATASET_THRESHOLD)
      self.ivf_centroid_multiplier = kwargs.get('ivf_centroid_multiplier', self.DEF_IVF_CENTROID_MULTIPLIER)
      self.max_centroids = kwargs.get('max_centroids', self.DEF_MAX_CENTROIDS)
      self.token_estimation_sample_size = kwargs.get('token_estimation_sample_size', self.DEF_TOKEN_ESTIMATION_SAMPLE_SIZE)
      self.token_estimation_multiplier = kwargs.get('token_estimation_multiplier', self.DEF_TOKEN_ESTIMATION_MULTIPLIER)
      self.similarity_threshold = kwargs.get('similarity_threshold', self.DEF_SIMILARITY_THRESHOLD)
      self.low_similarity_scope_factor = kwargs.get('low_similarity_scope_factor', self.DEF_LOW_SIMILARITY_SCOPE_FACTOR)
      self.max_chunk_overlap = kwargs.get('max_chunk_overlap', self.DEF_MAX_CHUNK_OVERLAP)
      self.overlap_ratio = kwargs.get('overlap_ratio', self.DEF_OVERLAP_RATIO)
      self.heading_search_limit = kwargs.get('heading_search_limit', self.DEF_HEADING_SEARCH_LIMIT)
      self.entity_extraction_limit = kwargs.get('entity_extraction_limit', self.DEF_ENTITY_EXTRACTION_LIMIT)
      self.default_dir_permissions = kwargs.get('default_dir_permissions', self.DEF_DEFAULT_DIR_PERMISSIONS)
      self.default_code_language = kwargs.get('default_code_language', self.DEF_DEFAULT_CODE_LANGUAGE)
      self.additional_stopword_languages = kwargs.get('additional_stopword_languages', self.DEF_ADDITIONAL_STOPWORD_LANGUAGES)
      
      # BM25/Hybrid search configuration
      self.enable_hybrid_search = kwargs.get('enable_hybrid_search', self.DEF_ENABLE_HYBRID_SEARCH)
      self.vector_weight = kwargs.get('vector_weight', self.DEF_VECTOR_WEIGHT)
      self.bm25_k1 = kwargs.get('bm25_k1', self.DEF_BM25_K1)
      self.bm25_b = kwargs.get('bm25_b', self.DEF_BM25_B)
      self.bm25_min_token_length = kwargs.get('bm25_min_token_length', self.DEF_BM25_MIN_TOKEN_LENGTH)
      self.bm25_rebuild_threshold = kwargs.get('bm25_rebuild_threshold', self.DEF_BM25_REBUILD_THRESHOLD)

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
