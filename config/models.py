"""
Pydantic configuration models for CustomKB knowledgebase system.

Provides typed, validated configuration classes that replace the monolithic
KnowledgeBase god object's config handling. Supports loading from .cfg files
(ConfigParser INI format) and environment variable overrides.

Configuration hierarchy (highest to lowest priority):
1. Environment variables
2. .cfg file values
3. Built-in defaults
"""

import configparser
import os
from typing import Any

from pydantic import BaseModel, Field

VECTORDBS = os.getenv('VECTORDBS', '/var/lib/vectordbs')


def _get_env(var_name: str, default: Any, cast_type: type = str) -> Any:
  """Get environment variable with type casting.

  Args:
    var_name: Environment variable name.
    default: Default value if not set.
    cast_type: Type to cast the value to.

  Returns:
    Cast value or default.
  """
  value = os.getenv(var_name)
  if value is None:
    return default
  try:
    if cast_type is bool:
      return value.lower() in ('true', '1', 'yes')
    return cast_type(value)
  except (ValueError, TypeError):
    return default


class DefaultConfig(BaseModel):
  """Core model and processing settings (DEFAULT section)."""
  vector_model: str = 'text-embedding-3-small'
  vector_dimensions: int = 1536
  vector_chunks: int = 200
  db_min_tokens: int = 100
  db_max_tokens: int = 200
  query_model: str = 'claude-sonnet-4-5'
  query_role: str = 'You are a helpful assistant.'
  query_top_k: int = 50
  query_context_scope: int = 4
  query_temperature: float = 0.0
  query_max_tokens: int = 4000
  reference_format: str = 'xml'
  query_prompt_template: str = 'default'
  query_context_files: list[str] = Field(default_factory=list)


class ApiConfig(BaseModel):
  """External API interaction parameters (API section)."""
  api_call_delay_seconds: float = 0.05
  api_max_retries: int = 20
  api_max_concurrency: int = 8
  api_min_concurrency: int = 3
  backoff_exponent: int = 2
  backoff_jitter: float = 0.1


class LimitsConfig(BaseModel):
  """Resource and security constraints (LIMITS section)."""
  max_file_size_mb: int = 100
  max_query_file_size_mb: int = 1
  memory_cache_size: int = 10000
  cache_memory_limit_mb: int = 500
  api_key_min_length: int = 20
  max_query_length: int = 10000
  max_config_value_length: int = 1000
  max_json_size: int = 10000


class PerformanceConfig(BaseModel):
  """Optimization and tuning settings (PERFORMANCE section)."""
  embedding_batch_size: int = 100
  checkpoint_interval: int = 10
  commit_frequency: int = 1000
  io_thread_pool_size: int = 4
  cache_thread_pool_size: int = 4
  file_processing_batch_size: int = 500
  sql_batch_size: int = 500
  reference_batch_size: int = 5
  query_cache_ttl_days: int = 7
  default_editor: str = 'joe'
  use_memory_mapped_faiss: bool = False


class AlgorithmsConfig(BaseModel):
  """Algorithm-specific thresholds and parameters (ALGORITHMS section)."""
  high_dimension_threshold: int = 1536
  small_dataset_threshold: int = 1000
  medium_dataset_threshold: int = 100000
  ivf_centroid_multiplier: int = 4
  max_centroids: int = 256
  token_estimation_sample_size: int = 10
  token_estimation_multiplier: float = 1.3
  similarity_threshold: float = 0.6
  low_similarity_scope_factor: float = 0.5
  max_chunk_overlap: int = 100
  overlap_ratio: float = 0.5
  heading_search_limit: int = 200
  entity_extraction_limit: int = 500
  default_dir_permissions: int = 0o770
  default_code_language: str = 'python'
  # Language detection
  language_detection_enabled: bool = False
  language_detection_confidence: float = 0.95
  language_detection_sample_size: int = 3072
  additional_stopword_languages: list[str] = Field(
    default_factory=lambda: ['indonesian', 'french', 'german', 'swedish']
  )
  # Encoding detection
  auto_detect_encoding: bool = True
  default_encoding: str = 'utf-8'
  encoding_fallbacks: list[str] = Field(
    default_factory=lambda: ['utf-8', 'windows-1252', 'latin-1', 'cp1252']
  )
  # BM25/Hybrid search
  enable_hybrid_search: bool = False
  hybrid_fusion_method: str = 'rrf'
  rrf_k: int = 60
  vector_weight: float = 0.7
  bm25_weight: float = 0.3
  bm25_k1: float = 1.2
  bm25_b: float = 0.75
  bm25_min_token_length: int = 2
  bm25_rebuild_threshold: int = 1000
  bm25_max_results: int = 1000
  # Query enhancement
  enable_query_enhancement: bool = True
  query_enhancement_synonyms: bool = True
  query_enhancement_spelling: bool = True
  max_synonyms_per_word: int = 2
  query_enhancement_cache_ttl_days: int = 30
  spelling_correction_threshold: float = 0.8
  synonym_relevance_threshold: float = 0.6
  # Categorization
  enable_categorization: bool = False
  # Reranking
  enable_reranking: bool = True
  reranking_model: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
  reranking_top_k: int = 20
  reranking_batch_size: int = 32
  reranking_device: str = 'cpu'
  reranking_cache_size: int = 1000
  # FAISS GPU
  faiss_gpu_batch_size: int = 1024
  faiss_gpu_use_float16: bool = True
  faiss_gpu_memory_buffer_gb: float = 4.0
  faiss_gpu_memory_limit_mb: int = 0
  faiss_nprobe: int = 32


class KBConfig(BaseModel):
  """Complete knowledgebase configuration.

  Aggregates all config sections into a single validated model.
  Provides factory methods for loading from .cfg files and KB names.

  Example:
    >>> config = KBConfig.from_cfg('okusiassociates2')
    >>> print(config.default.vector_model)
    'bge-m3'
    >>> print(config.api.api_max_retries)
    20
  """
  default: DefaultConfig = Field(default_factory=DefaultConfig)
  api: ApiConfig = Field(default_factory=ApiConfig)
  limits: LimitsConfig = Field(default_factory=LimitsConfig)
  performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
  algorithms: AlgorithmsConfig = Field(default_factory=AlgorithmsConfig)

  @classmethod
  def from_cfg(cls, kb_name_or_path: str) -> 'KBConfig':
    """Load configuration from a knowledgebase name or .cfg file path.

    Resolution order:
    1. If path ends with .cfg, use it directly
    2. Otherwise, resolve as KB name: $VECTORDBS/<name>/<name>.cfg

    Environment variables override .cfg values (same names, uppercased).

    Args:
      kb_name_or_path: KB name (e.g., 'okusiassociates2') or path to .cfg file.

    Returns:
      Populated KBConfig instance.

    Raises:
      FileNotFoundError: If the .cfg file cannot be found.
    """
    cfg_path = cls._resolve_cfg_path(kb_name_or_path)
    if cfg_path is None:
      raise FileNotFoundError(
        f"Cannot find config for '{kb_name_or_path}'. "
        f"Looked in {VECTORDBS}/"
      )

    parser = configparser.ConfigParser()
    try:
      parser.read(cfg_path)
    except configparser.Error:
      # Corrupted config file — return defaults
      return cls()

    default = cls._load_default_section(parser)
    api = cls._load_api_section(parser)
    limits = cls._load_limits_section(parser)
    performance = cls._load_performance_section(parser)
    algorithms = cls._load_algorithms_section(parser)

    return cls(
      default=default,
      api=api,
      limits=limits,
      performance=performance,
      algorithms=algorithms,
    )

  @staticmethod
  def _resolve_cfg_path(kb_name_or_path: str) -> str | None:
    """Resolve a KB name or path to an actual .cfg file path.

    Args:
      kb_name_or_path: KB name or .cfg file path.

    Returns:
      Resolved path or None if not found.
    """
    if kb_name_or_path.endswith('.cfg') and os.path.isfile(kb_name_or_path):
      return kb_name_or_path

    # Strip path components and .cfg extension
    name = os.path.basename(kb_name_or_path)
    if name.endswith('.cfg'):
      name = name[:-4]

    cfg_path = os.path.join(VECTORDBS, name, f'{name}.cfg')
    if os.path.isfile(cfg_path):
      return cfg_path

    return None

  @classmethod
  def _load_default_section(cls, parser: configparser.ConfigParser) -> DefaultConfig:
    """Load DEFAULT section from parsed config."""
    df = parser['DEFAULT']
    ctx_str = df.get('query_context_files', '')
    ctx_files = [f.strip() for f in ctx_str.split(',') if f.strip()] if ctx_str else []

    # Environment override for context files
    ctx_env = os.getenv('QUERY_CONTEXT_FILES')
    if ctx_env is not None:
      ctx_files = [f.strip() for f in ctx_env.split(',') if f.strip()]

    return DefaultConfig(
      vector_model=_get_env('VECTOR_MODEL', df.get('vector_model', 'text-embedding-3-small')),
      vector_dimensions=_get_env('VECTOR_DIMENSIONS', _cfg_int(df, 'vector_dimensions', 1536), int),
      vector_chunks=_get_env('VECTOR_CHUNKS', _cfg_int(df, 'vector_chunks', 200), int),
      db_min_tokens=_get_env('DB_MIN_TOKENS', _cfg_int(df, 'db_min_tokens', 100), int),
      db_max_tokens=_get_env('DB_MAX_TOKENS', _cfg_int(df, 'db_max_tokens', 200), int),
      query_model=_get_env('QUERY_MODEL', df.get('query_model', 'claude-sonnet-4-5')),
      query_role=_get_env('QUERY_ROLE', df.get('query_role', 'You are a helpful assistant.')),
      query_top_k=_get_env('QUERY_TOP_K', _cfg_int(df, 'query_top_k', 50), int),
      query_context_scope=_get_env('QUERY_CONTEXT_SCOPE', _cfg_int(df, 'query_context_scope', 4), int),
      query_temperature=_get_env('QUERY_TEMPERATURE', _cfg_float(df, 'query_temperature', 0.0), float),
      query_max_tokens=_get_env('QUERY_MAX_TOKENS', _cfg_int(df, 'query_max_tokens', 4000), int),
      reference_format=_get_env('REFERENCE_FORMAT', df.get('reference_format', 'xml')),
      query_prompt_template=_get_env('QUERY_PROMPT_TEMPLATE', df.get('query_prompt_template', 'default')),
      query_context_files=ctx_files,
    )

  @classmethod
  def _load_api_section(cls, parser: configparser.ConfigParser) -> ApiConfig:
    """Load API section from parsed config."""
    section = parser['API'] if 'API' in parser else parser['DEFAULT']
    return ApiConfig(
      api_call_delay_seconds=_get_env('API_CALL_DELAY_SECONDS', _cfg_float(section, 'api_call_delay_seconds', 0.05), float),
      api_max_retries=_get_env('API_MAX_RETRIES', _cfg_int(section, 'api_max_retries', 20), int),
      api_max_concurrency=_get_env('API_MAX_CONCURRENCY', _cfg_int(section, 'api_max_concurrency', 8), int),
      api_min_concurrency=_get_env('API_MIN_CONCURRENCY', _cfg_int(section, 'api_min_concurrency', 3), int),
      backoff_exponent=_get_env('BACKOFF_EXPONENT', _cfg_int(section, 'backoff_exponent', 2), int),
      backoff_jitter=_get_env('BACKOFF_JITTER', _cfg_float(section, 'backoff_jitter', 0.1), float),
    )

  @classmethod
  def _load_limits_section(cls, parser: configparser.ConfigParser) -> LimitsConfig:
    """Load LIMITS section from parsed config."""
    section = parser['LIMITS'] if 'LIMITS' in parser else parser['DEFAULT']
    return LimitsConfig(
      max_file_size_mb=_get_env('MAX_FILE_SIZE_MB', _cfg_int(section, 'max_file_size_mb', 100), int),
      max_query_file_size_mb=_get_env('MAX_QUERY_FILE_SIZE_MB', _cfg_int(section, 'max_query_file_size_mb', 1), int),
      memory_cache_size=_get_env('MEMORY_CACHE_SIZE', _cfg_int(section, 'memory_cache_size', 10000), int),
      cache_memory_limit_mb=_get_env('CACHE_MEMORY_LIMIT_MB', _cfg_int(section, 'cache_memory_limit_mb', 500), int),
      api_key_min_length=_get_env('API_KEY_MIN_LENGTH', _cfg_int(section, 'api_key_min_length', 20), int),
      max_query_length=_get_env('MAX_QUERY_LENGTH', _cfg_int(section, 'max_query_length', 10000), int),
      max_config_value_length=_get_env('MAX_CONFIG_VALUE_LENGTH', _cfg_int(section, 'max_config_value_length', 1000), int),
      max_json_size=_get_env('MAX_JSON_SIZE', _cfg_int(section, 'max_json_size', 10000), int),
    )

  @classmethod
  def _load_performance_section(cls, parser: configparser.ConfigParser) -> PerformanceConfig:
    """Load PERFORMANCE section from parsed config."""
    section = parser['PERFORMANCE'] if 'PERFORMANCE' in parser else parser['DEFAULT']
    return PerformanceConfig(
      embedding_batch_size=_get_env('EMBEDDING_BATCH_SIZE', _cfg_int(section, 'embedding_batch_size', 100), int),
      checkpoint_interval=_get_env('CHECKPOINT_INTERVAL', _cfg_int(section, 'checkpoint_interval', 10), int),
      commit_frequency=_get_env('COMMIT_FREQUENCY', _cfg_int(section, 'commit_frequency', 1000), int),
      io_thread_pool_size=_get_env('IO_THREAD_POOL_SIZE', _cfg_int(section, 'io_thread_pool_size', 4), int),
      cache_thread_pool_size=_get_env('CACHE_THREAD_POOL_SIZE', _cfg_int(section, 'cache_thread_pool_size', 4), int),
      file_processing_batch_size=_get_env('FILE_PROCESSING_BATCH_SIZE', _cfg_int(section, 'file_processing_batch_size', 500), int),
      sql_batch_size=_get_env('SQL_BATCH_SIZE', _cfg_int(section, 'sql_batch_size', 500), int),
      reference_batch_size=_get_env('REFERENCE_BATCH_SIZE', _cfg_int(section, 'reference_batch_size', 5), int),
      query_cache_ttl_days=_get_env('QUERY_CACHE_TTL_DAYS', _cfg_int(section, 'query_cache_ttl_days', 7), int),
      default_editor=_get_env('DEFAULT_EDITOR', section.get('default_editor', 'joe') if hasattr(section, 'get') else 'joe'),
      use_memory_mapped_faiss=_get_env('USE_MEMORY_MAPPED_FAISS', _cfg_bool(section, 'use_memory_mapped_faiss', False), bool),
    )

  @classmethod
  def _load_algorithms_section(cls, parser: configparser.ConfigParser) -> AlgorithmsConfig:
    """Load ALGORITHMS section from parsed config."""
    section = parser['ALGORITHMS'] if 'ALGORITHMS' in parser else parser['DEFAULT']
    def _g(key, default):
      return section.get(key, default) if hasattr(section, 'get') else default

    # List fields
    stopword_str = _g('additional_stopword_languages', 'indonesian,french,german,swedish')
    stopwords = [lang.strip() for lang in stopword_str.split(',') if lang.strip()]

    fallback_str = _g('encoding_fallbacks', 'utf-8,windows-1252,latin-1,cp1252')
    fallbacks = [enc.strip() for enc in fallback_str.split(',') if enc.strip()]

    return AlgorithmsConfig(
      high_dimension_threshold=_get_env('HIGH_DIMENSION_THRESHOLD', _cfg_int(section, 'high_dimension_threshold', 1536), int),
      small_dataset_threshold=_get_env('SMALL_DATASET_THRESHOLD', _cfg_int(section, 'small_dataset_threshold', 1000), int),
      medium_dataset_threshold=_get_env('MEDIUM_DATASET_THRESHOLD', _cfg_int(section, 'medium_dataset_threshold', 100000), int),
      ivf_centroid_multiplier=_get_env('IVF_CENTROID_MULTIPLIER', _cfg_int(section, 'ivf_centroid_multiplier', 4), int),
      max_centroids=_get_env('MAX_CENTROIDS', _cfg_int(section, 'max_centroids', 256), int),
      token_estimation_sample_size=_get_env('TOKEN_ESTIMATION_SAMPLE_SIZE', _cfg_int(section, 'token_estimation_sample_size', 10), int),
      token_estimation_multiplier=_get_env('TOKEN_ESTIMATION_MULTIPLIER', _cfg_float(section, 'token_estimation_multiplier', 1.3), float),
      similarity_threshold=_get_env('SIMILARITY_THRESHOLD', _cfg_float(section, 'similarity_threshold', 0.6), float),
      low_similarity_scope_factor=_get_env('LOW_SIMILARITY_SCOPE_FACTOR', _cfg_float(section, 'low_similarity_scope_factor', 0.5), float),
      max_chunk_overlap=_get_env('MAX_CHUNK_OVERLAP', _cfg_int(section, 'max_chunk_overlap', 100), int),
      overlap_ratio=_get_env('OVERLAP_RATIO', _cfg_float(section, 'overlap_ratio', 0.5), float),
      heading_search_limit=_get_env('HEADING_SEARCH_LIMIT', _cfg_int(section, 'heading_search_limit', 200), int),
      entity_extraction_limit=_get_env('ENTITY_EXTRACTION_LIMIT', _cfg_int(section, 'entity_extraction_limit', 500), int),
      default_dir_permissions=_get_env('DEFAULT_DIR_PERMISSIONS', _cfg_int(section, 'default_dir_permissions', 0o770), int),
      default_code_language=_get_env('DEFAULT_CODE_LANGUAGE', _g('default_code_language', 'python')),
      language_detection_enabled=_get_env('LANGUAGE_DETECTION_ENABLED', _cfg_bool(section, 'language_detection_enabled', False), bool),
      language_detection_confidence=_get_env('LANGUAGE_DETECTION_CONFIDENCE', _cfg_float(section, 'language_detection_confidence', 0.95), float),
      language_detection_sample_size=_get_env('LANGUAGE_DETECTION_SAMPLE_SIZE', _cfg_int(section, 'language_detection_sample_size', 3072), int),
      additional_stopword_languages=stopwords,
      auto_detect_encoding=_get_env('AUTO_DETECT_ENCODING', _cfg_bool(section, 'auto_detect_encoding', True), bool),
      default_encoding=_get_env('DEFAULT_ENCODING', _g('default_encoding', 'utf-8')),
      encoding_fallbacks=fallbacks,
      enable_hybrid_search=_get_env('ENABLE_HYBRID_SEARCH', _cfg_bool(section, 'enable_hybrid_search', False), bool),
      hybrid_fusion_method=_get_env('HYBRID_FUSION_METHOD', _g('hybrid_fusion_method', 'rrf')),
      rrf_k=_get_env('RRF_K', _cfg_int(section, 'rrf_k', 60), int),
      vector_weight=_get_env('VECTOR_WEIGHT', _cfg_float(section, 'vector_weight', 0.7), float),
      bm25_weight=_get_env('BM25_WEIGHT', _cfg_float(section, 'bm25_weight', 0.3), float),
      bm25_k1=_get_env('BM25_K1', _cfg_float(section, 'bm25_k1', 1.2), float),
      bm25_b=_get_env('BM25_B', _cfg_float(section, 'bm25_b', 0.75), float),
      bm25_min_token_length=_get_env('BM25_MIN_TOKEN_LENGTH', _cfg_int(section, 'bm25_min_token_length', 2), int),
      bm25_rebuild_threshold=_get_env('BM25_REBUILD_THRESHOLD', _cfg_int(section, 'bm25_rebuild_threshold', 1000), int),
      bm25_max_results=_get_env('BM25_MAX_RESULTS', _cfg_int(section, 'bm25_max_results', 1000), int),
      enable_query_enhancement=_get_env('ENABLE_QUERY_ENHANCEMENT', _cfg_bool(section, 'enable_query_enhancement', True), bool),
      query_enhancement_synonyms=_get_env('QUERY_ENHANCEMENT_SYNONYMS', _cfg_bool(section, 'query_enhancement_synonyms', True), bool),
      query_enhancement_spelling=_get_env('QUERY_ENHANCEMENT_SPELLING', _cfg_bool(section, 'query_enhancement_spelling', True), bool),
      max_synonyms_per_word=_get_env('MAX_SYNONYMS_PER_WORD', _cfg_int(section, 'max_synonyms_per_word', 2), int),
      query_enhancement_cache_ttl_days=_get_env('QUERY_ENHANCEMENT_CACHE_TTL_DAYS', _cfg_int(section, 'query_enhancement_cache_ttl_days', 30), int),
      spelling_correction_threshold=_get_env('SPELLING_CORRECTION_THRESHOLD', _cfg_float(section, 'spelling_correction_threshold', 0.8), float),
      synonym_relevance_threshold=_get_env('SYNONYM_RELEVANCE_THRESHOLD', _cfg_float(section, 'synonym_relevance_threshold', 0.6), float),
      enable_categorization=_get_env('ENABLE_CATEGORIZATION', _cfg_bool(section, 'enable_categorization', False), bool),
      enable_reranking=_get_env('ENABLE_RERANKING', _cfg_bool(section, 'enable_reranking', True), bool),
      reranking_model=_get_env('RERANKING_MODEL', _g('reranking_model', 'cross-encoder/ms-marco-MiniLM-L-6-v2')),
      reranking_top_k=_get_env('RERANKING_TOP_K', _cfg_int(section, 'reranking_top_k', 20), int),
      reranking_batch_size=_get_env('RERANKING_BATCH_SIZE', _cfg_int(section, 'reranking_batch_size', 32), int),
      reranking_device=_get_env('RERANKING_DEVICE', _g('reranking_device', 'cpu')),
      reranking_cache_size=_get_env('RERANKING_CACHE_SIZE', _cfg_int(section, 'reranking_cache_size', 1000), int),
      faiss_gpu_batch_size=_get_env('FAISS_GPU_BATCH_SIZE', _cfg_int(section, 'faiss_gpu_batch_size', 1024), int),
      faiss_gpu_use_float16=_get_env('FAISS_GPU_USE_FLOAT16', _cfg_bool(section, 'faiss_gpu_use_float16', True), bool),
      faiss_gpu_memory_buffer_gb=_get_env('FAISS_GPU_MEMORY_BUFFER_GB', _cfg_float(section, 'faiss_gpu_memory_buffer_gb', 4.0), float),
      faiss_gpu_memory_limit_mb=_get_env('FAISS_GPU_MEMORY_LIMIT_MB', _cfg_int(section, 'faiss_gpu_memory_limit_mb', 0), int),
      faiss_nprobe=_get_env('FAISS_NPROBE', _cfg_int(section, 'faiss_nprobe', 32), int),
    )


def _cfg_int(section: Any, key: str, default: int) -> int:
  """Safely get an integer from a config section."""
  try:
    if hasattr(section, 'getint'):
      return section.getint(key, fallback=default)
    val = section.get(key)
    return int(val) if val is not None else default
  except (ValueError, TypeError):
    return default


def _cfg_float(section: Any, key: str, default: float) -> float:
  """Safely get a float from a config section."""
  try:
    if hasattr(section, 'getfloat'):
      return section.getfloat(key, fallback=default)
    val = section.get(key)
    return float(val) if val is not None else default
  except (ValueError, TypeError):
    return default


def _cfg_bool(section: Any, key: str, default: bool) -> bool:
  """Get a boolean from a config section.

  Raises ValueError for invalid boolean values (fail-fast behavior).
  """
  if hasattr(section, 'getboolean'):
    return section.getboolean(key, fallback=default)
  val = section.get(key)
  if val is None:
    return default
  return str(val).lower() in ('true', '1', 'yes')

#fin
