#!/usr/bin/env python
"""
Configuration management for CustomKB knowledgebase system.

This module provides the core configuration infrastructure for CustomKB, handling:
- Configuration file loading and validation with security checks
- Environment variable overrides with type conversion
- Domain-style knowledgebase naming (e.g., 'example.com.cfg')
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

Configuration is backed by Pydantic models defined in config.models.
"""

import os
import time

from config.models import (
  AlgorithmsConfig,
  ApiConfig,
  DefaultConfig,
  KBConfig,
  LimitsConfig,
  PerformanceConfig,
)
from utils.logging_config import get_logger
from utils.text_utils import split_filepath

# Initialize vector database directory
VECTORDBS = os.getenv('VECTORDBS', '/var/lib/vectordbs')
if not os.path.exists(VECTORDBS):
  try:
    os.makedirs(VECTORDBS, mode=0o770, exist_ok=True)
  except OSError as e:
    raise OSError(f"Failed to create directory {VECTORDBS}: {e}") from e

logger = get_logger(__name__)


def get_kb_name(kb_input: str) -> str | None:
  """
  Extract and validate knowledgebase name from user input.

  Strips any path components and .cfg extension, then validates that
  the knowledgebase exists as a subdirectory in VECTORDBS.

  Args:
      kb_input: User input which may include paths or .cfg extension

  Returns:
      Clean knowledgebase name if valid, None otherwise
  """
  if not kb_input:
    logger.error("Knowledgebase name cannot be empty")
    return None

  # Remove any path components (like basename command)
  kb_name = os.path.basename(kb_input)

  # Remove .cfg extension if present
  if kb_name.endswith('.cfg'):
    kb_name = kb_name[:-4]

  # Validate KB name is not empty after cleaning
  if not kb_name:
    logger.error("Knowledgebase name cannot be empty after removing path/extension")
    return None

  # Check if knowledgebase directory exists
  kb_dir = os.path.join(VECTORDBS, kb_name)
  if not os.path.isdir(kb_dir):
    logger.error(f"Knowledgebase '{kb_name}' not found in {VECTORDBS}")

    # List available KBs for helpful error message
    try:
      available_kbs = [d for d in os.listdir(VECTORDBS)
                      if os.path.isdir(os.path.join(VECTORDBS, d))
                      and not d.startswith('.')]
      if available_kbs:
        logger.info(f"Available knowledgebases: {', '.join(sorted(available_kbs))}")
    except OSError:
      pass  # Ignore errors listing directory

    return None

  return kb_name


def get_fq_cfg_filename(cfgfile: str) -> str | None:
  """
  Resolve knowledgebase name to its configuration file path.

  This function now requires that knowledgebases exist as subdirectories
  within VECTORDBS. It strips any path components and .cfg extensions
  from the input to get a clean KB name.

  Args:
      cfgfile: Knowledgebase name (paths and .cfg extensions will be stripped)

  Returns:
      Full path to configuration file if KB exists, None otherwise

  Examples:
      >>> get_fq_cfg_filename('okusimail')
      '/var/lib/vectordbs/okusimail/okusimail.cfg'
      >>> get_fq_cfg_filename('/path/to/okusimail.cfg')
      '/var/lib/vectordbs/okusimail/okusimail.cfg'
  """
  # If the input already ends in .cfg and exists on disk, return it directly.
  # This supports absolute paths, relative paths, and sibling-directory references.
  if cfgfile.endswith('.cfg') and os.path.isfile(cfgfile):
    logger.debug(f"Using existing config file path: '{cfgfile}'")
    return cfgfile

  # Get clean KB name
  kb_name = get_kb_name(cfgfile)
  if not kb_name:
    return None

  # Construct config file path
  config_path = os.path.join(VECTORDBS, kb_name, f"{kb_name}.cfg")

  # Verify config file exists
  if not os.path.isfile(config_path):
    logger.error(f"Configuration file not found: {config_path}")
    logger.info(f"Expected to find {kb_name}.cfg in {os.path.join(VECTORDBS, kb_name)}/")
    return None

  logger.debug(f"Resolved '{cfgfile}' to '{config_path}'")
  return config_path


def _build_defaults_map() -> dict[str, tuple[type, object]]:
  """Build the DEF_* defaults map from Pydantic model defaults.

  Returns:
    Dict mapping DEF_NAME to (type, default_value) tuples.
  """
  from pydantic_core import PydanticUndefined

  mapping = {}
  for model_cls in (DefaultConfig, ApiConfig, LimitsConfig, PerformanceConfig, AlgorithmsConfig):
    for field_name, field_info in model_cls.model_fields.items():
      def_key = f'DEF_{field_name.upper()}'
      default = field_info.default
      if (default is PydanticUndefined or default is None) and field_info.default_factory is not None:
        default = field_info.default_factory()
      field_type = type(default) if default is not None else str
      mapping[def_key] = (field_type, default)
  return mapping


# Pre-compute the defaults map once at module level
_DEFAULTS_MAP = _build_defaults_map()


class KnowledgeBase:
  """
  Configuration and resource manager for a CustomKB knowledgebase instance.

  Backed by Pydantic models (KBConfig) for validated configuration. Provides
  flat attribute access for backward compatibility — all config parameters are
  accessible directly (e.g., kb.vector_model, kb.query_top_k).

  Configuration hierarchy: env vars > .cfg file values > built-in defaults.

  Runtime state (sql_connection, sql_cursor) is managed separately from
  configuration and is NOT part of the Pydantic model.

  Attributes:
      knowledge_base_name: The base name of this knowledgebase
      knowledge_base_db: Path to the SQLite database file
      knowledge_base_vector: Path to the FAISS vector index file
      _config: The underlying KBConfig Pydantic model
      sql_connection: SQLite connection (runtime state, initially None)
      sql_cursor: SQLite cursor (runtime state, initially None)

  Example:
      >>> kb = KnowledgeBase('myproject.cfg')
      >>> print(kb.vector_model)
      'text-embedding-3-small'
      >>> print(kb.knowledge_base_db)
      '/var/lib/vectordbs/myproject/myproject.db'
  """

  def __init__(self, kb_base: str, **kwargs):
    """
    Initialize a KnowledgeBase with configuration from file or defaults.

    Args:
        kb_base: Path to configuration file (e.g., 'myproject.cfg') or
            base name of the knowledgebase. If a .cfg file is provided,
            configuration will be loaded from it. Otherwise, defaults or
            kwargs will be used.
        **kwargs: Additional parameters to override configuration values.

    Raises:
        EnvironmentError: If required directories cannot be created.
        ValueError: If configuration values fail type conversion.
    """
    # Set DEF_* defaults for backward compatibility
    for def_key, (_type, default_val) in _DEFAULTS_MAP.items():
      setattr(self, def_key, default_val)

    self.start_time = int(time.time())

    # Resolve config file path
    cfg_path = kb_base
    if not kb_base.endswith('.cfg'):
      resolved_cfg = get_fq_cfg_filename(kb_base)
      if resolved_cfg:
        cfg_path = resolved_cfg

    # Load config via Pydantic models
    if cfg_path.endswith('.cfg') and os.path.isfile(cfg_path):
      self._config = KBConfig.from_cfg(cfg_path)
    else:
      self._config = KBConfig()

    # Flatten all config attributes onto self
    self._flatten_config()

    # Apply kwargs overrides
    for key, value in kwargs.items():
      if hasattr(self, key):
        setattr(self, key, value)

    # Legacy alias: vector_weight mirrors hybrid_vector_weight when loaded from cfg
    if 'vector_weight' not in kwargs and hasattr(self, 'hybrid_vector_weight'):
      self.vector_weight = self.hybrid_vector_weight

    # Set up database and vector file paths
    directory, basename, extension, fqfn = split_filepath(cfg_path)
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

    # Runtime state — NOT part of config
    self.sql_connection = None
    self.sql_cursor = None

  def _flatten_config(self) -> None:
    """Flatten KBConfig sub-models onto self for backward-compatible access."""
    sections = [
      self._config.default,
      self._config.api,
      self._config.limits,
      self._config.performance,
      self._config.algorithms,
    ]
    for section in sections:
      for field_name in type(section).model_fields:
        setattr(self, field_name, getattr(section, field_name))

    # Legacy aliases from the algorithms section
    if hasattr(self, 'vector_weight'):
      self.hybrid_vector_weight = self.vector_weight
    if hasattr(self, 'bm25_weight'):
      self.hybrid_bm25_weight = self.bm25_weight

  def save_config(self, output_to: str | None = None) -> None:
    """
    Save current configuration to a file or print to stderr.

    Args:
        output_to: Path to save config to. If None, prints to stderr.
    """
    if output_to:
      with open(output_to, 'w') as filehandle:
        print(f"# {self.knowledge_base_name}", file=filehandle)
        print("[DEFAULT]", file=filehandle)
        attrs = vars(self)
        for key, value in attrs.items():
          print(f"{key} = {value}", file=filehandle)
    else:
      import sys
      print(f"# {self.knowledge_base_name}", file=sys.stderr)
      attrs = vars(self)
      for key, value in attrs.items():
        print(f"{key} = {value}", file=sys.stderr)

#fin
