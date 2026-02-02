"""
Unit tests for config/models.py — Pydantic configuration models.
"""

import os
import tempfile
from unittest.mock import patch

import pytest

from config.models import (
  AlgorithmsConfig,
  ApiConfig,
  DefaultConfig,
  KBConfig,
  LimitsConfig,
  PerformanceConfig,
)


class TestDefaultConfig:
  """Test DefaultConfig model defaults."""

  def test_defaults(self):
    cfg = DefaultConfig()
    assert cfg.vector_model == 'text-embedding-3-small'
    assert cfg.vector_dimensions == 1536
    assert cfg.vector_chunks == 200
    assert cfg.db_min_tokens == 100
    assert cfg.db_max_tokens == 200
    assert cfg.query_model == 'claude-sonnet-4-5'
    assert cfg.query_top_k == 50
    assert cfg.query_temperature == 0.0
    assert cfg.query_max_tokens == 4000
    assert cfg.query_context_files == []


class TestApiConfig:
  """Test ApiConfig model defaults."""

  def test_defaults(self):
    cfg = ApiConfig()
    assert cfg.api_call_delay_seconds == 0.05
    assert cfg.api_max_retries == 20
    assert cfg.api_max_concurrency == 8
    assert cfg.backoff_exponent == 2


class TestLimitsConfig:
  """Test LimitsConfig model defaults."""

  def test_defaults(self):
    cfg = LimitsConfig()
    assert cfg.max_file_size_mb == 100
    assert cfg.memory_cache_size == 10000
    assert cfg.max_query_length == 10000


class TestPerformanceConfig:
  """Test PerformanceConfig model defaults."""

  def test_defaults(self):
    cfg = PerformanceConfig()
    assert cfg.embedding_batch_size == 100
    assert cfg.sql_batch_size == 500
    assert cfg.default_editor == 'joe'
    assert cfg.use_memory_mapped_faiss is False


class TestAlgorithmsConfig:
  """Test AlgorithmsConfig model defaults."""

  def test_defaults(self):
    cfg = AlgorithmsConfig()
    assert cfg.enable_hybrid_search is False
    assert cfg.bm25_k1 == 1.2
    assert cfg.bm25_b == 0.75
    assert cfg.vector_weight == 0.7
    assert cfg.enable_reranking is True
    assert cfg.additional_stopword_languages == ['indonesian', 'french', 'german', 'swedish']
    assert cfg.encoding_fallbacks == ['utf-8', 'windows-1252', 'latin-1', 'cp1252']


class TestKBConfig:
  """Test KBConfig loading from .cfg files."""

  def test_defaults(self):
    """Test KBConfig with all defaults."""
    cfg = KBConfig()
    assert cfg.default.vector_model == 'text-embedding-3-small'
    assert cfg.api.api_max_retries == 20
    assert cfg.limits.max_file_size_mb == 100
    assert cfg.performance.embedding_batch_size == 100
    assert cfg.algorithms.enable_hybrid_search is False

  def test_from_cfg_file(self):
    """Test loading from a .cfg file path."""
    with tempfile.TemporaryDirectory() as tmpdir:
      cfg_path = os.path.join(tmpdir, 'test.cfg')
      with open(cfg_path, 'w') as f:
        f.write("""[DEFAULT]
vector_model = bge-m3
vector_dimensions = 1024
query_model = gpt-4o
query_top_k = 30
query_temperature = 0.1

[API]
api_max_retries = 10
api_max_concurrency = 16

[LIMITS]
max_file_size_mb = 200

[PERFORMANCE]
embedding_batch_size = 500

[ALGORITHMS]
enable_hybrid_search = true
vector_weight = 0.8
bm25_k1 = 1.5
""")
      cfg = KBConfig.from_cfg(cfg_path)
      assert cfg.default.vector_model == 'bge-m3'
      assert cfg.default.vector_dimensions == 1024
      assert cfg.default.query_model == 'gpt-4o'
      assert cfg.default.query_top_k == 30
      assert cfg.default.query_temperature == pytest.approx(0.1)
      assert cfg.api.api_max_retries == 10
      assert cfg.api.api_max_concurrency == 16
      assert cfg.limits.max_file_size_mb == 200
      assert cfg.performance.embedding_batch_size == 500
      assert cfg.algorithms.enable_hybrid_search is True
      assert cfg.algorithms.vector_weight == pytest.approx(0.8)
      assert cfg.algorithms.bm25_k1 == pytest.approx(1.5)

  def test_from_cfg_kb_name(self):
    """Test loading from a KB name resolved via VECTORDBS."""
    with tempfile.TemporaryDirectory() as tmpdir:
      kb_dir = os.path.join(tmpdir, 'myproject')
      os.makedirs(kb_dir)
      cfg_path = os.path.join(kb_dir, 'myproject.cfg')
      with open(cfg_path, 'w') as f:
        f.write("""[DEFAULT]
vector_model = custom-model
vector_dimensions = 768
""")
      with patch('config.models.VECTORDBS', tmpdir):
        cfg = KBConfig.from_cfg('myproject')
        assert cfg.default.vector_model == 'custom-model'
        assert cfg.default.vector_dimensions == 768

  def test_from_cfg_not_found(self):
    """Test FileNotFoundError for missing KB."""
    with tempfile.TemporaryDirectory() as tmpdir, \
         patch('config.models.VECTORDBS', tmpdir), \
         pytest.raises(FileNotFoundError):
      KBConfig.from_cfg('nonexistent')

  def test_env_override(self):
    """Test that environment variables override .cfg values."""
    with tempfile.TemporaryDirectory() as tmpdir:
      cfg_path = os.path.join(tmpdir, 'test.cfg')
      with open(cfg_path, 'w') as f:
        f.write("""[DEFAULT]
vector_model = from-file
query_top_k = 30
""")
      with patch.dict(os.environ, {
        'VECTOR_MODEL': 'from-env',
        'QUERY_TOP_K': '100',
      }):
        cfg = KBConfig.from_cfg(cfg_path)
        assert cfg.default.vector_model == 'from-env'
        assert cfg.default.query_top_k == 100

  def test_missing_sections_use_defaults(self):
    """Test that missing sections fall back to defaults."""
    with tempfile.TemporaryDirectory() as tmpdir:
      cfg_path = os.path.join(tmpdir, 'minimal.cfg')
      with open(cfg_path, 'w') as f:
        f.write("""[DEFAULT]
vector_model = test
""")
      cfg = KBConfig.from_cfg(cfg_path)
      assert cfg.default.vector_model == 'test'
      assert cfg.api.api_max_retries == 20
      assert cfg.limits.max_file_size_mb == 100
      assert cfg.performance.embedding_batch_size == 100
      assert cfg.algorithms.enable_hybrid_search is False

  def test_query_context_files(self):
    """Test parsing of comma-separated query_context_files."""
    with tempfile.TemporaryDirectory() as tmpdir:
      cfg_path = os.path.join(tmpdir, 'ctx.cfg')
      with open(cfg_path, 'w') as f:
        f.write("""[DEFAULT]
query_context_files = file1.txt,file2.txt,file3.txt
""")
      cfg = KBConfig.from_cfg(cfg_path)
      assert cfg.default.query_context_files == ['file1.txt', 'file2.txt', 'file3.txt']

  def test_stopword_languages_parsing(self):
    """Test parsing of comma-separated stopword languages."""
    with tempfile.TemporaryDirectory() as tmpdir:
      cfg_path = os.path.join(tmpdir, 'langs.cfg')
      with open(cfg_path, 'w') as f:
        f.write("""[DEFAULT]
vector_model = test

[ALGORITHMS]
additional_stopword_languages = spanish,italian,portuguese
""")
      cfg = KBConfig.from_cfg(cfg_path)
      assert cfg.algorithms.additional_stopword_languages == ['spanish', 'italian', 'portuguese']

  def test_production_kb(self):
    """Test loading the okusiassociates2 test KB."""
    cfg = KBConfig.from_cfg('okusiassociates2')
    assert cfg.default.vector_model == 'bge-m3'
    assert cfg.default.vector_dimensions == 1024
    assert cfg.default.query_model == 'claude-sonnet-4-5'
    assert cfg.algorithms.enable_hybrid_search is True
    assert cfg.api.api_max_retries == 20

#fin
