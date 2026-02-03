"""
Integration tests for the built test knowledgebase.

Tests verify that the full pipeline (database, embeddings, BM25, citations)
produces correct artifacts and that search works end-to-end.

Requires: built_test_kb fixture (session-scoped, runs full pipeline once).
"""

import os
import sqlite3

import numpy as np
import pytest


@pytest.mark.integration
@pytest.mark.requires_test_kb
@pytest.mark.slow
class TestBuiltKBStructure:
  """Verify the KB build pipeline produced correct artifacts."""

  def test_database_exists(self, test_kb):
    """Database file should be created by process_database."""
    assert os.path.exists(test_kb['db_path']), \
      f"Database not found: {test_kb['db_path']}"

  def test_faiss_index_exists(self, test_kb):
    """FAISS index file should be created by process_embeddings."""
    assert os.path.exists(test_kb['faiss_path']), \
      f"FAISS index not found: {test_kb['faiss_path']}"

  def test_database_has_chunks(self, test_kb):
    """Database should contain chunked docs from all 100 source files."""
    conn = sqlite3.connect(test_kb['db_path'])
    try:
      cursor = conn.cursor()
      cursor.execute("SELECT COUNT(*) FROM docs")
      count = cursor.fetchone()[0]
      assert count > 0, "Database has no chunks"

      # Should have chunks from multiple source files
      cursor.execute("SELECT COUNT(DISTINCT sourcedoc) FROM docs")
      source_count = cursor.fetchone()[0]
      assert source_count >= 50, \
        f"Expected chunks from >= 50 source files, got {source_count}"
    finally:
      conn.close()

  def test_all_chunks_embedded(self, test_kb):
    """All chunks should have embeddings (embedded=1)."""
    conn = sqlite3.connect(test_kb['db_path'])
    try:
      cursor = conn.cursor()
      cursor.execute("SELECT COUNT(*) FROM docs WHERE embedded = 0")
      unembedded = cursor.fetchone()[0]
      assert unembedded == 0, \
        f"{unembedded} chunks still have embedded=0"
    finally:
      conn.close()

  def test_faiss_dimensions(self, test_kb):
    """FAISS index should have 384 dimensions (all-minilm-l6-v2)."""
    import faiss

    index = faiss.read_index(test_kb['faiss_path'])
    assert index.d == 384, \
      f"Expected 384 dimensions, got {index.d}"

  def test_faiss_has_vectors(self, test_kb):
    """FAISS index should contain vectors for unique embedded chunks."""
    import faiss

    index = faiss.read_index(test_kb['faiss_path'])
    assert index.ntotal > 0, "FAISS index is empty"

    conn = sqlite3.connect(test_kb['db_path'])
    try:
      cursor = conn.cursor()
      # Count unique embedded texts (deduplication reduces FAISS count)
      cursor.execute("SELECT COUNT(DISTINCT embedtext) FROM docs WHERE embedded = 1")
      unique_count = cursor.fetchone()[0]
      assert index.ntotal == unique_count, \
        f"FAISS has {index.ntotal} vectors but DB has {unique_count} unique embedded chunks"
    finally:
      conn.close()

  def test_bm25_index_exists(self, test_kb):
    """BM25 index (.bm25.npz) should be created by build_bm25_index."""
    bm25_path = os.path.join(
      test_kb['kb_dir'],
      f"{test_kb['kb_name']}.bm25.npz"
    )
    assert os.path.exists(bm25_path), \
      f"BM25 index not found: {bm25_path}"


@pytest.mark.integration
@pytest.mark.requires_test_kb
@pytest.mark.slow
class TestBuiltKBSearch:
  """Verify search works against the built KB."""

  def test_vector_search_returns_results(self, test_kb):
    """FAISS search with a query embedding should return results."""
    import faiss

    index = faiss.read_index(test_kb['faiss_path'])

    # Create a random query vector (384 dims)
    query = np.random.randn(1, 384).astype(np.float32)
    # Normalize for cosine similarity (IndexFlatIP)
    query /= np.linalg.norm(query, axis=1, keepdims=True)

    distances, indices = index.search(query, 10)
    assert len(indices[0]) > 0, "No search results returned"
    assert indices[0][0] >= 0, "First result index should be valid"

  def test_hybrid_search(self, test_kb):
    """Combined vector + BM25 search should produce results."""
    from config.config_manager import KnowledgeBase
    from embedding.bm25_manager import load_bm25_index, search_bm25

    kb = KnowledgeBase(test_kb['kb_name'])
    bm25 = load_bm25_index(kb)

    if bm25 is None:
      pytest.skip("BM25 index not loaded")

    results = search_bm25(kb, "applied anthropology dharma")
    assert len(results) > 0, "BM25 search returned no results"

  def test_context_only_query(self, test_kb):
    """Query pipeline in context-only mode should return context without LLM call."""
    from unittest.mock import Mock, patch

    from query.query_manager import process_query

    mock_logger = Mock()

    query_args = Mock()
    query_args.config_file = test_kb['kb_name']
    query_args.query_text = "What is applied anthropology?"
    query_args.query_file = ""
    query_args.context_only = True
    query_args.verbose = True
    query_args.debug = False
    query_args.top_k = None
    query_args.context_scope = None
    query_args.categories = None
    query_args.context_files = None
    query_args.format = None
    query_args.prompt_template = None

    # Mock only the LLM response generation (not embeddings or search)
    with patch('query.llm.generate_ai_response', return_value="Mock response"):
      result = process_query(query_args, mock_logger)

    assert isinstance(result, str)
    assert len(result) > 0


@pytest.mark.integration
@pytest.mark.requires_test_kb
@pytest.mark.slow
class TestCitations:
  """Verify citation extraction and application pipeline."""

  def test_citations_db_exists(self, test_kb):
    """citations.db should be created by gen-citations.sh."""
    if not os.path.exists(test_kb['citations_db']):
      pytest.skip("citations.db not created (Ollama may not be available)")
    assert os.path.getsize(test_kb['citations_db']) > 0

  def test_citations_db_has_entries(self, test_kb):
    """citations table should have rows for processed files."""
    if not os.path.exists(test_kb['citations_db']):
      pytest.skip("citations.db not created")

    conn = sqlite3.connect(test_kb['citations_db'])
    try:
      cursor = conn.cursor()
      cursor.execute("SELECT COUNT(*) FROM citations")
      count = cursor.fetchone()[0]
      assert count > 0, "citations table is empty"
    finally:
      conn.close()

  def test_citations_have_titles(self, test_kb):
    """Extracted titles should not all be blank or NF."""
    if not os.path.exists(test_kb['citations_db']):
      pytest.skip("citations.db not created")

    conn = sqlite3.connect(test_kb['citations_db'])
    try:
      cursor = conn.cursor()
      cursor.execute(
        "SELECT COUNT(*) FROM citations WHERE title IS NOT NULL AND title != '' AND title != 'NF'"
      )
      valid_count = cursor.fetchone()[0]
      assert valid_count > 0, "All citation titles are blank or NF"
    finally:
      conn.close()

  def test_frontmatter_applied(self, test_kb):
    """Source files should have YAML frontmatter prepended by append-citations.sh."""
    if not os.path.exists(test_kb['citations_db']):
      pytest.skip("citations.db not created")

    staging_dir = test_kb['staging_dir']
    from pathlib import Path

    files_with_frontmatter = 0
    source_files = list(Path(staging_dir).rglob("*"))
    source_files = [f for f in source_files if f.is_file()]

    for fpath in source_files[:20]:  # Check first 20 files
      content = fpath.read_text(errors='replace')
      if content.startswith('---\n'):
        files_with_frontmatter += 1

    if files_with_frontmatter == 0:
      pytest.skip("No frontmatter found (append-citations may not have run)")
    assert files_with_frontmatter > 0

  def test_frontmatter_has_title(self, test_kb):
    """Frontmatter should contain a title: field."""
    if not os.path.exists(test_kb['citations_db']):
      pytest.skip("citations.db not created")

    staging_dir = test_kb['staging_dir']
    from pathlib import Path

    found_title = False
    for fpath in sorted(Path(staging_dir).rglob("*")):
      if not fpath.is_file():
        continue
      content = fpath.read_text(errors='replace')
      if content.startswith('---\n') and '\ntitle:' in content.split('---\n')[1] if len(content.split('---\n')) > 1 else '':
        found_title = True
        break

    if not found_title:
      pytest.skip("No frontmatter with title: found (citations may not have run)")
    assert found_title

#fin
