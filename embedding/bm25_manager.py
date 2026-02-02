#!/usr/bin/env python
"""
BM25 index management for CustomKB hybrid retrieval system.

This module handles BM25 index creation, serialization, and loading for keyword-based
retrieval that complements the vector-based semantic search.
"""

import json
import os
import sqlite3
from typing import TYPE_CHECKING, Any

import numpy as np
from rank_bm25 import BM25Okapi

from utils.logging_config import get_logger
from utils.text_utils import tokenize_for_bm25

if TYPE_CHECKING:
  from config.config_manager import KnowledgeBase

logger = get_logger(__name__)

def build_bm25_index(kb: 'KnowledgeBase') -> BM25Okapi | None:
  """
  Build BM25 index from database and save to disk.

  Args:
      kb: The KnowledgeBase instance with database connection.

  Returns:
      BM25Okapi instance if successful, None otherwise.
  """
  try:
    cursor = kb.sql_cursor
    if cursor is None:
      logger.warning("No database cursor available. Cannot build BM25 index.")
      return None
    cursor.execute("""
      SELECT id, bm25_tokens, doc_length
      FROM docs
      WHERE keyphrase_processed = 1 AND bm25_tokens IS NOT NULL AND bm25_tokens != ''
      ORDER BY id
    """)

    corpus = []
    doc_ids = []
    total_length = 0

    for doc_id, tokens_str, doc_length in cursor.fetchall():
      if tokens_str and tokens_str.strip():
        tokens = tokens_str.split()
        if tokens:  # Only add non-empty token lists
          corpus.append(tokens)
          doc_ids.append(doc_id)
          total_length += doc_length or len(tokens)

    if not corpus:
      logger.warning("No BM25 tokens found in database. Ensure documents are processed with BM25 enabled.")
      return None

    # Create BM25 index with configurable parameters
    k1 = getattr(kb, 'bm25_k1', 1.2)
    b = getattr(kb, 'bm25_b', 0.75)

    logger.info(f"Building BM25 index with k1={k1}, b={b}")
    bm25 = BM25Okapi(corpus, k1=k1, b=b)

    # Extract BM25 internal data for NPZ serialization
    bm25_path = get_bm25_index_path(kb)

    # Convert idf dict to two arrays for efficient storage
    idf_terms = list(bm25.idf.keys())
    idf_scores = [bm25.idf[term] for term in idf_terms]

    # Save arrays with NPZ (efficient binary format)
    # doc_freqs is a list of dicts needed by get_scores()
    np.savez(
      bm25_path,
      idf_terms=np.array(idf_terms, dtype=object),
      idf_scores=np.array(idf_scores, dtype=np.float32),
      doc_len=np.array(bm25.doc_len, dtype=np.int32),
      doc_ids=np.array(doc_ids, dtype=np.int32),
      doc_freqs=np.array(bm25.doc_freqs, dtype=object)
    )

    # Save metadata with JSON (human-readable, secure)
    metadata_path = bm25_path.replace('.bm25', '.bm25.json')
    metadata = {
      'total_docs': len(doc_ids),
      'total_tokens': total_length,
      'avgdl': float(bm25.avgdl),
      'corpus_size': int(bm25.corpus_size),
      'k1': k1,
      'b': b,
      'version': '2.0'  # Version 2.0 for NPZ format
    }

    with open(metadata_path, 'w') as f:
      json.dump(metadata, f, indent=2)

    logger.info(f"Built BM25 index with {len(doc_ids)} documents, saved to {bm25_path}")
    return bm25

  except sqlite3.Error as e:
    logger.error(f"Database error building BM25 index: {e}")
    return None
  except (ImportError, OSError, ValueError, RuntimeError) as e:
    logger.error(f"Error building BM25 index: {e}")
    return None

def load_bm25_index(kb: 'KnowledgeBase') -> dict[str, Any] | None:
  """
  Load BM25 index from disk (NPZ format with JSON metadata).

  Args:
      kb: The KnowledgeBase instance.

  Returns:
      Dictionary containing BM25 data if successful, None otherwise.
  """
  bm25_path = get_bm25_index_path(kb)
  # np.savez() automatically adds .npz extension, so the actual file is *.bm25.npz
  npz_path = bm25_path + '.npz'
  metadata_path = bm25_path + '.json'

  # Try loading NPZ format first (version 2.0)
  if os.path.exists(npz_path) and os.path.exists(metadata_path):
    try:
      # Load arrays from NPZ
      npz_data = np.load(npz_path, allow_pickle=True)

      # Load metadata from JSON
      with open(metadata_path) as f:
        metadata = json.load(f)

      # Reconstruct idf dictionary
      idf_terms = npz_data['idf_terms']
      idf_scores = npz_data['idf_scores']
      idf = {term: float(score) for term, score in zip(idf_terms, idf_scores, strict=False)}

      # Reconstruct BM25Okapi object
      bm25 = BM25Okapi.__new__(BM25Okapi)
      bm25.idf = idf
      bm25.doc_len = npz_data['doc_len'].tolist()
      bm25.avgdl = metadata['avgdl']
      bm25.corpus_size = metadata['corpus_size']
      bm25.k1 = metadata['k1']
      bm25.b = metadata['b']

      # Restore doc_freqs (required for get_scores())
      if 'doc_freqs' in npz_data:
        bm25.doc_freqs = npz_data['doc_freqs'].tolist()
      else:
        # Old index format missing doc_freqs - needs rebuild
        logger.warning("BM25 index missing doc_freqs. Run 'customkb bm25 <kb_name> --force' to rebuild.")
        return None

      # Convert doc_ids back to list
      doc_ids = npz_data['doc_ids'].tolist()

      bm25_data = {
        'bm25': bm25,
        'doc_ids': doc_ids,
        'total_docs': metadata['total_docs'],
        'total_tokens': metadata['total_tokens'],
        'k1': metadata['k1'],
        'b': metadata['b'],
        'version': metadata.get('version', '2.0')
      }

      logger.debug(f"Loaded BM25 index (NPZ) with {metadata['total_docs']} documents")
      return bm25_data

    except (FileNotFoundError, json.JSONDecodeError, KeyError, ValueError, TypeError, OSError) as e:
      logger.warning(f"Failed to load NPZ format BM25 index: {e}")

  # Legacy pickle format is no longer supported (removed for security)
  # Check if there's an old pickle file that needs migration
  if os.path.exists(bm25_path) and not os.path.exists(metadata_path):
    logger.warning("=" * 70)
    logger.warning("LEGACY BM25 INDEX FORMAT DETECTED")
    logger.warning("=" * 70)
    logger.warning(f"The BM25 index at {bm25_path} uses the old pickle format.")
    logger.warning("For security reasons, this format is no longer supported.")
    logger.warning("")
    logger.info("Auto-migration option: The index will be rebuilt automatically")
    logger.info("during the next 'customkb bm25' operation, or you can rebuild now:")
    logger.info("  customkb bm25 <kb_name> --force")
    logger.warning("")
    logger.warning("Note: BM25 hybrid search will be disabled until rebuild completes.")
    logger.warning("=" * 70)

    # Rename the legacy file to prevent repeated warnings
    try:
      legacy_backup = bm25_path + '.legacy.backup'
      if not os.path.exists(legacy_backup):
        import shutil
        shutil.move(bm25_path, legacy_backup)
        logger.info(f"Legacy index backed up to: {legacy_backup}")
    except (OSError, PermissionError, FileNotFoundError) as e:
      logger.debug(f"Could not backup legacy index: {e}")

    return None

  logger.debug(f"BM25 index not found at {npz_path}")
  return None

def get_bm25_index_path(kb: 'KnowledgeBase') -> str:
  """
  Get the file path for the BM25 index.

  Args:
      kb: The KnowledgeBase instance.

  Returns:
      Path to the BM25 index file.
  """
  return kb.knowledge_base_vector.replace('.faiss', '.bm25')

def rebuild_bm25_if_needed(kb: 'KnowledgeBase') -> bool:
  """
  Check if BM25 index needs rebuilding and rebuild if necessary.

  Args:
      kb: The KnowledgeBase instance.

  Returns:
      True if rebuild was performed or not needed, False if rebuild failed.
  """
  try:
    cursor = kb.sql_cursor
    cursor.execute("SELECT COUNT(*) FROM docs WHERE keyphrase_processed = 0")
    unprocessed = cursor.fetchone()[0]

    # Get rebuild threshold from config
    threshold = getattr(kb, 'bm25_rebuild_threshold', 1000)

    if unprocessed > threshold:
      logger.info(f"BM25 rebuild needed: {unprocessed} unprocessed documents (threshold: {threshold})")
      bm25 = build_bm25_index(kb)
      return bm25 is not None
    else:
      logger.debug(f"BM25 rebuild not needed: {unprocessed} unprocessed documents (threshold: {threshold})")
      return True

  except sqlite3.Error as e:
    logger.error(f"Error checking BM25 rebuild status: {e}")
    return False

def get_bm25_scores(kb: 'KnowledgeBase', query_text: str, bm25_data: dict[str, Any], max_results: int = None) -> list[tuple[int, float]]:
  """
  Get BM25 scores for a query with result limiting.

  Args:
      kb: The KnowledgeBase instance.
      query_text: The query text.
      bm25_data: Loaded BM25 index data.
      max_results: Maximum results to return (overrides KB config).

  Returns:
      List of (doc_id, score) tuples sorted by score descending.
  """
  try:
    # Tokenize query using same method as indexing
    language = getattr(kb, 'language', 'en')
    query_tokens, _ = tokenize_for_bm25(query_text, language)
    query_tokens = query_tokens.split()

    if not query_tokens:
      logger.warning("Query tokenization resulted in empty token list")
      return []

    # Get BM25 scores
    bm25 = bm25_data['bm25']
    doc_ids = bm25_data['doc_ids']

    scores = bm25.get_scores(query_tokens)

    # Get max results limit from parameter or config
    limit = max_results if max_results is not None else getattr(kb, 'bm25_max_results', 1000)

    # Apply limit efficiently for large result sets
    if limit > 0:  # 0 means unlimited
      # Use heapq for efficient top-k selection
      import heapq

      # Find indices of positive scores
      positive_scores = [(i, float(score)) for i, score in enumerate(scores) if score > 0]

      if len(positive_scores) > limit:
        # Get top k results efficiently
        top_indices = heapq.nlargest(limit, positive_scores, key=lambda x: x[1])
        doc_scores = [(doc_ids[i], score) for i, score in top_indices]
        doc_scores.sort(key=lambda x: x[1], reverse=True)

        logger.info(f"BM25 results limited from {len(positive_scores)} to {limit} "
                   f"(index size: {len(doc_ids)}, query: '{query_text[:30]}...')")
      else:
        # All positive scores fit within limit
        doc_scores = [(doc_ids[i], score) for i, score in positive_scores]
        doc_scores.sort(key=lambda x: x[1], reverse=True)
    else:
      # No limit - return all positive scores
      doc_scores = [(doc_id, float(score)) for doc_id, score in zip(doc_ids, scores, strict=False) if score > 0]
      doc_scores.sort(key=lambda x: x[1], reverse=True)

    logger.debug(f"BM25 search returned {len(doc_scores)} results for query: {query_text[:50]}...")
    return doc_scores

  except (ValueError, TypeError, KeyError, IndexError) as e:
    logger.error(f"Error computing BM25 scores: {e}")
    return []

def ensure_bm25_index(kb: 'KnowledgeBase') -> bool:
  """
  Ensure BM25 index exists and is up to date.

  Args:
      kb: The KnowledgeBase instance.

  Returns:
      True if index is available, False otherwise.
  """
  # Check if hybrid search is enabled
  if not getattr(kb, 'enable_hybrid_search', False):
    logger.debug("Hybrid search disabled, skipping BM25 index check")
    return False

  # Try to load existing index
  bm25_data = load_bm25_index(kb)
  if bm25_data:
    # Check if rebuild is needed
    return rebuild_bm25_if_needed(kb)
  else:
    # No index exists, build one
    logger.info("No BM25 index found, building new index")
    bm25 = build_bm25_index(kb)
    return bm25 is not None


def search_bm25(kb: 'KnowledgeBase', query_text: str, max_results: int = 10) -> list[tuple[int, float]]:
  """
  Search BM25 index with automatic index loading and validation.

  This is a convenience wrapper around get_bm25_scores that handles
  index loading and existence validation automatically.

  Args:
      kb: The KnowledgeBase instance.
      query_text: The query text.
      max_results: Maximum number of results to return.

  Returns:
      List of (doc_id, score) tuples sorted by score descending.
      Returns empty list if index doesn't exist or search fails.
  """
  # Check if hybrid search is enabled
  if not getattr(kb, 'enable_hybrid_search', False):
    logger.debug("Hybrid search disabled")
    return []

  # Load BM25 index
  bm25_data = load_bm25_index(kb)
  if not bm25_data:
    logger.warning("BM25 index not found. Run 'customkb bm25 <kb_name>' to build it.")
    return []

  # Perform search
  return get_bm25_scores(kb, query_text, bm25_data, max_results)


#fin
