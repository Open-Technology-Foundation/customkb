#!/usr/bin/env python
"""
Search functionality for CustomKB query system.

This module handles vector similarity search, hybrid search,
category filtering, and document retrieval operations.
"""

import os
import sqlite3
from typing import Any

import numpy as np

# FAISS loaded lazily on first use (avoids ~7s import-time GPU init)
_faiss = None
_FAISS_GPU_AVAILABLE = None


def _get_faiss():
  """Get FAISS module, loading lazily on first use."""
  global _faiss, _FAISS_GPU_AVAILABLE
  if _faiss is None:
    from utils.faiss_loader import get_faiss
    _faiss, _FAISS_GPU_AVAILABLE = get_faiss()
  return _faiss, _FAISS_GPU_AVAILABLE

from utils.exceptions import DatabaseError, SearchError
from utils.logging_config import get_logger
from utils.security_utils import validate_table_name

logger = get_logger(__name__)


def get_context_range(index_start: int, context_n: int) -> list[int]:
  """
  Calculate the start and end indices for context retrieval.

  Args:
      index_start: The starting index.
      context_n: The number of context items to retrieve.

  Returns:
      A list containing the start and end indices.
  """
  if context_n < 1:
    context_n = 1

  half_context = (context_n - 1) // 2
  start_index = max(0, index_start - half_context)
  end_index = start_index + context_n
  start_index = max(0, end_index - context_n)

  return [start_index, end_index - 1]


def fetch_document_by_id(kb: Any, doc_id: int) -> tuple[int, int, str] | None:
  """
  Fetch a document by its ID from the database.

  Args:
      kb: KnowledgeBase instance
      doc_id: Document ID to fetch

  Returns:
      Tuple of (id, sid, sourcedoc) or None if not found
  """
  try:
    # Validate table name for security
    table_name = getattr(kb, 'table_name', 'docs')
    if not validate_table_name(table_name):
      raise DatabaseError(f"Invalid table name: {table_name}")

    # Get the document's location info for context retrieval
    query = f"SELECT id, sid, sourcedoc FROM {table_name} WHERE id = ? LIMIT 1"
    kb.sql_cursor.execute(query, (int(doc_id),))
    result = kb.sql_cursor.fetchone()

    if result:
      return result
    else:
      logger.warning(f"Document ID {doc_id} not found in database")
      return None

  except sqlite3.Error as e:
    logger.error(f"Database error fetching document {doc_id}: {e}")
    raise DatabaseError(f"Failed to fetch document: {e}") from e


async def filter_results_by_category(kb: Any, results: list[tuple[int, float]],
                                    categories: list[str]) -> list[tuple[int, float]]:
  """
  Filter search results by category if categorization is enabled.

  Args:
      kb: KnowledgeBase instance
      results: List of (doc_id, score) tuples
      categories: List of category names to filter by

  Returns:
      Filtered list of (doc_id, score) tuples
  """
  if not categories:
    return results

  # Check if categorization is enabled
  if not getattr(kb, 'enable_categorization', False):
    logger.warning("Category filtering requested but enable_categorization=false in config")
    return results

  if not hasattr(kb, 'sql_cursor') or not kb.sql_cursor:
    logger.warning("No database connection for category filtering")
    return results

  try:
    # Validate table name
    table_name = getattr(kb, 'table_name', 'docs')
    if not validate_table_name(table_name):
      raise DatabaseError(f"Invalid table name: {table_name}")

    # Extract document IDs
    doc_ids = [doc_id for doc_id, _ in results]
    if not doc_ids:
      return results

    # First check if category columns exist
    kb.sql_cursor.execute(f"PRAGMA table_info({table_name})")
    columns = {col[1] for col in kb.sql_cursor.fetchall()}

    if 'primary_category' not in columns and 'categories' not in columns:
      logger.warning(f"Category columns not found in {table_name} table. Run 'customkb categorize --import' first.")
      return results

    # Create placeholders for IN query
    placeholders = ','.join(['?'] * len(doc_ids))
    category_placeholders = ','.join(['?'] * len(categories))

    # Build query based on available columns
    conditions = []
    params = doc_ids.copy()

    if 'primary_category' in columns:
      conditions.append(f"primary_category IN ({category_placeholders})")
      params.extend(categories)

    if 'categories' in columns:
      for cat in categories:
        conditions.append("categories LIKE ?")
        params.append(f'%{cat}%')

    if not conditions:
      return results

    # Query for documents that match any of the specified categories
    query = f"""
      SELECT id FROM {table_name}
      WHERE id IN ({placeholders})
      AND ({' OR '.join(conditions)})
    """

    # Execute query with parameters
    kb.sql_cursor.execute(query, params)

    # Get matching document IDs
    matching_ids = {row[0] for row in kb.sql_cursor.fetchall()}

    # Filter original results
    filtered_results = [(doc_id, score) for doc_id, score in results
                       if doc_id in matching_ids]

    logger.info(f"Category filtering: {len(results)} -> {len(filtered_results)} results")
    return filtered_results

  except sqlite3.Error as e:
    logger.error(f"Category filtering failed: {e}")
    # Return unfiltered results rather than failing completely
    return results


async def perform_vector_search(kb: Any, query_embedding: np.ndarray,
                               top_k: int = 10) -> list[tuple[int, float]]:
  """
  Perform vector similarity search using FAISS index.

  Args:
      kb: KnowledgeBase instance
      query_embedding: Query embedding vector
      top_k: Number of top results to return

  Returns:
      List of (doc_id, similarity_score) tuples
  """
  try:
    # Load FAISS index
    if not hasattr(kb, 'faiss_index') or kb.faiss_index is None:
      if not os.path.exists(kb.knowledge_base_vector):
        raise SearchError(f"Vector index not found: {kb.knowledge_base_vector}")

      logger.debug(f"Loading FAISS index from {kb.knowledge_base_vector}")
      faiss, _ = _get_faiss()
      kb.faiss_index = faiss.read_index(kb.knowledge_base_vector)

    # Set nprobe for IVF indexes (requires downcast to expose nprobe)
    try:
      faiss, _ = _get_faiss()
      ivf_index = faiss.extract_index_ivf(kb.faiss_index)
      if ivf_index is not None:
        nprobe = getattr(kb, 'faiss_nprobe', 32)
        if ivf_index.nprobe != nprobe:
          ivf_index.nprobe = nprobe
          logger.info(f"Set FAISS nprobe={nprobe}")
    except RuntimeError:
      pass  # Not an IVF index

    # Ensure query embedding is correct shape and type
    if query_embedding.ndim == 1:
      query_embedding = query_embedding.reshape(1, -1)

    if query_embedding.dtype != np.float32:
      query_embedding = query_embedding.astype(np.float32)

    # Perform search
    distances, indices = kb.faiss_index.search(query_embedding, top_k)

    # Convert distances to similarity scores based on metric type
    # Detect index metric by checking the underlying index type
    index = kb.faiss_index
    # Unwrap IndexIDMap if present
    inner_index = index.index if hasattr(index, 'index') else index

    # Check if using Inner Product (IP) or L2 distance
    uses_ip_metric = (
      'IP' in type(inner_index).__name__ or
      (hasattr(inner_index, 'metric_type') and inner_index.metric_type == 1)  # 1 = METRIC_INNER_PRODUCT
    )

    # For Inner Product, higher scores are already better (no conversion needed)
    # For L2 distance, convert to similarity (higher is better)
    similarities = distances[0] if uses_ip_metric else 1.0 / (1.0 + distances[0])

    # Create results list
    results = []
    for idx, similarity in zip(indices[0], similarities, strict=False):
      if idx != -1:  # FAISS returns -1 for empty slots
        results.append((int(idx), float(similarity)))

    logger.debug(f"Vector search found {len(results)} results")
    return results

  except (FileNotFoundError, OSError, RuntimeError, ValueError) as e:
    logger.error(f"Vector search failed: {e}")
    raise SearchError(f"Vector search error: {e}") from e


async def perform_bm25_search(kb: Any, query_text: str,
                             top_k: int = 10) -> list[tuple[int, float]]:
  """
  Perform BM25 keyword search if enabled.

  Args:
      kb: KnowledgeBase instance
      query_text: Original query text
      top_k: Number of top results to return

  Returns:
      List of (doc_id, bm25_score) tuples
  """
  if not getattr(kb, 'enable_hybrid_search', False):
    return []

  try:
    # Import BM25 functionality
    from embedding.bm25_manager import search_bm25

    # Perform BM25 search
    bm25_results = search_bm25(kb, query_text, top_k * 2)  # Get more for merging

    if not bm25_results:
      logger.debug("No BM25 results found")
      return []

    logger.debug(f"BM25 search found {len(bm25_results)} results")
    return bm25_results

  except ImportError:
    logger.warning("BM25 search not available")
    return []
  except (ValueError, RuntimeError, AttributeError) as e:
    logger.error(f"BM25 search failed: {e}")
    return []


def merge_search_results_rrf(vector_results: list[tuple[int, float]],
                             bm25_results: list[tuple[int, float]],
                             k: int = 60) -> list[tuple[int, float]]:
  """
  Merge results using Reciprocal Rank Fusion (RRF).

  RRF is more robust than weighted averaging because it uses rank positions
  rather than raw scores, making it invariant to score scale differences
  between rankers.

  Formula: RRF_score(d) = sum(1 / (k + rank_i(d))) for each ranker i

  Args:
      vector_results: Vector search results (doc_id, score), sorted by relevance
      bm25_results: BM25 search results (doc_id, score), sorted by relevance
      k: Ranking constant (default 60, standard in literature)
          Higher k = more weight to lower-ranked documents

  Returns:
      Merged results sorted by RRF score (doc_id, rrf_score)
  """
  rrf_scores: dict[int, float] = {}

  # Add vector search contributions (results should already be sorted by score)
  for rank, (doc_id, _) in enumerate(vector_results, 1):
    rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank)

  # Add BM25 contributions
  for rank, (doc_id, _) in enumerate(bm25_results, 1):
    rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank)

  # Sort by RRF score descending
  merged_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

  logger.debug(f"RRF merged {len(vector_results)} vector + {len(bm25_results)} BM25 "
              f"= {len(merged_results)} total results")

  return merged_results


def merge_search_results_weighted(vector_results: list[tuple[int, float]],
                                  bm25_results: list[tuple[int, float]],
                                  vector_weight: float = 0.7,
                                  bm25_weight: float = 0.3) -> list[tuple[int, float]]:
  """
  Merge vector and BM25 search results with weighted scoring (legacy method).

  Note: RRF (merge_search_results_rrf) is generally preferred as it's more
  robust to score scale differences between rankers.

  Args:
      vector_results: Vector search results
      bm25_results: BM25 search results
      vector_weight: Weight for vector scores
      bm25_weight: Weight for BM25 scores

  Returns:
      Merged and sorted list of (doc_id, combined_score) tuples
  """
  # Normalize weights
  total_weight = vector_weight + bm25_weight
  vector_weight /= total_weight
  bm25_weight /= total_weight

  # Convert to dictionaries for easy lookup
  vector_scores = dict(vector_results)
  bm25_scores = dict(bm25_results)

  # Get all unique document IDs
  all_doc_ids = set(vector_scores.keys()) | set(bm25_scores.keys())

  # Normalize scores to [0, 1] range
  if vector_scores:
    max_vector_score = max(vector_scores.values())
    if max_vector_score > 0:
      vector_scores = {doc_id: score / max_vector_score
                      for doc_id, score in vector_scores.items()}

  if bm25_scores:
    max_bm25_score = max(bm25_scores.values())
    if max_bm25_score > 0:
      bm25_scores = {doc_id: score / max_bm25_score
                    for doc_id, score in bm25_scores.items()}

  # Combine scores
  combined_results = []
  for doc_id in all_doc_ids:
    vector_score = vector_scores.get(doc_id, 0.0)
    bm25_score = bm25_scores.get(doc_id, 0.0)

    combined_score = (vector_weight * vector_score +
                     bm25_weight * bm25_score)

    combined_results.append((doc_id, combined_score))

  # Sort by combined score (descending)
  combined_results.sort(key=lambda x: x[1], reverse=True)

  logger.debug(f"Weighted merged {len(vector_results)} vector + {len(bm25_results)} BM25 "
              f"= {len(combined_results)} total results")

  return combined_results


def merge_search_results(vector_results: list[tuple[int, float]],
                        bm25_results: list[tuple[int, float]],
                        vector_weight: float = 0.7,
                        bm25_weight: float = 0.3,
                        fusion_method: str = "rrf",
                        rrf_k: int = 60) -> list[tuple[int, float]]:
  """
  Merge vector and BM25 search results using specified fusion method.

  Args:
      vector_results: Vector search results
      bm25_results: BM25 search results
      vector_weight: Weight for vector scores (used with 'weighted' method)
      bm25_weight: Weight for BM25 scores (used with 'weighted' method)
      fusion_method: Fusion algorithm - 'rrf' (recommended) or 'weighted'
      rrf_k: RRF ranking constant (default 60)

  Returns:
      Merged and sorted list of (doc_id, combined_score) tuples
  """
  if fusion_method == "rrf":
    return merge_search_results_rrf(vector_results, bm25_results, k=rrf_k)
  else:
    return merge_search_results_weighted(vector_results, bm25_results,
                                         vector_weight, bm25_weight)


async def perform_hybrid_search(kb: Any, query_text: str,
                               query_embedding: np.ndarray,
                               top_k: int = 10,
                               categories: list[str] = None,
                               rerank: bool = False) -> list[tuple[int, float]]:
  """
  Perform hybrid search combining vector similarity and BM25 keyword search.

  Args:
      kb: KnowledgeBase instance
      query_text: Original query text
      query_embedding: Query embedding vector
      top_k: Number of top results to return
      categories: Optional category filter
      rerank: Whether to apply reranking

  Returns:
      List of (doc_id, score) tuples sorted by relevance
  """
  try:
    # Perform vector search
    vector_results = await perform_vector_search(kb, query_embedding, top_k)

    # Perform BM25 search if enabled
    bm25_results = await perform_bm25_search(kb, query_text, top_k)

    # Merge results if we have both
    if vector_results and bm25_results:
      # Get fusion method and parameters from config
      fusion_method = getattr(kb, 'hybrid_fusion_method', 'rrf')
      vector_weight = getattr(kb, 'hybrid_vector_weight', 0.7)
      bm25_weight = getattr(kb, 'hybrid_bm25_weight', 0.3)
      rrf_k = getattr(kb, 'rrf_k', 60)

      results = merge_search_results(
        vector_results, bm25_results,
        vector_weight=vector_weight,
        bm25_weight=bm25_weight,
        fusion_method=fusion_method,
        rrf_k=rrf_k
      )
    elif vector_results:
      results = vector_results
    elif bm25_results:
      results = bm25_results
    else:
      logger.warning("No search results found")
      return []

    # Apply category filtering if specified
    if categories:
      results = await filter_results_by_category(kb, results, categories)

    # Apply reranking if enabled
    if rerank and getattr(kb, 'enable_reranking', False):
      try:
        from embedding.rerank_manager import rerank_search_results
        results = await rerank_search_results(kb, query_text, results)
        logger.debug("Applied reranking to search results")
      except (ImportError, RuntimeError, ValueError) as e:
        logger.warning(f"Reranking failed: {e}")

    # Limit to top_k results
    results = results[:top_k]

    logger.info(f"Hybrid search completed: {len(results)} results")
    return results

  except (ValueError, RuntimeError, OSError, sqlite3.Error) as e:
    logger.error(f"Hybrid search failed: {e}")
    raise SearchError(f"Search error: {e}") from e


async def process_reference_batch(kb: Any, batch: list[tuple[int, float]]) -> list[list[Any]]:
  """
  Process a batch of search results to retrieve document content with context.

  Args:
      kb: KnowledgeBase instance
      batch: List of (doc_id, distance) tuples

  Returns:
      List of document data lists with surrounding context
  """
  if not batch:
    return []

  references = []
  context_scope = int(kb.query_context_scope)

  # Validate table name once
  table_name = getattr(kb, 'table_name', 'docs')
  if not validate_table_name(table_name):
    raise DatabaseError(f"Invalid table name: {table_name}")

  try:
    # Check if category columns exist
    kb.sql_cursor.execute(f"PRAGMA table_info({table_name})")
    columns = {col[1] for col in kb.sql_cursor.fetchall()}
    has_primary_category = 'primary_category' in columns
    has_categories = 'categories' in columns

    # Build query with optional category columns
    select_fields = ["id", "sid", "sourcedoc", "originaltext", "metadata"]
    if has_primary_category:
      select_fields.append("primary_category")
    if has_categories:
      select_fields.append("categories")

    for doc_id, distance in batch:
      # Adjust context scope based on similarity (configurable thresholds)
      similarity_threshold = getattr(kb, 'similarity_threshold', 0.6)
      scope_factor = getattr(kb, 'low_similarity_scope_factor', 0.5)
      local_context_scope = max(int(context_scope * scope_factor), 1) if distance < similarity_threshold else context_scope

      # Get document location info
      doc_info = fetch_document_by_id(kb, doc_id)
      if not doc_info:
        continue

      doc_id, sid, sourcedoc = doc_info
      stsid, endsid = get_context_range(sid, local_context_scope)

      # Fetch all documents in the context range
      query = f"""
        SELECT {', '.join(select_fields)}
        FROM {table_name}
        WHERE sourcedoc=? AND sid>=? AND sid<=?
        ORDER BY sid
        LIMIT ?
      """

      kb.sql_cursor.execute(query, (sourcedoc, int(stsid), int(endsid), local_context_scope))
      refrows = kb.sql_cursor.fetchall()

      if refrows:
        for r in refrows:
          # Extract base fields
          rid, rsid, rsrc, originaltext, metadata = r[:5]

          # Extract category fields if they exist
          primary_category = None
          categories = None
          idx = 5
          if has_primary_category:
            primary_category = r[idx] if idx < len(r) else None
            idx += 1
          if has_categories:
            categories = r[idx] if idx < len(r) else None

          # Structure: [rid, rsrc, rsid, originaltext, distance, metadata, primary_category, categories]
          references.append([rid, rsrc, rsid, originaltext, distance, metadata, primary_category, categories])
          logger.debug(f"Context doc: id={rid}, sid={rsid}, src={rsrc}, distance={distance}")
        logger.debug('---')
      else:
        logger.warning(f'No rows found for {sourcedoc} with sid range {stsid}-{endsid}')

    return references

  except sqlite3.Error as e:
    logger.error(f"Database error processing reference batch: {e}")
    raise DatabaseError(f"Failed to process reference batch: {e}") from e


#fin
