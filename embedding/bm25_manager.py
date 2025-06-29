#!/usr/bin/env python
"""
BM25 index management for CustomKB hybrid retrieval system.

This module handles BM25 index creation, serialization, and loading for keyword-based
retrieval that complements the vector-based semantic search.
"""

import os
import pickle
import sqlite3
from typing import List, Tuple, Optional, Dict, Any
from rank_bm25 import BM25Okapi

from utils.logging_utils import get_logger
from utils.text_utils import tokenize_for_bm25

logger = get_logger(__name__)

def build_bm25_index(kb: 'KnowledgeBase') -> Optional[BM25Okapi]:
  """
  Build BM25 index from database and save to disk.
  
  Args:
      kb: The KnowledgeBase instance with database connection.
      
  Returns:
      BM25Okapi instance if successful, None otherwise.
  """
  try:
    cursor = kb.sql_cursor
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
    
    # Save index and metadata
    bm25_path = get_bm25_index_path(kb)
    bm25_data = {
      'bm25': bm25,
      'doc_ids': doc_ids,
      'total_docs': len(doc_ids),
      'total_tokens': total_length,
      'k1': k1,
      'b': b,
      'version': '1.0'
    }
    
    with open(bm25_path, 'wb') as f:
      pickle.dump(bm25_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    logger.info(f"Built BM25 index with {len(doc_ids)} documents, saved to {bm25_path}")
    return bm25
    
  except sqlite3.Error as e:
    logger.error(f"Database error building BM25 index: {e}")
    return None
  except Exception as e:
    logger.error(f"Error building BM25 index: {e}")
    return None

def load_bm25_index(kb: 'KnowledgeBase') -> Optional[Dict[str, Any]]:
  """
  Load BM25 index from disk.
  
  Args:
      kb: The KnowledgeBase instance.
      
  Returns:
      Dictionary containing BM25 data if successful, None otherwise.
  """
  bm25_path = get_bm25_index_path(kb)
  
  if not os.path.exists(bm25_path):
    logger.debug(f"BM25 index not found at {bm25_path}")
    return None
  
  try:
    with open(bm25_path, 'rb') as f:
      bm25_data = pickle.load(f)
    
    # Validate loaded data
    required_keys = ['bm25', 'doc_ids', 'total_docs']
    if not all(key in bm25_data for key in required_keys):
      logger.error("Invalid BM25 index file: missing required keys")
      return None
    
    logger.debug(f"Loaded BM25 index with {bm25_data['total_docs']} documents")
    return bm25_data
    
  except (pickle.PickleError, IOError) as e:
    logger.error(f"Error loading BM25 index: {e}")
    return None
  except Exception as e:
    logger.error(f"Unexpected error loading BM25 index: {e}")
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

def get_bm25_scores(kb: 'KnowledgeBase', query_text: str, bm25_data: Dict[str, Any], max_results: int = None) -> List[Tuple[int, float]]:
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
      doc_scores = [(doc_id, float(score)) for doc_id, score in zip(doc_ids, scores) if score > 0]
      doc_scores.sort(key=lambda x: x[1], reverse=True)
    
    logger.debug(f"BM25 search returned {len(doc_scores)} results for query: {query_text[:50]}...")
    return doc_scores
    
  except Exception as e:
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

#fin