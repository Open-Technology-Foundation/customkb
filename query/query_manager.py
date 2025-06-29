#!/usr/bin/env python
"""
Query management for CustomKB.
Handles semantic searching and response generation.
"""

import os
import sys
import re
import numpy as np
import faiss
import sqlite3
import xml.sax.saxutils
import argparse
import asyncio
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple, Dict, Any, Optional, Set
from datetime import datetime

from utils.logging_utils import setup_logging, get_logger, elapsed_time
from utils.text_utils import clean_text
from config.config_manager import KnowledgeBase, get_fq_cfg_filename
from database.db_manager import connect_to_database, close_database

# Import AI clients with validation
from openai import OpenAI, AsyncOpenAI
from anthropic import Anthropic, AsyncAnthropic

# Import reranking functionality
from embedding.rerank_manager import rerank_search_results
from utils.security_utils import validate_api_key, safe_log_error

def load_and_validate_api_keys():
  """Load and validate API keys securely."""
  # Load OpenAI API key
  openai_key = os.getenv('OPENAI_API_KEY')
  if not openai_key:
    raise EnvironmentError("OPENAI_API_KEY environment variable not set.")
  
  if not validate_api_key(openai_key, 'sk-', 40):
    raise ValueError("Invalid OpenAI API key format")
  
  # Load Anthropic API key
  anthropic_key = os.getenv('ANTHROPIC_API_KEY')
  if not anthropic_key:
    raise EnvironmentError("ANTHROPIC_API_KEY environment variable not set.")
  
  if not validate_api_key(anthropic_key, 'sk-ant-', 95):
    raise ValueError("Invalid Anthropic API key format")
  
  return openai_key, anthropic_key

try:
  OPENAI_API_KEY, ANTHROPIC_API_KEY = load_and_validate_api_keys()
  openai_client = OpenAI(api_key=OPENAI_API_KEY)
  async_openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
  anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY)
  async_anthropic_client = AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
except (EnvironmentError, ValueError) as e:
  # Don't use safe_log_error during module initialization
  # as logging may not be set up yet
  print(f"ERROR: API key validation failed: {e}", file=sys.stderr)
  raise

# Llama client
llama_client = OpenAI(api_key='ollama', base_url='http://localhost:11434/v1')

logger = get_logger(__name__)

# Cache settings
CACHE_DIR = os.path.join(os.getenv('VECTORDBS', '/var/lib/vectordbs'), '.query_cache')
os.makedirs(CACHE_DIR, exist_ok=True)
# Cache TTL will be loaded from KB config in functions

def get_cache_key(query_text: str, model: str) -> str:
  """
  Generate a cache key for a query.
  
  Args:
      query_text: The query text.
      model: The model used for embedding.
      
  Returns:
      A cache key string.
  """
  text_hash = hashlib.md5(query_text.encode('utf-8')).hexdigest()
  return f"{model}_{text_hash}"

def get_cached_query_embedding(query_text: str, model: str, kb=None) -> Optional[List[float]]:
  """
  Retrieve a cached query embedding if it exists.
  
  Args:
      query_text: The query text.
      model: The model used for embedding.
      kb: KnowledgeBase instance for configuration (optional).
      
  Returns:
      The cached embedding or None if not found or expired.
  """
  cache_key = get_cache_key(query_text, model)
  cache_file = os.path.join(CACHE_DIR, f"{cache_key}.json")
  
  if os.path.exists(cache_file):
    try:
      # Check if cache is expired
      file_time = os.path.getmtime(cache_file)
      cache_ttl_days = getattr(kb, 'query_cache_ttl_days', 7) if kb else 7
      cache_ttl_seconds = cache_ttl_days * 24 * 3600
      if time.time() - file_time > cache_ttl_seconds:
        os.remove(cache_file)
        return None
        
      with open(cache_file, 'r') as f:
        return json.load(f)
    except (json.JSONDecodeError, IOError):
      return None
  
  return None

def save_query_embedding_to_cache(query_text: str, model: str, embedding: List[float]) -> None:
  """
  Save a query embedding to the cache.
  
  Args:
      query_text: The query text.
      model: The model used for embedding.
      embedding: The embedding vector.
  """
  cache_key = get_cache_key(query_text, model)
  cache_file = os.path.join(CACHE_DIR, f"{cache_key}.json")
  
  try:
    with open(cache_file, 'w') as f:
      json.dump(embedding, f)
  except IOError as e:
    logger.warning(f"Failed to cache query embedding: {e}")

def get_enhancement_cache_key(query_text: str) -> str:
  """
  Generate a cache key for query enhancement.
  
  Args:
      query_text: The original query text.
      
  Returns:
      A cache key string for enhancement.
  """
  text_hash = hashlib.md5(query_text.encode('utf-8')).hexdigest()
  return f"enhancement_{text_hash}"

def get_cached_enhanced_query(query_text: str, kb=None) -> Optional[str]:
  """
  Retrieve a cached enhanced query if it exists and is not expired.
  
  Args:
      query_text: The original query text.
      kb: KnowledgeBase instance for configuration (optional).
      
  Returns:
      The cached enhanced query or None if not found or expired.
  """
  cache_key = get_enhancement_cache_key(query_text)
  cache_file = os.path.join(CACHE_DIR, f"{cache_key}.txt")
  
  if os.path.exists(cache_file):
    try:
      # Check if cache is expired
      file_time = os.path.getmtime(cache_file)
      cache_ttl_days = getattr(kb, 'query_enhancement_cache_ttl_days', 30) if kb else 30
      cache_ttl_seconds = cache_ttl_days * 24 * 3600
      if time.time() - file_time > cache_ttl_seconds:
        os.remove(cache_file)
        return None
        
      with open(cache_file, 'r', encoding='utf-8') as f:
        return f.read().strip()
    except (IOError, UnicodeDecodeError):
      return None
  
  return None

def save_enhanced_query_to_cache(original_query: str, enhanced_query: str) -> None:
  """
  Save an enhanced query to the cache.
  
  Args:
      original_query: The original query text.
      enhanced_query: The enhanced query text.
  """
  cache_key = get_enhancement_cache_key(original_query)
  cache_file = os.path.join(CACHE_DIR, f"{cache_key}.txt")
  
  try:
    with open(cache_file, 'w', encoding='utf-8') as f:
      f.write(enhanced_query)
  except IOError as e:
    logger.warning(f"Failed to cache enhanced query: {e}")

def normalize_query(query: str) -> str:
  """
  Normalize query text for better processing.
  
  Args:
      query: The input query text.
      
  Returns:
      Normalized query text.
  """
  # Remove extra whitespace
  query = re.sub(r'\s+', ' ', query.strip())
  
  # Handle common abbreviations and expansions
  replacements = {
    r'\bdb\b': 'database',
    r'\bconfig\b': 'configuration',
    r'\bapi\b': 'API',
    r'\bui\b': 'user interface',
    r'\bml\b': 'machine learning',
    r'\bai\b': 'artificial intelligence',
    r'\bdocs?\b': 'documentation',
    r'\bimpl\b': 'implementation',
    r'\bperf\b': 'performance'
  }
  
  for pattern, replacement in replacements.items():
    query = re.sub(pattern, replacement, query, flags=re.IGNORECASE)
  
  return query

def get_synonyms_for_word(word: str, max_synonyms: int = 2, relevance_threshold: float = 0.6) -> List[str]:
  """
  Get relevant synonyms for a word using NLTK WordNet.
  
  Args:
      word: The word to find synonyms for.
      max_synonyms: Maximum number of synonyms to return.
      relevance_threshold: Minimum relevance score for synonyms.
      
  Returns:
      List of relevant synonyms.
  """
  try:
    from nltk.corpus import wordnet
    import nltk
    
    # Download wordnet if not available
    try:
      wordnet.synsets('test')
    except LookupError:
      nltk.download('wordnet', quiet=True)
      nltk.download('omw-1.4', quiet=True)
    
    synonyms = set()
    word_lower = word.lower()
    
    # Get synsets for the word
    synsets = wordnet.synsets(word_lower)
    
    if not synsets:
      return []
    
    # Get synonyms from the first few synsets (most common meanings)
    for synset in synsets[:3]:  # Limit to first 3 synsets for relevance
      for lemma in synset.lemmas():
        synonym = lemma.name().replace('_', ' ')
        if (synonym != word_lower and 
            len(synonym) > 2 and 
            synonym.isalpha() and
            len(synonyms) < max_synonyms):
          synonyms.add(synonym)
    
    return list(synonyms)[:max_synonyms]
    
  except Exception as e:
    logger.debug(f"Error getting synonyms for '{word}': {e}")
    return []

def correct_spelling(word: str, vocabulary: Optional[Set[str]] = None) -> str:
  """
  Simple spelling correction using edit distance.
  
  Args:
      word: The word to potentially correct.
      vocabulary: Set of known correct words (optional).
      
  Returns:
      Corrected word or original if no good correction found.
  """
  if not vocabulary or len(word) < 4:
    return word
  
  def edit_distance(s1: str, s2: str) -> int:
    """Calculate edit distance between two strings."""
    if len(s1) > len(s2):
      s1, s2 = s2, s1
    
    distances = list(range(len(s1) + 1))
    for i2, c2 in enumerate(s2):
      new_distances = [i2 + 1]
      for i1, c1 in enumerate(s1):
        if c1 == c2:
          new_distances.append(distances[i1])
        else:
          new_distances.append(1 + min(distances[i1], distances[i1 + 1], new_distances[-1]))
      distances = new_distances
    
    return distances[-1]
  
  word_lower = word.lower()
  best_match = word
  min_distance = len(word) // 3  # Allow up to 1/3 of word length in edits
  
  # Check words with similar length first
  candidates = [v for v in vocabulary if abs(len(v) - len(word)) <= 2]
  
  for candidate in candidates:
    if candidate.lower() == word_lower:
      continue
      
    distance = edit_distance(word_lower, candidate.lower())
    if distance < min_distance:
      min_distance = distance
      best_match = candidate
  
  return best_match

def expand_synonyms(query: str, kb: Optional['KnowledgeBase'] = None) -> str:
  """
  Expand query with synonyms for better semantic matching.
  
  Args:
      query: The input query text.
      kb: KnowledgeBase instance for configuration.
      
  Returns:
      Query expanded with relevant synonyms.
  """
  if not kb or not getattr(kb, 'query_enhancement_synonyms', True):
    return query
  
  max_synonyms = getattr(kb, 'max_synonyms_per_word', 2)
  relevance_threshold = getattr(kb, 'synonym_relevance_threshold', 0.6)
  
  words = query.split()
  expanded_words = words.copy()  # Keep original words
  
  # Only expand content words (longer than 3 characters, not common words)
  skip_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'this', 'that', 'these', 'those'}
  
  for word in words:
    if len(word) > 3 and word.lower() not in skip_words:
      synonyms = get_synonyms_for_word(word, max_synonyms, relevance_threshold)
      expanded_words.extend(synonyms)
  
  # Remove duplicates while preserving order
  seen = set()
  unique_words = []
  for word in expanded_words:
    word_lower = word.lower()
    if word_lower not in seen:
      seen.add(word_lower)
      unique_words.append(word)
  
  return ' '.join(unique_words)

def apply_spelling_correction(query: str, kb: Optional['KnowledgeBase'] = None) -> str:
  """
  Apply spelling correction to query words.
  
  Args:
      query: The input query text.
      kb: KnowledgeBase instance for configuration and vocabulary.
      
  Returns:
      Query with spelling corrections applied.
  """
  if not kb or not getattr(kb, 'query_enhancement_spelling', True):
    return query
  
  # For now, implement basic corrections for common technical terms
  # In a full implementation, this could use the KB's vocabulary
  common_corrections = {
    'databse': 'database',
    'cofigure': 'configure',
    'cofig': 'config',
    'configuraton': 'configuration',
    'conection': 'connection',
    'querry': 'query',
    'serach': 'search',
    'performace': 'performance',
    'optmization': 'optimization',
    'algorythm': 'algorithm',
    'machien': 'machine',
    'learing': 'learning',
    'retreive': 'retrieve',
    'retreival': 'retrieval'
  }
  
  words = query.split()
  corrected_words = []
  
  for word in words:
    word_lower = word.lower()
    if word_lower in common_corrections:
      # Preserve original case
      if word.isupper():
        corrected_words.append(common_corrections[word_lower].upper())
      elif word.istitle():
        corrected_words.append(common_corrections[word_lower].title())
      else:
        corrected_words.append(common_corrections[word_lower])
    else:
      corrected_words.append(word)
  
  return ' '.join(corrected_words)

def enhance_query(query: str, kb: Optional['KnowledgeBase'] = None) -> str:
  """
  Main query enhancement pipeline that applies normalization, spelling correction, and synonym expansion.
  
  Args:
      query: The original query text.
      kb: KnowledgeBase instance for configuration.
      
  Returns:
      Enhanced query text optimized for better retrieval.
  """
  if not kb or not getattr(kb, 'enable_query_enhancement', True):
    return query
  
  # Check cache for previously enhanced query
  cached_enhanced = get_cached_enhanced_query(query, kb)
  if cached_enhanced:
    logger.debug(f"Using cached enhanced query for: '{query}'")
    return cached_enhanced
  
  try:
    # Step 1: Normalize the query
    enhanced = normalize_query(query)
    
    # Step 2: Apply spelling correction
    enhanced = apply_spelling_correction(enhanced, kb)
    
    # Step 3: Expand with synonyms
    enhanced = expand_synonyms(enhanced, kb)
    
    # Cache the enhanced query if it changed
    if enhanced != query:
      save_enhanced_query_to_cache(query, enhanced)
      logger.debug(f"Query enhanced and cached: '{query}' -> '{enhanced}'")
    
    return enhanced
    
  except Exception as e:
    logger.warning(f"Error enhancing query '{query}': {e}")
    return query  # Fallback to original query on error

def get_context_range(index_start: int, context_n: int) -> List[int]:
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

async def get_query_embedding(query_text: str, model: str, kb: Optional['KnowledgeBase'] = None) -> np.ndarray:
  """
  Get embedding for a query, using cache if available.
  
  Args:
      query_text: The query text.
      model: The model to use for embedding.
      kb: KnowledgeBase instance for configuration.
      
  Returns:
      Numpy array containing the embedding vector.
  """
  clean_query = clean_text(query_text)
  
  # Apply query enhancement if enabled
  enhanced_query = enhance_query(clean_query, kb)
  
  # Log enhancement if there was a change
  if enhanced_query != clean_query and enhanced_query != query_text:
    logger.info(f"Query enhanced: '{clean_query}' -> '{enhanced_query}'")
  
  # Use enhanced query for caching and embedding generation
  query_for_embedding = enhanced_query
  cached_embedding = get_cached_query_embedding(query_for_embedding, model, kb)
  
  if cached_embedding:
    logger.info("Using cached query embedding")
    embedding = cached_embedding
  else:
    response = await async_openai_client.embeddings.create(
      input=query_for_embedding, 
      model=model
    )
    embedding = response.data[0].embedding
    save_query_embedding_to_cache(query_for_embedding, model, embedding)
    
  return np.array(embedding, dtype=np.float32).reshape(1, -1)

def read_context_file(file_path: str) -> Tuple[str, str]:
  """
  Read a context file and return its content and base name.
  
  Args:
      file_path: The path to the context file.
      
  Returns:
      A tuple containing the file content and base name.
  """
  try:
    with open(file_path, 'r') as f:
      file_content = f.read().strip()
    file_content = xml.sax.saxutils.escape(file_content)
    base_name, _ = os.path.splitext(os.path.basename(file_path.strip()))
    return file_content, base_name
  except Exception as e:
    logger.error(f"Error reading context file {file_path}: {e}")
    return "", ""

def fetch_document_by_id(kb: KnowledgeBase, doc_id: int) -> Optional[Tuple[int, int, str]]:
  """
  Fetch a document by its ID.
  
  Args:
      kb: The KnowledgeBase instance.
      doc_id: The document ID.
      
  Returns:
      A tuple containing the document ID, sid, and source document, or None if not found.
  """
  try:
    kb.sql_cursor.execute("SELECT id, sid, sourcedoc FROM docs WHERE id=? LIMIT 1;", (int(doc_id),))
    rows = kb.sql_cursor.fetchall()
    logger.debug(f'Query result for id={doc_id}: {rows}')
    
    if not rows:
      logger.warning(f'No rows found {doc_id=}')
      return None
      
    return rows[0]
  except sqlite3.Error as e:
    logger.error(f"SQLite error: {e}")
    return None


async def perform_hybrid_search(kb: KnowledgeBase, query_text: str, 
                               query_vector: np.ndarray, index: faiss.Index) -> List[Tuple[int, float]]:
  """
  Perform hybrid vector + BM25 search or fallback to vector-only.
  
  Args:
      kb: The KnowledgeBase instance.
      query_text: The original query text.
      query_vector: The query embedding vector.
      index: The FAISS index.
      
  Returns:
      List of (doc_id, distance) tuples sorted by relevance.
  """
  k = kb.query_top_k
  
  # Always perform vector search
  distances, indices = index.search(query_vector, k * 2)  # Get extra for potential merging
  
  # Check if hybrid search is enabled
  if not getattr(kb, 'enable_hybrid_search', False):
    logger.debug("Hybrid search disabled, using vector-only results")
    return [(int(indices[0][i]), float(distances[0][i])) for i in range(min(k, len(indices[0]))) if indices[0][i] >= 0]
  
  # Try to ensure BM25 index exists and perform hybrid search
  try:
    from embedding.bm25_manager import ensure_bm25_index, load_bm25_index, get_bm25_scores
    
    # Ensure BM25 index exists (build if needed)
    if not ensure_bm25_index(kb):
      logger.info("BM25 index not available, falling back to vector-only search")
      return [(int(indices[0][i]), float(distances[0][i])) for i in range(min(k, len(indices[0]))) if indices[0][i] >= 0]
    
    # Load the BM25 index
    bm25_data = load_bm25_index(kb)
    if not bm25_data:
      logger.info("BM25 index could not be loaded, falling back to vector-only search")
      return [(int(indices[0][i]), float(distances[0][i])) for i in range(min(k, len(indices[0]))) if indices[0][i] >= 0]
    
    # Get BM25 scores
    bm25_results = get_bm25_scores(kb, query_text, bm25_data)
    
    if not bm25_results:
      logger.info("No BM25 results, falling back to vector-only search")
      return [(int(indices[0][i]), float(distances[0][i])) for i in range(min(k, len(indices[0]))) if indices[0][i] >= 0]
    
    # Combine vector and BM25 scores
    vector_weight = getattr(kb, 'vector_weight', 0.7)
    bm25_weight = 1 - vector_weight
    
    combined_scores = {}
    
    # Add vector results (convert distance to similarity)
    for i in range(len(indices[0])):
      idx = int(indices[0][i])
      if idx >= 0:  # Valid index
        distance = float(distances[0][i])
        similarity = 1 / (1 + distance)  # Convert distance to similarity
        combined_scores[idx] = vector_weight * similarity
    
    # Add BM25 results (normalize and add to combined scores)
    max_bm25 = max(score for _, score in bm25_results) if bm25_results else 1.0
    if max_bm25 > 0:
      for doc_id, score in bm25_results:
        normalized_score = score / max_bm25
        if doc_id in combined_scores:
          combined_scores[doc_id] += bm25_weight * normalized_score
        else:
          combined_scores[doc_id] = bm25_weight * normalized_score
    
    # Sort by combined score and convert back to distance format
    sorted_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:k]
    
    # Convert similarity back to distance for consistency with existing code
    final_results = []
    for doc_id, combined_similarity in sorted_results:
      # Convert similarity back to distance (inverse of the conversion above)
      distance = max(0, (1 / combined_similarity) - 1) if combined_similarity > 0 else float('inf')
      final_results.append((doc_id, distance))
    
    logger.info(f"Hybrid search combined {len(indices[0])} vector results with {len(bm25_results)} BM25 results")
    return final_results
    
  except Exception as e:
    logger.error(f"Error in hybrid search, falling back to vector-only: {e}")
    return [(int(indices[0][i]), float(distances[0][i])) for i in range(min(k, len(indices[0]))) if indices[0][i] >= 0]

async def process_reference_batch(kb: KnowledgeBase, batch: List[Tuple[int, float]]) -> List[List[Any]]:
  """
  Process a batch of document references asynchronously.
  
  Args:
      kb: The KnowledgeBase instance.
      batch: A list of (doc_id, distance) tuples.
      
  Returns:
      A list of reference documents.
  """
  references = []
  context_scope = int(kb.query_context_scope)
  
  for idx, distance in batch:
    # Adjust context scope based on similarity (configurable thresholds)
    similarity_threshold = getattr(kb, 'similarity_threshold', 0.6)
    scope_factor = getattr(kb, 'low_similarity_scope_factor', 0.5)
    if distance < similarity_threshold:
      local_context_scope = max(int(context_scope * scope_factor), 1)
    else:
      local_context_scope = context_scope
      
    doc_info = fetch_document_by_id(kb, idx)
    if not doc_info:
      continue
      
    doc_id, sid, sourcedoc = doc_info
    stsid, endsid = get_context_range(sid, local_context_scope)
    
    # Modify to also fetch metadata
    kb.sql_cursor.execute(
      "SELECT id, sid, sourcedoc, originaltext, metadata FROM docs "
      "WHERE sourcedoc=? AND sid>=? AND sid<=? "
      "ORDER BY sid LIMIT ?",
      (sourcedoc, int(stsid), int(endsid), local_context_scope))
    refrows = kb.sql_cursor.fetchall()
    
    if refrows:
      for r in refrows:
        rid, rsid, rsrc, originaltext, metadata = r
        references.append([rid, rsrc, rsid, originaltext, distance, metadata])
        logger.info(f"{rid=} | {rsid=} | {rsrc=} | {distance=}")
      logger.info('---')
    else:
      logger.warning(f'No rows found for {sourcedoc} with sid range {stsid}-{endsid}')
      
  return references

async def process_query_async(args: argparse.Namespace, logger) -> str:
  """
  Execute a semantic query asynchronously on the CustomKB knowledge base.

  Args:
      args: Command-line arguments.
      logger: Initialized logger instance.

  Returns:
      The query response or context.
  """
  # Get configuration file
  cfgfile = get_fq_cfg_filename(args.config_file)
  if not cfgfile:
    return "Error: Configuration file not found."

  logger.info(f"Knowledgebase config: {cfgfile}")

  # Initialize knowledge base early to access configuration
  kb = KnowledgeBase(cfgfile)

  # Get query text
  query_text = args.query_text
  if args.query_file:
    try:
      from utils.security_utils import validate_file_path, sanitize_query_text
      
      # Validate the query file path
      try:
        validated_query_file = validate_file_path(args.query_file, ['.txt', '.md', '.query'])
      except ValueError as e:
        logger.error(f"Invalid query file path: {e}")
        return f"Error: Invalid query file path: {e}"
      
      # Check file size limit for query files
      try:
        file_size = os.path.getsize(validated_query_file)
        # Get configurable max query file size
        max_query_file_size_mb = getattr(kb, 'max_query_file_size_mb', 1)
        max_query_file_size = max_query_file_size_mb * 1024 * 1024  # Convert MB to bytes
        if file_size > max_query_file_size:
          logger.error(f"Query file too large: {file_size} bytes (max: {max_query_file_size})")
          return f"Error: Query file too large (max 1MB)"
      except OSError as e:
        logger.error(f"Cannot access query file: {e}")
        return f"Error: Cannot access query file: {e}"
      
      with open(validated_query_file, 'r') as file:
        additional_query = file.read()
        # Sanitize the loaded query text
        # Use configurable max query length
        max_query_length = getattr(kb, 'max_query_length', 10000)
        additional_query = sanitize_query_text(additional_query, max_query_length)
        query_text = additional_query + f"\n{query_text}"
        
    except IOError as e:
      logger.error(f"Error reading file: {e}")
      return f"Error reading query file: {e}"

  logger.info(f"Query: {query_text}")

  # Check if only context is requested
  return_context_only = args.context_only
  if return_context_only:
    logger.warning("Returning context only")

  if args.verbose:
    kb.save_config()

  logger.info(f"Knowledgebase db: {kb.knowledge_base_db}")

  # Check if database exists
  if not os.path.exists(kb.knowledge_base_db):
    return f"Error: Database {kb.knowledge_base_db} does not exist"

  # Connect to database
  connect_to_database(kb)
  kb.sql_connection.commit()

  # Check if vector database exists
  if not os.path.exists(kb.knowledge_base_vector):
    close_database(kb)
    return f"Error: Vector Database {kb.knowledge_base_vector} does not yet exist!"

  # Load FAISS index
  index = faiss.read_index(kb.knowledge_base_vector)
  
  # Move index to GPU if available and index size permits
  ngpus = faiss.get_num_gpus()
  use_gpu = False
  
  if ngpus > 0:
    # Check index size
    index_size_mb = os.path.getsize(kb.knowledge_base_vector) / (1024 * 1024)
    logger.info(f"FAISS index size: {index_size_mb:.1f} MB")
    
    # Only use GPU for indexes that fit comfortably (leave 4GB buffer for temp memory)
    gpu_memory_limit_mb = 19 * 1024  # 19GB limit for 23GB GPU
    
    if index_size_mb < gpu_memory_limit_mb:
      try:
        logger.info(f"GPU detected, moving FAISS index to GPU (found {ngpus} GPU(s))")
        res = faiss.StandardGpuResources()
        # Configure GPU resources
        co = faiss.GpuClonerOptions()
        co.useFloat16 = getattr(kb, 'faiss_gpu_use_float16', True)
        # Move index to GPU
        index = faiss.index_cpu_to_gpu(res, 0, index, co)
        logger.info("FAISS index loaded on GPU")
        use_gpu = True
      except RuntimeError as e:
        logger.warning(f"Failed to load index on GPU, falling back to CPU: {e}")
        # Index remains on CPU
    else:
      logger.info(f"Index too large for GPU ({index_size_mb:.1f} MB > {gpu_memory_limit_mb} MB limit), using CPU")
  
  if not use_gpu:
    logger.info("Using CPU for FAISS search")

  # Generate query embedding asynchronously
  query_vector = await get_query_embedding(query_text, kb.vector_model, kb)

  # Perform hybrid search (vector + BM25) or fallback to vector-only
  search_results = await perform_hybrid_search(kb, query_text, query_vector, index)
  
  # Apply reranking if enabled
  if getattr(kb, 'enable_reranking', False):
    try:
      logger.info(f"Applying reranking to top {getattr(kb, 'reranking_top_k', 20)} results")
      search_results = await rerank_search_results(kb, query_text, search_results)
      logger.info("Reranking completed successfully")
    except Exception as e:
      logger.error(f"Reranking failed, continuing with original results: {e}")
      # Continue with original search results if reranking fails
  
  if not search_results:
    logger.warning("No search results returned")
    close_database(kb)
    return "No relevant documents found."
  
  # Extract indices and distances from search results
  indices = [[result[0] for result in search_results]]
  distances = [[result[1] for result in search_results]]
  
  logger.info(f"Search returned {len(search_results)} results")
  logger.debug(f"{distances[0][:5]=}\n  {indices[0][:5]=}\n")

  # Check database connection
  if kb.sql_cursor.connection is None:
    logger.error("Database connection is not open.")
    close_database(kb)
    return "Error: Database connection is not open."

  # Prepare batch processing (configurable batch size)
  batch_size = getattr(kb, 'reference_batch_size', 5)  # Process documents at a time
  reference_batches = []
  for i in range(0, len(indices[0]), batch_size):
    batch = [(int(indices[0][j]), float(distances[0][j])) 
             for j in range(i, min(i+batch_size, len(indices[0])))]
    reference_batches.append(batch)
  
  # Process batches concurrently
  tasks = [process_reference_batch(kb, batch) for batch in reference_batches]
  reference_lists = await asyncio.gather(*tasks)
  
  # Flatten and sort references
  reference = []
  for ref_list in reference_lists:
    reference.extend(ref_list)
  
  # Remove duplicates and sort
  seen_ids = set()
  unique_reference = []
  for item in reference:
    if item[0] not in seen_ids:
      seen_ids.add(item[0])
      unique_reference.append(item)
  
  # Sort by distance then source and sid
  unique_reference.sort(key=lambda x: (x[4], x[1], x[2]))
  
  # Close database connection
  close_database(kb)

  # Read context files in parallel
  context_files_content = []
  if kb.query_context_files:
    # Get configurable context file I/O thread pool size
    max_workers = getattr(kb, 'io_thread_pool_size', 4)
    with ThreadPoolExecutor(max_workers=min(max_workers, len(kb.query_context_files))) as executor:
      context_files_content = list(executor.map(
        read_context_file, 
        [file for file in kb.query_context_files if file]
      ))

  # Build reference string
  reference_string = build_reference_string(kb, unique_reference, context_files_content)

  logger.info(f"context_length={int(len(reference_string) / 1024)}KB, {return_context_only=}")

  # Return context only if requested
  if return_context_only:
    logger.info(f"Elapsed Time: {elapsed_time(kb.start_time)}")
    return reference_string

  # Generate AI response
  return await generate_ai_response(kb, reference_string, query_text)

def process_query(args: argparse.Namespace, logger) -> str:
  """
  Execute a semantic query on the CustomKB knowledge base and generate an AI-based response.
  
  Performs vector similarity search against the knowledge base using the query text,
  retrieves relevant context, and optionally generates a response using AI models
  (OpenAI, Anthropic Claude, or Meta Llama).
  
  Args:
      args: Command-line arguments containing:
          config_file: Path to knowledge base configuration
          query_text: The text to search for
          query_file: Optional path to file with additional query text
          context_only: Flag to return only the context without generating a response
          role: Custom system role for the LLM
          model: LLM model to use
          top_k: Number of top results to return
          context_scope: Number of segments to include for each result
          temperature: Model temperature setting
          max_tokens: Maximum tokens for the response
          verbose: Enable verbose output
          debug: Enable debug output
      logger: Initialized logger instance

  Returns:
      The AI-generated response or retrieved context, depending on the context_only flag.
  """
  return asyncio.run(process_query_async(args, logger))

def build_reference_string(kb: KnowledgeBase, reference: List[List[Any]], 
                          context_files_content: List[Tuple[str, str]] = None) -> str:
  """
  Build a reference string from the retrieved documents.

  Args:
      kb: The KnowledgeBase instance.
      reference: List of reference documents.
      context_files_content: Pre-loaded context files content.

  Returns:
      The formatted reference string.
  """
  reference_string = ''

  # Add context files if specified
  if context_files_content:
    for file_content, base_name in context_files_content:
      if file_content and base_name:
        reference_string += f'<reference src="{xml.sax.saxutils.escape(base_name)}">\n'
        reference_string += f"{file_content}\n</reference>\n\n"

  # Add reference documents
  src = old_src = ''
  sid = old_sid = 0
  end_context = ''

  logger.info(f'Processing {len(reference)} reference items')

  for item in reference:
    src = item[1]
    sid = item[2]
    rtext = item[3].strip("\n")
    rtext = xml.sax.saxutils.escape(rtext)
    similarity = item[4] if len(item) > 4 else 1.0
    
    # Intelligent path truncation for display
    display_src = src
    if len(src) > 50 and '/' in src:
      # Show last 2-3 directories: .../parent/dir/file.txt
      parts = src.split('/')
      if len(parts) > 3:
        display_src = '.../' + '/'.join(parts[-3:])
      else:
        display_src = src
    
    # Extract metadata if available
    metadata_str = item[5] if len(item) > 5 else None
    metadata_attrs = ""
    
    if metadata_str:
      try:
        # Safely parse metadata using JSON instead of ast.literal_eval
        from utils.security_utils import safe_json_loads
        
        # Try to parse as JSON first (safer)
        try:
          # Use configurable max JSON size
          max_json_size = getattr(kb, 'max_json_size', 10000)
          metadata = safe_json_loads(metadata_str, max_json_size)
        except ValueError:
          # Fallback: if it's not JSON, try to convert Python dict string to JSON
          # This handles cases where metadata was stored as Python dict strings
          try:
            # Replace Python dict syntax with JSON syntax
            json_str = metadata_str.replace("'", '"').replace('True', 'true').replace('False', 'false').replace('None', 'null')
            metadata = safe_json_loads(json_str, max_json_size)
          except ValueError:
            logger.warning(f"Could not parse metadata: {metadata_str[:100]}...")
            metadata = {}
        
        # Add relevant metadata as attributes
        metadata_elems = []
        for key, value in metadata.items():
          if key in ['heading', 'section_type', 'source', 'char_length', 'word_count']:
            safe_value = xml.sax.saxutils.escape(str(value))
            metadata_elems.append(f'<meta name="{key}">{safe_value}</meta>')
        
        metadata_attrs = " ".join(metadata_elems)
      except (ValueError, SyntaxError, TypeError) as e:
        logger.warning(f"Error parsing metadata: {e}")
    
    # Add similarity score as an attribute
    similarity_attr = f'<meta name="similarity">{1.0 - similarity:.4f}</meta>'

    if src != old_src or sid != (old_sid + 1):
      # Close previous context if needed
      if end_context:
        reference_string += end_context
      
      # Open new context with metadata (use display_src for readability)
      reference_string += f'<context src="{xml.sax.saxutils.escape(display_src)}:{sid}">\n'
      
      # Add metadata elements if available
      if metadata_attrs or similarity_attr:
        reference_string += f'<metadata>\n{metadata_attrs}\n{similarity_attr}\n</metadata>\n'

    end_context = f'</context>\n\n'
    old_src = src
    old_sid = sid
    reference_string += rtext + "\n"

  reference_string += end_context

  return reference_string

async def generate_ai_response(kb: KnowledgeBase, reference_string: str, query_text: str) -> str:
  """
  Generate an AI response based on the reference string and query text.

  Args:
      kb: The KnowledgeBase instance.
      reference_string: The formatted reference string.
      query_text: The user's query text.

  Returns:
      The AI-generated response.
  """
  # Replace datetime placeholder in query role
  kb.query_role = kb.query_role.replace('{{datetime}}', datetime.now().isoformat())

  from utils.logging_utils import log_model_operation, log_operation_error, OperationLogger
  
  # Generate response using the appropriate model
  try:
    with OperationLogger(logger, "ai_response_generation", 
                        model=kb.query_model, 
                        temperature=kb.query_temperature,
                        max_tokens=kb.query_max_tokens) as op_logger:
      
      if kb.query_model.startswith('gpt'):
        op_logger.add_context(provider="openai", model_type="gpt")
        response = await async_openai_client.chat.completions.create(
          model=kb.query_model,
          messages=[
            {"role": "system", "content": kb.query_role},
            {"role": "user", "content": f"{reference_string}\n\n{query_text}"}
          ],
          temperature=kb.query_temperature,
          max_tokens=kb.query_max_tokens,
          stop=None
        )
        
        response_content = response.choices[0].message.content
        op_logger.add_context(
          response_tokens=len(response_content.split()) if response_content else 0,
          input_tokens=len(f"{kb.query_role}\n{reference_string}\n{query_text}".split())
        )
        
        logger.info(f"Elapsed Time: {elapsed_time(kb.start_time)}")
        return response_content

      elif kb.query_model.startswith('o1') or kb.query_model.startswith('o3'):
        op_logger.add_context(provider="openai", model_type="o1")
        response = await async_openai_client.chat.completions.create(
          model=kb.query_model,
          messages=[
            {"role": "user", "content": f"{kb.query_role}\n\n{reference_string}\n\n{query_text}\n"}
          ],
        )
        
        response_content = response.choices[0].message.content
        op_logger.add_context(
          response_tokens=len(response_content.split()) if response_content else 0
        )
        
        logger.info(f"Elapsed Time: {elapsed_time(kb.start_time)}")
        return response_content

      elif kb.query_model.startswith('claude'):
        op_logger.add_context(provider="anthropic", model_type="claude")
        message = await async_anthropic_client.messages.create(
          max_tokens=kb.query_max_tokens,
          messages=[{"role": "user", "content": f"{reference_string}\n\n{query_text}"}],
          model=kb.query_model,
          system=kb.query_role,
          temperature=kb.query_temperature
        )
        
        response_content = message.content[0].text
        op_logger.add_context(
          response_tokens=len(response_content.split()) if response_content else 0,
          input_tokens=len(f"{kb.query_role}\n{reference_string}\n{query_text}".split())
        )
        
        logger.info(f"Elapsed Time: {elapsed_time(kb.start_time)}")
        return response_content

      else:
        # Fallback to synchronous for non-async clients (llama)
        op_logger.add_context(provider="ollama", model_type="llama")
        response = llama_client.chat.completions.create(
          model=kb.query_model,
          messages=[
            {"role": "system", "content": kb.query_role},
            {"role": "user", "content": f"{reference_string}\n\n{query_text}"}
          ],
          temperature=kb.query_temperature,
          max_tokens=kb.query_max_tokens,
          stop=None
        )
        
        response_content = response.choices[0].message.content
        op_logger.add_context(
          response_tokens=len(response_content.split()) if response_content else 0,
          input_tokens=len(f"{kb.query_role}\n{reference_string}\n{query_text}".split())
        )
        
        logger.info(f"Elapsed Time: {elapsed_time(kb.start_time)}")
        return response_content
  except Exception as e:
    log_operation_error(logger, "ai_response_generation", e,
                       model=kb.query_model,
                       temperature=kb.query_temperature,
                       max_tokens=kb.query_max_tokens,
                       query_length=len(query_text),
                       context_length=len(reference_string))
    return f"Error: Failed to generate response: {e}"


#fin
