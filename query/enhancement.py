#!/usr/bin/env python
"""
Query enhancement functionality for CustomKB.

This module handles query preprocessing, normalization, synonym expansion,
spelling correction, and caching of enhanced queries.
"""

import re
import os
import json
import hashlib
from typing import Optional, Set, List, Any
from pathlib import Path

from utils.logging_config import get_logger
from utils.text_utils import clean_text
from utils.exceptions import ProcessingError

logger = get_logger(__name__)

# Enhancement cache directory
ENHANCEMENT_CACHE_DIR = os.path.join(os.getenv('VECTORDBS', '/var/lib/vectordbs'), '.query_enhancement_cache')
os.makedirs(ENHANCEMENT_CACHE_DIR, exist_ok=True)


def normalize_query(query: str) -> str:
  """
  Normalize a query string for better matching.
  
  Args:
      query: Input query string
      
  Returns:
      Normalized query string
  """
  if not query:
    return ""
  
  # Basic text cleaning
  normalized = clean_text(query)
  
  # Additional query-specific normalization
  # Remove extra whitespace
  normalized = re.sub(r'\s+', ' ', normalized).strip()
  
  # Standardize quotes
  normalized = re.sub(r'[""\u201c\u201d]', '"', normalized)
  normalized = re.sub(r"['\u2018\u2019]", "'", normalized)
  
  # Remove redundant punctuation
  normalized = re.sub(r'[.]{2,}', '.', normalized)
  normalized = re.sub(r'[?]{2,}', '?', normalized)
  normalized = re.sub(r'[!]{2,}', '!', normalized)
  
  logger.debug(f"Query normalized: '{query}' -> '{normalized}'")
  return normalized


def get_synonyms_for_word(word: str, max_synonyms: int = 2, 
                         relevance_threshold: float = 0.6) -> List[str]:
  """
  Get synonyms for a word using various methods.
  
  Args:
      word: Word to find synonyms for
      max_synonyms: Maximum number of synonyms to return
      relevance_threshold: Minimum relevance score for synonyms
      
  Returns:
      List of synonyms
  """
  if not word or len(word) < 2:
    return []
  
  synonyms = []
  
  try:
    # Try using NLTK WordNet if available
    import nltk
    from nltk.corpus import wordnet
    
    # Ensure WordNet is downloaded
    try:
      nltk.data.find('corpora/wordnet')
    except LookupError:
      logger.debug("WordNet not found, downloading...")
      nltk.download('wordnet', quiet=True)
    
    # Get synsets for the word
    synsets = wordnet.synsets(word.lower())
    
    for synset in synsets[:3]:  # Limit to first 3 synsets
      for lemma in synset.lemmas():
        synonym = lemma.name().replace('_', ' ')
        if (synonym.lower() != word.lower() and 
            synonym not in synonyms and 
            len(synonym) > 1):
          synonyms.append(synonym)
          if len(synonyms) >= max_synonyms:
            break
      if len(synonyms) >= max_synonyms:
        break
    
  except (ImportError, Exception) as e:
    logger.debug(f"WordNet synonym lookup failed: {e}")
  
  # Fallback: Use simple morphological variants
  if not synonyms and len(word) > 3:
    word_lower = word.lower()
    
    # Common suffix variations
    if word_lower.endswith('ing'):
      base = word_lower[:-3]
      synonyms.extend([base, base + 'ed', base + 'er'])
    elif word_lower.endswith('ed'):
      base = word_lower[:-2]
      synonyms.extend([base, base + 'ing', base + 'er'])
    elif word_lower.endswith('er'):
      base = word_lower[:-2]
      synonyms.extend([base, base + 'ing', base + 'ed'])
    elif word_lower.endswith('s') and len(word_lower) > 4:
      base = word_lower[:-1]
      synonyms.append(base)
    
    # Filter and limit
    synonyms = [s for s in synonyms if len(s) > 2 and s != word_lower][:max_synonyms]
  
  logger.debug(f"Found synonyms for '{word}': {synonyms}")
  return synonyms


def correct_spelling(word: str, vocabulary: Optional[Set[str]] = None) -> str:
  """
  Attempt to correct spelling of a word.
  
  Args:
      word: Word to correct
      vocabulary: Optional vocabulary set for validation
      
  Returns:
      Corrected word or original if no correction found
  """
  if not word or len(word) < 2:
    return word
  
  # If vocabulary is provided, check if word is already correct
  if vocabulary and word.lower() in vocabulary:
    return word
  
  try:
    # Try using textblob if available
    from textblob import TextBlob
    
    blob = TextBlob(word)
    corrected = str(blob.correct())
    
    # Only return correction if it's significantly different
    if corrected.lower() != word.lower() and len(corrected) > 1:
      logger.debug(f"Spelling correction: '{word}' -> '{corrected}'")
      return corrected
    
  except (ImportError, Exception) as e:
    logger.debug(f"Spelling correction failed: {e}")
  
  # Fallback: Simple corrections for common typos
  word_lower = word.lower()
  
  # Common double letter corrections
  if len(word_lower) > 3:
    # Remove double letters that are likely typos
    for i in range(len(word_lower) - 1):
      if word_lower[i] == word_lower[i + 1] and word_lower[i] in 'acefghiklmnoprstuwy':
        candidate = word_lower[:i] + word_lower[i+1:]
        if vocabulary and candidate in vocabulary:
          logger.debug(f"Simple correction: '{word}' -> '{candidate}'")
          return candidate
  
  return word


def expand_synonyms(query: str, kb: Optional[Any] = None) -> str:
  """
  Expand query with synonyms for better matching.
  
  Args:
      query: Original query string
      kb: Optional KnowledgeBase for configuration
      
  Returns:
      Query expanded with synonyms
  """
  if not query:
    return query
  
  # Check if synonym expansion is enabled
  enable_synonyms = True
  if kb:
    enable_synonyms = getattr(kb, 'enable_synonym_expansion', True)
  
  if not enable_synonyms:
    return query
  
  # Split query into words
  words = re.findall(r'\b\w+\b', query.lower())
  
  if len(words) == 0:
    return query
  
  # Configuration
  max_synonyms_per_word = getattr(kb, 'max_synonyms_per_word', 1) if kb else 1
  min_word_length = getattr(kb, 'synonym_min_word_length', 4) if kb else 4
  
  # Expand only key words (longer words, not common stop words)
  stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'must'}
  
  expanded_terms = []
  
  for word in words:
    if len(word) >= min_word_length and word not in stop_words:
      synonyms = get_synonyms_for_word(word, max_synonyms_per_word)
      if synonyms:
        # Add original word and synonyms
        expanded_terms.append(f"({word} OR {' OR '.join(synonyms)})")
      else:
        expanded_terms.append(word)
    else:
      expanded_terms.append(word)
  
  if expanded_terms:
    expanded_query = ' '.join(expanded_terms)
    logger.debug(f"Synonym expansion: '{query}' -> '{expanded_query}'")
    return expanded_query
  
  return query


def apply_spelling_correction(query: str, kb: Optional[Any] = None) -> str:
  """
  Apply spelling correction to query terms.
  
  Args:
      query: Original query string
      kb: Optional KnowledgeBase for configuration
      
  Returns:
      Query with corrected spelling
  """
  if not query:
    return query
  
  # Check if spelling correction is enabled
  enable_correction = True
  if kb:
    enable_correction = getattr(kb, 'enable_spelling_correction', True)
  
  if not enable_correction:
    return query
  
  # Extract words from query
  words = re.findall(r'\b\w+\b', query)
  
  if not words:
    return query
  
  corrected_words = []
  corrections_made = 0
  
  for word in words:
    if len(word) > 2:  # Only correct words longer than 2 characters
      corrected = correct_spelling(word)
      corrected_words.append(corrected)
      if corrected.lower() != word.lower():
        corrections_made += 1
    else:
      corrected_words.append(word)
  
  if corrections_made > 0:
    # Reconstruct query maintaining original structure
    corrected_query = query
    for original, corrected in zip(words, corrected_words):
      if original != corrected:
        # Use word boundaries to replace whole words only
        pattern = r'\b' + re.escape(original) + r'\b'
        corrected_query = re.sub(pattern, corrected, corrected_query, flags=re.IGNORECASE)
    
    logger.debug(f"Spelling correction applied: {corrections_made} words corrected")
    return corrected_query
  
  return query


def get_enhancement_cache_key(query_text: str) -> str:
  """
  Generate cache key for enhanced query.
  
  Args:
      query_text: Original query text
      
  Returns:
      Cache key string
  """
  return hashlib.sha256(query_text.encode()).hexdigest()


def get_cached_enhanced_query(query_text: str, kb=None) -> Optional[str]:
  """
  Retrieve enhanced query from cache.
  
  Args:
      query_text: Original query text
      kb: Optional KnowledgeBase for configuration
      
  Returns:
      Cached enhanced query or None
  """
  try:
    cache_key = get_enhancement_cache_key(query_text)
    cache_file = os.path.join(ENHANCEMENT_CACHE_DIR, f"{cache_key}.json")
    
    if os.path.exists(cache_file):
      # Check cache TTL
      cache_ttl = getattr(kb, 'enhancement_cache_ttl', 3600) if kb else 3600  # 1 hour default
      
      file_age = time.time() - os.path.getmtime(cache_file)
      if file_age < cache_ttl:
        with open(cache_file, 'r') as f:
          cache_data = json.load(f)
        
        if cache_data.get('original') == query_text:
          logger.debug("Using cached enhanced query")
          return cache_data.get('enhanced')
      else:
        # Remove expired cache
        os.remove(cache_file)
        logger.debug("Removed expired enhancement cache")
    
  except Exception as e:
    logger.debug(f"Enhancement cache retrieval failed: {e}")
  
  return None


def save_enhanced_query_to_cache(original_query: str, enhanced_query: str) -> None:
  """
  Save enhanced query to cache.
  
  Args:
      original_query: Original query text
      enhanced_query: Enhanced query text
  """
  if original_query == enhanced_query:
    return  # No point caching if no enhancement was made
  
  try:
    cache_key = get_enhancement_cache_key(original_query)
    cache_file = os.path.join(ENHANCEMENT_CACHE_DIR, f"{cache_key}.json")
    
    cache_data = {
      'original': original_query,
      'enhanced': enhanced_query,
      'timestamp': time.time()
    }
    
    with open(cache_file, 'w') as f:
      json.dump(cache_data, f)
    
    logger.debug("Enhanced query saved to cache")
    
  except Exception as e:
    logger.debug(f"Enhancement cache save failed: {e}")


def enhance_query(query: str, kb: Optional[Any] = None) -> str:
  """
  Apply all query enhancements.
  
  Args:
      query: Original query string
      kb: Optional KnowledgeBase for configuration
      
  Returns:
      Enhanced query string
  """
  if not query:
    return query
  
  # Check cache first
  cached_enhanced = get_cached_enhanced_query(query, kb)
  if cached_enhanced:
    return cached_enhanced
  
  try:
    # Start with normalization
    enhanced = normalize_query(query)
    
    # Apply spelling correction if enabled
    if getattr(kb, 'enable_spelling_correction', False) if kb else False:
      enhanced = apply_spelling_correction(enhanced, kb)
    
    # Apply synonym expansion if enabled
    if getattr(kb, 'enable_synonym_expansion', False) if kb else False:
      enhanced = expand_synonyms(enhanced, kb)
    
    # Cache the result if enhancement was made
    if enhanced != query:
      save_enhanced_query_to_cache(query, enhanced)
      logger.debug(f"Query enhanced: '{query}' -> '{enhanced}'")
    
    return enhanced
    
  except Exception as e:
    logger.error(f"Query enhancement failed: {e}")
    return query  # Return original query on error


def clear_enhancement_cache() -> int:
  """
  Clear the query enhancement cache.
  
  Returns:
      Number of files removed
  """
  removed_count = 0
  
  try:
    for file_path in Path(ENHANCEMENT_CACHE_DIR).glob("*.json"):
      try:
        file_path.unlink()
        removed_count += 1
      except OSError:
        pass
    
    logger.info(f"Cleared {removed_count} enhancement cache files")
    
  except Exception as e:
    logger.error(f"Failed to clear enhancement cache: {e}")
  
  return removed_count


def get_enhancement_stats() -> dict:
  """
  Get enhancement cache statistics.
  
  Returns:
      Dictionary with cache statistics
  """
  stats = {
    'cache_dir': ENHANCEMENT_CACHE_DIR,
    'cache_files': 0,
    'total_size_bytes': 0,
    'oldest_file': None,
    'newest_file': None
  }
  
  try:
    cache_files = list(Path(ENHANCEMENT_CACHE_DIR).glob("*.json"))
    stats['cache_files'] = len(cache_files)
    
    if cache_files:
      total_size = sum(f.stat().st_size for f in cache_files)
      stats['total_size_bytes'] = total_size
      
      # Get oldest and newest files
      file_times = [(f, f.stat().st_mtime) for f in cache_files]
      file_times.sort(key=lambda x: x[1])
      
      stats['oldest_file'] = file_times[0][1] if file_times else None
      stats['newest_file'] = file_times[-1][1] if file_times else None
    
  except Exception as e:
    logger.error(f"Failed to get enhancement stats: {e}")
  
  return stats


#fin