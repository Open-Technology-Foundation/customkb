#!/usr/bin/env python
"""
Query management for CustomKB.

NOTE: This module is being refactored. New code should import from:
- query.search for search functionality
- query.enhancement for query preprocessing
- query.embedding for query embeddings
- query.response for AI response generation
- query.processing for main orchestration

This file maintains backward compatibility during the transition.
All imports below will trigger deprecation warnings after 2025-08-30.
"""

import warnings
import argparse
import asyncio
import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Set

from utils.logging_config import setup_logging, get_logger, elapsed_time
from config.config_manager import KnowledgeBase, get_fq_cfg_filename

# Import from new refactored modules
from .search import (
    get_context_range as _get_context_range,
    fetch_document_by_id as _fetch_document_by_id,
    filter_results_by_category as _filter_results_by_category,
    perform_hybrid_search as _perform_hybrid_search,
    process_reference_batch as _process_reference_batch,
    perform_vector_search,
    perform_bm25_search,
    merge_search_results
)

from .enhancement import (
    normalize_query as _normalize_query,
    get_synonyms_for_word as _get_synonyms_for_word,
    correct_spelling as _correct_spelling,
    expand_synonyms as _expand_synonyms,
    apply_spelling_correction as _apply_spelling_correction,
    enhance_query as _enhance_query,
    get_enhancement_cache_key as _get_enhancement_cache_key,
    get_cached_enhanced_query as _get_cached_enhanced_query,
    save_enhanced_query_to_cache as _save_enhanced_query_to_cache,
    clear_enhancement_cache,
    get_enhancement_stats
)

from .embedding import (
    get_cache_key as _get_cache_key,
    get_cached_query_embedding as _get_cached_query_embedding,
    save_query_embedding_to_cache as _save_query_embedding_to_cache,
    get_query_embedding as _get_query_embedding,
    generate_query_embedding,
    clear_query_cache,
    get_query_cache_stats,
    validate_embedding_dimensions
)

from .response import (
    load_and_validate_api_keys as _load_and_validate_api_keys,
    _is_reasoning_model,
    format_messages_for_responses_api as _format_messages_for_responses_api,
    _extract_content_from_response,
    generate_ai_response as _generate_ai_response,
    get_prompt_template,
    generate_openai_response,
    generate_anthropic_response,
    generate_google_response,
    generate_xai_response,
    generate_llama_response,
    initialize_clients
)

from .processing import (
    read_context_file as _read_context_file,
    build_reference_string as _build_reference_string,
    process_query as _process_query,
    process_query_async as _process_query_async,
    validate_query_args,
    prepare_query_context,
    batch_process_queries
)

logger = get_logger(__name__)

# Initialize clients for backward compatibility
initialize_clients()

# Deprecation warning helper
def _deprecation_warning(func_name: str, new_module: str):
    """Issue deprecation warning for function usage."""
    warnings.warn(
        f"Importing '{func_name}' from query.query_manager is deprecated. "
        f"Import from query.{new_module} instead. "
        f"This compatibility layer will be removed after 2025-08-30.",
        DeprecationWarning,
        stacklevel=3
    )

# Backward compatibility wrapper functions - Search module
def get_context_range(index_start: int, context_n: int) -> List[int]:
    """Calculate the start and end indices for context retrieval."""
    _deprecation_warning('get_context_range', 'search')
    return _get_context_range(index_start, context_n)

def fetch_document_by_id(kb: KnowledgeBase, doc_id: int) -> Optional[Tuple[int, int, str]]:
    """Fetch a document by its ID from the database."""
    _deprecation_warning('fetch_document_by_id', 'search')
    return _fetch_document_by_id(kb, doc_id)

async def filter_results_by_category(kb: KnowledgeBase, results: List[Tuple[int, float]], 
                                    categories: List[str]) -> List[Tuple[int, float]]:
    """Filter search results by category if categorization is enabled."""
    _deprecation_warning('filter_results_by_category', 'search')
    return await _filter_results_by_category(kb, results, categories)

async def perform_hybrid_search(kb: KnowledgeBase, query_text: str, 
                               query_embedding: np.ndarray,
                               top_k: int = 10,
                               categories: List[str] = None,
                               rerank: bool = False) -> List[Tuple[int, float]]:
    """Perform hybrid search combining vector similarity and BM25 keyword search."""
    _deprecation_warning('perform_hybrid_search', 'search')
    return await _perform_hybrid_search(kb, query_text, query_embedding, top_k, categories, rerank)

async def process_reference_batch(kb: KnowledgeBase, batch: List[Tuple[int, float]]) -> List[List[Any]]:
    """Process a batch of search results to retrieve document content."""
    _deprecation_warning('process_reference_batch', 'search')
    return await _process_reference_batch(kb, batch)

# Backward compatibility wrapper functions - Enhancement module
def normalize_query(query: str) -> str:
    """Normalize a query string for better matching."""
    _deprecation_warning('normalize_query', 'enhancement')
    return _normalize_query(query)

def get_synonyms_for_word(word: str, max_synonyms: int = 2, 
                         relevance_threshold: float = 0.6) -> List[str]:
    """Get synonyms for a word using various methods."""
    _deprecation_warning('get_synonyms_for_word', 'enhancement')
    return _get_synonyms_for_word(word, max_synonyms, relevance_threshold)

def correct_spelling(word: str, vocabulary: Optional[Set[str]] = None) -> str:
    """Attempt to correct spelling of a word."""
    _deprecation_warning('correct_spelling', 'enhancement')
    return _correct_spelling(word, vocabulary)

def expand_synonyms(query: str, kb: Optional[KnowledgeBase] = None) -> str:
    """Expand query with synonyms for better matching."""
    _deprecation_warning('expand_synonyms', 'enhancement')
    return _expand_synonyms(query, kb)

def apply_spelling_correction(query: str, kb: Optional[KnowledgeBase] = None) -> str:
    """Apply spelling correction to query terms."""
    _deprecation_warning('apply_spelling_correction', 'enhancement')
    return _apply_spelling_correction(query, kb)

def enhance_query(query: str, kb: Optional[KnowledgeBase] = None) -> str:
    """Apply all query enhancements."""
    _deprecation_warning('enhance_query', 'enhancement')
    return _enhance_query(query, kb)

def get_enhancement_cache_key(query_text: str) -> str:
    """Generate cache key for enhanced query."""
    _deprecation_warning('get_enhancement_cache_key', 'enhancement')
    return _get_enhancement_cache_key(query_text)

def get_cached_enhanced_query(query_text: str, kb=None) -> Optional[str]:
    """Retrieve enhanced query from cache."""
    _deprecation_warning('get_cached_enhanced_query', 'enhancement')
    return _get_cached_enhanced_query(query_text, kb)

def save_enhanced_query_to_cache(original_query: str, enhanced_query: str) -> None:
    """Save enhanced query to cache."""
    _deprecation_warning('save_enhanced_query_to_cache', 'enhancement')
    return _save_enhanced_query_to_cache(original_query, enhanced_query)

# Backward compatibility wrapper functions - Embedding module
def get_cache_key(query_text: str, model: str) -> str:
    """Generate a cache key for query embeddings."""
    _deprecation_warning('get_cache_key', 'embedding')
    return _get_cache_key(query_text, model)

def get_cached_query_embedding(query_text: str, model: str, kb=None) -> Optional[List[float]]:
    """Retrieve cached query embedding."""
    _deprecation_warning('get_cached_query_embedding', 'embedding')
    return _get_cached_query_embedding(query_text, model, kb)

def save_query_embedding_to_cache(query_text: str, model: str, embedding: List[float]) -> None:
    """Save query embedding to cache."""
    _deprecation_warning('save_query_embedding_to_cache', 'embedding')
    return _save_query_embedding_to_cache(query_text, model, embedding)

async def get_query_embedding(query_text: str, model: str, kb: Optional[KnowledgeBase] = None) -> np.ndarray:
    """Get embedding for a query, using cache if available."""
    _deprecation_warning('get_query_embedding', 'embedding')
    return await _get_query_embedding(query_text, model, kb)

# Backward compatibility wrapper functions - Response module
def load_and_validate_api_keys():
    """Load and validate API keys securely."""
    _deprecation_warning('load_and_validate_api_keys', 'response')
    return _load_and_validate_api_keys()

def format_messages_for_responses_api(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Format messages for OpenAI Responses API."""
    _deprecation_warning('format_messages_for_responses_api', 'response')
    return _format_messages_for_responses_api(messages)

async def generate_ai_response(kb: KnowledgeBase, reference_string: str, query_text: str,
                              prompt_template: str = None) -> str:
    """Generate AI response using the configured model."""
    _deprecation_warning('generate_ai_response', 'response')
    return await _generate_ai_response(kb, reference_string, query_text, prompt_template)

# Backward compatibility wrapper functions - Processing module
def read_context_file(file_path: str) -> Tuple[str, str]:
    """Read additional context from a file."""
    _deprecation_warning('read_context_file', 'processing')
    return _read_context_file(file_path)

def build_reference_string(kb: KnowledgeBase, reference: List[List[Any]], 
                          context_files_content: List[Tuple[str, str]] = None,
                          debug: bool = False, format_type: str = None) -> str:
    """Build a reference string from the retrieved documents."""
    _deprecation_warning('build_reference_string', 'processing')
    return _build_reference_string(kb, reference, context_files_content, debug, format_type)

def process_query(args: argparse.Namespace, logger) -> str:
    """Execute a semantic query synchronously."""
    _deprecation_warning('process_query', 'processing')
    return _process_query(args, logger)

async def process_query_async(args: argparse.Namespace, logger) -> str:
    """Execute a semantic query asynchronously."""
    _deprecation_warning('process_query_async', 'processing')
    return await _process_query_async(args, logger)

# Export all public functions
__all__ = [
    # Search functions
    'get_context_range',
    'fetch_document_by_id',
    'filter_results_by_category',
    'perform_hybrid_search',
    'process_reference_batch',
    'perform_vector_search',
    'perform_bm25_search',
    'merge_search_results',
    
    # Enhancement functions
    'normalize_query',
    'get_synonyms_for_word',
    'correct_spelling',
    'expand_synonyms',
    'apply_spelling_correction',
    'enhance_query',
    'get_enhancement_cache_key',
    'get_cached_enhanced_query',
    'save_enhanced_query_to_cache',
    'clear_enhancement_cache',
    'get_enhancement_stats',
    
    # Embedding functions
    'get_cache_key',
    'get_cached_query_embedding',
    'save_query_embedding_to_cache',
    'get_query_embedding',
    'generate_query_embedding',
    'clear_query_cache',
    'get_query_cache_stats',
    'validate_embedding_dimensions',
    
    # Response functions
    'load_and_validate_api_keys',
    '_is_reasoning_model',
    'format_messages_for_responses_api',
    '_extract_content_from_response',
    'generate_ai_response',
    'get_prompt_template',
    'generate_openai_response',
    'generate_anthropic_response',
    'generate_google_response',
    'generate_xai_response',
    'generate_llama_response',
    'initialize_clients',
    
    # Processing functions
    'read_context_file',
    'build_reference_string',
    'process_query',
    'process_query_async',
    'validate_query_args',
    'prepare_query_context',
    'batch_process_queries',
]

#fin