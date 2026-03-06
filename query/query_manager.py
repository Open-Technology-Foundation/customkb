#!/usr/bin/env python
"""
Query management re-export hub for CustomKB.

This module provides a single import point for all query subsystem functions.
The actual implementations live in dedicated submodules:
- query.search — FAISS vector search, BM25, hybrid search, result processing
- query.enhancement — query normalization, spelling correction, synonym expansion
- query.embedding — query embedding generation and caching
- query.response — LLM response generation (OpenAI, Anthropic, Google, xAI)
- query.processing — top-level query orchestration and context building
- query.llm — unified LLM provider interface via LiteLLM
"""

from .embedding import (
  clear_query_cache,
  generate_query_embedding,
  get_cache_key,
  get_cached_query_embedding,
  get_query_cache_stats,
  get_query_embedding,
  save_query_embedding_to_cache,
  validate_embedding_dimensions,
)
from .enhancement import (
  apply_spelling_correction,
  clear_enhancement_cache,
  correct_spelling,
  enhance_query,
  expand_synonyms,
  get_cached_enhanced_query,
  get_enhancement_cache_key,
  get_enhancement_stats,
  get_synonyms_for_word,
  normalize_query,
  save_enhanced_query_to_cache,
)
from .llm import generate_ai_response
from .processing import (
  batch_process_queries,
  build_reference_string,
  prepare_query_context,
  process_query,
  process_query_async,
  read_context_file,
  validate_query_args,
)
from .response import (
  _extract_content_from_response,
  _is_reasoning_model,
  format_messages_for_responses_api,
  generate_anthropic_response,
  generate_google_response,
  generate_llama_response,
  generate_openai_response,
  generate_xai_response,
  get_prompt_template,
  initialize_clients,
  load_and_validate_api_keys,
)
from .search import (
  fetch_document_by_id,
  filter_results_by_category,
  get_context_range,
  merge_search_results,
  perform_bm25_search,
  perform_hybrid_search,
  perform_vector_search,
  process_reference_batch,
)

__all__ = [
  # Search
  'get_context_range',
  'fetch_document_by_id',
  'filter_results_by_category',
  'perform_hybrid_search',
  'process_reference_batch',
  'perform_vector_search',
  'perform_bm25_search',
  'merge_search_results',
  # Enhancement
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
  # Embedding
  'get_cache_key',
  'get_cached_query_embedding',
  'save_query_embedding_to_cache',
  'get_query_embedding',
  'generate_query_embedding',
  'clear_query_cache',
  'get_query_cache_stats',
  'validate_embedding_dimensions',
  # Response
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
  # Processing
  'read_context_file',
  'build_reference_string',
  'process_query',
  'process_query_async',
  'validate_query_args',
  'prepare_query_context',
  'batch_process_queries',
]

# fin
