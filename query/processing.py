#!/usr/bin/env python
"""
Main query processing orchestration for CustomKB.

This module coordinates the query pipeline: embedding generation,
search execution, context assembly, and response generation.
"""

import os
import asyncio
import argparse
from typing import List, Tuple, Any, Optional

from utils.logging_config import get_logger
from utils.text_utils import clean_text
from config.config_manager import KnowledgeBase, get_fq_cfg_filename
from database.db_manager import connect_to_database, close_database
from utils.exceptions import QueryError, ProcessingError

from .embedding import get_query_embedding
from .search import perform_hybrid_search, process_reference_batch
from .response import generate_ai_response

logger = get_logger(__name__)


def read_context_file(file_path: str) -> Tuple[str, str]:
  """
  Read additional context from a file.
  
  Args:
      file_path: Path to the context file
      
  Returns:
      Tuple of (file_content, file_name)
  """
  try:
    with open(file_path, 'r', encoding='utf-8') as f:
      content = f.read()
    
    # Get just the filename for reference
    file_name = os.path.basename(file_path)
    
    logger.debug(f"Read context file: {file_name} ({len(content)} characters)")
    return content, file_name
    
  except Exception as e:
    logger.error(f"Failed to read context file {file_path}: {e}")
    raise ProcessingError(f"Context file read failed: {e}") from e


def build_reference_string(kb: Any, reference: List[List[Any]], 
                          context_files_content: List[Tuple[str, str]] = None,
                          debug: bool = False, format_type: str = None) -> str:
  """
  Build a reference string from the retrieved documents.

  Args:
      kb: The KnowledgeBase instance.
      reference: List of reference documents.
      context_files_content: Pre-loaded context files content.
      debug: Enable debug information in output.
      format_type: Output format type ('xml', 'json', 'markdown', 'plain').

  Returns:
      Formatted reference string.
  """
  if not reference:
    return ""
  
  try:
    # Import formatter
    from query.formatters import format_references
    
    # Determine format type
    if not format_type:
      format_type = getattr(kb, 'reference_format', 'xml')
    
    # Format the references
    reference_string = format_references(
      references=reference,
      format_type=format_type,
      context_files=context_files_content,
      debug=debug
    )
    
    logger.debug(f"Built reference string: {len(reference_string)} characters, format: {format_type}")
    return reference_string
    
  except Exception as e:
    logger.error(f"Failed to build reference string: {e}")
    # Fallback to simple text format
    return build_simple_reference_string(reference, context_files_content)


def build_simple_reference_string(reference: List[List[Any]], 
                                 context_files_content: List[Tuple[str, str]] = None) -> str:
  """
  Build a simple text reference string as fallback.
  
  Args:
      reference: List of reference documents
      context_files_content: Optional context files
      
  Returns:
      Simple formatted reference string
  """
  parts = []
  
  # Add context files if provided
  if context_files_content:
    for content, filename in context_files_content:
      parts.append(f"=== Context File: {filename} ===")
      parts.append(content)
      parts.append("")
  
  # Add search results
  if reference:
    parts.append("=== Search Results ===")
    for i, ref in enumerate(reference, 1):
      if len(ref) >= 4:  # [id, sid, sourcedoc, originaltext, ...]
        sourcedoc = ref[2] if ref[2] else "Unknown"
        text = ref[3] if ref[3] else ""
        parts.append(f"--- Result {i}: {sourcedoc} ---")
        parts.append(text)
        parts.append("")
  
  return "\n".join(parts)


async def process_query_async(args: argparse.Namespace, logger) -> str:
  """
  Execute a semantic query asynchronously.
  
  This is the main async query processing function that orchestrates
  the entire query pipeline.
  
  Args:
      args: Command-line arguments
      logger: Logger instance
      
  Returns:
      The AI-generated response or retrieved context
  """
  try:
    # Load configuration
    config_file = get_fq_cfg_filename(args.config_file)
    kb = KnowledgeBase(config_file)
    
    if args.verbose:
      kb.save_config()
    
    # Extract query parameters
    query_text = args.query_text
    return_context_only = args.context_only if hasattr(args, 'context_only') else False
    
    # Get top_k with proper None handling
    top_k = getattr(args, 'top_k', None)
    if top_k is None:
      top_k = kb.query_top_k
    
    # Get context_scope with proper None handling
    context_scope = getattr(args, 'context_scope', None)
    if context_scope is None:
      context_scope = kb.query_context_scope
    
    categories = getattr(args, 'categories', None)
    
    # Parse categories if provided as string
    if categories and isinstance(categories, str):
      categories = [cat.strip() for cat in categories.split(',')]
    
    logger.info(f"Processing query: '{query_text[:50]}{'...' if len(query_text) > 50 else ''}'")
    logger.info(f"Parameters: top_k={top_k}, context_scope={context_scope}, categories={categories}")
    
    # Connect to database
    connect_to_database(kb)
    
    try:
      # Generate query embedding
      logger.debug("Generating query embedding...")
      query_embedding = await get_query_embedding(query_text, kb.vector_model, kb)
      
      # Perform hybrid search
      logger.debug("Performing hybrid search...")
      search_results = await perform_hybrid_search(
        kb=kb,
        query_text=query_text,
        query_embedding=query_embedding,
        top_k=top_k,
        categories=categories,
        rerank=getattr(kb, 'enable_reranking', False)
      )
      
      if not search_results:
        return "No relevant results found for your query."
      
      logger.info(f"Found {len(search_results)} relevant results")
      
      # Process search results to get document content
      logger.debug("Processing reference batch...")
      reference_data = await process_reference_batch(kb, search_results)
      
      if not reference_data:
        return "No document content could be retrieved."
      
      # Read context files from configuration or CLI
      context_files_content = []
      
      # First, load context files from KB configuration
      if kb.query_context_files:
        for context_file in kb.query_context_files:
          try:
            content, filename = read_context_file(context_file)
            context_files_content.append((content, filename))
            logger.debug(f"Loaded context file from config: {filename}")
          except Exception as e:
            logger.warning(f"Failed to read context file {context_file}: {e}")
      
      # Then, add any CLI-provided context files (from --context-files)
      context_files_arg = getattr(args, 'context_files', None)
      if context_files_arg:
        for context_file in args.context_files:
          try:
            content, filename = read_context_file(context_file)
            context_files_content.append((content, filename))
            logger.debug(f"Loaded context file from CLI: {filename}")
          except Exception as e:
            logger.warning(f"Failed to read context file {context_file}: {e}")
      
      # Build reference string
      format_type = getattr(args, 'format', None)
      debug_mode = getattr(args, 'debug', False)
      
      reference_string = build_reference_string(
        kb=kb,
        reference=reference_data,
        context_files_content=context_files_content,
        debug=debug_mode,
        format_type=format_type
      )
      
      logger.info(f"Context length: {len(reference_string)} characters")
      
      # Return context only if requested
      if return_context_only:
        logger.info("Returning context only (no AI response generation)")
        return reference_string
      
      # Generate AI response
      logger.debug("Generating AI response...")
      prompt_template = getattr(args, 'prompt_template', None)
      response = await generate_ai_response(
        kb=kb,
        reference_string=reference_string,
        query_text=query_text,
        prompt_template=prompt_template
      )
      
      if not response:
        return "Failed to generate a response. Please try again."
      
      logger.info(f"Generated response: {len(response)} characters")
      return response
      
    finally:
      # Always close database connection
      close_database(kb)
  
  except Exception as e:
    logger.error(f"Query processing failed: {e}")
    raise QueryError(f"Query processing error: {e}") from e


def process_query(args: argparse.Namespace, logger) -> str:
  """
  Execute a semantic query synchronously.
  
  This is the main synchronous entry point that wraps the async function.
  
  Args:
      args: Command-line arguments containing:
          config_file: Path to knowledgebase configuration
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
  try:
    return asyncio.run(process_query_async(args, logger))
  except KeyboardInterrupt:
    logger.info("Query processing interrupted by user")
    return "Query processing was interrupted."
  except Exception as e:
    logger.error(f"Query processing failed: {e}")
    return f"Error processing query: {e}"


def validate_query_args(args: argparse.Namespace) -> bool:
  """
  Validate query arguments.
  
  Args:
      args: Command-line arguments
      
  Returns:
      True if valid, raises exception otherwise
  """
  if not hasattr(args, 'query_text') or not args.query_text:
    raise ValueError("Query text is required")
  
  if not hasattr(args, 'config_file') or not args.config_file:
    raise ValueError("Configuration file is required")
  
  # Validate top_k
  if hasattr(args, 'top_k') and args.top_k is not None:
    if args.top_k <= 0:
      raise ValueError("top_k must be positive")
    if args.top_k > 1000:
      logger.warning(f"Large top_k value: {args.top_k}")
  
  # Validate context_scope
  if hasattr(args, 'context_scope') and args.context_scope is not None:
    if args.context_scope <= 0:
      raise ValueError("context_scope must be positive")
  
  # Validate temperature
  if hasattr(args, 'temperature') and args.temperature is not None:
    if not 0.0 <= args.temperature <= 2.0:
      raise ValueError("temperature must be between 0.0 and 2.0")
  
  # Validate max_tokens
  if hasattr(args, 'max_tokens') and args.max_tokens is not None:
    if args.max_tokens <= 0:
      raise ValueError("max_tokens must be positive")
  
  return True


def prepare_query_context(query_text: str, context_files: List[str] = None) -> Tuple[str, List[Tuple[str, str]]]:
  """
  Prepare query context by combining query text with context files.
  
  Args:
      query_text: Original query text
      context_files: Optional list of context file paths
      
  Returns:
      Tuple of (enhanced_query, context_files_content)
  """
  enhanced_query = clean_text(query_text)
  context_files_content = []
  
  if context_files:
    for context_file in context_files:
      try:
        content, filename = read_context_file(context_file)
        context_files_content.append((content, filename))
        
        # Optionally enhance query with context file information
        enhanced_query += f"\n\nAdditional context from {filename}:\n{content[:500]}..."
        
      except Exception as e:
        logger.warning(f"Could not read context file {context_file}: {e}")
  
  return enhanced_query, context_files_content


async def batch_process_queries(queries: List[str], config_file: str, 
                               **kwargs) -> List[Tuple[str, str]]:
  """
  Process multiple queries in batch.
  
  Args:
      queries: List of query strings
      config_file: Path to configuration file
      **kwargs: Additional query parameters
      
  Returns:
      List of (query, response) tuples
  """
  results = []
  
  # Load configuration once
  kb = KnowledgeBase(config_file)
  
  try:
    connect_to_database(kb)
    
    for i, query in enumerate(queries, 1):
      logger.info(f"Processing query {i}/{len(queries)}: {query[:50]}...")
      
      try:
        # Create mock args object
        class MockArgs:
          def __init__(self, query_text, config_file, **kwargs):
            self.query_text = query_text
            self.config_file = config_file
            self.context_only = kwargs.get('context_only', False)
            self.top_k = kwargs.get('top_k', 10)
            self.context_scope = kwargs.get('context_scope', 3)
            self.categories = kwargs.get('categories', None)
            self.format = kwargs.get('format', None)
            self.prompt_template = kwargs.get('prompt_template', None)
            self.debug = kwargs.get('debug', False)
            self.verbose = kwargs.get('verbose', False)
        
        args = MockArgs(query, config_file, **kwargs)
        response = await process_query_async(args, logger)
        results.append((query, response))
        
      except Exception as e:
        logger.error(f"Failed to process query '{query}': {e}")
        results.append((query, f"Error: {e}"))
    
  finally:
    close_database(kb)
  
  logger.info(f"Batch processing complete: {len(results)} queries processed")
  return results


#fin