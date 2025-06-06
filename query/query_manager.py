#!/usr/bin/env python
"""
Query management for CustomKB.
Handles semantic searching and response generation.
"""

import os
import sys
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
CACHE_TTL = 3600 * 24 * 7  # 7 days in seconds

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

def get_cached_query_embedding(query_text: str, model: str) -> Optional[List[float]]:
  """
  Retrieve a cached query embedding if it exists.
  
  Args:
      query_text: The query text.
      model: The model used for embedding.
      
  Returns:
      The cached embedding or None if not found or expired.
  """
  cache_key = get_cache_key(query_text, model)
  cache_file = os.path.join(CACHE_DIR, f"{cache_key}.json")
  
  if os.path.exists(cache_file):
    try:
      # Check if cache is expired
      file_time = os.path.getmtime(cache_file)
      if time.time() - file_time > CACHE_TTL:
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

async def get_query_embedding(query_text: str, model: str) -> np.ndarray:
  """
  Get embedding for a query, using cache if available.
  
  Args:
      query_text: The query text.
      model: The model to use for embedding.
      
  Returns:
      Numpy array containing the embedding vector.
  """
  clean_query = clean_text(query_text)
  cached_embedding = get_cached_query_embedding(clean_query, model)
  
  if cached_embedding:
    logger.info("Using cached query embedding")
    embedding = cached_embedding
  else:
    response = await async_openai_client.embeddings.create(
      input=clean_query, 
      model=model
    )
    embedding = response.data[0].embedding
    save_query_embedding_to_cache(clean_query, model, embedding)
    
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
    # Adjust context scope based on similarity
    if distance < 0.6:
      local_context_scope = max(int(context_scope / 2), 1)
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
        max_query_file_size = 1024 * 1024  # 1MB limit for query files
        if file_size > max_query_file_size:
          logger.error(f"Query file too large: {file_size} bytes (max: {max_query_file_size})")
          return f"Error: Query file too large (max 1MB)"
      except OSError as e:
        logger.error(f"Cannot access query file: {e}")
        return f"Error: Cannot access query file: {e}"
      
      with open(validated_query_file, 'r') as file:
        additional_query = file.read()
        # Sanitize the loaded query text
        additional_query = sanitize_query_text(additional_query)
        query_text = additional_query + f"\n{query_text}"
        
    except IOError as e:
      logger.error(f"Error reading file: {e}")
      return f"Error reading query file: {e}"

  logger.info(f"Query: {query_text}")

  # Check if only context is requested
  return_context_only = args.context_only
  if return_context_only:
    logger.warning("Returning context only")

  # Initialize knowledge base
  kb = KnowledgeBase(cfgfile)
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

  # Generate query embedding asynchronously
  query_vector = await get_query_embedding(query_text, kb.vector_model)

  # Search for similar vectors
  distances, indices = index.search(query_vector, kb.query_top_k)
  logger.info(f"{distances[0]=}\n  {indices[0]=}\n")

  # Check database connection
  if kb.sql_cursor.connection is None:
    logger.error("Database connection is not open.")
    close_database(kb)
    return "Error: Database connection is not open."

  # Prepare batch processing
  batch_size = 5  # Process 5 documents at a time
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
    with ThreadPoolExecutor(max_workers=min(4, len(kb.query_context_files))) as executor:
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
    
    # Extract metadata if available
    metadata_str = item[5] if len(item) > 5 else None
    metadata_attrs = ""
    
    if metadata_str:
      try:
        # Safely parse metadata using JSON instead of ast.literal_eval
        from utils.security_utils import safe_json_loads
        
        # Try to parse as JSON first (safer)
        try:
          metadata = safe_json_loads(metadata_str)
        except ValueError:
          # Fallback: if it's not JSON, try to convert Python dict string to JSON
          # This handles cases where metadata was stored as Python dict strings
          try:
            # Replace Python dict syntax with JSON syntax
            json_str = metadata_str.replace("'", '"').replace('True', 'true').replace('False', 'false').replace('None', 'null')
            metadata = safe_json_loads(json_str)
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
      
      # Open new context with metadata
      reference_string += f'<context src="{xml.sax.saxutils.escape(src)}:{sid}">\n'
      
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
