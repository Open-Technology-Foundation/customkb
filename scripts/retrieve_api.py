#!/usr/bin/env python
"""
Retrieval API for nanochat.

Thin HTTP wrapper around CustomKB's hybrid search pipeline.
Exposes a POST endpoint that accepts {"query": "..."} and returns
{"context": "..."} for use by nanochat's --retrieve-url flag.

Usage:
    .venv/bin/python scripts/retrieve_api.py --kb okusiassociates --port 8100
"""

import argparse
import logging
import sys
import time

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

# CustomKB imports (run from /ai/scripts/customkb with its venv)
from config.config_manager import KnowledgeBase, get_fq_cfg_filename
from database.connection import connect_to_database, close_database
from query.embedding import get_query_embedding
from query.search import perform_hybrid_search, process_reference_batch
from query.processing import build_reference_string

logging.basicConfig(
  level=logging.INFO,
  format='%(asctime)s %(levelname)s %(name)s: %(message)s',
  datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger('retrieve_api')

# Globals set at startup
kb = None
max_chars = 3000
top_k = 5


class QueryRequest(BaseModel):
  query: str


class QueryResponse(BaseModel):
  context: str


app = FastAPI(title='Nanochat Retrieval API')


@app.get('/health')
def health():
  return {
    'status': 'ok',
    'kb': kb.knowledge_base_name if kb else None,
    'top_k': top_k,
    'max_chars': max_chars,
  }


@app.post('/', response_model=QueryResponse)
async def retrieve(req: QueryRequest):
  query_text = req.query.strip()
  if not query_text:
    return QueryResponse(context='')

  t0 = time.perf_counter()

  # Embed the query
  query_embedding = await get_query_embedding(query_text, kb.vector_model, kb)
  t_embed = time.perf_counter()

  # Hybrid search (FAISS + BM25, RRF fusion)
  search_results = await perform_hybrid_search(
    kb=kb,
    query_text=query_text,
    query_embedding=query_embedding,
    top_k=top_k,
    rerank=False,
  )
  t_search = time.perf_counter()

  if not search_results:
    logger.info(f'No results for: {query_text[:60]}')
    return QueryResponse(context='')

  # Fetch document content
  reference_data = await process_reference_batch(kb, search_results)
  if not reference_data:
    return QueryResponse(context='')

  # Build plain-text context string (no extra context files)
  context = build_reference_string(
    kb=kb,
    reference=reference_data,
    context_files_content=[],
    format_type='plain',
  )

  # Truncate to max_chars
  if len(context) > max_chars:
    context = context[:max_chars]

  t_total = time.perf_counter()
  logger.info(
    f'Query: {query_text[:60]!r} | '
    f'embed={1000*(t_embed-t0):.0f}ms search={1000*(t_search-t_embed):.0f}ms '
    f'total={1000*(t_total-t0):.0f}ms | '
    f'{len(search_results)} hits, {len(context)} chars'
  )

  return QueryResponse(context=context)


def main():
  global kb, max_chars, top_k

  parser = argparse.ArgumentParser(description='Nanochat retrieval API')
  parser.add_argument('--kb', required=True, help='Knowledgebase name (e.g. okusiassociates)')
  parser.add_argument('--port', type=int, default=8100, help='Listen port (default: 8100)')
  parser.add_argument('--host', default='0.0.0.0', help='Listen host (default: 0.0.0.0)')
  parser.add_argument('--top-k', type=int, default=5, help='Number of results (default: 5)')
  parser.add_argument('--max-chars', type=int, default=3000, help='Max context chars (default: 3000)')
  args = parser.parse_args()

  top_k = args.top_k
  max_chars = args.max_chars

  # Resolve and load KB config
  config_file = get_fq_cfg_filename(args.kb)
  if not config_file:
    logger.error(f'Knowledgebase not found: {args.kb}')
    sys.exit(1)

  kb = KnowledgeBase(config_file)
  logger.info(f'Loaded KB: {kb.knowledge_base_name} ({config_file})')

  # Connect to database (stays open for the lifetime of the server)
  connect_to_database(kb)
  logger.info(f'Database connected: {kb.knowledge_base_db}')

  # Warm the embedding model with a dummy query
  import asyncio
  logger.info(f'Warming embedding model: {kb.vector_model}')
  asyncio.run(get_query_embedding('warmup', kb.vector_model, kb))
  logger.info('Embedding model ready')

  logger.info(f'Starting retrieval API on {args.host}:{args.port} (top_k={top_k}, max_chars={max_chars})')

  uvicorn.run(app, host=args.host, port=args.port, log_level='warning')

  # Cleanup on shutdown
  close_database(kb)


if __name__ == '__main__':
  main()

#fin
