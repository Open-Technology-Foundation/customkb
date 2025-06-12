"""
Mock data generators for CustomKB tests.
Provides realistic test data for databases, configurations, and API responses.
"""

import json
import tempfile
import os
from typing import Dict, List, Any, Optional
from pathlib import Path


class MockDataGenerator:
  """Generate mock data for testing CustomKB components."""
  
  @staticmethod
  def create_sample_config(
    kb_name: str = "test_kb",
    vector_model: str = "text-embedding-3-small",
    vector_dimensions: int = 1536,
    include_new_sections: bool = True,
    **kwargs
  ) -> str:
    """
    Create a sample configuration file content.
    
    Args:
        kb_name: Knowledge base name
        vector_model: Embedding model to use
        vector_dimensions: Vector dimensions
        include_new_sections: Whether to include new config sections
        **kwargs: Additional config parameters
        
    Returns:
        Configuration file content as string
    """
    config_content = f"""[DEFAULT]
# Test configuration for {kb_name}
vector_model = {vector_model}
vector_dimensions = {vector_dimensions}
vector_chunks = {kwargs.get('vector_chunks', 200)}

db_min_tokens = {kwargs.get('db_min_tokens', 100)}
db_max_tokens = {kwargs.get('db_max_tokens', 200)}

query_model = {kwargs.get('query_model', 'gpt-4o')}
query_max_tokens = {kwargs.get('query_max_tokens', 4000)}
query_top_k = {kwargs.get('query_top_k', 50)}
query_context_scope = {kwargs.get('query_context_scope', 4)}
query_temperature = {kwargs.get('query_temperature', 0.1)}
query_role = {kwargs.get('query_role', 'You are a helpful assistant.')}
"""
    
    if 'query_context_files' in kwargs:
      config_content += f"query_context_files = {kwargs['query_context_files']}\n"
    
    # Add new configuration sections for comprehensive testing
    if include_new_sections:
      config_content += f"""
[API]
# API performance and rate limiting - test values
api_call_delay_seconds = {kwargs.get('api_call_delay_seconds', 0.01)}
api_max_retries = {kwargs.get('api_max_retries', 3)}
api_max_concurrency = {kwargs.get('api_max_concurrency', 2)}
api_min_concurrency = {kwargs.get('api_min_concurrency', 1)}
backoff_exponent = {kwargs.get('backoff_exponent', 2)}
backoff_jitter = {kwargs.get('backoff_jitter', 0.1)}

[LIMITS]
# File and memory limits - test values
max_file_size_mb = {kwargs.get('max_file_size_mb', 10)}
max_query_file_size_mb = {kwargs.get('max_query_file_size_mb', 1)}
memory_cache_size = {kwargs.get('memory_cache_size', 100)}
api_key_min_length = {kwargs.get('api_key_min_length', 20)}
max_query_length = {kwargs.get('max_query_length', 1000)}
max_config_value_length = {kwargs.get('max_config_value_length', 500)}
max_json_size = {kwargs.get('max_json_size', 1000)}

[PERFORMANCE]
# Processing and caching performance - test values
embedding_batch_size = {kwargs.get('embedding_batch_size', 10)}
checkpoint_interval = {kwargs.get('checkpoint_interval', 2)}
commit_frequency = {kwargs.get('commit_frequency', 10)}
io_thread_pool_size = {kwargs.get('io_thread_pool_size', 2)}
file_processing_batch_size = {kwargs.get('file_processing_batch_size', 5)}
sql_batch_size = {kwargs.get('sql_batch_size', 10)}
reference_batch_size = {kwargs.get('reference_batch_size', 2)}
query_cache_ttl_days = {kwargs.get('query_cache_ttl_days', 1)}
default_editor = {kwargs.get('default_editor', 'nano')}

[ALGORITHMS]
# Algorithm parameters and thresholds - test values
high_dimension_threshold = {kwargs.get('high_dimension_threshold', 1024)}
small_dataset_threshold = {kwargs.get('small_dataset_threshold', 10)}
medium_dataset_threshold = {kwargs.get('medium_dataset_threshold', 100)}
ivf_centroid_multiplier = {kwargs.get('ivf_centroid_multiplier', 2)}
max_centroids = {kwargs.get('max_centroids', 64)}
token_estimation_sample_size = {kwargs.get('token_estimation_sample_size', 3)}
token_estimation_multiplier = {kwargs.get('token_estimation_multiplier', 1.2)}
similarity_threshold = {kwargs.get('similarity_threshold', 0.7)}
low_similarity_scope_factor = {kwargs.get('low_similarity_scope_factor', 0.5)}
max_chunk_overlap = {kwargs.get('max_chunk_overlap', 50)}
overlap_ratio = {kwargs.get('overlap_ratio', 0.3)}
heading_search_limit = {kwargs.get('heading_search_limit', 100)}
entity_extraction_limit = {kwargs.get('entity_extraction_limit', 200)}
default_dir_permissions = {kwargs.get('default_dir_permissions', 493)}
default_code_language = {kwargs.get('default_code_language', 'python')}
additional_stopword_languages = {kwargs.get('additional_stopword_languages', 'french,german')}
"""
    
    return config_content

  @staticmethod
  def create_sample_texts() -> List[str]:
    """Create sample text documents for testing."""
    return [
      "# Introduction to Machine Learning\nMachine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed.",
      
      "## Natural Language Processing\nNLP combines computational linguistics with statistical machine learning and deep learning models to enable computers to process and analyze large amounts of natural language data.",
      
      "### Vector Embeddings\nVector embeddings are dense vector representations of words, phrases, or documents that capture semantic meaning in a high-dimensional space.",
      
      "```python\ndef calculate_similarity(vec1, vec2):\n    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))\n```",
      
      "The quick brown fox jumps over the lazy dog. This sentence contains every letter of the English alphabet at least once.",
      
      "Knowledge bases are structured repositories of information that can be searched and queried to retrieve relevant context for various applications.",
      
      "FAISS (Facebook AI Similarity Search) is a library for efficient similarity search and clustering of dense vectors, optimized for large-scale applications.",
      
      "OpenAI's embedding models convert text into numerical vectors that capture semantic relationships between different pieces of content.",
      
      "## Database Design\nSQLite provides a lightweight, file-based database solution that's perfect for applications requiring structured data storage without the overhead of a full database server.",
      
      "Semantic search goes beyond keyword matching to understand the intent and contextual meaning of search queries, providing more relevant results."
    ]

  @staticmethod
  def create_mock_embedding_response(texts: List[str], dimensions: int = 1536) -> Dict[str, Any]:
    """
    Create a mock OpenAI embedding API response.
    
    Args:
        texts: List of texts to create embeddings for
        dimensions: Embedding vector dimensions
        
    Returns:
        Mock API response dictionary
    """
    import numpy as np
    
    embeddings = []
    for i, text in enumerate(texts):
      # Create deterministic but realistic-looking embeddings
      np.random.seed(hash(text) % 2**32)
      embedding = np.random.normal(0, 0.1, dimensions).tolist()
      embeddings.append({
        "object": "embedding",
        "index": i,
        "embedding": embedding
      })
    
    return {
      "object": "list",
      "data": embeddings,
      "model": "text-embedding-3-small",
      "usage": {
        "prompt_tokens": sum(len(text.split()) for text in texts),
        "total_tokens": sum(len(text.split()) for text in texts)
      }
    }

  @staticmethod
  def create_mock_chat_response(content: str, model: str = "gpt-4o") -> Dict[str, Any]:
    """
    Create a mock OpenAI chat completion response.
    
    Args:
        content: Response content
        model: Model name
        
    Returns:
        Mock API response dictionary
    """
    return {
      "id": "chatcmpl-test123",
      "object": "chat.completion",
      "created": 1677652288,
      "model": model,
      "choices": [{
        "index": 0,
        "message": {
          "role": "assistant",
          "content": content
        },
        "finish_reason": "stop"
      }],
      "usage": {
        "prompt_tokens": 50,
        "completion_tokens": len(content.split()),
        "total_tokens": 50 + len(content.split())
      }
    }

  @staticmethod
  def create_mock_anthropic_response(content: str, model: str = "claude-3-sonnet-20240229") -> Dict[str, Any]:
    """
    Create a mock Anthropic message response.
    
    Args:
        content: Response content
        model: Model name
        
    Returns:
        Mock API response dictionary
    """
    return {
      "id": "msg_test123",
      "type": "message",
      "role": "assistant",
      "content": [{"type": "text", "text": content}],
      "model": model,
      "stop_reason": "end_turn",
      "stop_sequence": None,
      "usage": {
        "input_tokens": 50,
        "output_tokens": len(content.split())
      }
    }

  @staticmethod
  def create_database_rows(texts: List[str], source_doc: str = "test.txt") -> List[tuple]:
    """
    Create database rows for testing.
    
    Args:
        texts: List of text chunks
        source_doc: Source document name
        
    Returns:
        List of database row tuples
    """
    rows = []
    for i, text in enumerate(texts):
      # (id, sid, sourcedoc, originaltext, embedtext, embedded, language, metadata)
      metadata = json.dumps({
        "char_length": len(text),
        "word_count": len(text.split()),
        "source": source_doc
      })
      
      clean_text = text.lower().replace('\n', ' ').strip()
      
      row = (
        i + 1,  # id
        i,      # sid
        source_doc,  # sourcedoc
        text,   # originaltext
        clean_text,  # embedtext
        0,      # embedded (not embedded yet)
        "en",   # language
        metadata  # metadata
      )
      rows.append(row)
    
    return rows


class TestDataManager:
  """Manage temporary test data files and directories."""
  
  def __init__(self):
    self.temp_dirs = []
    self.temp_files = []
    
  def create_temp_dir(self, prefix: str = "customkb_test_") -> str:
    """Create a temporary directory for testing."""
    temp_dir = tempfile.mkdtemp(prefix=prefix)
    self.temp_dirs.append(temp_dir)
    return temp_dir
    
  def create_temp_config(self, config_content: str, suffix: str = ".cfg") -> str:
    """Create a temporary configuration file."""
    fd, temp_file = tempfile.mkstemp(suffix=suffix, text=True)
    self.temp_files.append(temp_file)
    
    with os.fdopen(fd, 'w') as f:
      f.write(config_content)
    
    return temp_file
    
  def create_temp_text_file(self, content: str, filename: str = "test.txt") -> str:
    """Create a temporary text file."""
    if not self.temp_dirs:
      self.create_temp_dir()
    
    temp_file = os.path.join(self.temp_dirs[0], filename)
    with open(temp_file, 'w') as f:
      f.write(content)
    
    self.temp_files.append(temp_file)
    return temp_file
    
  def cleanup(self):
    """Clean up all temporary files and directories."""
    import shutil
    
    for temp_file in self.temp_files:
      try:
        if os.path.exists(temp_file):
          os.unlink(temp_file)
      except OSError:
        pass
    
    for temp_dir in self.temp_dirs:
      try:
        if os.path.exists(temp_dir):
          shutil.rmtree(temp_dir)
      except OSError:
        pass
    
    self.temp_dirs.clear()
    self.temp_files.clear()


def create_mock_knowledge_base(config_file: str, temp_path: Path) -> 'KnowledgeBase':
  """
  Create a mock KnowledgeBase instance with test data.
  
  Args:
      config_file: Path to configuration file
      temp_path: Temporary directory path for database files
      
  Returns:
      Configured KnowledgeBase instance
  """
  import sqlite3
  from config.config_manager import KnowledgeBase
  
  # Create KnowledgeBase instance
  kb = KnowledgeBase(config_file)
  
  # Override paths to use temp directory
  kb.knowledge_base_db = str(temp_path / "test.db")
  kb.knowledge_base_vector = str(temp_path / "test.faiss")
  
  # Create mock database
  conn = sqlite3.connect(kb.knowledge_base_db)
  cursor = conn.cursor()
  
  # Create tables
  cursor.execute("""
    CREATE TABLE IF NOT EXISTS docs (
      id INTEGER PRIMARY KEY,
      sid INTEGER,
      sourcedoc TEXT,
      originaltext TEXT,
      embedtext TEXT,
      embedded INTEGER DEFAULT 0,
      language TEXT,
      metadata TEXT,
      bm25_tokens TEXT,
      doc_length INTEGER DEFAULT 0,
      keyphrase_processed INTEGER DEFAULT 0
    )
  """)
  
  # Insert mock data
  mock_gen = MockDataGenerator()
  texts = mock_gen.create_sample_texts()[:5]  # Use first 5 texts
  rows = mock_gen.create_database_rows(texts)
  
  for row in rows:
    cursor.execute("""
      INSERT INTO docs (id, sid, sourcedoc, originaltext, embedtext, 
                       embedded, language, metadata)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, row)
  
  conn.commit()
  conn.close()
  
  # Create mock FAISS index
  import numpy as np
  import faiss
  
  dimension = kb.vector_dimensions
  index = faiss.IndexFlatL2(dimension)
  
  # Add some random vectors
  vectors = np.random.normal(0, 0.1, (len(texts), dimension)).astype('float32')
  index.add(vectors)
  
  # Save index
  faiss.write_index(index, kb.knowledge_base_vector)
  
  # Set up mock SQL connection for query
  kb.sql_connection = sqlite3.connect(kb.knowledge_base_db)
  kb.sql_cursor = kb.sql_connection.cursor()
  
  return kb

#fin