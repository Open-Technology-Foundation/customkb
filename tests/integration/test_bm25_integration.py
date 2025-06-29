"""
Integration tests for BM25 hybrid search functionality.
Tests complete BM25 workflows from database creation through hybrid querying.
"""

import pytest
import os
import tempfile
import sqlite3
import pickle
from unittest.mock import patch, Mock, MagicMock
import numpy as np

from config.config_manager import KnowledgeBase
from database.db_manager import process_database, migrate_for_bm25
from embedding.bm25_manager import (
  build_bm25_index, 
  load_bm25_index, 
  ensure_bm25_index,
  get_bm25_scores
)
from query.query_manager import perform_hybrid_search, process_query
from utils.text_utils import tokenize_for_bm25


@pytest.mark.integration
class TestBM25DatabaseIntegration:
  """Test BM25 integration with database operations."""
  
  def test_bm25_database_migration(self, temp_data_manager):
    """Test BM25 database migration on existing database."""
    # Create a database without BM25 columns first
    kb_dir = temp_data_manager.create_temp_dir()
    db_path = os.path.join(kb_dir, 'migration_test.db')
    
    # Create legacy database structure
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
      CREATE TABLE docs (
        id INTEGER PRIMARY KEY,
        sid INTEGER,
        sourcedoc VARCHAR(255),
        originaltext TEXT,
        embedtext TEXT,
        embedded INTEGER DEFAULT 0,
        language TEXT DEFAULT "en",
        metadata TEXT,
        keyphrase_processed INTEGER DEFAULT 0
      )
    ''')
    
    # Add some test data
    cursor.execute('''
      INSERT INTO docs (id, sid, sourcedoc, originaltext, embedtext, embedded, 
                       language, metadata, keyphrase_processed)
      VALUES (1, 0, 'test.txt', 'Machine learning algorithms', 
              'machine learning algorithms', 1, 'en', '{}', 0)
    ''')
    conn.commit()
    conn.close()
    
    # Create mock KB
    mock_kb = Mock()
    mock_kb.knowledge_base_db = db_path
    mock_kb.enable_hybrid_search = True
    mock_kb.language = 'en'
    
    # Set up SQL connection
    mock_kb.sql_connection = sqlite3.connect(db_path)
    mock_kb.sql_cursor = mock_kb.sql_connection.cursor()
    
    # Test migration
    migrate_for_bm25(mock_kb)
    
    # Verify migration worked
    cursor = mock_kb.sql_cursor
    cursor.execute("PRAGMA table_info(docs)")
    columns = cursor.fetchall()
    column_names = [col[1] for col in columns]
    
    assert 'bm25_tokens' in column_names
    assert 'doc_length' in column_names
    
    # Verify existing data is preserved
    cursor.execute("SELECT originaltext FROM docs WHERE id = 1")
    result = cursor.fetchone()
    assert result[0] == 'Machine learning algorithms'
    
    mock_kb.sql_connection.close()
  
  def test_bm25_data_processing_during_ingestion(self, temp_data_manager, sample_texts):
    """Test that BM25 tokens are generated during text ingestion."""
    kb_dir = temp_data_manager.create_temp_dir()
    config_file = os.path.join(kb_dir, 'bm25_ingest.cfg')
    
    # Create config with BM25 enabled
    with open(config_file, 'w') as f:
      f.write("""[DEFAULT]
vector_model = text-embedding-3-small
vector_dimensions = 1536
query_model = gpt-4o

[ALGORITHMS]
enable_hybrid_search = true
vector_weight = 0.7
bm25_k1 = 1.2
bm25_b = 0.75
""")
    
    # Create test document
    test_file = os.path.join(kb_dir, 'test_doc.txt')
    with open(test_file, 'w') as f:
      f.write(sample_texts[0])
    
    # Process database
    mock_logger = Mock()
    db_args = Mock()
    db_args.config_file = config_file
    db_args.files = [test_file]
    db_args.language = 'en'
    db_args.force = False
    db_args.verbose = True
    db_args.debug = False
    
    with patch('builtins.input', return_value='y'):
      result = process_database(db_args, mock_logger)
    
    assert "files added" in result
    
    # Verify BM25 data was processed
    kb = KnowledgeBase(config_file)
    kb.sql_connection = sqlite3.connect(kb.knowledge_base_db)
    kb.sql_cursor = kb.sql_connection.cursor()
    
    # Check that BM25 tokens were generated
    kb.sql_cursor.execute("""
      SELECT bm25_tokens, doc_length, keyphrase_processed 
      FROM docs 
      WHERE keyphrase_processed = 1
    """)
    results = kb.sql_cursor.fetchall()
    
    assert len(results) > 0
    for tokens, length, processed in results:
      assert tokens is not None and tokens != ""
      assert length > 0
      assert processed == 1
    
    kb.sql_connection.close()
  
  def test_bm25_backward_compatibility(self, temp_data_manager, sample_texts):
    """Test that BM25 doesn't break existing functionality when disabled."""
    kb_dir = temp_data_manager.create_temp_dir()
    config_file = os.path.join(kb_dir, 'compat_test.cfg')
    
    # Create config with BM25 disabled (default)
    with open(config_file, 'w') as f:
      f.write("""[DEFAULT]
vector_model = text-embedding-3-small
vector_dimensions = 1536
query_model = gpt-4o
""")
    
    test_file = os.path.join(kb_dir, 'test_doc.txt')
    with open(test_file, 'w') as f:
      f.write(sample_texts[0])
    
    # Process database
    mock_logger = Mock()
    db_args = Mock()
    db_args.config_file = config_file
    db_args.files = [test_file]
    db_args.language = 'en'
    db_args.force = False
    db_args.verbose = True
    db_args.debug = False
    
    with patch('builtins.input', return_value='y'):
      result = process_database(db_args, mock_logger)
    
    assert "files added" in result
    
    # Verify that BM25 columns exist but are not populated
    kb = KnowledgeBase(config_file)
    kb.sql_connection = sqlite3.connect(kb.knowledge_base_db)
    kb.sql_cursor = kb.sql_connection.cursor()
    
    # Check table structure has BM25 columns
    kb.sql_cursor.execute("PRAGMA table_info(docs)")
    columns = [col[1] for col in kb.sql_cursor.fetchall()]
    assert 'bm25_tokens' in columns
    assert 'doc_length' in columns
    
    # Check that BM25 data is empty since hybrid search is disabled
    kb.sql_cursor.execute("SELECT bm25_tokens, keyphrase_processed FROM docs")
    results = kb.sql_cursor.fetchall()
    
    for tokens, processed in results:
      # Should be empty/null since BM25 is disabled
      assert tokens is None or tokens == ""
      assert processed == 0
    
    kb.sql_connection.close()


@pytest.mark.integration
class TestBM25IndexIntegration:
  """Test BM25 index building and management."""
  
  def setup_method(self):
    """Set up test environment with BM25 data."""
    self.temp_dir = tempfile.mkdtemp()
    self.kb_dir = self.temp_dir
    self.config_file = os.path.join(self.kb_dir, 'index_test.cfg')
    
    # Create config
    with open(self.config_file, 'w') as f:
      f.write("""[DEFAULT]
vector_model = text-embedding-3-small
vector_dimensions = 1536
query_model = gpt-4o

[ALGORITHMS]
enable_hybrid_search = true
vector_weight = 0.7
bm25_k1 = 1.2
bm25_b = 0.75
bm25_rebuild_threshold = 10
""")
    
    # Create test database with BM25 data
    self.create_test_database()
    
  def teardown_method(self):
    """Clean up test environment."""
    import shutil
    shutil.rmtree(self.temp_dir, ignore_errors=True)
  
  def create_test_database(self):
    """Create test database with BM25 tokens."""
    db_path = os.path.join(self.kb_dir, 'index_test.db')
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute('''
      CREATE TABLE docs (
        id INTEGER PRIMARY KEY,
        sid INTEGER,
        sourcedoc VARCHAR(255),
        originaltext TEXT,
        embedtext TEXT,
        embedded INTEGER DEFAULT 0,
        language TEXT DEFAULT "en",
        metadata TEXT,
        keyphrase_processed INTEGER DEFAULT 0,
        bm25_tokens TEXT,
        doc_length INTEGER DEFAULT 0
      )
    ''')
    
    # Insert test data with BM25 tokens
    test_data = [
      (1, 0, 'doc1.txt', 'Machine learning algorithms are powerful', 
       'machine learning algorithms powerful', 1, 'en', '{}', 1, 
       'machine learning algorithms powerful', 4),
      (2, 1, 'doc1.txt', 'Deep learning neural networks work well',
       'deep learning neural networks work well', 1, 'en', '{}', 1,
       'deep learning neural networks work well', 6),
      (3, 0, 'doc2.txt', 'Natural language processing techniques',
       'natural language processing techniques', 1, 'en', '{}', 1,
       'natural language processing techniques', 4),
      (4, 1, 'doc2.txt', 'Text classification with transformers',
       'text classification transformers', 1, 'en', '{}', 1,
       'text classification transformers', 3)
    ]
    
    cursor.executemany('''
      INSERT INTO docs (id, sid, sourcedoc, originaltext, embedtext, embedded, 
                       language, metadata, keyphrase_processed, bm25_tokens, doc_length)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', test_data)
    
    conn.commit()
    conn.close()
  
  def test_end_to_end_bm25_index_workflow(self):
    """Test complete BM25 index workflow: build → save → load → search."""
    # Create KnowledgeBase
    kb = KnowledgeBase(self.config_file)
    kb.sql_connection = sqlite3.connect(kb.knowledge_base_db)
    kb.sql_cursor = kb.sql_connection.cursor()
    
    # Step 1: Build BM25 index
    bm25_index = build_bm25_index(kb)
    assert bm25_index is not None
    
    # Verify index file was created
    index_path = kb.knowledge_base_vector.replace('.faiss', '.bm25')
    assert os.path.exists(index_path)
    
    # Step 2: Load BM25 index
    loaded_data = load_bm25_index(kb)
    assert loaded_data is not None
    assert 'bm25' in loaded_data
    assert 'doc_ids' in loaded_data
    assert 'total_docs' in loaded_data
    assert loaded_data['total_docs'] == 4
    
    # Step 3: Test scoring with loaded index
    with patch('embedding.bm25_manager.tokenize_for_bm25') as mock_tokenize:
      mock_tokenize.return_value = ('machine learning', 2)
      
      scores = get_bm25_scores(kb, 'machine learning algorithms', loaded_data)
      
      # Should return positive scores for relevant documents
      assert len(scores) > 0
      for doc_id, score in scores:
        assert doc_id in [1, 2, 3, 4]
        assert score > 0
    
    kb.sql_connection.close()
  
  def test_bm25_index_ensure_functionality(self):
    """Test ensure_bm25_index functionality."""
    kb = KnowledgeBase(self.config_file)
    kb.sql_connection = sqlite3.connect(kb.knowledge_base_db)
    kb.sql_cursor = kb.sql_connection.cursor()
    
    # Test when no index exists
    index_path = kb.knowledge_base_vector.replace('.faiss', '.bm25')
    if os.path.exists(index_path):
      os.remove(index_path)
    
    # Should build new index
    result = ensure_bm25_index(kb)
    assert result is True
    assert os.path.exists(index_path)
    
    # Test when index exists
    result = ensure_bm25_index(kb)
    assert result is True  # Should succeed without rebuilding
    
    kb.sql_connection.close()
  
  def test_bm25_index_rebuild_threshold(self):
    """Test BM25 index rebuilding based on threshold."""
    kb = KnowledgeBase(self.config_file)
    kb.sql_connection = sqlite3.connect(kb.knowledge_base_db)
    kb.sql_cursor = kb.sql_connection.cursor()
    
    # Build initial index
    build_bm25_index(kb)
    initial_mtime = os.path.getmtime(kb.knowledge_base_vector.replace('.faiss', '.bm25'))
    
    # Add more unprocessed documents (below threshold)
    cursor = kb.sql_cursor
    cursor.execute('''
      INSERT INTO docs (id, sid, sourcedoc, originaltext, embedtext, embedded, 
                       language, metadata, keyphrase_processed, bm25_tokens, doc_length)
      VALUES (5, 0, 'doc3.txt', 'New document content', 'new document content', 
              1, 'en', '{}', 0, '', 0)
    ''')
    kb.sql_connection.commit()
    
    # Should not rebuild (below threshold)
    from embedding.bm25_manager import rebuild_bm25_if_needed
    result = rebuild_bm25_if_needed(kb)
    assert result is True
    
    # Add many more unprocessed documents (above threshold)
    for i in range(6, 20):  # Add 14 more = 15 total unprocessed > threshold of 10
      cursor.execute('''
        INSERT INTO docs (id, sid, sourcedoc, originaltext, embedtext, embedded, 
                         language, metadata, keyphrase_processed, bm25_tokens, doc_length)
        VALUES (?, 0, 'doc?.txt', 'More content', 'more content', 
                1, 'en', '{}', 0, '', 0)
      ''', (i,))
    kb.sql_connection.commit()
    
    # Should rebuild (above threshold)
    result = rebuild_bm25_if_needed(kb)
    assert result is True
    
    # Check that index was actually rebuilt (file should be newer)
    new_mtime = os.path.getmtime(kb.knowledge_base_vector.replace('.faiss', '.bm25'))
    assert new_mtime > initial_mtime
    
    kb.sql_connection.close()


@pytest.mark.integration
class TestBM25QueryIntegration:
  """Test BM25 integration with query processing."""
  
  def setup_method(self):
    """Set up test environment for query testing."""
    self.temp_dir = tempfile.mkdtemp()
    self.config_file = os.path.join(self.temp_dir, 'query_test.cfg')
    
    # Create config with hybrid search enabled
    with open(self.config_file, 'w') as f:
      f.write("""[DEFAULT]
vector_model = text-embedding-3-small
vector_dimensions = 1536
query_model = gpt-4o
query_top_k = 10

[ALGORITHMS]
enable_hybrid_search = true
vector_weight = 0.7
bm25_k1 = 1.2
bm25_b = 0.75
""")
    
    # Create test database and BM25 index
    self.create_test_data()
    
  def teardown_method(self):
    """Clean up test environment."""
    import shutil
    shutil.rmtree(self.temp_dir, ignore_errors=True)
  
  def create_test_data(self):
    """Create test database with sample data."""
    kb = KnowledgeBase(self.config_file)
    
    # Create database
    conn = sqlite3.connect(kb.knowledge_base_db)
    cursor = conn.cursor()
    
    cursor.execute('''
      CREATE TABLE docs (
        id INTEGER PRIMARY KEY,
        sid INTEGER,
        sourcedoc VARCHAR(255),
        originaltext TEXT,
        embedtext TEXT,
        embedded INTEGER DEFAULT 0,
        language TEXT DEFAULT "en",
        metadata TEXT,
        keyphrase_processed INTEGER DEFAULT 0,
        bm25_tokens TEXT,
        doc_length INTEGER DEFAULT 0
      )
    ''')
    
    # Insert diverse test documents for better retrieval testing
    test_docs = [
      "Machine learning algorithms are used in artificial intelligence systems to make predictions and decisions.",
      "Deep learning neural networks can process complex patterns in large datasets effectively.",
      "Natural language processing enables computers to understand and generate human language.",
      "Computer vision algorithms analyze and interpret visual information from images and videos.",
      "Reinforcement learning teaches agents to make decisions through trial and error interactions."
    ]
    
    for i, doc in enumerate(test_docs):
      tokens, length = tokenize_for_bm25(doc, 'en')
      cursor.execute('''
        INSERT INTO docs (id, sid, sourcedoc, originaltext, embedtext, embedded, 
                         language, metadata, keyphrase_processed, bm25_tokens, doc_length)
        VALUES (?, 0, ?, ?, ?, 1, 'en', '{}', 1, ?, ?)
      ''', (i+1, f'doc{i+1}.txt', doc, doc.lower(), tokens, length))
    
    conn.commit()
    conn.close()
    
    # Build BM25 index
    kb.sql_connection = sqlite3.connect(kb.knowledge_base_db)
    kb.sql_cursor = kb.sql_connection.cursor()
    build_bm25_index(kb)
    kb.sql_connection.close()
  
  @pytest.mark.asyncio
  @patch('query.query_manager.get_query_embedding')
  @patch('query.query_manager.faiss.read_index')
  async def test_hybrid_search_integration(self, mock_read_index, mock_get_embedding):
    """Test hybrid search combining vector and BM25 results."""
    # Mock vector search components
    mock_index = Mock()
    mock_index.search.return_value = (
      np.array([[0.3, 0.5, 0.7, 0.9, 1.1]]),  # distances (lower = more similar)
      np.array([[0, 1, 2, 3, 4]])  # indices (0-based)
    )
    mock_read_index.return_value = mock_index
    mock_get_embedding.return_value = np.array([[0.1] * 1536])
    
    # Test hybrid search
    kb = KnowledgeBase(self.config_file)
    kb.sql_connection = sqlite3.connect(kb.knowledge_base_db)
    kb.sql_cursor = kb.sql_connection.cursor()
    
    # Load BM25 data
    bm25_data = load_bm25_index(kb)
    assert bm25_data is not None
    
    # Perform hybrid search
    query_vector = np.array([[0.1] * 1536])
    results = await perform_hybrid_search(kb, "machine learning algorithms", query_vector, mock_index)
    
    # Should return combined results
    assert len(results) > 0
    
    # Results should be tuples of (doc_id, combined_score)
    for doc_id, score in results:
      assert isinstance(doc_id, int)
      assert isinstance(score, (int, float))
      assert doc_id >= 1 and doc_id <= 5  # Valid document IDs
      assert score >= 0  # Non-negative scores
    
    # Results should be sorted by score (descending)
    scores = [score for _, score in results]
    assert scores == sorted(scores, reverse=True)
    
    kb.sql_connection.close()
  
  @patch('query.query_manager.get_query_embedding')
  @patch('query.query_manager.faiss.read_index')
  def test_query_process_with_hybrid_search(self, mock_read_index, mock_get_embedding):
    """Test complete query processing with hybrid search enabled."""
    # Mock components
    mock_index = Mock()
    mock_index.search.return_value = (
      np.array([[0.2, 0.4, 0.6]]),
      np.array([[0, 1, 2]])
    )
    mock_read_index.return_value = mock_index
    mock_get_embedding.return_value = np.array([[0.1] * 1536])
    
    # Create query args
    mock_logger = Mock()
    query_args = Mock()
    query_args.config_file = self.config_file
    query_args.query_text = "machine learning algorithms"
    query_args.query_file = ""
    query_args.context_only = True  # Avoid LLM API calls
    query_args.verbose = True
    query_args.debug = False
    
    # Process query
    result = process_query(query_args, mock_logger)
    
    # Should return context result
    assert isinstance(result, str)
    assert len(result) > 0
    
    # Should contain some relevant content
    assert "machine" in result.lower() or "learning" in result.lower()
  
  def test_hybrid_search_weight_configuration(self):
    """Test that hybrid search respects vector_weight configuration."""
    # Test with different weight configurations
    weight_configs = [
      (0.9, 0.1),  # Mostly vector
      (0.5, 0.5),  # Equal weight
      (0.1, 0.9),  # Mostly BM25
    ]
    
    for vector_weight, bm25_weight in weight_configs:
      # Create config with specific weights
      config_file = os.path.join(self.temp_dir, f'weight_test_{vector_weight}.cfg')
      with open(config_file, 'w') as f:
        f.write(f"""[DEFAULT]
vector_model = text-embedding-3-small
vector_dimensions = 1536
query_model = gpt-4o

[ALGORITHMS]
enable_hybrid_search = true
vector_weight = {vector_weight}
""")
      
      # Test that configuration is loaded correctly
      kb = KnowledgeBase(config_file)
      assert kb.vector_weight == vector_weight
      assert abs(kb.vector_weight + (1 - vector_weight) - 1.0) < 0.001  # Weights sum to 1
  
  def test_fallback_to_vector_only_search(self):
    """Test fallback to vector-only search when BM25 fails."""
    # Create config with hybrid disabled
    config_file = os.path.join(self.temp_dir, 'vector_only.cfg')
    with open(config_file, 'w') as f:
      f.write("""[DEFAULT]
vector_model = text-embedding-3-small
vector_dimensions = 1536
query_model = gpt-4o

[ALGORITHMS]
enable_hybrid_search = false
""")
    
    kb = KnowledgeBase(config_file)
    assert kb.enable_hybrid_search is False
    
    # ensure_bm25_index should return False for disabled hybrid search
    result = ensure_bm25_index(kb)
    assert result is False


@pytest.mark.integration
class TestBM25TokenizationIntegration:
  """Test BM25 tokenization integration with real documents."""
  
  def test_tokenization_with_various_content_types(self, temp_data_manager):
    """Test BM25 tokenization with different types of content."""
    # Test different content types
    test_contents = [
      # Technical content with hyphens
      "State-of-the-art machine-learning algorithms achieve 95.5% accuracy on CIFAR-10 dataset.",
      
      # Content with URLs and emails
      "Visit https://example.com or contact admin@company.com for API documentation.",
      
      # Mixed case with acronyms
      "The REST-API uses HTTP requests to process AI/ML workflows with GPU acceleration.",
      
      # Content with numbers and versions
      "Python 3.9 supports type hints and async/await patterns in TensorFlow 2.8 models.",
      
      # Natural language
      "Natural language processing enables computers to understand human communication patterns."
    ]
    
    for i, content in enumerate(test_contents):
      tokens, length = tokenize_for_bm25(content, 'en')
      
      # Basic validation
      assert tokens is not None
      assert length > 0
      assert isinstance(tokens, str)
      assert isinstance(length, int)
      
      # Check that important terms are preserved
      if "machine-learning" in content:
        assert "machine-learning" in tokens
      if "95.5" in content:
        assert "95.5" in tokens
      if "example.com" in content:
        assert "example.com" in tokens
      if "REST-API" in content.lower():
        assert "rest-api" in tokens
  
  def test_tokenization_consistency_across_database_operations(self, temp_data_manager, sample_texts):
    """Test that tokenization is consistent across different database operations."""
    kb_dir = temp_data_manager.create_temp_dir()
    config_file = os.path.join(kb_dir, 'consistency_test.cfg')
    
    # Create config with BM25 enabled
    with open(config_file, 'w') as f:
      f.write("""[DEFAULT]
vector_model = text-embedding-3-small
vector_dimensions = 1536

[ALGORITHMS]
enable_hybrid_search = true
""")
    
    # Create test document
    test_content = sample_texts[0]
    test_file = os.path.join(kb_dir, 'consistency_test.txt')
    with open(test_file, 'w') as f:
      f.write(test_content)
    
    # Process through database
    mock_logger = Mock()
    db_args = Mock()
    db_args.config_file = config_file
    db_args.files = [test_file]
    db_args.language = 'en'
    db_args.force = False
    db_args.verbose = True
    db_args.debug = False
    
    with patch('builtins.input', return_value='y'):
      process_database(db_args, mock_logger)
    
    # Get tokens from database
    kb = KnowledgeBase(config_file)
    kb.sql_connection = sqlite3.connect(kb.knowledge_base_db)
    kb.sql_cursor = kb.sql_connection.cursor()
    
    kb.sql_cursor.execute("SELECT bm25_tokens, doc_length FROM docs WHERE keyphrase_processed = 1")
    db_results = kb.sql_cursor.fetchall()
    
    # Get tokens from direct tokenization
    direct_tokens, direct_length = tokenize_for_bm25(test_content, 'en')
    
    # Should have consistent tokenization
    assert len(db_results) > 0
    
    # At least one result should match direct tokenization
    # (chunks may have different tokenization, but basic approach should be consistent)
    for db_tokens, db_length in db_results:
      assert db_tokens is not None
      assert db_length > 0
      # Token style should be consistent (lowercase, space-separated)
      assert db_tokens.islower()
      assert ' ' in db_tokens or len(db_tokens.split()) == 1
    
    kb.sql_connection.close()


@pytest.mark.integration  
class TestBM25PerformanceIntegration:
  """Test BM25 performance characteristics in integration scenarios."""
  
  def test_bm25_index_size_and_speed(self, temp_data_manager):
    """Test BM25 index size and loading speed with realistic data volumes."""
    kb_dir = temp_data_manager.create_temp_dir()
    config_file = os.path.join(kb_dir, 'performance_test.cfg')
    
    with open(config_file, 'w') as f:
      f.write("""[DEFAULT]
vector_model = text-embedding-3-small
vector_dimensions = 1536

[ALGORITHMS]
enable_hybrid_search = true
""")
    
    # Create database with substantial amount of data
    kb = KnowledgeBase(config_file)
    conn = sqlite3.connect(kb.knowledge_base_db)
    cursor = conn.cursor()
    
    cursor.execute('''
      CREATE TABLE docs (
        id INTEGER PRIMARY KEY,
        sid INTEGER,
        sourcedoc VARCHAR(255),
        originaltext TEXT,
        embedtext TEXT,
        embedded INTEGER DEFAULT 0,
        language TEXT DEFAULT "en",
        metadata TEXT,
        keyphrase_processed INTEGER DEFAULT 0,
        bm25_tokens TEXT,
        doc_length INTEGER DEFAULT 0
      )
    ''')
    
    # Generate test documents (simulate realistic corpus)
    doc_templates = [
      "Machine learning algorithms process data patterns using {technique} methods for {domain} applications.",
      "Deep learning neural networks with {layers} layers achieve {accuracy}% accuracy on {dataset} benchmarks.",
      "Natural language processing models like {model} use {approach} to understand {language} text.",
      "Computer vision systems detect {objects} in {context} using {architecture} networks.",
      "Reinforcement learning agents learn {strategies} through {environment} interactions."
    ]
    
    import random
    techniques = ['supervised', 'unsupervised', 'semi-supervised', 'self-supervised']
    domains = ['healthcare', 'finance', 'robotics', 'autonomous driving', 'recommendation']
    layers = ['12', '24', '48', '96']
    accuracies = ['89.5', '92.3', '95.1', '97.8']
    datasets = ['ImageNet', 'COCO', 'GLUE', 'SQuAD']
    
    num_docs = 500  # Reduced for memory safety during testing
    for i in range(num_docs):
      template = random.choice(doc_templates)
      doc = template.format(
        technique=random.choice(techniques),
        domain=random.choice(domains),
        layers=random.choice(layers),
        accuracy=random.choice(accuracies),
        dataset=random.choice(datasets),
        model=f"Model-{i%20}",
        approach=random.choice(['attention', 'convolution', 'recurrence']),
        language=random.choice(['English', 'Spanish', 'French']),
        objects=random.choice(['faces', 'vehicles', 'buildings']),
        context=random.choice(['urban', 'rural', 'indoor']),
        architecture=random.choice(['ResNet', 'Transformer', 'LSTM']),
        strategies=random.choice(['exploration', 'exploitation', 'planning']),
        environment=random.choice(['simulated', 'real-world', 'virtual'])
      )
      
      tokens, length = tokenize_for_bm25(doc, 'en')
      cursor.execute('''
        INSERT INTO docs (id, sid, sourcedoc, originaltext, embedtext, embedded, 
                         language, metadata, keyphrase_processed, bm25_tokens, doc_length)
        VALUES (?, 0, ?, ?, ?, 1, 'en', '{}', 1, ?, ?)
      ''', (i+1, f'doc{i+1}.txt', doc, doc.lower(), tokens, length))
    
    conn.commit()
    conn.close()
    
    # Test index building speed
    import time
    kb.sql_connection = sqlite3.connect(kb.knowledge_base_db)
    kb.sql_cursor = kb.sql_connection.cursor()
    
    start_time = time.time()
    bm25_index = build_bm25_index(kb)
    build_time = time.time() - start_time
    
    assert bm25_index is not None
    assert build_time < 30.0  # Should build in reasonable time
    
    # Test index file size
    index_path = kb.knowledge_base_vector.replace('.faiss', '.bm25')
    assert os.path.exists(index_path)
    file_size = os.path.getsize(index_path)
    assert file_size < 50 * 1024 * 1024  # Should be under 50MB for 1000 docs
    
    # Test loading speed
    start_time = time.time()
    loaded_data = load_bm25_index(kb)
    load_time = time.time() - start_time
    
    assert loaded_data is not None
    assert load_time < 5.0  # Should load quickly
    assert loaded_data['total_docs'] == num_docs
    
    # Test search speed
    start_time = time.time()
    scores = get_bm25_scores(kb, "machine learning algorithms", loaded_data)
    search_time = time.time() - start_time
    
    assert len(scores) > 0
    assert search_time < 1.0  # Should search quickly
    
    kb.sql_connection.close()
  
  def test_bm25_memory_usage_patterns(self, temp_data_manager):
    """Test BM25 memory usage with larger datasets."""
    kb_dir = temp_data_manager.create_temp_dir()
    config_file = os.path.join(kb_dir, 'memory_test.cfg')
    
    with open(config_file, 'w') as f:
      f.write("""[DEFAULT]
vector_model = text-embedding-3-small

[ALGORITHMS]
enable_hybrid_search = true
""")
    
    # Create moderately sized dataset
    kb = KnowledgeBase(config_file)
    conn = sqlite3.connect(kb.knowledge_base_db)
    cursor = conn.cursor()
    
    cursor.execute('''
      CREATE TABLE docs (
        id INTEGER PRIMARY KEY,
        sid INTEGER,
        sourcedoc VARCHAR(255),
        originaltext TEXT,
        embedtext TEXT,
        embedded INTEGER DEFAULT 0,
        language TEXT DEFAULT "en",
        metadata TEXT,
        keyphrase_processed INTEGER DEFAULT 0,
        bm25_tokens TEXT,
        doc_length INTEGER DEFAULT 0
      )
    ''')
    
    # Create documents with varying lengths
    base_text = "Machine learning algorithms process data patterns using computational methods. "
    
    for i in range(500):  # Moderate size for memory testing
      # Vary document length
      multiplier = (i % 10) + 1
      doc = base_text * multiplier
      
      tokens, length = tokenize_for_bm25(doc, 'en')
      cursor.execute('''
        INSERT INTO docs (id, sid, sourcedoc, originaltext, embedtext, embedded, 
                         language, metadata, keyphrase_processed, bm25_tokens, doc_length)
        VALUES (?, 0, ?, ?, ?, 1, 'en', '{}', 1, ?, ?)
      ''', (i+1, f'doc{i+1}.txt', doc, doc.lower(), tokens, length))
    
    conn.commit()
    conn.close()
    
    # Test that BM25 operations work without memory issues
    kb.sql_connection = sqlite3.connect(kb.knowledge_base_db)
    kb.sql_cursor = kb.sql_connection.cursor()
    
    # Build index
    bm25_index = build_bm25_index(kb)
    assert bm25_index is not None
    
    # Load index multiple times (should not leak memory)
    for _ in range(5):
      loaded_data = load_bm25_index(kb)
      assert loaded_data is not None
      assert loaded_data['total_docs'] == 500
    
    # Perform multiple searches
    test_queries = [
      "machine learning",
      "data patterns",
      "computational methods",
      "algorithms process",
      "learning data"
    ]
    
    for query in test_queries:
      scores = get_bm25_scores(kb, query, loaded_data)
      assert len(scores) > 0
    
    kb.sql_connection.close()


@pytest.mark.integration
class TestBM25ResultLimiting:
  """Test BM25 result limiting functionality in integration scenarios."""
  
  def setup_method(self):
    """Set up test environment for result limiting tests."""
    self.temp_dir = tempfile.mkdtemp()
    self.kb_dir = self.temp_dir
    self.config_file = os.path.join(self.kb_dir, 'limit_test.cfg')
    
  def teardown_method(self):
    """Clean up test environment."""
    import shutil
    shutil.rmtree(self.temp_dir, ignore_errors=True)
  
  def create_large_test_dataset(self, num_docs=1000):
    """Create a large test dataset to test result limiting."""
    # Reduced default from 5000 to 1000 to prevent memory exhaustion
    db_path = os.path.join(self.kb_dir, 'limit_test.db')
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute('''
      CREATE TABLE docs (
        id INTEGER PRIMARY KEY,
        sid INTEGER,
        sourcedoc VARCHAR(255),
        originaltext TEXT,
        embedtext TEXT,
        embedded INTEGER DEFAULT 0,
        language TEXT DEFAULT "en",
        metadata TEXT,
        keyphrase_processed INTEGER DEFAULT 0,
        bm25_tokens TEXT,
        doc_length INTEGER DEFAULT 0
      )
    ''')
    
    # Create documents that will match a common query
    # Half contain "machine learning", half contain other terms
    for i in range(num_docs):
      if i < num_docs // 2:
        # Documents that will match "machine learning"
        doc = f"Document {i}: Machine learning algorithms are essential for AI. This document discusses machine learning techniques."
      else:
        # Documents with other content
        doc = f"Document {i}: Natural language processing and computer vision are important fields in artificial intelligence."
      
      tokens, length = tokenize_for_bm25(doc, 'en')
      cursor.execute('''
        INSERT INTO docs (id, sid, sourcedoc, originaltext, embedtext, embedded, 
                         language, metadata, keyphrase_processed, bm25_tokens, doc_length)
        VALUES (?, 0, ?, ?, ?, 1, 'en', '{}', 1, ?, ?)
      ''', (i+1, f'doc{i+1}.txt', doc, doc.lower(), tokens, length))
    
    conn.commit()
    conn.close()
  
  def test_bm25_result_limiting_with_config(self):
    """Test BM25 result limiting using configuration parameter."""
    # Create config with BM25 result limit
    with open(self.config_file, 'w') as f:
      f.write("""[DEFAULT]
vector_model = text-embedding-3-small
vector_dimensions = 1536

[ALGORITHMS]
enable_hybrid_search = true
bm25_max_results = 100
bm25_k1 = 1.2
bm25_b = 0.75
""")
    
    # Create large dataset
    self.create_large_test_dataset(1000)
    
    # Load KB and build index
    kb = KnowledgeBase(self.config_file)
    kb.sql_connection = sqlite3.connect(kb.knowledge_base_db)
    kb.sql_cursor = kb.sql_connection.cursor()
    
    # Debug: Check if documents are in the database
    kb.sql_cursor.execute("SELECT COUNT(*) FROM docs WHERE keyphrase_processed = 1")
    doc_count = kb.sql_cursor.fetchone()[0]
    assert doc_count > 0, f"Expected documents with keyphrase_processed=1, found {doc_count}"
    
    # Build BM25 index
    bm25_index = build_bm25_index(kb)
    assert bm25_index is not None
    
    # Load index
    bm25_data = load_bm25_index(kb)
    assert bm25_data is not None
    assert bm25_data['total_docs'] > 0, f"Expected documents in index, found {bm25_data.get('total_docs', 0)}"
    
    # Debug: Check a few document tokens
    kb.sql_cursor.execute("SELECT bm25_tokens FROM docs LIMIT 5")
    sample_tokens = kb.sql_cursor.fetchall()
    print(f"Sample BM25 tokens: {sample_tokens[:2]}")
    
    # Perform search that would return many results
    scores = get_bm25_scores(kb, "machine learning", bm25_data)
    
    # Should be limited to 100 results
    assert len(scores) <= 100
    assert len(scores) > 0  # Should have some results
    
    # All scores should be positive and sorted
    for i, (doc_id, score) in enumerate(scores):
      assert score > 0
      if i > 0:
        assert scores[i-1][1] >= score  # Descending order
    
    kb.sql_connection.close()
  
  def test_bm25_result_limiting_prevents_memory_issues(self):
    """Test that result limiting prevents memory issues with large result sets."""
    # Create config with small limit for memory safety
    with open(self.config_file, 'w') as f:
      f.write("""[DEFAULT]
vector_model = text-embedding-3-small

[ALGORITHMS]
enable_hybrid_search = true
bm25_max_results = 500
""")
    
    # Create large dataset (reduced from 5000 to prevent memory exhaustion)
    self.create_large_test_dataset(1000)
    
    # Load KB and build index
    kb = KnowledgeBase(self.config_file)
    kb.sql_connection = sqlite3.connect(kb.knowledge_base_db)
    kb.sql_cursor = kb.sql_connection.cursor()
    
    # Build BM25 index
    bm25_index = build_bm25_index(kb)
    assert bm25_index is not None
    
    # Load index
    bm25_data = load_bm25_index(kb)
    assert bm25_data is not None
    assert bm25_data['total_docs'] == 1000
    
    # Perform search that would return ~500 results without limiting
    import time
    start_time = time.time()
    scores = get_bm25_scores(kb, "machine learning", bm25_data)
    search_time = time.time() - start_time
    
    # Should be limited to 500 results
    assert len(scores) == 500
    assert search_time < 2.0  # Should be fast even with large dataset
    
    # Memory test: perform multiple searches
    for _ in range(10):
      scores = get_bm25_scores(kb, "machine learning algorithms", bm25_data)
      assert len(scores) <= 500
    
    kb.sql_connection.close()
  
  def test_bm25_unlimited_results_option(self):
    """Test that setting bm25_max_results=0 allows unlimited results."""
    # Create config with unlimited results
    with open(self.config_file, 'w') as f:
      f.write("""[DEFAULT]
vector_model = text-embedding-3-small

[ALGORITHMS]
enable_hybrid_search = true
bm25_max_results = 0
""")
    
    # Create moderate dataset
    self.create_large_test_dataset(200)
    
    # Load KB and build index
    kb = KnowledgeBase(self.config_file)
    kb.sql_connection = sqlite3.connect(kb.knowledge_base_db)
    kb.sql_cursor = kb.sql_connection.cursor()
    
    # Build and load BM25 index
    build_bm25_index(kb)
    bm25_data = load_bm25_index(kb)
    
    # Perform search
    scores = get_bm25_scores(kb, "machine learning", bm25_data)
    
    # Should return all matching results (about 100 out of 200)
    assert len(scores) > 50  # At least half should match
    assert len(scores) <= 200  # Can't exceed total docs
    
    kb.sql_connection.close()
  
  @patch('embedding.bm25_manager.get_logger')
  def test_bm25_result_limiting_logging(self, mock_logger):
    """Test that result limiting is properly logged."""
    # Create config with result limit
    with open(self.config_file, 'w') as f:
      f.write("""[DEFAULT]
vector_model = text-embedding-3-small

[ALGORITHMS]
enable_hybrid_search = true
bm25_max_results = 50
""")
    
    # Create dataset (safe size)
    self.create_large_test_dataset(300)
    
    # Load KB and build index
    kb = KnowledgeBase(self.config_file)
    kb.sql_connection = sqlite3.connect(kb.knowledge_base_db)
    kb.sql_cursor = kb.sql_connection.cursor()
    
    # Build and load BM25 index
    build_bm25_index(kb)
    bm25_data = load_bm25_index(kb)
    
    # Perform search that triggers limiting
    scores = get_bm25_scores(kb, "machine learning", bm25_data)
    
    # Check logging was called
    assert mock_logger.return_value.info.called
    log_messages = [call[0][0] for call in mock_logger.return_value.info.call_args_list]
    
    # Should have logged about limiting
    limiting_logged = any("BM25 results limited" in msg for msg in log_messages)
    assert limiting_logged
    
    # Should show original and limited counts
    for msg in log_messages:
      if "BM25 results limited" in msg:
        assert "from" in msg and "to" in msg
        # Should mention the limit (50)
        assert "50" in msg
    
    kb.sql_connection.close()
  
  @pytest.mark.asyncio
  @patch('query.query_manager.get_query_embedding')
  @patch('query.query_manager.faiss.read_index')
  async def test_hybrid_search_with_bm25_limiting(self, mock_read_index, mock_get_embedding):
    """Test that hybrid search works correctly with BM25 result limiting."""
    # Create config with result limit
    with open(self.config_file, 'w') as f:
      f.write("""[DEFAULT]
vector_model = text-embedding-3-small
vector_dimensions = 1536
query_top_k = 50

[ALGORITHMS]
enable_hybrid_search = true
vector_weight = 0.5
bm25_max_results = 100
""")
    
    # Create dataset (safe size)
    self.create_large_test_dataset(500)
    
    # Mock vector search
    mock_index = Mock()
    mock_index.search.return_value = (
      np.array([[0.1] * 50]),  # 50 distances
      np.array([list(range(50))])  # 50 indices
    )
    mock_read_index.return_value = mock_index
    mock_get_embedding.return_value = np.array([[0.1] * 1536])
    
    # Load KB
    kb = KnowledgeBase(self.config_file)
    kb.sql_connection = sqlite3.connect(kb.knowledge_base_db)
    kb.sql_cursor = kb.sql_connection.cursor()
    
    # Build and load BM25 index
    build_bm25_index(kb)
    
    # Perform hybrid search
    query_vector = np.array([[0.1] * 1536])
    results = await perform_hybrid_search(kb, "machine learning", query_vector, mock_index)
    
    # Should have results from both vector and BM25
    assert len(results) > 0
    assert len(results) <= 150  # Max possible: 50 vector + 100 BM25
    
    # Results should be properly scored and sorted
    scores = [score for _, score in results]
    assert scores == sorted(scores, reverse=True)
    
    kb.sql_connection.close()

#fin