# CustomKB Development Guide

This guide provides comprehensive information for developers working on or contributing to CustomKB.

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Development Environment Setup](#development-environment-setup)
- [Project Structure](#project-structure)
- [Core Components](#core-components)
- [Building and Testing](#building-and-testing)
- [Coding Standards](#coding-standards)
- [API Reference](#api-reference)
- [Contributing Workflow](#contributing-workflow)
- [Performance Considerations](#performance-considerations)
- [Security Guidelines](#security-guidelines)
- [Troubleshooting](#troubleshooting)

## Architecture Overview

CustomKB implements a modular, three-tier architecture designed for scalability and maintainability:

```
┌─────────────────────────────────────────────────────────────┐
│                     CLI Interface (customkb.py)              │
├─────────────────────────────────────────────────────────────┤
│                      Core Components                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │   Config     │  │   Database   │  │    Embedding     │   │
│  │   Manager    │  │   Manager    │  │    Manager       │   │
│  └──────────────┘  └──────────────┘  └──────────────────┘   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │    Query     │  │    Model     │  │     Utils        │   │
│  │   Manager    │  │   Manager    │  │   (Logging,      │   │
│  └──────────────┘  └──────────────┘  │    Security,     │   │
│                                       │    Text)         │   │
│                                       └──────────────────┘   │
├─────────────────────────────────────────────────────────────┤
│                      Storage Layer                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │   SQLite     │  │    FAISS     │  │   File System    │   │
│  │  Database    │  │   Indices    │  │   (Config,Logs)  │   │
│  └──────────────┘  └──────────────┘  └──────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### Key Design Principles

1. **Separation of Concerns**: Each component has a single, well-defined responsibility
2. **Configuration-Driven**: Behavior customizable through config files and environment variables
3. **Security by Design**: Input validation, path sanitization, and API key protection
4. **Performance Optimized**: Batch processing, caching, and concurrent operations
5. **Fault Tolerant**: Checkpoint saving, retry logic, and graceful error handling

## Development Environment Setup

### Prerequisites

- Python 3.12 or higher
- Git 2.25+
- SQLite 3.45+
- 8GB RAM minimum (16GB recommended for large datasets)
- Ubuntu 24.04 LTS or compatible Linux distribution

### Initial Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/Open-Technology-Foundation/customkb.git
   cd customkb
   ```

2. **Create virtual environment**
   ```bash
   python3.12 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   # Core dependencies
   pip install -r requirements.txt
   
   # Development dependencies
   pip install -r requirements-test.txt
   ```

4. **Set up environment variables**
   ```bash
   # Create .env file (not tracked by git)
   cat > .env << EOF
   export OPENAI_API_KEY="your-api-key"
   export ANTHROPIC_API_KEY="your-api-key"
   export NLTK_DATA="${HOME}/nltk_data"
   export VECTORDBS="/var/lib/vectordbs"
   export PYTHONPATH="${PWD}:${PYTHONPATH}"
   EOF
   
   # Load environment
   source .env
   ```

5. **Initialize NLTK data**
   ```bash
   python -c "import nltk; nltk.download('all', download_dir='${NLTK_DATA}')"
   ```

6. **Install spaCy language model**
   ```bash
   python -m spacy download en_core_web_sm
   ```

### IDE Configuration

#### VS Code
```json
{
  "python.defaultInterpreterPath": ".venv/bin/python",
  "python.linting.enabled": true,
  "python.linting.flake8Enabled": true,
  "python.linting.mypyEnabled": true,
  "python.formatting.provider": "black",
  "editor.rulers": [100],
  "files.exclude": {
    "**/__pycache__": true,
    "**/*.pyc": true,
    ".venv": true
  }
}
```

#### PyCharm
- Set Project Interpreter to `.venv/bin/python`
- Enable flake8 and mypy inspections
- Set line length to 100 characters
- Configure Python indentation to 2 spaces

## Project Structure

```
customkb/
├── customkb.py              # Main CLI entry point
├── customkb                 # Executable script (symlink)
├── version.py               # Version management
├── version.sh               # Version update script
├── Models.json              # Model registry and aliases
│
├── config/                  # Configuration management
│   ├── __init__.py
│   └── config_manager.py    # KnowledgeBase class and config loading
│
├── database/                # Database operations
│   ├── __init__.py
│   └── db_manager.py        # Text processing and storage
│
├── embedding/               # Embedding generation
│   ├── __init__.py
│   └── embed_manager.py     # Vector embedding creation
│
├── query/                   # Query processing
│   ├── __init__.py
│   └── query_manager.py     # Search and response generation
│
├── models/                  # Model management
│   ├── __init__.py
│   └── model_manager.py     # Model registry and resolution
│
├── utils/                   # Utility functions
│   ├── __init__.py
│   ├── logging_utils.py     # Logging configuration
│   ├── security_utils.py    # Security validations
│   └── text_utils.py        # Text processing utilities
│
├── tests/                   # Test suite
│   ├── __init__.py
│   ├── conftest.py          # Pytest configuration
│   ├── unit/                # Unit tests
│   ├── integration/         # Integration tests
│   └── fixtures/            # Test data and mocks
│
├── docs/                    # Documentation (if needed)
├── scripts/                 # Utility scripts
└── examples/                # Example configurations
```

## Core Components

### Configuration Manager (`config/config_manager.py`)

Manages all configuration aspects of CustomKB:

```python
from config.config_manager import KnowledgeBase, get_fq_cfg_filename

# Load configuration
kb = KnowledgeBase('myproject.cfg')

# Access configuration values
print(kb.vector_model)        # 'text-embedding-3-small'
print(kb.knowledge_base_db)   # '/var/lib/vectordbs/myproject.db'

# Configuration hierarchy:
# 1. Environment variables (highest priority)
# 2. Config file values
# 3. Default values (lowest priority)
```

Key features:
- Domain-style naming support (e.g., 'example.com.cfg')
- Security validation for paths
- Type conversion and validation
- 40+ configurable parameters across 5 sections

### Database Manager (`database/db_manager.py`)

Handles all text processing and storage operations:

```python
from database.db_manager import process_database

# Process files into knowledge base
args = argparse.Namespace(
    config_file='myproject.cfg',
    files=['doc1.txt', 'doc2.md'],
    language='english',
    verbose=True
)
result = process_database(args, logger)
```

Key features:
- Multi-format support (Markdown, HTML, code, text)
- Intelligent chunking with overlap
- Entity preservation during cleaning
- Metadata extraction (headings, sections)
- Batch processing with transactions

### Embedding Manager (`embedding/embed_manager.py`)

Generates and manages vector embeddings:

```python
from embedding.embed_manager import process_embeddings

# Generate embeddings for all chunks
args = argparse.Namespace(
    config_file='myproject.cfg',
    reset_database=False,
    verbose=True
)
result = process_embeddings(args, logger)
```

Key features:
- Two-tier caching (memory + disk)
- Adaptive FAISS index selection
- Checkpoint saving for resilience
- Concurrent API calls with rate limiting
- Deduplication of embeddings

### Query Manager (`query/query_manager.py`)

Handles search queries and response generation:

```python
from query.query_manager import process_query

# Query the knowledge base
args = argparse.Namespace(
    config_file='myproject.cfg',
    query_text='How to configure embedding models?',
    context_only=False,
    model='gpt-4o-mini'
)
response = process_query(args, logger)
```

Key features:
- Semantic similarity search
- Context-aware retrieval
- Multi-model support (OpenAI, Anthropic, Llama)
- XML-formatted reference contexts
- Query caching with TTL

### Model Manager (`models/model_manager.py`)

Provides model registry and resolution:

```python
from models.model_manager import get_canonical_model, get_models_by_category

# Resolve model aliases
model_info = get_canonical_model('claude')
# Returns: {'model': 'claude-3-5-sonnet-20241022', 'category': 'llm', ...}

# Get models by category
llm_models = get_models_by_category('llm')
embed_models = get_models_by_category('embed')
```

## Building and Testing

### Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=. --cov-report=html tests/

# Run specific test categories
pytest tests/unit/
pytest tests/integration/

# Run with verbose output
pytest -v tests/

# Run specific test
pytest tests/unit/test_config_manager.py::test_domain_style_names
```

### Code Quality Checks

```bash
# Linting
flake8 . --max-line-length=100 --extend-ignore=E203,W503

# Type checking
mypy --ignore-missing-imports .

# Security scan
bandit -r . -x .venv,tests

# Code formatting (check only)
black --check --line-length=100 .

# All checks combined
make lint  # If Makefile is available
```

### Performance Testing

```bash
# Run performance benchmarks
pytest tests/performance/ -v

# Profile a specific operation
python -m cProfile -o profile.stats customkb.py query test.cfg "test query"
python -m pstats profile.stats
```

### Building Documentation

```bash
# Generate API documentation
pdoc --html --output-dir docs --force \
  config database embedding query models utils

# Build user documentation (if using Sphinx)
cd docs && make html
```

## Coding Standards

### Python Style Guide

1. **Indentation**: 2 spaces (not tabs)
2. **Line Length**: Maximum 100 characters
3. **Imports**: 
   ```python
   # Standard library
   import os
   import sys
   from typing import List, Optional
   
   # Third-party
   import numpy as np
   from langchain.text_splitter import RecursiveCharacterTextSplitter
   
   # Local imports
   from config.config_manager import KnowledgeBase
   from utils.logging_utils import get_logger
   ```

4. **Docstrings**: Google style with type hints
   ```python
   def process_text(text: str, max_length: int = 100) -> List[str]:
       """
       Process text into chunks of specified maximum length.
       
       Args:
           text: Input text to process.
           max_length: Maximum length of each chunk.
           
       Returns:
           List of text chunks.
           
       Raises:
           ValueError: If text is empty or max_length <= 0.
       """
   ```

5. **Error Handling**:
   ```python
   try:
       result = risky_operation()
   except SpecificException as e:
       logger.error(f"Operation failed: {e}")
       raise CustomException(f"Failed to process: {e}") from e
   ```

### Security Guidelines

1. **Input Validation**: Always validate user input
   ```python
   from utils.security_utils import validate_file_path, validate_safe_path
   
   # Validate file paths
   safe_path = validate_file_path(user_input, allowed_extensions=['.txt', '.md'])
   ```

2. **API Key Handling**: Never log or expose API keys
   ```python
   # Good
   logger.debug("Using API key: ***")
   
   # Bad
   logger.debug(f"Using API key: {api_key}")
   ```

3. **SQL Injection Prevention**: Use parameterized queries
   ```python
   # Good
   cursor.execute("SELECT * FROM chunks WHERE id = ?", (chunk_id,))
   
   # Bad
   cursor.execute(f"SELECT * FROM chunks WHERE id = {chunk_id}")
   ```

### Git Workflow

1. **Branch Naming**:
   - Features: `feature/add-embedding-cache`
   - Bugs: `fix/query-timeout-issue`
   - Improvements: `improve/database-performance`

2. **Commit Messages**:
   ```
   <type>: <subject>
   
   <body>
   
   <footer>
   ```
   
   Types: feat, fix, docs, style, refactor, test, chore
   
   Example:
   ```
   feat: add caching support for embeddings
   
   - Implement two-tier cache (memory + disk)
   - Add cache expiration logic
   - Include cache hit rate metrics
   
   Closes #123
   ```

3. **Pull Request Process**:
   - Create feature branch from main
   - Write tests for new functionality
   - Ensure all tests pass
   - Update documentation
   - Submit PR with clear description
   - Address review feedback

## API Reference

### High-Level API

```python
# Configuration Management
from config.config_manager import KnowledgeBase
kb = KnowledgeBase('config.cfg')

# Database Operations
from database.db_manager import process_database
process_database(args, logger)

# Embedding Generation
from embedding.embed_manager import process_embeddings
process_embeddings(args, logger)

# Query Processing
from query.query_manager import process_query
response = process_query(args, logger)
```

### Utility Functions

```python
# Logging
from utils.logging_utils import setup_logging, get_logger
logger = setup_logging(verbose=True, debug=False)

# Text Processing
from utils.text_utils import clean_text, enhanced_clean_text
cleaned = clean_text(raw_text, preserve_entities=True)

# Security
from utils.security_utils import validate_api_key, safe_log_error
is_valid = validate_api_key(key, min_length=20)
```

## Performance Considerations

### Optimizing Embedding Generation

1. **Batch Size Tuning**:
   ```ini
   [PERFORMANCE]
   embedding_batch_size = 200  # Increase for better throughput
   api_max_concurrency = 16    # More parallel requests
   ```

2. **Caching Strategy**:
   - Memory cache for frequently accessed embeddings
   - Disk cache for persistence across runs
   - TTL-based expiration for query cache

3. **Index Selection**:
   - Flat index: < 1,000 vectors (highest accuracy)
   - IVF index: 1,000 - 100,000 vectors (balanced)
   - IVFPQ index: > 100,000 vectors (memory efficient)

### Database Optimization

1. **Indexing**:
   ```sql
   CREATE INDEX idx_chunks_file_hash ON chunks(file_hash);
   CREATE INDEX idx_chunks_embedded ON chunks(embedded);
   ```

2. **Batch Operations**:
   ```python
   # Good: Batch insert
   cursor.executemany("INSERT INTO chunks VALUES (?, ?, ?)", chunk_data)
   
   # Bad: Individual inserts
   for chunk in chunks:
       cursor.execute("INSERT INTO chunks VALUES (?, ?, ?)", chunk)
   ```

### Memory Management

1. **Streaming Large Files**:
   ```python
   def process_large_file(filepath: str, chunk_size: int = 1024*1024):
       with open(filepath, 'r') as f:
           while chunk := f.read(chunk_size):
               yield process_chunk(chunk)
   ```

2. **Garbage Collection**:
   ```python
   import gc
   
   # Force collection after processing large dataset
   gc.collect()
   ```

## Security Guidelines

### API Key Management

1. **Environment Variables**: Store keys in environment
2. **Key Rotation**: Implement regular rotation schedule
3. **Access Control**: Limit key permissions to necessary operations

### Input Sanitization

1. **Path Validation**:
   ```python
   from utils.security_utils import validate_safe_path
   
   if not validate_safe_path(user_path, allowed_dir):
       raise SecurityError("Invalid path")
   ```

2. **Query Sanitization**:
   ```python
   # Limit query length
   if len(query) > kb.max_query_length:
       raise ValueError("Query too long")
   ```

### Data Protection

1. **Sensitive Data**: Never log sensitive information
2. **Encryption**: Use encryption for sensitive database fields
3. **Access Logs**: Maintain audit trail for data access

## Troubleshooting

### Common Development Issues

1. **Import Errors**:
   ```bash
   # Ensure PYTHONPATH includes project root
   export PYTHONPATH="${PWD}:${PYTHONPATH}"
   ```

2. **NLTK Data Missing**:
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   nltk.download('wordnet')
   ```

3. **API Rate Limits**:
   ```ini
   [API]
   api_call_delay_seconds = 0.1  # Increase delay
   api_max_concurrency = 4       # Reduce concurrency
   ```

### Debugging Tips

1. **Enable Debug Logging**:
   ```bash
   customkb query config.cfg "test" --debug
   ```

2. **Profile Performance**:
   ```python
   import cProfile
   cProfile.run('expensive_function()', 'profile.stats')
   ```

3. **Memory Profiling**:
   ```bash
   pip install memory_profiler
   python -m memory_profiler your_script.py
   ```

### Getting Help

1. **Documentation**: Check this guide and README.md
2. **Code Comments**: Review inline documentation
3. **Test Cases**: Look at test examples for usage patterns
4. **GitHub Issues**: Search existing issues or create new ones

## Version Management

CustomKB uses semantic versioning with automatic build numbers:

```bash
# Show current version
./version.sh

# Update versions
./version.sh patch  # Bug fixes: 0.1.1 → 0.1.2
./version.sh minor  # New features: 0.1.2 → 0.2.0
./version.sh major  # Breaking changes: 0.2.0 → 1.0.0
```

Build numbers increment automatically with each commit via git hooks.

---

For additional information, see:
- [README.md](README.md) - User documentation
- [CLAUDE.md](CLAUDE.md) - AI assistant instructions
- [CHANGELOG.md](CHANGELOG.md) - Version history