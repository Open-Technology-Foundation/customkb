# CustomKB Development Guide for AI Assistants

This document provides specific instructions and context for AI assistants (like Claude) working with the CustomKB codebase.

## Project Overview

CustomKB is a production-ready AI-powered knowledge base system that:
- Processes documents into searchable vector databases
- Generates embeddings using OpenAI/Anthropic models
- Performs semantic search with context-aware AI responses
- Supports multiple file formats and languages
- Implements enterprise-grade security and performance features

## Critical Context for AI Assistants

### Working Directory Structure
```
/ai/scripts/customkb/         # Project root
├── .venv/                    # Python virtual environment (DO NOT modify)
├── .gudang/                  # Backup directory (DO NOT check/modify)
├── customkb.py               # Main entry point
├── config/                   # Configuration management
├── database/                 # Database operations
├── embedding/                # Embedding generation
├── query/                    # Query processing
├── models/                   # Model management
├── utils/                    # Utility functions
└── tests/                    # Test suite
```

### Essential Commands

Always activate the virtual environment first:
```bash
source .venv/bin/activate
```

### Performance Optimization

CustomKB includes powerful optimization and maintenance commands to ensure peak performance:

#### Optimize Command

The `optimize` command automatically tunes configurations based on system resources and creates missing database indexes:

```bash
# Show all optimization tiers and their settings
customkb optimize --show-tiers

# Analyze all KBs and show size/performance recommendations
customkb optimize --analyze

# Preview optimizations for a specific KB (dry-run mode)
customkb optimize myproject --dry-run

# Apply optimizations to a specific KB
customkb optimize myproject

# Optimize all KBs in VECTORDBS
customkb optimize

# Override system memory detection (useful for containers)
customkb optimize --memory-gb 64
```

**Features:**
- Automatically detects system memory and applies tier-based settings
- Creates missing SQLite indexes for improved query performance
- Backs up configuration files before making changes
- Handles both old (`docs`) and new (`chunks`) database schemas
- Resolves KB names even when local directories have the same name

**Optimization Tiers:**
- **Low Memory (<16GB)**: Conservative settings to avoid memory pressure
  - Reduced batch sizes and cache limits
  - Minimal thread pools
  - Hybrid search disabled
- **Medium Memory (16-64GB)**: Balanced performance for most workloads
  - Moderate batch sizes and caching
  - Reasonable concurrency settings
- **High Memory (64-128GB)**: High performance for production use
  - Large batch sizes and extensive caching
  - Hybrid search enabled
  - Increased thread pools
- **Very High Memory (>128GB)**: Maximum performance for large deployments
  - Maximum batch sizes and cache limits
  - Full concurrency utilization
  - All features enabled

#### Verify Indexes Command

Check database index health and identify missing indexes:

```bash
# Verify indexes for a specific KB
customkb verify-indexes myproject

# Check from any directory (resolves KB name automatically)
cd /anywhere && customkb verify-indexes ollama
```

**Expected Indexes:**
- `idx_embedded`: Filters embedded vs non-embedded documents
- `idx_embedded_embedtext`: Speeds up embedded text queries
- `idx_keyphrase_processed`: Enables fast keyphrase searches
- `idx_sourcedoc`: Filters by source document
- `idx_sourcedoc_sid`: Compound queries on source and section

Missing indexes will be reported with suggestions to run `optimize`.

#### BM25 Hybrid Search

Enable keyword-based search alongside semantic search:

```bash
# Build BM25 index (requires enable_hybrid_search=true in config)
customkb bm25 myproject

# Force rebuild existing index
customkb bm25 myproject --force
```

**Requirements:**
1. Set `enable_hybrid_search=true` in `[ALGORITHMS]` section
2. For older databases, run `upgrade_bm25_tokens.py` first
3. Sufficient disk space for `.bm25` index file

**Benefits:**
- Combines keyword matching with semantic understanding
- Better results for exact term matches
- Improved performance on technical documentation

### Important Notes
- **Virtual Environment**: Always use `.venv/` - it contains all dependencies
- **Backups**: `.gudang/` contains backups - DO NOT process or modify
- **Logs**: Check `/var/lib/vectordbs/<kb_name>/logs/` for runtime logs
- **Security**: All user inputs must be validated using `utils.security_utils`

## Code Style Requirements

### Python Formatting
```python
# CRITICAL: Use 2-space indentation (not 4!)
def example_function():
  """Docstring with proper indentation."""
  if condition:
    do_something()  # 2 spaces per level
```

### Import Order
```python
# Standard library
import os
import sys
from typing import List, Optional

# Third-party packages
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Local imports (use relative imports)
from config.config_manager import KnowledgeBase
from utils.logging_utils import get_logger
```

### Docstring Format
```python
def process_text(text: str, max_tokens: int = 500) -> List[str]:
  """
  Process text into chunks with specified token limit.
  
  Args:
      text: Input text to process.
      max_tokens: Maximum tokens per chunk.
      
  Returns:
      List of processed text chunks.
      
  Raises:
      ValueError: If text is empty or max_tokens <= 0.
  """
```

### File Endings
Always end Python files with:
```python
#fin
```

## Architecture Guidelines

### Configuration Hierarchy
1. **Environment variables** (highest priority)
2. **Configuration file values**
3. **Default values** (lowest priority)

Example:
```python
# This order is enforced by get_env() in text_utils.py
value = get_env('VECTOR_MODEL',                    # 1. Check environment
                config.get('vector_model',         # 2. Check config file
                          'text-embedding-3-small'))  # 3. Use default
```

### Security Patterns

Always validate user input:
```python
from utils.security_utils import validate_file_path, validate_api_key

# File path validation (default - strict security)
safe_path = validate_file_path(user_input, allowed_extensions=['.cfg'])

# Knowledge base config validation (allows absolute paths and relative traversal)
kb_config_path = validate_file_path(config_input, ['.cfg'], 
                                   allow_absolute=True, 
                                   allow_relative_traversal=True)

# API key validation
if not validate_api_key(api_key, min_length=20):
  raise ValueError("Invalid API key format")
```

**Path Validation Parameters:**
- `allow_absolute`: Set to `True` for trusted user input like KB config paths
- `allow_relative_traversal`: Set to `True` to allow `../` in relative paths
- Use defaults (`False`) for untrusted input like web requests

### Error Handling

Use specific exceptions with context:
```python
try:
  result = database_operation()
except sqlite3.DatabaseError as e:
  logger.error(f"Database operation failed: {e}")
  raise DatabaseError(f"Failed to process chunk: {e}") from e
```

## Knowledge Base Configuration Paths

CustomKB supports three flexible ways to specify knowledge base configuration files:

### Case 1: Absolute Path to Config File
```bash
customkb query /var/lib/vectordbs/myproject/myproject.cfg 'search query'
customkb query /home/user/projects/kb/config.cfg 'search query'
```
- Direct path to `.cfg` file anywhere in the filesystem
- KB files (`.db`, `.faiss`, `logs/`) created in same directory as config
- Most explicit and reliable method

### Case 2: Knowledge Base Name (searches VECTORDBS)
```bash
customkb query myproject 'search query'
```
- Searches `$VECTORDBS` directory for `myproject/myproject.cfg`
- Default `VECTORDBS=/var/lib/vectordbs`
- Convenient for standard installations

### Case 3: Relative Path with Traversal
```bash
# From /var/lib/vectordbs/okusiassociates/
customkb query ../okusimail/okusimail.cfg 'search query'
```
- Allows `../` to reference sibling directories
- Useful for projects with multiple related knowledge bases
- Maintains security while allowing legitimate navigation

### Implementation Notes

The path resolution uses enhanced security validation:
```python
# In config_manager.py
validated_cfgfile = validate_file_path(cfgfile, ['.cfg', ''], 
                                      allow_absolute=True, 
                                      allow_relative_traversal=True)
```

This approach:
- **Allows** absolute paths and relative traversal for KB configs (trusted user input)
- **Maintains** strict validation for other security checks (dangerous characters, etc.)
- **Preserves** VECTORDBS search functionality for backward compatibility

## Common Optimization Scenarios

### Optimizing a Large Knowledge Base

For knowledge bases over 1GB:

```bash
# First analyze the current state
customkb optimize mylargeproject --analyze

# Preview optimizations
customkb optimize mylargeproject --dry-run

# Apply optimizations and create indexes
customkb optimize mylargeproject

# Verify all indexes were created
customkb verify-indexes mylargeproject
```

### Migrating from Old Database Schema

For databases created with older CustomKB versions:

```bash
# Check current indexes
customkb verify-indexes oldproject

# Upgrade database schema for BM25 support
python upgrade_bm25_tokens.py oldproject

# Enable hybrid search in config
customkb edit oldproject
# Set: enable_hybrid_search = true

# Build BM25 index
customkb bm25 oldproject

# Optimize for current system
customkb optimize oldproject
```

### Container/VM Deployment

When running in containers with memory limits:

```bash
# Override detected memory for container limits
customkb optimize --memory-gb 8  # For 8GB container

# Show what settings would be applied
customkb optimize --show-tiers | grep "Low Memory" -A15
```

## Common Development Tasks

### Adding a New Configuration Parameter

1. Add to `KnowledgeBase.__init__()` in `config/config_manager.py`:
```python
self.NEW_CONFIG = {
  'DEF_MY_PARAMETER': (int, 42),  # (type, default_value)
}
```

2. Add to config file template:
```ini
[SECTION]
my_parameter = 42  # Description of parameter
```

3. Load in `load_config()`:
```python
self.my_parameter = get_env('MY_PARAMETER',
  section.getint('my_parameter', fallback=self.DEF_MY_PARAMETER), int)
```

### BM25 Result Limiting

To prevent memory exhaustion from large BM25 result sets:

```ini
[ALGORITHMS]
# Maximum BM25 results to process (0 = unlimited)
bm25_max_results = 1000
```

This parameter limits the number of BM25 search results to prevent crashes when processing large knowledge bases. The limiting is applied efficiently using heap-based selection to maintain the highest-scoring results.

### Cache Thread Pool Configuration

The system includes a dedicated thread pool for cache operations to prevent resource leaks:

```python
# In config/config_manager.py PERFORMANCE_CONFIG section:
'DEF_CACHE_THREAD_POOL_SIZE': (int, 4),  # Dedicated thread pool for cache operations
```

Configuration usage:
```ini
[PERFORMANCE]
cache_thread_pool_size = 4  # Thread pool size for embedding cache operations
memory_cache_size = 10000   # Maximum number of embeddings in memory cache
```

**Thread Pool Management:**
- Lazy initialization: ThreadPoolExecutor created only when needed
- Proper lifecycle: Automatic cleanup on process exit using `atexit`
- Resource safety: Single shared executor prevents resource leaks
- Configuration: Thread pool size configurable per knowledge base

**Cache Performance:**
- Thread-safe LRU eviction with optimal performance
- Atomic metrics tracking for monitoring
- Configurable memory cache size
- Automatic promotion from disk to memory cache

### Adding a New CLI Command

1. Add parser in `customkb.py`:
```python
# Add new subparser
new_parser = subparsers.add_parser(
  'newcommand',
  description=textwrap.dedent(process_new.__doc__),
  formatter_class=argparse.RawDescriptionHelpFormatter,
)
new_parser.add_argument('config_file', help='Knowledge base configuration')
```

2. Add command handling:
```python
elif args.command == 'newcommand':
  result = process_new(args, logger)
  print(result)
```

3. Implement function in appropriate manager module

### Working with Models

Always use model resolution:
```python
from models.model_manager import get_canonical_model

# User might provide alias
user_model = 'claude'  # or 'gpt4', 'sonnet', etc.

# Resolve to canonical name
model_info = get_canonical_model(user_model)
actual_model = model_info['model']  # 'claude-3-5-sonnet-20241022'
```

## Testing Guidelines

### Running Tests
```bash
# Always in virtual environment
source .venv/bin/activate

# Run all tests
pytest tests/

# Run specific test
pytest tests/unit/test_config_manager.py -v

# Run with coverage
pytest --cov=. tests/
```

### Writing Tests
```python
# tests/unit/test_new_feature.py
import pytest
from unittest.mock import Mock, patch

def test_feature_success():
  """Test successful feature execution."""
  # Arrange
  mock_kb = Mock()
  mock_kb.some_param = 'value'
  
  # Act
  result = new_feature(mock_kb)
  
  # Assert
  assert result == expected_value
```

## Performance Optimization

### Batch Processing
```python
# Good: Process in batches
for i in range(0, len(items), batch_size):
  batch = items[i:i + batch_size]
  process_batch(batch)
  
# Bad: Process one at a time
for item in items:
  process_item(item)
```

### Thread-Safe Caching Patterns
```python
# Use the CacheThreadManager for thread-safe operations
from embedding.embed_manager import cache_manager

# Thread-safe cache operations
cache_manager.add_to_memory_cache(key, embedding, kb)
cached_value = cache_manager.get_from_memory_cache(key)

# Performance monitoring
metrics = cache_manager.get_metrics()
print(f"Cache hit ratio: {metrics['cache_hit_ratio']:.2%}")
print(f"Cache size: {metrics['cache_size']}/{metrics['max_cache_size']}")

# Configuration
cache_manager.configure(max_workers=8, memory_cache_size=20000)
```

### Embedding Cache Architecture
The embedding cache system uses a thread-safe two-tier approach:

1. **Memory Cache**: Fast LRU cache with configurable size
2. **Disk Cache**: Persistent JSON-based storage
3. **Thread Pool**: Managed ThreadPoolExecutor for async disk writes

**Thread Safety Guarantees:**
- All cache operations use RLock protection
- No race conditions in LRU eviction
- Metrics tracking is atomic
- ThreadPoolExecutor lifecycle is properly managed

**Performance Features:**
- Automatic cache hit/miss metrics
- Configurable thread pool size
- LRU eviction with optimal performance
- Backward compatibility with deprecation warnings

## Debugging Tips

### Enable Debug Logging
```python
# In code
logger = get_logger(__name__)
logger.debug(f"Processing chunk: {chunk_id}")

# From CLI
customkb query config.cfg "test" --debug
```

### Check Configuration Loading
```python
# Verify configuration values
kb = KnowledgeBase('config.cfg')
print(f"Vector model: {kb.vector_model}")
print(f"Database path: {kb.knowledge_base_db}")
```

### Database Inspection
```bash
# Open SQLite database
sqlite3 /var/lib/vectordbs/myproject.db

# Useful queries
.tables
SELECT COUNT(*) FROM chunks;
SELECT * FROM chunks LIMIT 5;
SELECT DISTINCT file_hash FROM chunks;
```

## Common Pitfalls to Avoid

1. **Wrong Indentation**: Always use 2 spaces, not 4
2. **Missing Imports**: Check relative imports from other modules
3. **Hardcoded Paths**: Use `VECTORDBS` environment variable
4. **Unvalidated Input**: Always validate user-provided paths and data
   - Use `validate_file_path()` with appropriate parameters for the use case
   - KB config paths: `allow_absolute=True, allow_relative_traversal=True`
   - Untrusted input: use defaults (strict validation)
5. **Synchronous API Calls**: Use async patterns from embed_manager.py
6. **Memory Leaks**: Clear large objects, use generators for big files
7. **SQL Injection**: Use parameterized queries, never string formatting

## Quick Reference

### Key Environment Variables
```bash
OPENAI_API_KEY       # OpenAI API key
ANTHROPIC_API_KEY    # Anthropic API key
VECTORDBS           # Base directory for KBs (default: /var/lib/vectordbs)
NLTK_DATA           # NLTK data directory
VECTOR_MODEL        # Override embedding model
QUERY_MODEL         # Override query model
```

### Configuration Sections
- `[DEFAULT]`: Core model and processing settings
- `[API]`: Rate limiting and concurrency
- `[LIMITS]`: Resource constraints
- `[PERFORMANCE]`: Optimization parameters
- `[ALGORITHMS]`: Processing thresholds and BM25 settings

### Model Categories
- `llm`: Language models for responses
- `embed`: Embedding models for vectors
- `image`: Image generation models
- `tts`: Text-to-speech models
- `stt`: Speech-to-text models

## Getting Help

1. **Check existing code**: Similar patterns are likely implemented
2. **Read test cases**: Tests show expected behavior
3. **Review logs**: Enable debug mode for detailed information
4. **Search codebase**: Use grep/ripgrep for finding implementations

Remember: The codebase prioritizes clarity, security, and performance. When in doubt, follow existing patterns in the codebase.

#fin