# CustomKB Development Guide

## Overview

CustomKB is an AI-powered knowledge base system that enables semantic search and context-aware responses using vector embeddings and large language models. This document provides guidelines for development, code style, and usage.

## Development Environment

### Setup
```bash
# Create and activate virtual environment
python -m venv .venv && source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt 

# Install development dependencies
pip install flake8 mypy pytest pytest-cov pdoc
```

### Memories

- python venv for this codebase is .venv/
- once again, remember. .gudang/ and .venv/ don't need to be checked

### Testing & Quality
```bash
# Linting
flake8 .

# Type checking
mypy --ignore-missing-imports .

# Run tests
pytest tests/

# Run specific test
pytest tests/test_file.py::test_function -v

# Test coverage
pytest --cov=. tests/
```

### Core Technologies
- **Python 3.12+**: Core language
- **SQLite**: Structured storage for text and metadata
- **FAISS**: Vector indices for similarity search
- **OpenAI, Anthropic, Meta**: API integration for embeddings and LLMs
- **NLTK, spaCy**: Text processing and NLP
- **asyncio**: Asynchronous operations

## Application Structure

### Main Components
- **config_manager.py**: Configuration handling and KnowledgeBase class
- **db_manager.py**: Database operations and text processing
- **embed_manager.py**: Vector embedding generation
- **query_manager.py**: Semantic search and response generation
- **model_manager.py**: AI model handling and resolution
- **utils/**: Text processing, logging, and utility functions

### Command Line Interface
```bash
customkb <command> <config_file> [options]
```

### Commands

1. **database**: Process text files into the knowledge base
   ```bash
   customkb database <config_file> [files...]
   ```
   - `-l, --language`: Language for stopwords (default: english)

2. **embed**: Generate embeddings for stored text
   ```bash
   customkb embed <config_file> [options]
   ```
   - `-r, --reset-database`: Reset embedding status flags

3. **query**: Semantic search with AI response
   ```bash
   customkb query <config_file> <query_text> [options]
   ```
   - `-Q, --query_file`: Load additional query text from file
   - `-c, --context-only`: Return only context without AI response
   - `-R, --role`: Set custom system role
   - `-m, --model`: Specify LLM model to use
   - `-k, --top-k`: Number of results to return
   - `-s, --context-scope`: Context segments per result
   - `-t, --temperature`: Model temperature
   - `-M, --max-tokens`: Maximum output tokens

4. **edit**: Edit configuration file
   ```bash
   customkb edit <config_file>
   ```

5. **help**: Display usage information
   ```bash
   customkb help
   ```

### Example Usage
```bash
# Process files
customkb database okusiassociates.cfg *.txt *.md

# Generate embeddings
customkb embed okusiassociates.cfg

# Query with AI response
customkb query okusiassociates.cfg "What are the key features?"

# Get context only (no AI response)
customkb query okusiassociates.cfg "Key features?" -c
```

## Code Style Guidelines

### Python Style
- 2-space indentation, 100 character line limit
- Triple double quotes for docstrings with type annotations
- Imports: stdlib → third-party → local (grouped by relationship)
- Use relative imports for local modules
- Constants: Define at top of files, use UPPER_CASE
- Always end scripts with '\n#fin\n'
- snake_case for variables/functions, PascalCase for classes
- F-strings for string formatting

### Documentation
- Function docstrings with Args/Returns/Raises sections
- Type hints for all parameters and return values
- Use typing module: Optional[Type], List[Type], Dict[KeyType, ValueType]

### Error Handling
- Catch specific exceptions with context in error messages
- Use appropriate logging levels (DEBUG, INFO, WARNING, ERROR)
- Track elapsed time for performance monitoring

### Design Patterns
- Async operations with proper error handling and timeouts
- Exponential backoff for external API calls
- Comprehensive logging with colorlog
- Environment variable configuration with override hierarchy
- Model configuration via Models.json (aliases, categories, parameters)

## Configuration

### Config File Format (.cfg)
```ini
[DEFAULT]
vector_model = text-embedding-3-large
vector_dimensions = 1536
vector_chunks = 500

db_min_tokens = 300
db_max_tokens = 500

query_model = gpt-4o
query_max_tokens = 4096
query_top_k = 50
query_context_scope = 4
query_temperature = 0.1
query_role = You are a helpful assistant.

[API]
# API performance and rate limiting
api_call_delay_seconds = 0.05
api_max_retries = 20
api_max_concurrency = 8
api_min_concurrency = 3
backoff_exponent = 2
backoff_jitter = 0.1

[LIMITS]
# File and memory limits
max_file_size_mb = 100
max_query_file_size_mb = 1
memory_cache_size = 10000
api_key_min_length = 20
max_query_length = 10000
max_config_value_length = 1000
max_json_size = 10000

[PERFORMANCE]
# Processing and caching performance
embedding_batch_size = 100
checkpoint_interval = 10
commit_frequency = 1000
io_thread_pool_size = 4
file_processing_batch_size = 500
sql_batch_size = 500
reference_batch_size = 5
query_cache_ttl_days = 7
default_editor = joe

[ALGORITHMS]
# Algorithm parameters and thresholds
high_dimension_threshold = 1536
small_dataset_threshold = 1000
medium_dataset_threshold = 100000
ivf_centroid_multiplier = 4
max_centroids = 256
token_estimation_sample_size = 10
token_estimation_multiplier = 1.3
similarity_threshold = 0.6
low_similarity_scope_factor = 0.5
max_chunk_overlap = 100
overlap_ratio = 0.5
heading_search_limit = 200
entity_extraction_limit = 500
default_dir_permissions = 0o770
default_code_language = python
additional_stopword_languages = indonesian,french,german,swedish
```

### Environment Variables
- `OPENAI_API_KEY`: OpenAI API key
- `ANTHROPIC_API_KEY`: Anthropic API key
- `VECTORDBS`: Base directory for vector databases (default: /var/lib/vectordbs)
- `NLTK_DATA`: NLTK data directory
- Configuration overrides (e.g., `VECTOR_MODEL`, `QUERY_MODEL`)

## Project Standards

### Shell Scripts
- Shebang '#!/bin/bash'
- Always use `set -euo pipefail` for error handling
- 2-space indentation
- Descriptive variable names with `declare` statements
- Use `[[` over `[` for conditionals
- Always end scripts with '\n#fin\n'

### Ignored Files and Directories
- .venv/
- __pycache__/
- *.pyc
- tmp/, temp/, .temp/
- .gudang/

### Backups
- .gudang/ directory for complete codebase backups
- Format: .gudang/YYYYMMDD_hhmmss/
- Create checkpoint: `rsync -a --exclude='.gudang' --exclude='sessions' /project/path/ .gudang/$(date +%Y%m%d_%H%M%S)/`

## Coding Principles
- K.I.S.S. (Keep It Simple, Stupid)
- "The best process is no process"
- "Everything should be made as simple as possible, but not simpler"
- Favor clarity over cleverness
- Optimize for maintainability and extensibility

## Developer Tech Stack
- Ubuntu 24.04 LTS
- Bash 5.2
- Python 3.12
- SQLite 3.45
- APIs: OpenAI, Anthropic Claude, Meta Llama
- Text processing: NLTK, spaCy, BeautifulSoup
- Key libraries: anthropic, openai, nltk, numpy, faiss-cpu, langchain

#fin