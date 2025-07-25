# CustomKB: AI-Powered Knowledge Base System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Version](https://img.shields.io/badge/version-0.1.1-green.svg)](https://github.com/Open-Technology-Foundation/customkb)

CustomKB is a production-ready knowledge base system that transforms your document collections into AI-powered, searchable knowledge repositories. It combines state-of-the-art embedding models with efficient vector search to deliver contextually relevant answers from your data.

## 🌟 Key Features

### Intelligent Document Processing
- **Multi-Format Support**: Process Markdown, HTML, code files, and plain text with format-specific optimization
- **Smart Chunking**: Configurable token-based chunking that preserves context and document structure
- **Metadata Extraction**: Automatic extraction of headings, sections, and entities
- **Multi-Language Support**: Stopword filtering and text normalization for 27+ languages
- **Full Path Storage**: Stores complete file paths to properly handle duplicate filenames across directories

### Advanced Vector Search
- **High-Performance Storage**: SQLite for structured data + FAISS for vector indices
- **Adaptive Indexing**: Automatically selects optimal index type based on dataset size
- **Semantic Search**: Find relevant content based on meaning, not just keywords
- **Relevance Scoring**: Context-aware ranking with configurable similarity thresholds
- **Hybrid Search**: Combines vector similarity with BM25 keyword matching for optimal results
- **Cross-Encoder Reranking**: Enabled by default - uses advanced models to improve accuracy by 20-40%

### AI-Powered Responses
- **Multi-Model Support**: 
  - OpenAI GPT (4o, 4o-mini, o1 series)
  - Anthropic Claude (3.0, 3.5, 3.7)
  - Meta Llama (via Ollama integration)
- **Context Management**: XML-formatted reference contexts for precise prompting
- **Customizable Behavior**: Fine-tune temperature, token limits, and system roles

### Enterprise-Ready Features
- **Security First**: Input validation, path traversal prevention, API key protection
- **Performance Optimized**: Batch processing, caching, concurrent API calls
- **Resilient Design**: Checkpoint saving, automatic retries, graceful error handling
- **Comprehensive Logging**: Per-knowledge-base logs with performance metrics

## 🚀 Quick Start

### Prerequisites

- Python 3.12 or higher
- SQLite 3.45+
- API keys for your chosen embedding/LLM providers
- 4GB+ RAM recommended for large datasets

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Open-Technology-Foundation/customkb.git
   cd customkb
   ```

2. **Set up Python environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**
   ```bash
   export OPENAI_API_KEY="your-openai-api-key"
   export ANTHROPIC_API_KEY="your-anthropic-api-key"
   export NLTK_DATA="$HOME/nltk_data"  # Required for text processing
   export VECTORDBS="/var/lib/vectordbs"  # Optional: default KB storage location
   ```

### Basic Usage Example

1. **Create a knowledge base directory**:
   ```bash
   mkdir -p /var/lib/vectordbs/myproject
   ```

2. **Create a configuration file** (`/var/lib/vectordbs/myproject/myproject.cfg`):
   ```ini
   [DEFAULT]
   vector_model = text-embedding-3-small
   query_model = gpt-4o-mini
   db_min_tokens = 200
   db_max_tokens = 400
   ```

3. **Process your documents**:
   ```bash
   customkb database myproject docs/*.md *.txt
   ```

4. **Generate embeddings**:
   ```bash
   customkb embed myproject
   ```

5. **Query your knowledge base**:
   ```bash
   customkb query myproject "How do I configure the vector model?"
   ```

> **💡 Knowledge Base Resolution**: CustomKB requires all knowledge bases to be organized within the VECTORDBS directory:
> - **Simple KB name**: `myproject` → `/var/lib/vectordbs/myproject/myproject.cfg`
> - **Auto-stripping**: Paths and `.cfg` extensions are automatically removed
> - **Helpful errors**: Lists available KBs when one isn't found

## 📖 Detailed Documentation

### Command Reference

#### `database` - Import Documents
```bash
customkb database <config> [files...] [options]
```
Processes text files and stores them in the knowledge base.

**Options:**
- `-l, --language LANG`: Stopword language (default: english)
- `-f, --force`: Reprocess files already in database
- `-v, --verbose`: Enable detailed output
- `-d, --debug`: Enable debug logging

**Example:**
```bash
customkb database myproject ~/documents/**/*.md -l english
```

#### `embed` - Generate Embeddings
```bash
customkb embed <config> [options]
```
Creates vector embeddings for all text chunks in the database.

**Options:**
- `-r, --reset-database`: Reset embedding status to reprocess all chunks
- `-v, --verbose`: Show progress information

**Example:**
```bash
customkb embed myproject --verbose
```

#### `query` - Search Knowledge Base
```bash
customkb query <config> "<query>" [options]
```
Performs semantic search and generates AI responses.

**Options:**
- `-c, --context-only`: Return search results without AI response
- `-m, --model MODEL`: Override configured LLM model
- `-k, --top-k N`: Number of results to retrieve (default: 50)
- `-s, --context-scope N`: Context segments per result (default: 4)
- `-t, --temperature T`: Response creativity (0.0-2.0)
- `-M, --max-tokens N`: Maximum response length
- `-R, --role "ROLE"`: Custom system prompt

**Examples:**
```bash
# Simple query with KB name
customkb query myproject "What are the main features?"

# With .cfg extension (automatically stripped)
customkb query myproject.cfg "How to install?"

# Context-only search
customkb query myproject "authentication" --context-only

# Custom model and parameters
customkb query myproject "Explain the architecture" \
  --model claude-3-5-sonnet-20241022 \
  --temperature 0.7 \
  --max-tokens 2000

# All KB files must be in VECTORDBS directory:
# /var/lib/vectordbs/myproject/myproject.cfg
# /var/lib/vectordbs/myproject/myproject.db
# /var/lib/vectordbs/myproject/myproject.faiss
```

#### `edit` - Modify Configuration
```bash
customkb edit <config>
```
Opens the configuration file in your default editor.

#### `optimize` - Performance Optimization
```bash
customkb optimize [config] [options]
```
Automatically optimizes knowledge base performance based on system resources.

**Options:**
- `--dry-run`: Preview changes without applying them
- `--analyze`: Show KB size analysis and recommendations
- `--show-tiers`: Display optimization settings for all memory tiers
- `--memory-gb N`: Override system memory detection

**Features:**
- Detects system memory and applies tier-based optimizations
- Creates missing SQLite indexes for faster queries
- Backs up configuration files before changes
- Handles directory name conflicts intelligently

**Examples:**
```bash
# Show all optimization tiers
customkb optimize --show-tiers

# Analyze all KBs
customkb optimize --analyze

# Preview changes for a specific KB
customkb optimize myproject --dry-run

# Apply optimizations
customkb optimize myproject

# Optimize all KBs in VECTORDBS
customkb optimize
```

#### `verify-indexes` - Check Database Health
```bash
customkb verify-indexes <config>
```
Verifies that all performance-critical indexes exist in the database.

**Expected Indexes:**
- `idx_embedded`: Filters embedded vs non-embedded documents
- `idx_embedded_embedtext`: Speeds up embedded text queries
- `idx_keyphrase_processed`: Enables fast keyphrase searches
- `idx_sourcedoc`: Filters by source document
- `idx_sourcedoc_sid`: Compound queries on source and section

**Example:**
```bash
customkb verify-indexes myproject
# Output shows which indexes are present/missing
```

#### `bm25` - Build Hybrid Search Index
```bash
customkb bm25 <config> [options]
```
Builds or rebuilds the BM25 index for keyword-based hybrid search.

**Options:**
- `--force`: Force rebuild even if index exists

**Requirements:**
- Set `enable_hybrid_search=true` in `[ALGORITHMS]` section
- For older databases, run `upgrade_bm25_tokens.py` first

**Example:**
```bash
# Build BM25 index
customkb bm25 myproject

# Force rebuild
customkb bm25 myproject --force
```

#### `help` - Show Usage
```bash
customkb help
```
Displays comprehensive usage information.

#### `version` - Display Version
```bash
customkb version [--build]
```
Shows version information with optional build number.

### Configuration Guide

CustomKB uses INI-style configuration files with five sections:

#### [DEFAULT] - Core Settings
```ini
[DEFAULT]
# Embedding model configuration
vector_model = text-embedding-3-small  # Options: ada-002, 3-small, 3-large
vector_dimensions = 1536               # Must match model output
vector_chunks = 500                    # Max chunks to process per batch

# Text processing parameters
db_min_tokens = 200                    # Minimum chunk size
db_max_tokens = 400                    # Maximum chunk size

# Query configuration
query_model = gpt-4o-mini              # LLM for responses
query_max_tokens = 4096                # Max response length
query_top_k = 50                       # Results to retrieve
query_context_scope = 4                # Context segments per result
query_temperature = 0.1                # Response creativity (0-2)
query_role = You are a helpful assistant who provides accurate, detailed answers.
```

#### [API] - External Service Settings
```ini
[API]
api_call_delay_seconds = 0.05          # Rate limiting delay
api_max_retries = 20                   # Retry attempts
api_max_concurrency = 8                # Parallel API calls
api_min_concurrency = 3                # Minimum parallel calls
backoff_exponent = 2                   # Exponential backoff factor
backoff_jitter = 0.1                   # Randomization factor
```

#### [LIMITS] - Resource Constraints
```ini
[LIMITS]
max_file_size_mb = 100                 # Maximum file size
max_query_file_size_mb = 1             # Max query file size
memory_cache_size = 10000              # Cache entries
api_key_min_length = 20                # Security validation
max_query_length = 10000               # Query text limit
```

#### [PERFORMANCE] - Optimization
```ini
[PERFORMANCE]
embedding_batch_size = 100             # Embeddings per batch
checkpoint_interval = 10               # Save progress frequency
commit_frequency = 1000                # Database commit interval
io_thread_pool_size = 4                # Parallel I/O threads
query_cache_ttl_days = 7               # Cache expiration
```

#### [ALGORITHMS] - Processing Thresholds
```ini
[ALGORITHMS]
similarity_threshold = 0.6             # Minimum relevance score
low_similarity_scope_factor = 0.5      # Scope reduction factor
max_chunk_overlap = 100                # Token overlap between chunks
heading_search_limit = 200             # Characters to scan for headings
```

### Model Support

#### Embedding Models
- **OpenAI**: text-embedding-ada-002, text-embedding-3-small, text-embedding-3-large
- **Anthropic**: claude-3-embed (coming soon)

#### Language Models
- **OpenAI GPT-4**: gpt-4o, gpt-4o-mini, gpt-4-turbo
- **OpenAI o1**: o1-preview, o1-mini
- **Anthropic Claude**: claude-3-5-sonnet, claude-3-5-haiku, claude-3-opus
- **Meta Llama**: llama-3.1-8b, llama-3.1-70b (via Ollama)

### Knowledge Base Organization

#### Standardized KB Structure

All knowledge bases MUST be organized as subdirectories within the VECTORDBS directory:

```bash
# Set the KB base directory (default: /var/lib/vectordbs)
export VECTORDBS="/var/lib/vectordbs"

# Create a new knowledge base
mkdir -p $VECTORDBS/myproject

# Process documents - KB name only
customkb database myproject docs/*.md

# Generate embeddings
customkb embed myproject

# Query the knowledge base
customkb query myproject "search terms"
```

#### KB Name Resolution

The system automatically handles various input formats:
```bash
# All of these resolve to the same KB:
customkb query myproject "search"           # Simple name
customkb query myproject.cfg "search"       # With .cfg extension
customkb query /path/to/myproject "search"  # With path (stripped)
customkb query ../myproject.cfg "search"    # Relative path (stripped)

# Result: Uses $VECTORDBS/myproject/myproject.cfg
```

#### Required Directory Structure

```
$VECTORDBS/
├── myproject/
│   ├── myproject.cfg    # Configuration file
│   ├── myproject.db     # SQLite database
│   ├── myproject.faiss  # Vector index
│   └── logs/            # Runtime logs
```

#### Multi-Project Setup Example

Organize related knowledge bases for easy cross-referencing:
```
/var/lib/vectordbs/
├── company-docs/
│   ├── company-docs.cfg
│   ├── company-docs.db
│   └── company-docs.faiss
├── api-reference/
│   ├── api-reference.cfg
│   ├── api-reference.db
│   └── api-reference.faiss
└── customer-support/
    ├── customer-support.cfg
    ├── customer-support.db
    └── customer-support.faiss

# Query any KB using just the name
customkb query company-docs "HR policies"
customkb query api-reference "authentication"
customkb query customer-support "tickets"
```

### Performance Tuning

#### For Large Datasets (100k+ documents)
```ini
[PERFORMANCE]
embedding_batch_size = 200
api_max_concurrency = 16
checkpoint_interval = 5

[ALGORITHMS]
medium_dataset_threshold = 100000
ivf_centroid_multiplier = 4
```

#### For High-Accuracy Requirements
```ini
[DEFAULT]
vector_model = text-embedding-3-large
query_top_k = 100
query_context_scope = 6

[ALGORITHMS]
similarity_threshold = 0.7
enable_hybrid_search = true
enable_reranking = true
reranking_model = cross-encoder/ms-marco-MiniLM-L-6-v2
reranking_top_k = 30
```

#### For Fast Response Times
```ini
[DEFAULT]
vector_model = text-embedding-3-small
query_top_k = 20
query_model = gpt-4o-mini

[ALGORITHMS]
enable_reranking = false  # Disable for lowest latency

[PERFORMANCE]
query_cache_ttl_days = 30
```

## 🔧 Advanced Usage

### Reranking Configuration

Cross-encoder reranking is enabled by default to maximize search accuracy. However, you may want to adjust or disable it based on your needs:

#### When to Disable Reranking
```ini
[ALGORITHMS]
enable_reranking = false  # Disable when you need:
# - Ultra-low latency (<100ms response time)
# - Resource-constrained environments (RPi, edge devices)
# - High query volume (>100 QPS)
# - Simple keyword-based searches
```

#### Optimizing Reranking Performance
```ini
[ALGORITHMS]
enable_reranking = true
reranking_model = cross-encoder/ms-marco-MiniLM-L-6-v2  # Lightweight model
reranking_top_k = 10         # Reduce for faster processing
reranking_batch_size = 64    # Increase for GPU processing
reranking_device = cuda      # Use GPU if available
reranking_cache_size = 5000  # Increase cache for repeated queries
```

### Domain-Style Knowledge Bases
```bash
# Create knowledge base with domain naming
mkdir -p /var/lib/vectordbs/example.com
cat > /var/lib/vectordbs/example.com/example.com.cfg <<EOF
[DEFAULT]
vector_model = text-embedding-3-small
query_model = gpt-4o-mini
EOF

customkb database example.com ~/example-docs/*.md

# The system will create:
# /var/lib/vectordbs/example.com/example.com.db
# /var/lib/vectordbs/example.com/example.com.faiss
```

### Multi-Language Processing
```bash
# Process French documents
customkb database multilang docs/*.txt --language french

# Supported languages: english, french, german, spanish, italian, 
# portuguese, dutch, swedish, norwegian, danish, finnish, russian,
# turkish, arabic, chinese, japanese, korean, indonesian, and more
```

### Batch Processing Scripts
```bash
#!/bin/bash
# process_knowledge_bases.sh

for config in configs/*.cfg; do
    echo "Processing $config..."
    customkb database "$config" data/*.txt
    customkb embed "$config"
done
```

### Integration Examples

#### Python Integration
```python
import subprocess
import json

def query_knowledge_base(kb_name, query_text):
    """Query CustomKB from Python code."""
    result = subprocess.run(
        ['customkb', 'query', kb_name, query_text, '--context-only'],
        capture_output=True,
        text=True
    )
    return result.stdout

# Example usage
context = query_knowledge_base('myproject', 'How to install?')
print(context)
```

#### Shell Pipeline
```bash
# Extract all Python code examples
customkb query myproject.cfg "Python code examples" --context-only | \
  grep -A 5 -B 5 "```python"
```

## 🚀 Performance Optimization

CustomKB automatically optimizes performance based on available system resources. The `optimize` command applies tier-based settings:

### Memory Tiers

#### Low Memory Systems (<16GB)
- **Use Case**: Development machines, small VMs
- **Settings**: Conservative to prevent memory exhaustion
- **Features**: Basic search functionality, limited caching
- **Example Settings**:
  ```ini
  memory_cache_size = 50000
  embedding_batch_size = 375
  reference_batch_size = 15
  enable_hybrid_search = false
  ```

#### Medium Memory Systems (16-64GB)
- **Use Case**: Standard production servers
- **Settings**: Balanced performance and resource usage
- **Features**: Good query performance, moderate caching
- **Example Settings**:
  ```ini
  memory_cache_size = 100000
  embedding_batch_size = 562
  reference_batch_size = 22
  enable_hybrid_search = false
  ```

#### High Memory Systems (64-128GB)
- **Use Case**: Dedicated KB servers, high-traffic applications
- **Settings**: High performance with extensive caching
- **Features**: Hybrid search enabled, large batch processing
- **Example Settings**:
  ```ini
  memory_cache_size = 150000
  embedding_batch_size = 750
  reference_batch_size = 30
  enable_hybrid_search = true
  ```

#### Very High Memory Systems (>128GB)
- **Use Case**: Enterprise deployments, ML workstations
- **Settings**: Maximum performance, all features enabled
- **Features**: Full GPU acceleration, maximum concurrency
- **Example Settings**:
  ```ini
  memory_cache_size = 200000
  embedding_batch_size = 1125
  reference_batch_size = 45
  enable_hybrid_search = true
  ```

### Optimization Best Practices

1. **Run optimization after installation**:
   ```bash
   customkb optimize --analyze  # See current state
   customkb optimize           # Apply to all KBs
   ```

2. **Verify database health**:
   ```bash
   customkb verify-indexes myproject
   ```

3. **Monitor performance**:
   - Check logs for processing times
   - Use `--analyze` to track KB growth
   - Adjust settings based on usage patterns

4. **Container deployments**:
   ```bash
   # Override detection for container limits
   customkb optimize --memory-gb 8
   ```

## 🛠️ Troubleshooting

### Common Issues

**"Knowledge base 'name' not found"**
- Ensure the knowledge base directory exists in VECTORDBS
- Check if VECTORDBS environment variable is set correctly
- The error message will list available knowledge bases

**"API rate limit exceeded"**
- Adjust `api_call_delay_seconds` in configuration
- Reduce `api_max_concurrency` for stricter rate limits

**"Out of memory during embedding"**
- Reduce `embedding_batch_size` in configuration
- Process files in smaller batches

**"Low similarity scores"**
- Check if documents and queries are in the same language
- Consider using a more powerful embedding model
- Adjust `similarity_threshold` in configuration

### Debug Mode

Enable comprehensive logging:
```bash
customkb query myproject.cfg "test query" --debug
```

Check logs in:
```
/var/lib/vectordbs/<kb_name>/logs/<kb_name>.log
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-test.txt

# Run tests
pytest tests/

# Check code style
flake8 .
mypy --ignore-missing-imports .
```

## 📄 License

CustomKB is released under the MIT License. See [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

CustomKB is built on excellent open-source projects:
- [FAISS](https://github.com/facebookresearch/faiss) - Efficient similarity search
- [LangChain](https://github.com/langchain-ai/langchain) - Text splitting utilities
- [NLTK](https://www.nltk.org/) - Natural language processing
- [spaCy](https://spacy.io/) - Advanced NLP features

## 📞 Support

- **Documentation**: [CustomKB Docs](https://customkb.readthedocs.io)
- **Issues**: [GitHub Issues](https://github.com/Open-Technology-Foundation/customkb/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Open-Technology-Foundation/customkb/discussions)

---

*CustomKB is actively maintained by the Open Technology Foundation*