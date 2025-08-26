# CustomKB: AI-Powered Knowledge Base System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Version](https://img.shields.io/badge/version-0.8.0-green.svg)](https://github.com/Open-Technology-Foundation/customkb)

CustomKB is a production-ready knowledge base system that transforms document collections into AI-powered, searchable knowledge repositories. It combines state-of-the-art embedding models, vector search, and language models to deliver contextually relevant answers from your data.

## Key Features

### Document Processing
- **Multi-format support**: Process Markdown, HTML, code files, and plain text with format-specific optimization
- **Smart chunking**: Configurable token-based chunking that preserves context and document structure
- **Metadata extraction**: Automatic extraction of headings, sections, and document structure
- **Multi-language support**: Stopword filtering and text normalization for 27+ languages
- **Full path preservation**: Complete file paths stored to handle duplicate filenames across directories

### Vector Search & Retrieval
- **Dual storage system**: SQLite for structured data, FAISS for vector indices
- **Adaptive indexing**: Automatic selection of optimal index type based on dataset size
- **Semantic search**: Content retrieval based on meaning, not just keywords
- **Hybrid search**: Combined vector similarity with BM25 keyword matching for optimal results
- **Cross-encoder reranking**: Advanced models improve retrieval accuracy by 20-40%
- **Configurable relevance**: Adjustable similarity thresholds and result filtering

### AI Integration
- **Multi-provider support**:
  - OpenAI: GPT-4o, GPT-4o-mini, o3 series
  - Anthropic: Claude 4.0 Opus/Sonnet, Claude 3.5/3.7 Sonnet/Haiku
  - Google: Gemini 2.0/2.5 Flash/Pro
  - xAI: Grok 4.0
  - Local models via Ollama: Llama, Gemma, Mistral, Qwen, DeepSeek
- **Flexible context formatting**: XML, JSON, Markdown, or plain text reference formats
- **Customizable behavior**: Fine-tune temperature, token limits, and system roles
- **Prompt templates**: Pre-configured styles for different response types

### Enterprise Features
- **Security**: Input validation, path traversal prevention, API key protection
- **Performance optimization**: Batch processing, caching, concurrent API calls, GPU acceleration
- **Resilience**: Checkpoint saving, automatic retries, graceful error handling
- **Monitoring**: Per-knowledge-base logging with performance metrics
- **Resource management**: Memory-based optimization tiers, configurable thread pools

## Quick Start

### Prerequisites

- Python 3.12 or higher
- SQLite 3.45+
- 4GB+ RAM recommended
- NVIDIA GPU with CUDA (optional, for accelerated reranking)
- API keys for chosen embedding/LLM providers

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

4. **Set up NLTK data**
   ```bash
   sudo ./setup/nltk_setup.py download cleanup
   ```

5. **Configure environment variables**
   ```bash
   export OPENAI_API_KEY="your-openai-api-key"
   export ANTHROPIC_API_KEY="your-anthropic-api-key"
   export GEMINI_API_KEY="your-gemini-api-key"      # Optional
   export XAI_API_KEY="your-xai-api-key"            # Optional
   export NLTK_DATA="$HOME/nltk_data"               # Required for text processing
   export VECTORDBS="/var/lib/vectordbs"            # Default KB storage location
   ```

### Basic Usage

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

3. **Process documents**:
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

## Command Reference

### Core Commands

#### `database` - Import Documents
```bash
customkb database <knowledge_base> [files...] [options]
```

Process text files and store them in the knowledge base.

**Options:**
- `-l, --language LANG`: Language for stopwords (en, fr, de, etc.)
- `--detect-language`: Auto-detect language per file
- `-f, --force`: Force reprocess existing files
- `-v, --verbose`: Enable detailed output
- `-d, --debug`: Enable debug logging

**Examples:**
```bash
# Process all markdown files
customkb database myproject ~/documents/**/*.md

# Process with French stopwords
customkb database myproject docs/*.txt -l french

# Force reprocess with language detection
customkb database myproject --force --detect-language docs/
```

#### `embed` - Generate Embeddings
```bash
customkb embed <knowledge_base> [options]
```

Create vector embeddings for all text chunks in the database.

**Options:**
- `-r, --reset-database`: Reset embedding status flags
- `-v, --verbose`: Show progress information

#### `query` - Search Knowledge Base
```bash
customkb query <knowledge_base> "<query>" [options]
```

Perform semantic search and generate AI responses.

**Options:**
- `-c, --context-only`: Return only context without AI response
- `-m, --model MODEL`: AI model to use
- `-k, --top-k N`: Number of search results to return
- `-s, --context-scope N`: Context segments per result
- `-t, --temperature T`: Model temperature (0.0-2.0)
- `-M, --max-tokens N`: Maximum response tokens
- `-R, --role "ROLE"`: Custom system role
- `-f, --format FORMAT`: Output format (xml, json, markdown, plain)
- `-p, --prompt-template TEMPLATE`: Prompt style

**Examples:**
```bash
# Simple query
customkb query myproject "What are the main features?"

# Context-only search
customkb query myproject "authentication" -c -k 10

# Custom model with specific format
customkb query myproject "Explain the architecture" \
  --model claude-3-5-sonnet-latest \
  --temperature 0.7 \
  --format json \
  --prompt-template technical
```

### Optimization Commands

#### `optimize` - Performance Tuning
```bash
customkb optimize [knowledge_base] [options]
```

Automatically optimize knowledge base performance based on system resources.

**Options:**
- `--dry-run`: Preview changes without applying
- `--analyze`: Analyze and show recommendations
- `--show-tiers`: Display all memory tier settings
- `--memory-gb N`: Override detected memory (GB)

**Examples:**
```bash
# Show optimization tiers
customkb optimize --show-tiers

# Analyze all knowledge bases
customkb optimize --analyze

# Optimize specific KB
customkb optimize myproject

# Container deployment with memory override
customkb optimize --memory-gb 8
```

#### `verify-indexes` - Database Health Check
```bash
customkb verify-indexes <knowledge_base>
```

Verify that all performance-critical indexes exist in the database.

**Expected Indexes:**
- `idx_embedded`: Filters embedded vs non-embedded documents
- `idx_embedded_embedtext`: Speeds up embedded text queries
- `idx_keyphrase_processed`: Enables fast keyphrase searches
- `idx_sourcedoc`: Filters by source document
- `idx_sourcedoc_sid`: Compound queries on source and section

#### `bm25` - Build Hybrid Search Index
```bash
customkb bm25 <knowledge_base> [options]
```

Build BM25 index for keyword-based hybrid search.

**Options:**
- `--force`: Force rebuild existing index

**Requirements:**
- Set `enable_hybrid_search=true` in configuration
- For older databases, run `scripts/upgrade_bm25_tokens.py` first

### Utility Commands

#### `categorize` - Document Categorization
```bash
customkb categorize <knowledge_base> [options]
```

Categorize documents and enable category-based filtering.

**Options:**
- `--import`: Import categories into database schema
- `--force`: Force re-categorization
- `--model MODEL`: Model for categorization
- `--batch-size N`: Documents per batch
- `--categories "CAT1,CAT2"`: Custom categories

#### `edit` - Modify Configuration
```bash
customkb edit <knowledge_base>
```

Open the configuration file in your default editor.

#### `version` - Display Version
```bash
customkb version [--build]
```

Show version information with optional build number.

#### `help` - Show Help
```bash
customkb help
```

Display comprehensive usage information.

## Knowledge Base Architecture

### Directory Structure

All knowledge bases must be organized as subdirectories within the VECTORDBS directory:

```
$VECTORDBS/                          # Default: /var/lib/vectordbs
├── myproject/                       # Knowledge base directory
│   ├── myproject.cfg               # Configuration file (required)
│   ├── myproject.db                # SQLite database
│   ├── myproject.faiss             # FAISS vector index
│   ├── myproject.bm25              # BM25 index (optional)
│   ├── logs/                       # Runtime logs
│   │   └── myproject.log
│   ├── staging.text/               # Processed documents (optional)
│   └── embed_data/                 # Source documents (optional)
```

### Knowledge Base Resolution

The system automatically handles various input formats:

```bash
# All of these resolve to the same KB:
customkb query myproject "search"           # Simple name
customkb query myproject.cfg "search"       # With .cfg extension
customkb query /path/to/myproject "search"  # With path (stripped)

# Result: Uses $VECTORDBS/myproject/myproject.cfg
```

When a KB isn't found, the system provides helpful feedback with available knowledge bases.

## Configuration

CustomKB uses INI-style configuration files with environment variable overrides.

### Configuration Hierarchy

1. **Environment variables** (highest priority)
2. **Configuration file values**
3. **Default values** (lowest priority)

### Configuration Sections

#### [DEFAULT] - Core Settings
```ini
[DEFAULT]
# Embedding model configuration
vector_model = text-embedding-3-small  # Options: ada-002, 3-small, 3-large, gemini-embedding-001
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

# Output formatting
reference_format = xml                 # Options: xml, json, markdown, plain
query_prompt_template = default        # Template style for prompts

# Additional context files (comma-separated)
query_context_files = /path/to/glossary.md,/path/to/reference.txt
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
cache_thread_pool_size = 4             # Cache operation threads
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
reference_batch_size = 20              # References per query batch
```

#### [ALGORITHMS] - Processing Settings
```ini
[ALGORITHMS]
similarity_threshold = 0.6             # Minimum relevance score
low_similarity_scope_factor = 0.5      # Scope reduction factor
max_chunk_overlap = 100                # Token overlap between chunks
heading_search_limit = 200             # Characters to scan for headings

# Hybrid search settings
enable_hybrid_search = true            # Enable BM25 + vector search
bm25_weight = 0.5                     # Balance between BM25 and vector
bm25_max_results = 1000               # Limit BM25 results for memory safety

# Reranking settings
enable_reranking = true                # Enable cross-encoder reranking
reranking_model = cross-encoder/ms-marco-MiniLM-L-6-v2
reranking_top_k = 30                  # Results to rerank
reranking_batch_size = 32             # Batch size for reranking
reranking_device = auto               # Device: auto, cuda, cpu
reranking_cache_size = 10000          # Cache size for reranking

# Categorization settings
enable_categorization = false          # Enable category-based filtering
```

## Model Support

### Language Models (LLMs)

**OpenAI:**
- GPT-4o / GPT-4o-mini (flagship multimodal, 128k context)
- o3 / o3-mini / o3-pro (advanced reasoning models)
- o4-mini (multimodal reasoning with tools)

**Anthropic:**
- Claude 4.0 Opus/Sonnet (advanced coding capabilities, 200k context)
- Claude 3.7 Sonnet (extended thinking capability)
- Claude 3.5 Sonnet/Haiku (balanced performance)

**Google:**
- Gemini 2.5 Pro/Flash/Lite (thinking models)
- Gemini 2.0 Pro/Flash (2M context window)

**xAI:**
- Grok 4.0 / 4.0-heavy (PhD-level reasoning)

**Local Models (via Ollama):**
- Llama 3.1/3.2/3.3 (8B to 70B parameters)
- Gemma 3 (4B/12B/27B)
- Mistral 7B
- Qwen 2.5
- DeepSeek R1
- CodeLlama 13B
- Phi-4 14B

### Embedding Models

**OpenAI:**
- text-embedding-3-large (3072 dimensions, best performance)
- text-embedding-3-small (1536 dimensions, 5x cheaper than ada-002)
- text-embedding-ada-002 (1536 dimensions, legacy)

**Google:**
- gemini-embedding-001 (configurable: 768/1536/3072 dimensions)
  - Superior MTEB benchmark scores (68% vs 64.6% for OpenAI)
  - Supports up to 30k tokens (vs 8k for OpenAI)
  - Matryoshka Representation Learning maintains quality at lower dimensions

### Model Configuration

Use convenient aliases instead of full model names:
```bash
# Using aliases
customkb query myproject "test" -m gpt4o      # → gpt-4o
customkb query myproject "test" -m sonnet     # → claude-sonnet-4-0
customkb query myproject "test" -m gemini2    # → gemini-2.0-flash

# List all models and aliases
customkb query myproject "test" -m list
```

## Performance Optimization

### Memory-Based Optimization Tiers

CustomKB automatically optimizes based on available system memory:

**Low Memory (<16GB):**
- Conservative settings to avoid memory pressure
- Reduced batch sizes and cache limits
- Minimal thread pools
- Hybrid search disabled

**Medium Memory (16-64GB):**
- Balanced performance for most workloads
- Moderate batch sizes and caching
- Reasonable concurrency settings

**High Memory (64-128GB):**
- High performance for production use
- Large batch sizes and extensive caching
- Hybrid search enabled
- Increased thread pools

**Very High Memory (>128GB):**
- Maximum performance for large deployments
- Maximum batch sizes and cache limits
- Full concurrency utilization
- All features enabled

### GPU Acceleration

For NVIDIA GPUs with CUDA support:

```bash
# Check GPU availability
python scripts/benchmark_gpu.py

# Monitor GPU usage during operations
./scripts/gpu_monitor.sh

# Configure for GPU acceleration
[ALGORITHMS]
reranking_device = cuda
reranking_batch_size = 64
```

### Best Practices

1. **Initial Setup:**
   ```bash
   # Analyze system and apply optimizations
   customkb optimize --analyze
   customkb optimize myproject
   
   # Verify database health
   customkb verify-indexes myproject
   ```

2. **Large Datasets (100k+ documents):**
   - Use `embedding_batch_size = 200+`
   - Enable hybrid search for better performance
   - Consider GPU acceleration for reranking

3. **Container Deployments:**
   ```bash
   # Override memory detection
   customkb optimize --memory-gb 8
   ```

4. **Production Deployments:**
   - Enable comprehensive logging
   - Set up regular index verification
   - Monitor embedding cache hit rates
   - Use dedicated vector model API keys

## Advanced Features

### Prompt Templates

Customize how queries are presented to LLMs:

```bash
# Available templates
customkb query myproject "question" --prompt-template <template>
```

**Templates:**
- `default`: Simple format with minimal instructions
- `instructive`: Clear instructions with explicit guidelines
- `scholarly`: Academic style with emphasis on citations
- `concise`: Minimal, direct responses
- `analytical`: Structured analytical approach
- `conversational`: Friendly, conversational tone
- `technical`: Technical depth with precise terminology

### Reference Output Formats

Control how search results are formatted:

```bash
# XML (default)
customkb query myproject "search" --format xml

# JSON for programmatic access
customkb query myproject "search" --format json

# Markdown for human readability
customkb query myproject "search" --format markdown

# Plain text for simplicity
customkb query myproject "search" --format plain
```

### Multi-Language Processing

```bash
# Process documents with specific language
customkb database multilang docs/*.txt --language french

# Auto-detect language per file
customkb database multilang docs/ --detect-language

# Supported languages:
# arabic, chinese, danish, dutch, english, finnish, french,
# german, indonesian, italian, japanese, korean, norwegian,
# portuguese, russian, spanish, swedish, turkish, and more
```

### Category Filtering

```bash
# First, categorize your knowledge base
customkb categorize myproject --import

# Enable categorization in config
[ALGORITHMS]
enable_categorization = true

# Query with category filters
customkb query myproject "query text" --category "Technical"
customkb query myproject "query text" --categories "Technical,Legal,Finance"
```

## Creating a Knowledge Base

### Complete Workflow Example

A step-by-step guide to creating a production-ready knowledge base:

#### 1. Data Preparation

```bash
# Create workspace
mkdir -p ~/workshop/myproject

# Collect and convert source materials
cd ~/workshop/myproject

# Convert PDFs to text
for pdf in *.pdf; do
    pdftotext "$pdf" "${pdf%.pdf}.txt"
done

# Convert HTML to markdown
for html in *.html; do
    pandoc -f html -t markdown "$html" -o "${html%.html}.md"
done
```

#### 2. Data Staging

```bash
# Create staging directory
mkdir -p /var/lib/vectordbs/myproject/staging.text

# Copy processed documents
cp ~/workshop/myproject/*.{md,txt} /var/lib/vectordbs/myproject/staging.text/

# Add metadata headers (optional)
cd /var/lib/vectordbs/myproject/staging.text
for file in *.md; do
    if ! grep -q "^---" "$file"; then
        echo -e "---\nsource: $file\ndate: $(date -r "$file" +%Y-%m-%d)\n---\n" | \
        cat - "$file" > temp && mv temp "$file"
    fi
done
```

#### 3. Configuration

```bash
cat > /var/lib/vectordbs/myproject/myproject.cfg << 'EOF'
[DEFAULT]
# Embedding model configuration
vector_model = text-embedding-3-small
vector_dimensions = 1536

# Text processing
db_min_tokens = 200
db_max_tokens = 400

# Query settings
query_model = gpt-4o-mini
query_max_tokens = 4096
query_top_k = 30
query_temperature = 0.1

# System role
query_role = |
  You are an expert assistant with deep knowledge of the documents in this knowledge base.
  Provide accurate, detailed answers based on the provided context.
  Always cite specific sources when making claims.

[ALGORITHMS]
enable_hybrid_search = true
enable_reranking = true

[PERFORMANCE]
embedding_batch_size = 100
checkpoint_interval = 10
EOF
```

#### 4. Processing Pipeline

```bash
# Import documents
customkb database myproject staging.text/*.md staging.text/*.txt

# Generate embeddings
customkb embed myproject --verbose

# Build search indexes
customkb bm25 myproject
customkb optimize myproject

# Verify indexes
customkb verify-indexes myproject
```

#### 5. Testing

```bash
# Test basic query
customkb query myproject "What are the main topics covered?"

# Test with context only
customkb query myproject "List all document types" --context-only

# Test with different models
customkb query myproject "Summarize the key findings" \
  --model claude-3-5-sonnet-latest
```

## Integration with Dejavu2-CLI

CustomKB integrates seamlessly with [dejavu2-cli](https://github.com/Open-Technology-Foundation/dejavu2-cli) for enhanced AI interactions.

### Using CustomKB as dv2 Knowledge Base

```bash
# Query with dv2 using CustomKB knowledge base
dv2 -k /var/lib/vectordbs/myproject/myproject.cfg "What are the key features?"

# Use pre-configured templates (Agents)
dv2 -T askOkusi "How to establish a PMA company in Indonesia?"
```

### Creating Custom dv2 Agents

Edit Agents.json to create custom templates:
```bash
dv2 --edit-templates
```

Example agent configuration:
```json
{
  "MyExpert": {
    "model": "claude-3-5-sonnet-latest",
    "temperature": 0.2,
    "max_tokens": 8000,
    "knowledge_base": "/var/lib/vectordbs/myproject/myproject.cfg",
    "system_prompt": "You are an expert on my project documentation..."
  }
}
```

## Utility Scripts

CustomKB includes utility scripts in the `scripts/` directory:

### Performance Scripts
- `optimize_kb_performance.py`: Apply memory-based optimization tiers
- `show_optimization_tiers.py`: Display settings for different memory tiers
- `emergency_optimize.py`: Apply conservative settings after crashes
- `performance_analyzer.py`: Analyze KB performance metrics

### GPU Scripts
- `benchmark_gpu.py`: Benchmark GPU vs CPU performance
- `gpu_monitor.sh`: Real-time GPU usage monitoring
- `gpu_env.sh`: GPU environment setup

### Maintenance Scripts
- `rebuild_bm25_filtered.py`: Create filtered BM25 indexes
- `upgrade_bm25_tokens.py`: Upgrade older databases for BM25
- `diagnose_crashes.py`: Diagnose crash issues
- `emergency_cleanup.sh`: Emergency cleanup operations

### Security
- `security-check.sh`: Run security scans on dependencies

## Citations Extraction System

CustomKB includes an advanced citation extraction system in `utils/citations/`:

```bash
# Extract citations from documents
cd utils/citations
./gen-citations.sh /var/lib/vectordbs/myproject/embed_data.text/

# Apply citations as YAML frontmatter
./append-citations.sh /var/lib/vectordbs/myproject/embed_data.text/
```

Features:
- Parallel processing (1-20+ workers)
- AI-powered citation extraction
- SQLite storage for citations
- YAML frontmatter generation
- Comprehensive error handling

## Testing

CustomKB includes a comprehensive test suite:

```bash
# Install test dependencies
pip install -r requirements-test.txt

# Run all tests
python run_tests.py

# Run specific test categories
python run_tests.py --unit          # Unit tests only
python run_tests.py --integration   # Integration tests
python run_tests.py --performance   # Performance benchmarks

# Safe mode with memory limits
python run_tests.py --safe --memory-limit 1024

# With coverage report
python run_tests.py --coverage --html
```

## Troubleshooting

### Common Issues

**"Knowledge base 'name' not found"**
- Ensure KB directory exists in `$VECTORDBS`
- Check VECTORDBS environment variable
- Error message lists available KBs

**"API rate limit exceeded"**
- Adjust `api_call_delay_seconds` in configuration
- Reduce `api_max_concurrency`

**"Out of memory during embedding"**
- Reduce `embedding_batch_size`
- Run `customkb optimize --analyze`

**"Low similarity scores"**
- Check document/query language match
- Consider more powerful embedding model
- Adjust `similarity_threshold`

### Debug Mode

Enable comprehensive logging:
```bash
# Debug mode for commands
customkb query myproject "test query" --debug

# Check logs
tail -f /var/lib/vectordbs/myproject/logs/myproject.log
```

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Check code style
flake8 .
mypy --ignore-missing-imports .
```

## License

CustomKB is released under the MIT License. See [LICENSE](LICENSE) file for details.

## Acknowledgments

CustomKB is built on excellent open-source projects:
- [FAISS](https://github.com/facebookresearch/faiss) - Efficient similarity search
- [LangChain](https://github.com/langchain-ai/langchain) - Text splitting utilities
- [NLTK](https://www.nltk.org/) - Natural language processing
- [spaCy](https://spacy.io/) - Advanced NLP features
- [Sentence Transformers](https://www.sbert.net/) - Cross-encoder reranking

## Support

- **Documentation**: [docs/](docs/) directory
- **Issues**: [GitHub Issues](https://github.com/Open-Technology-Foundation/customkb/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Open-Technology-Foundation/customkb/discussions)

---

*CustomKB is actively maintained by the Open Technology Foundation*