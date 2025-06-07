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

### Advanced Vector Search
- **High-Performance Storage**: SQLite for structured data + FAISS for vector indices
- **Adaptive Indexing**: Automatically selects optimal index type based on dataset size
- **Semantic Search**: Find relevant content based on meaning, not just keywords
- **Relevance Scoring**: Context-aware ranking with configurable similarity thresholds

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

1. **Create a configuration file** (`myproject.cfg`):
   ```ini
   [DEFAULT]
   vector_model = text-embedding-3-small
   query_model = gpt-4o-mini
   db_min_tokens = 200
   db_max_tokens = 400
   ```

2. **Process your documents**:
   ```bash
   customkb database myproject.cfg docs/*.md *.txt
   ```

3. **Generate embeddings**:
   ```bash
   customkb embed myproject.cfg
   ```

4. **Query your knowledge base**:
   ```bash
   customkb query myproject.cfg "How do I configure the vector model?"
   ```

> **💡 Flexible Path Handling**: CustomKB supports multiple ways to specify knowledge base locations:
> - **Direct config path**: `/path/to/project.cfg`
> - **Relative traversal**: `../sibling-project/config.cfg`
> - **KB name search**: `myproject` (searches `$VECTORDBS`)

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
customkb database myproject.cfg ~/documents/**/*.md -l english
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
customkb embed myproject.cfg --verbose
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
# Simple query with config file
customkb query myproject.cfg "What are the main features?"

# Absolute path to knowledge base
customkb query /var/lib/vectordbs/docs/docs.cfg "How to install?"

# Relative path with directory traversal
customkb query ../other-project/config.cfg "What's new?"

# KB name search (searches $VECTORDBS directory)
customkb query myproject "What are the main features?"

# Context-only search
customkb query myproject.cfg "authentication" --context-only

# Custom model and parameters
customkb query myproject.cfg "Explain the architecture" \
  --model claude-3-5-sonnet-20241022 \
  --temperature 0.7 \
  --max-tokens 2000
```

#### `edit` - Modify Configuration
```bash
customkb edit <config>
```
Opens the configuration file in your default editor.

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

#### Configuration Path Flexibility

CustomKB supports three different approaches for organizing and accessing knowledge bases:

**1. Absolute Paths** - Direct file system access:
```bash
# Config file anywhere in the filesystem
customkb query /home/user/projects/docs/docs.cfg "search terms"
customkb query /var/lib/vectordbs/company/hr.cfg "policy questions"

# KB files created in same directory as config:
# /home/user/projects/docs/docs.db
# /home/user/projects/docs/docs.faiss
# /home/user/projects/docs/logs/
```

**2. Relative Paths with Directory Traversal** - Navigate between related KBs:
```bash
# From /var/lib/vectordbs/project-a/ directory
customkb query ../project-b/config.cfg "cross-project search"
customkb query ../../archives/old-docs.cfg "historical data"
```

**3. Knowledge Base Name Search** - Convenient VECTORDBS lookup:
```bash
# Searches $VECTORDBS/myproject/myproject.cfg
customkb query myproject "search terms"

# Set custom search location
export VECTORDBS="/custom/kb/location"
customkb query docs "search terms"  # Looks in /custom/kb/location/docs/docs.cfg
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

# Query any KB from any location
customkb query company-docs "HR policies"
customkb query ../api-reference/api-reference.cfg "authentication"
customkb query /var/lib/vectordbs/customer-support/customer-support.cfg "tickets"
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
```

#### For Fast Response Times
```ini
[DEFAULT]
vector_model = text-embedding-3-small
query_top_k = 20
query_model = gpt-4o-mini

[PERFORMANCE]
query_cache_ttl_days = 30
```

## 🔧 Advanced Usage

### Domain-Style Knowledge Bases
```bash
# Create knowledge base with domain naming
customkb database example.com.cfg ~/example-docs/*.md

# The system will create:
# /var/lib/vectordbs/example.com.db
# /var/lib/vectordbs/example.com.faiss
```

### Multi-Language Processing
```bash
# Process French documents
customkb database multilang.cfg docs/*.txt --language french

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

def query_knowledge_base(config_file, query_text):
    """Query CustomKB from Python code."""
    result = subprocess.run(
        ['customkb', 'query', config_file, query_text, '--context-only'],
        capture_output=True,
        text=True
    )
    return result.stdout

# Example usage
context = query_knowledge_base('myproject.cfg', 'How to install?')
print(context)
```

#### Shell Pipeline
```bash
# Extract all Python code examples
customkb query myproject.cfg "Python code examples" --context-only | \
  grep -A 5 -B 5 "```python"
```

## 🛠️ Troubleshooting

### Common Issues

**"Configuration file not found"**
- Ensure the .cfg file exists and has correct permissions
- Check if VECTORDBS environment variable is set correctly

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