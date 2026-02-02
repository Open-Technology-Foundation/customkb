# CustomKB: Production-Ready AI Knowledgebase System

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Version](https://img.shields.io/badge/version-1.1.0-green.svg)](https://github.com/Open-Technology-Foundation/customkb)
[![Security](https://img.shields.io/badge/security-hardened-brightgreen.svg)](README.md#security)

CustomKB transforms your documents into AI-powered, searchable knowledgebases with state-of-the-art embedding models, vector search, and language models to deliver contextually relevant answers from your data.

## Table of Contents

- [Key Features](#key-features)
- [How It Works](#how-it-works)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Core Commands](#core-commands)
- [Configuration](#configuration)
- [Advanced Features](#advanced-features)
- [Security](#security)
- [Performance Optimization](#performance-optimization)
- [Troubleshooting](#troubleshooting)
- [FAQ](#frequently-asked-questions)
- [Contributing](#contributing)
- [Support & Community](#support--community)
- [License](#license)

## Key Features

### Core Capabilities
- **Semantic Search**: Find information by meaning, not just keywords
- **Multi-Provider AI**: OpenAI, Anthropic, Google, xAI, and local models via Ollama
- **Universal Document Support**: Process Markdown, HTML, code, PDFs, and plain text
- **27+ Language Support**: Multi-language processing with automatic detection
- **Hybrid Search**: Combines vector similarity with BM25 keyword matching
- **Cross-Encoder Reranking**: Boosts accuracy by 20-40% with advanced models
- **Enterprise Security**: Hardened against injection attacks, safe serialization (no pickle), input validation, path protection

### Performance & Scale
- **Memory-Optimized Tiers**: Automatically adapts from 4GB to 128GB+ systems
- **GPU Acceleration**: CUDA support for faster reranking
- **Concurrent Processing**: Batch operations with configurable thread pools
- **Smart Caching**: Two-tier cache system with LRU eviction
- **Production Ready**: Checkpoint saving, automatic retries, graceful error handling

## How It Works

CustomKB follows a three-stage pipeline to transform your documents into an intelligent knowledgebase:

```
1. Document Processing
   ├─ Text extraction (Markdown, HTML, PDF, code, plain text)
   ├─ Language detection (27+ languages)
   ├─ Intelligent chunking (200-400 tokens, context-aware)
   └─ Metadata extraction (filenames, categories, timestamps)

2. Embedding Generation
   ├─ Vector embeddings via OpenAI, Google, or local models
   ├─ Batch processing with checkpoints
   ├─ FAISS index creation for fast similarity search
   └─ Optional BM25 index for hybrid search

3. Semantic Search & Query
   ├─ Query embedding generation
   ├─ Vector similarity search (k-NN via FAISS)
   ├─ Optional: Hybrid search (vector + BM25 keyword matching)
   ├─ Optional: Cross-encoder reranking for precision
   ├─ Context assembly from top results
   └─ LLM response generation with retrieved context
```

**Why This Approach Works:**
- **Semantic Understanding**: Vector embeddings capture meaning, not just keywords
- **Hybrid Accuracy**: Combining vector and keyword search catches both conceptual and exact matches
- **Reranking Precision**: Cross-encoders evaluate query-document pairs for superior relevance
- **Efficient Retrieval**: FAISS enables sub-millisecond search across millions of vectors

## Prerequisites

- **Python**: 3.12 or higher
- **SQLite**: 3.45+ (usually included with Python)
- **RAM**: 4GB+ (8GB+ recommended for optimal performance)
- **GPU** (optional): NVIDIA GPU with CUDA 11 or 12 for acceleration
- **API Keys**: For your chosen AI providers (OpenAI, Anthropic, Google, xAI)

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/Open-Technology-Foundation/customkb.git
cd customkb
```

### 2. Setup Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Install FAISS

```bash
# Automatic installation (detects GPU and CUDA version)
./setup/install_faiss.sh

# Or manual installation:
# CPU-only: pip install -r requirements-faiss-cpu.txt
# GPU (CUDA 12): pip install -r requirements-faiss-gpu-cu12.txt
# GPU (CUDA 11): pip install -r requirements-faiss-gpu-cu11.txt

# Force specific variant:
# FAISS_VARIANT=cpu ./setup/install_faiss.sh
```

### 4. Install NLTK Data

```bash
sudo ./setup/nltk_setup.py download cleanup
```

### 5. Setup Knowledgebase Directory

Choose between system-wide or user-local installation:

**Option A: System-wide (requires sudo)**
```bash
sudo mkdir -p /var/lib/vectordbs
sudo chown $USER:$USER /var/lib/vectordbs
export VECTORDBS="/var/lib/vectordbs"
```

**Option B: User-local (no sudo required, recommended)**
```bash
mkdir -p "$HOME/knowledgebases"
export VECTORDBS="$HOME/knowledgebases"
```

Add to your shell profile (`~/.bashrc`, `~/.zshrc`, etc.):
```bash
export VECTORDBS="$HOME/knowledgebases"  # or /var/lib/vectordbs
```

### 6. Configure API Keys

```bash
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
export GOOGLE_API_KEY="your-google-key"      # Optional
export XAI_API_KEY="your-xai-key"            # Optional
```

Add these to your shell profile for persistence.

## Quick Start

### Create Your First Knowledgebase

```bash
# 1. Create knowledgebase directory
mkdir -p "$VECTORDBS/myproject"

# 2. Create configuration
cat > "$VECTORDBS/myproject/myproject.cfg" << 'EOF'
[DEFAULT]
vector_model = text-embedding-3-small
query_model = gpt-4o-mini
db_min_tokens = 200
db_max_tokens = 400
EOF

# 3. Process documents (from your project directory)
customkb database myproject docs/*.md *.txt

# 4. Generate embeddings
customkb embed myproject

# 5. Query your knowledgebase
customkb query myproject "What are the main features?"
```

**That's it!** Your knowledgebase is ready to answer questions about your documents.

## Core Commands

### `database` - Import Documents

```bash
customkb database <kb_name> [files...] [options]
```

Process and store text files in the knowledgebase.

**Options:**
- `-l, --language`: Stopwords language (en, fr, de, etc.)
- `--detect-language`: Auto-detect language per file
- `-f, --force`: Reprocess existing files
- `-v, --verbose`: Detailed output

**Examples:**
```bash
# Process all markdown files
customkb database myproject ~/docs/**/*.md

# Auto-detect language for multilingual docs
customkb database myproject ~/docs/ --detect-language

# Force reprocess existing files
customkb database myproject ~/docs/*.md --force
```

### `embed` - Generate Embeddings

```bash
customkb embed <kb_name> [options]
```

Create vector embeddings for all text chunks.

**Options:**
- `-r, --reset-database`: Reset embedding status
- `-v, --verbose`: Show progress

**Examples:**
```bash
# Generate embeddings with progress
customkb embed myproject --verbose

# Reset and regenerate all embeddings
customkb embed myproject --reset-database
```

### `query` - Search & Ask Questions

```bash
customkb query <kb_name> "<question>" [options]
```

Perform semantic search and generate AI responses.

**Options:**
- `-c, --context-only`: Return only context, no AI response
- `-m, --model`: AI model to use
- `-k, --top-k`: Number of results (default: 50)
- `-t, --temperature`: Response creativity (0-2)
- `-f, --format`: Output format (xml, json, markdown, plain)
- `-p, --prompt-template`: Response style template

**Examples:**
```bash
# Simple query
customkb query myproject "How does authentication work?"

# Advanced query with specific model
customkb query myproject "Explain the architecture" \
  --model claude-sonnet-4-5 \
  --format json \
  --prompt-template technical

# Get context only (no LLM response)
customkb query myproject "Find authentication docs" --context-only
```

## Configuration

CustomKB uses INI-style configuration with environment variable overrides.

### Priority Order

1. **Environment variables** (highest)
2. Configuration file (`.cfg`)
3. Default values (lowest)

### Example Configuration

```ini
[DEFAULT]
# Models
vector_model = text-embedding-3-small
query_model = gpt-4o-mini

# Text Processing
db_min_tokens = 200          # Minimum chunk size
db_max_tokens = 400          # Maximum chunk size

# Query Settings
query_max_tokens = 4096      # Max tokens in LLM response
query_top_k = 30             # Number of chunks to retrieve
query_temperature = 0.1      # LLM creativity (0=precise, 2=creative)
query_role = You are a helpful expert assistant.

# Output Format
reference_format = json      # xml, json, markdown, plain
query_prompt_template = technical  # Response style

[ALGORITHMS]
# Search Configuration
similarity_threshold = 0.6   # Minimum similarity score (0-1)
enable_hybrid_search = true  # Combine vector + keyword search
bm25_weight = 0.5           # Weight for BM25 in hybrid mode
bm25_max_results = 1000     # Max results from BM25

# Reranking
enable_reranking = true      # Use cross-encoder for precision
reranking_model = cross-encoder/ms-marco-MiniLM-L-6-v2
reranking_top_k = 30         # Rerank top N results

[PERFORMANCE]
# Optimization
embedding_batch_size = 100   # Chunks per batch
cache_thread_pool_size = 4   # Concurrent cache operations
memory_cache_size = 10000    # LRU cache entries
checkpoint_interval = 10      # Save every N batches

[API]
# Rate Limiting
api_call_delay_seconds = 0.05  # Delay between API calls
api_max_concurrency = 8        # Parallel API requests
api_max_retries = 20           # Retry attempts for failed calls
```

### Configuration Tips

- **`db_min_tokens`/`db_max_tokens`**: Controls chunk size. Smaller = more precise, larger = more context
- **`similarity_threshold`**: Lower (0.5) for broader results, higher (0.7) for strict relevance
- **`enable_hybrid_search`**: Enable for technical docs, disable for narrative content
- **`query_temperature`**: 0.0-0.3 for factual, 0.7-1.0 for creative responses

## Advanced Features

### Supported Models

#### Language Models (LLMs)

**OpenAI**
- GPT-5.x series (5, 5-mini, 5-nano, 5-pro, 5.1, 5.2)
- GPT-4.1, GPT-4.1-mini, GPT-4.1-nano (1M context)
- GPT-4o, GPT-4o-mini (128k context)
- o3, o4-mini (advanced reasoning)

**Anthropic**
- Claude Opus 4.5 (200k context, extended thinking)
- Claude Sonnet 4.5, Haiku 4.5 (200k context)
- Claude Opus 4.1 (200k context)

**Google**
- Gemini 3.x Pro/Flash (preview)
- Gemini 2.5 Pro/Flash/Flash-Lite (thinking models, 1M+ context)
- Gemini 1.5 Pro/Flash-8B

**xAI**
- Grok 4, Grok 4-fast (256k-2M context, reasoning)

**Local (Ollama)**
- Llama 3.3 (8B-70B)
- Gemma 3 (4B-27B)
- DeepSeek R1, Qwen 2.5, Mistral, Phi-4

#### Embedding Models

**OpenAI**
- `text-embedding-3-large` (3072 dims, best quality)
- `text-embedding-3-small` (1536 dims, cost-effective)
- `text-embedding-ada-002` (1536 dims, legacy)

**Google**
- `gemini-embedding-001` (768/1536/3072 dims)
  - 68% MTEB score vs 64.6% for OpenAI
  - 30k token context vs 8k
  - Matryoshka Representation Learning

### Prompt Templates

Customize response styles:

```bash
customkb query myproject "question" --prompt-template <template>
```

Available templates:
- `default`: Balanced, helpful responses
- `instructive`: Step-by-step explanations
- `scholarly`: Academic, citation-rich
- `concise`: Brief, to-the-point
- `analytical`: Deep analysis with reasoning
- `conversational`: Friendly, approachable
- `technical`: Precise, developer-focused

### Output Formats

Control how results are formatted:

```bash
# JSON for APIs
customkb query myproject "search" --format json

# XML with structured references
customkb query myproject "search" --format xml

# Markdown for documentation
customkb query myproject "search" --format markdown

# Plain text
customkb query myproject "search" --format plain
```

### Category Filtering

Organize and filter results by categories:

```bash
# Categorize documents
customkb categorize myproject --import

# Query with category filters
customkb query myproject "query" --categories "Technical,Legal"
```

### Multi-Language Support

```bash
# Process with specific language
customkb database myproject docs/*.txt --language french

# Auto-detect languages (recommended for multilingual docs)
customkb database myproject docs/ --detect-language
```

Supported languages: English, French, German, Spanish, Italian, Portuguese, Dutch, Swedish, Norwegian, Danish, Finnish, Russian, Turkish, Arabic, Hebrew, Japanese, Chinese, Korean, and more.

## Security

CustomKB implements enterprise-grade security measures to protect your data and systems.

### Security Features

**Safe Serialization**
- ✓ Zero pickle deserialization vulnerabilities
- ✓ JSON format for reranking cache (human-readable, secure)
- ✓ JSON format for categorization checkpoints
- ✓ NPZ + JSON hybrid for BM25 indexes (efficient + secure)
- ✓ Automatic migration from legacy pickle formats

**Injection Prevention**
- ✓ SQL injection protection via table name validation
- ✓ Path traversal protection in file operations
- ✓ Input validation for all user-provided parameters
- ✓ Parameterized queries for database operations

**API Security**
- ✓ API key validation and secure storage
- ✓ Environment variable-based configuration
- ✓ No API keys in logs or error messages
- ✓ Secure credential handling

**Data Protection**
- ✓ Database integrity checks
- ✓ Atomic operations with rollback support
- ✓ Backup support for critical operations
- ✓ File permission validation

### Security Best Practices

When deploying CustomKB in production:

1. **API Keys**: Store in environment variables, never in code or config files
2. **File Permissions**: Restrict knowledgebase directories to application user only
3. **Network Access**: Run on localhost or behind authentication proxy
4. **Updates**: Regularly check for security patches
5. **Backups**: Enable automatic backups before migrations

### Reporting Security Issues

If you discover a security vulnerability:

1. **Do not** create a public GitHub issue
2. Email security concerns to: [Create issue for security contact]
3. Include:
   - Steps to reproduce
   - Potential impact assessment
   - Suggested remediation (if any)
4. Allow reasonable time for patching before public disclosure

See commit history for detailed security update information.

## Performance Optimization

### Auto-Optimization

```bash
# Analyze system and show recommendations
customkb optimize --analyze

# Apply optimizations automatically
customkb optimize myproject

# Preview changes without applying
customkb optimize myproject --dry-run
```

### Memory Tiers

CustomKB automatically configures based on available memory:

| Memory | Tier | Features | Batch Size | Cache Size |
|--------|------|----------|------------|------------|
| <16GB | Low | Conservative, no hybrid search | 50 | 5,000 |
| 16-64GB | Medium | Balanced, moderate caching | 100 | 10,000 |
| 64-128GB | High | Large batches, hybrid search | 200 | 20,000 |
| >128GB | Very High | Maximum performance | 300 | 50,000 |

### Database Indexes

```bash
# Verify performance indexes
customkb verify-indexes myproject

# Build BM25 hybrid search index
customkb bm25 myproject
```

### GPU Acceleration

CustomKB automatically detects and uses NVIDIA GPUs for:
- Cross-encoder reranking (20-40% faster)
- FAISS index search (GPU-enabled builds)

```bash
# Benchmark GPU vs CPU performance
./scripts/benchmark_gpu.py

# Monitor GPU usage during operations
./scripts/gpu_monitor.sh
```

## Troubleshooting

### Common Issues

**"Knowledgebase not found"**
```bash
# Verify KB exists
ls -la $VECTORDBS/

# Check for .cfg file
ls -la $VECTORDBS/myproject/myproject.cfg

# Error message shows available KBs
customkb query nonexistent "test"
```

**"API rate limit exceeded"**
```ini
# Increase delay between calls in config
api_call_delay_seconds = 0.1
api_max_concurrency = 4
```

**"Out of memory during embedding"**
```bash
# Run optimizer for your system
customkb optimize myproject

# Or manually reduce batch size in config
embedding_batch_size = 50
```

**"Low similarity scores" or poor results**
```ini
# Try lower threshold
similarity_threshold = 0.5

# Enable hybrid search
enable_hybrid_search = true

# Or use stronger embedding model
vector_model = text-embedding-3-large
```

**"Import failed: unsupported file type"**
```bash
# CustomKB supports: .md, .txt, .html, .pdf
# Convert other formats to supported types first

# For code files, use .txt extension or markdown fenced blocks
```

### Debug Mode

```bash
# Enable verbose logging
customkb query myproject "test" -v

# Check detailed logs
tail -f $VECTORDBS/myproject/logs/myproject.log

# Run diagnostics
./scripts/diagnose_crashes.py myproject
```

## Knowledgebase Structure

All knowledgebases live in `$VECTORDBS`:

```
$VECTORDBS/
├── myproject/
│   ├── myproject.cfg       # Configuration (required)
│   ├── myproject.db        # SQLite database with chunks
│   ├── myproject.faiss     # FAISS vector index
│   ├── myproject.bm25      # BM25 index (optional, for hybrid search)
│   ├── .rerank_cache/      # Reranking cache (optional)
│   └── logs/               # Runtime logs
```

### Name Resolution

The system intelligently resolves KB names:

```bash
# All resolve to the same KB:
customkb query myproject "test"
customkb query myproject.cfg "test"
customkb query $VECTORDBS/myproject "test"
customkb query $VECTORDBS/myproject/myproject.cfg "test"
# → All use $VECTORDBS/myproject/myproject.cfg
```

## Utility Scripts

Located in `scripts/` directory:

### Performance & Optimization
- `show_optimization_tiers.py` - Display memory tier settings
- `emergency_optimize.py` - Conservative recovery settings
- `clean_corrupted_cache.py` - Clean corrupted cache files

### GPU
- `benchmark_gpu.py` - Compare GPU vs CPU performance
- `gpu_monitor.sh` - Real-time GPU utilization monitoring
- `gpu_env.sh` - GPU environment setup

### Maintenance
- `rebuild_bm25_filtered.py` - Rebuild BM25 indexes with filters
- `upgrade_bm25_tokens.py` - Upgrade database for BM25 tokens
- `diagnose_crashes.py` - Analyze crash logs and system state
- `update_dependencies.py` - Check and update Python dependencies
- `security-check.sh` - Run security validation checks

## Testing

```bash
# Install test dependencies
pip install -r requirements-test.txt

# Run all tests
python run_tests.py

# Run specific test suites
python run_tests.py --unit         # Unit tests only
python run_tests.py --integration  # Integration tests only

# Run with safety limits (recommended for CI)
python run_tests.py --safe --memory-limit 2048

# Generate coverage report
python run_tests.py --coverage
```

## Frequently Asked Questions

### General

**Q: Can I use CustomKB without any API keys?**

A: Yes! Use local Ollama models for both embeddings and queries. No external API calls required. Performance depends on your local hardware.

**Q: How much does it cost to process documents?**

A: Costs vary by provider and model:
- OpenAI `text-embedding-3-small`: $0.02 per 1M tokens (~750k words)
- Google `gemini-embedding-001`: $0.15 per 1M tokens
- Local Ollama models: Free (just electricity)

Example: A 500-page technical manual (~250k tokens) costs about $0.005 to embed with OpenAI.

**Q: Is my data private and secure?**

A: Your documents stay local. Only text chunks are sent to API providers during embedding and query operations. The full document contents never leave your system. For maximum privacy, use local Ollama models.

**Q: What's the difference between CustomKB and vector databases like Pinecone?**

A: CustomKB is a complete RAG (Retrieval-Augmented Generation) system including:
- Document processing pipeline
- Embedding generation
- Vector + hybrid search
- LLM integration
- Response generation

Vector databases only handle storage and retrieval. You'd need to build the rest yourself.

### Technical

**Q: Can I use multiple embedding models in one knowledgebase?**

A: No, each knowledgebase uses one embedding model. To switch models, create a new KB or regenerate embeddings with `--reset-database`.

**Q: How do I update my knowledgebase when documents change?**

A: Re-run the database command with updated files:
```bash
customkb database myproject docs/*.md --force
customkb embed myproject
```

Only changed/new files are reprocessed.

**Q: What's the maximum knowledgebase size?**

A: Tested up to 10M+ chunks (~4GB database). FAISS scales to billions of vectors. Practical limits depend on your RAM and disk space.

**Q: Can I run CustomKB in a Docker container?**

A: Yes, though no official Docker image yet. Use a Python 3.12+ base image and install dependencies. Mount your `$VECTORDBS` directory as a volume.

**Q: Does CustomKB support real-time document monitoring?**

A: Not yet. You manually trigger document processing. Consider using filesystem watchers (inotify) to trigger updates automatically.

## Contributing

We welcome contributions from the community! Whether you're fixing bugs, adding features, improving documentation, or sharing ideas, your help makes CustomKB better for everyone.

### Ways to Contribute

- **Report Bugs**: [Open an issue](https://github.com/Open-Technology-Foundation/customkb/issues/new?template=bug_report.md)
- **Suggest Features**: [Open an issue](https://github.com/Open-Technology-Foundation/customkb/issues/new?template=feature_request.md)
- **Improve Documentation**: Fix typos, clarify instructions, add examples
- **Submit Code**: Bug fixes, new features, performance improvements
- **Share Knowledge**: Answer questions, write tutorials, create examples

### Quick Start for Contributors

1. **Fork the repository** on GitHub

2. **Clone your fork**
   ```bash
   git clone https://github.com/YOUR-USERNAME/customkb.git
   cd customkb
   ```

3. **Create a feature branch**
   ```bash
   git checkout -b feature/amazing-feature
   ```

4. **Set up development environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   pip install -r requirements-test.txt
   ```

5. **Make your changes**
   - Write clean, documented code
   - Follow existing code style
   - Add tests for new features
   - Update documentation as needed

6. **Run tests**
   ```bash
   python run_tests.py
   python run_tests.py --coverage
   ```

7. **Commit your changes**
   ```bash
   git add .
   git commit -m "Add amazing feature"
   ```

8. **Push to your fork**
   ```bash
   git push origin feature/amazing-feature
   ```

9. **Open a Pull Request**
   - Go to the original repository
   - Click "New Pull Request"
   - Select your branch
   - Describe your changes clearly

### Development Guidelines

- **Code Style**: Follow PEP 8 for Python code
- **Type Hints**: Use type annotations for function signatures
- **Testing**: Maintain or improve test coverage
- **Documentation**: Update README and docstrings
- **Commits**: Write clear, descriptive commit messages

### Code of Conduct

- Be respectful and inclusive
- Welcome newcomers and different perspectives
- Focus on what's best for the community
- Show empathy towards others

### Need Help?

- Join discussions in [GitHub Discussions](https://github.com/Open-Technology-Foundation/customkb/discussions)
- Ask questions in issues (label with `question`)
- Review existing PRs to see the process

## Support & Community

### Get Help

- **Documentation**: You're reading it! Check the sections above
- **Issues**: [GitHub Issues](https://github.com/Open-Technology-Foundation/customkb/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Open-Technology-Foundation/customkb/discussions)

### Stay Updated

- **Releases**: Watch the repository for release notifications
- **Changelog**: See commit history for version details
- **Security**: Check [Security](#security) section for vulnerability reporting

### Connect

- **GitHub**: [Open-Technology-Foundation/customkb](https://github.com/Open-Technology-Foundation/customkb)
- **Maintainer**: Indonesian Open Technology Foundation
- **License**: GPL-3.0 (see [LICENSE](LICENSE))

## Complete Example

### Building a Production Knowledgebase

Here's a complete workflow for creating a production-ready knowledgebase:

```bash
# 1. Setup environment
export VECTORDBS="$HOME/knowledgebases"
export OPENAI_API_KEY="your-key-here"

# 2. Create KB directory
mkdir -p "$VECTORDBS/techbase"
cd "$VECTORDBS/techbase"

# 3. Create optimized configuration
cat > techbase.cfg << 'EOF'
[DEFAULT]
vector_model = text-embedding-3-small
query_model = gpt-4o-mini
db_min_tokens = 250
db_max_tokens = 500

[ALGORITHMS]
enable_hybrid_search = true
enable_reranking = true
similarity_threshold = 0.65
bm25_weight = 0.5

[PERFORMANCE]
embedding_batch_size = 100
memory_cache_size = 20000
checkpoint_interval = 10
EOF

# 4. Process documents with language detection
customkb database techbase ~/docs/**/*.md --detect-language --verbose

# 5. Generate embeddings with progress
customkb embed techbase --verbose

# 6. Build hybrid search index
customkb bm25 techbase

# 7. Optimize for your system
customkb optimize techbase

# 8. Verify everything is set up correctly
customkb verify-indexes techbase

# 9. Test with sample queries
customkb query techbase "What are the best practices?" \
  --prompt-template technical \
  --format markdown

# 10. Test context-only retrieval
customkb query techbase "authentication implementation" \
  --context-only \
  --top-k 10
```

## Quick Reference

### Environment Variables

```bash
OPENAI_API_KEY       # OpenAI API key
ANTHROPIC_API_KEY    # Anthropic API key
GOOGLE_API_KEY       # Google/Gemini API key
XAI_API_KEY          # xAI API key
VECTORDBS            # Knowledgebase base directory
NLTK_DATA            # NLTK data location (optional)
```

### Model Aliases

```bash
# Embedding models
text-embedding-3-small   → OpenAI small (1536 dims)
text-embedding-3-large   → OpenAI large (3072 dims)
gemini-embedding-001     → Google Gemini (configurable dims)

# LLM models (examples)
gpt-5-mini               → OpenAI GPT-5 Mini (latest)
gpt-4o-mini              → OpenAI GPT-4o Mini (cost-effective)
claude-opus-4-5          → Anthropic Claude Opus 4.5
claude-sonnet-4-5        → Anthropic Claude Sonnet 4.5
gemini-2.5-flash         → Google Gemini 2.5 Flash
grok-4                   → xAI Grok 4
```

### Performance Tips

- **Large datasets**: Increase `embedding_batch_size` up to system limits
- **Technical content**: Enable `enable_hybrid_search = true`
- **GPU available**: Install FAISS GPU variant for 2-4x speedup
- **Low memory**: Run `customkb optimize` to adjust for your system
- **Better accuracy**: Enable reranking, lower similarity threshold
- **Faster queries**: Increase cache size, disable reranking for speed

## License

GPL-3.0 License - see [LICENSE](LICENSE) file for details.

**Copyright © 2025 Indonesian Open Technology Foundation**

---

**Actively maintained by the [Indonesian Open Technology Foundation](https://github.com/Open-Technology-Foundation)**

