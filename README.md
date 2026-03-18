# CustomKB: AI-Powered Knowledgebase System

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Version](https://img.shields.io/badge/version-1.1.0-green.svg)](https://github.com/Open-Technology-Foundation/customkb)
[![Security](https://img.shields.io/badge/security-hardened-brightgreen.svg)](README.md#security)

CustomKB transforms documents into AI-powered, searchable knowledgebases using vector embeddings, FAISS indexing, and LLM integration. It supports multiple AI providers (OpenAI, Anthropic, Google, xAI, Ollama) and delivers contextually relevant answers from your data.

## Table of Contents

- [How It Works](#how-it-works)
- [Key Features](#key-features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Core Commands](#core-commands)
- [Configuration](#configuration)
- [Architecture](#architecture)
- [Advanced Features](#advanced-features)
- [Security](#security)
- [Performance Optimization](#performance-optimization)
- [MCP Server Integration](#mcp-server-integration)
- [Troubleshooting](#troubleshooting)
- [FAQ](#frequently-asked-questions)
- [Contributing](#contributing)
- [License](#license)

## How It Works

CustomKB follows a three-stage pipeline:

```
1. Document Processing (customkb database)
   ├─ Text extraction (Markdown, HTML, PDF, code, plain text)
   ├─ Language detection (27+ languages)
   ├─ Intelligent chunking (configurable token ranges, context-aware)
   └─ Metadata extraction (filenames, categories, timestamps)

2. Embedding Generation (customkb embed)
   ├─ Vector embeddings via OpenAI, Google, or local models
   ├─ Batch processing with checkpoints and retry logic
   ├─ FAISS index creation (auto-selects Flat/IVF/HNSW by dataset size)
   └─ Two-tier caching (memory LRU + disk)

3. Semantic Query (customkb query)
   ├─ Query embedding + optional enhancement (spelling, synonyms)
   ├─ Vector similarity search (k-NN via FAISS)
   ├─ Optional hybrid search (vector + BM25 keyword matching)
   ├─ Optional cross-encoder reranking for precision
   ├─ Context assembly from top results
   └─ LLM response generation with retrieved context
```

## Key Features

| Category | Features |
|----------|----------|
| **Search** | Semantic search by meaning; hybrid vector + BM25 keyword search; cross-encoder reranking (+20-40% accuracy) |
| **AI Providers** | OpenAI, Anthropic, Google, xAI, and local models via Ollama |
| **Documents** | Markdown, HTML, PDF, code files, plain text; 27+ languages with auto-detection |
| **Performance** | Auto-tuning memory tiers (4GB–128GB+); GPU acceleration (CUDA 11/12); concurrent batch processing; two-tier LRU cache |
| **Security** | Zero pickle (JSON-only serialization); SQL injection protection; path traversal prevention; parameterized queries; no API keys in logs |
| **Reliability** | Checkpoint saving; exponential backoff with retry; graceful error handling; atomic DB operations |

## Prerequisites

- **Python** 3.12+
- **SQLite** 3.45+ (included with Python)
- **RAM** 4GB+ (8GB+ recommended)
- **GPU** (optional): NVIDIA with CUDA 11 or 12
- **API Keys**: For chosen AI providers (OpenAI, Anthropic, Google, xAI) — or use local Ollama models for zero API dependency

## Installation

### 1. Clone and Install

```bash
git clone https://github.com/Open-Technology-Foundation/customkb.git
cd customkb

# Install uv if not already available
# See: https://docs.astral.sh/uv/getting-started/installation/

# Install dependencies (auto-creates .venv)
uv sync --extra faiss-gpu-cu12 --extra mcp --extra test

# CPU-only (no NVIDIA GPU):
# uv sync --extra faiss-cpu --extra mcp --extra test

# Or auto-detect FAISS variant:
# ./setup/install_faiss.sh
```

### 2. Install NLTK Data

```bash
sudo ./setup/nltk_setup.py download cleanup
```

### 3. Setup Knowledgebase Directory

```bash
# System-wide (requires sudo)
sudo mkdir -p /var/lib/vectordbs && sudo chown $USER:$USER /var/lib/vectordbs
export VECTORDBS="/var/lib/vectordbs"

# Or user-local (no sudo)
mkdir -p "$HOME/knowledgebases"
export VECTORDBS="$HOME/knowledgebases"
```

Add to your shell profile (`~/.bashrc`):

```bash
export VECTORDBS="/var/lib/vectordbs"  # or $HOME/knowledgebases
```

### 4. Configure API Keys

```bash
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
export GOOGLE_API_KEY="your-google-key"      # Optional
export XAI_API_KEY="your-xai-key"            # Optional
```

## Quick Start

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

# 3. Process documents
customkb database myproject docs/*.md *.txt

# 4. Generate embeddings
customkb embed myproject

# 5. Query
customkb query myproject "What are the main features?"
```

## Core Commands

### `database` — Import Documents

```bash
customkb database <kb_name> [files...] [options]
```

| Option | Description |
|--------|-------------|
| `-l, --language` | Stopwords language (en, fr, de, etc.) |
| `--detect-language` | Auto-detect language per file |
| `-f, --force` | Reprocess existing files |
| `-v / -q` | Verbose / quiet output |

### `embed` — Generate Embeddings

```bash
customkb embed <kb_name> [options]
```

| Option | Description |
|--------|-------------|
| `-r, --reset-database` | Reset embedding status and regenerate all |
| `-v / -q` | Verbose / quiet output |

### `query` — Search & Ask Questions

```bash
customkb query <kb_name> "<question>" [options]
```

| Option | Description |
|--------|-------------|
| `-c, --context-only` | Return context without AI response |
| `-m, --model` | AI model to use |
| `-k, --top-k` | Number of results (default: 50) |
| `-s, --context-scope` | Context segments per result |
| `-t, --temperature` | Response creativity (0.0–2.0) |
| `-f, --format` | Output format: xml, json, markdown, plain |
| `-p, --prompt-template` | Style: default, instructive, scholarly, concise, analytical, conversational, technical |
| `--categories` | Filter by categories (comma-separated) |
| `--context-files` | Additional context files to include |

### `categorize` — AI-Powered Categorization

```bash
customkb categorize <kb_name> [options]
```

| Option | Description |
|--------|-------------|
| `-S, --sample N` | Process only N sample articles |
| `-f, --full` | Process all articles |
| `--fresh` | Ignore checkpoint, reprocess all |
| `--import` | Import categories to database |
| `--list` | List existing categories and counts |
| `-m, --model` | AI model (default: `claude-haiku-4-5`) |
| `-s, --sampling T-M-B` | Chunk sampling (e.g., `5-10-5`) |
| `-c, --confidence-threshold` | Minimum confidence (default: 0.5) |
| `-D, --no-dedup` | Disable category deduplication |
| `--dedup-threshold` | Similarity threshold for dedup (default: 85.0) |

### Other Commands

| Command | Description |
|---------|-------------|
| `customkb edit <kb_name>` | Open KB config in `$EDITOR` |
| `customkb optimize <kb_name>` | Auto-optimize performance settings |
| `customkb optimize --analyze` | Show system recommendations |
| `customkb verify-indexes <kb_name>` | Check database index health |
| `customkb bm25 <kb_name>` | Build BM25 hybrid search index |
| `customkb convert-encoding <files>` | Convert files to UTF-8 |
| `customkb version` | Show version information |

## Configuration

CustomKB uses INI-style configuration with environment variable overrides.

**Priority order**: Environment variables > `.cfg` file > defaults.

### Example Configuration

```ini
[DEFAULT]
# Models
vector_model = text-embedding-3-small
query_model = claude-sonnet-4-6
# vector_dimensions — leave commented out for dynamic detection.
# Set explicitly for Matryoshka models (e.g., 1536 for gemini-embedding-001).

# Text Processing
db_min_tokens = 200          # Minimum chunk size
db_max_tokens = 400          # Maximum chunk size

# Query Settings
query_max_tokens = 4000      # Max tokens in LLM response
query_top_k = 50             # Chunks to retrieve
query_temperature = 0.0      # 0=precise, 2=creative
query_role = You are a helpful assistant.

# Output
reference_format = xml       # xml, json, markdown, plain
query_prompt_template = default

[ALGORITHMS]
similarity_threshold = 0.6   # Minimum similarity (0–1)
enable_hybrid_search = true  # Vector + keyword search
bm25_weight = 0.3            # BM25 weight in hybrid mode
bm25_max_results = 1000
enable_reranking = true
reranking_model = cross-encoder/ms-marco-MiniLM-L-6-v2
reranking_top_k = 20

[PERFORMANCE]
embedding_batch_size = 100
cache_thread_pool_size = 4
memory_cache_size = 10000
checkpoint_interval = 10

[API]
api_call_delay_seconds = 0.05
api_max_concurrency = 8
api_max_retries = 20
```

### Configuration Tips

- **`vector_dimensions`**: Leave unset for dynamic detection (probes the model on first embed). Set explicitly for Matryoshka models like `gemini-embedding-001` to select output dimensions (768/1536/3072). After the first embed, the actual dimensions are auto-synced back to the `.cfg` file.
- **`db_min_tokens`/`db_max_tokens`**: Smaller chunks = more precise retrieval; larger = more context per result
- **`similarity_threshold`**: Lower (0.5) for broader recall, higher (0.7) for strict relevance
- **`enable_hybrid_search`**: Recommended for technical documentation
- **`query_temperature`**: 0.0 (default) for factual answers, 0.7–1.0 for creative responses

## Architecture

```
customkb                     # Bash wrapper (activates .venv, calls customkb.py)
customkb.py                  # CLI entry point, match/case command dispatch
├── config/
│   ├── config_manager.py    # KnowledgeBase class, config loading, name resolution
│   └── models.py            # Pydantic config models with env var overrides
├── categorize/
│   ├── categorize_manager.py    # LLM-based document categorization
│   ├── category_deduplicator.py # Fuzzy deduplication via rapidfuzz
│   └── import_to_db.py          # Category import pipeline to SQLite
├── database/
│   ├── db_manager.py        # Document processing pipeline
│   ├── connection.py        # SQLite connection with WAL + PRAGMA tuning
│   ├── chunking.py          # Text chunking with overlap (langchain splitters)
│   ├── index_manager.py     # Database index creation and verification
│   └── migrations.py        # Schema migrations with version tracking
├── embedding/
│   ├── embed_manager.py     # Embedding orchestration, FAISS index management
│   ├── providers.py         # Embedding provider abstraction (OpenAI, Google, local)
│   ├── litellm_provider.py  # LiteLLM unified embedding interface
│   ├── bm25_manager.py      # BM25 index for hybrid search
│   ├── rerank_manager.py    # Cross-encoder reranking with score caching
│   ├── cache.py             # Thread-safe two-tier cache (memory LRU + disk)
│   ├── batch.py             # Batch progress tracking with ETA
│   └── index.py             # FAISS index type auto-selection
├── query/
│   ├── query_manager.py     # Re-export hub for all query submodules
│   ├── processing.py        # Query orchestration pipeline
│   ├── search.py            # FAISS vector search, hybrid search, result assembly
│   ├── response.py          # LLM response generation (multi-provider)
│   ├── llm.py               # Unified LLM interface via LiteLLM
│   ├── embedding.py         # Query embedding generation and caching
│   ├── enhancement.py       # Spelling correction, synonym expansion
│   ├── formatters.py        # Output formatting (XML, JSON, Markdown, plain)
│   └── prompt_templates.py  # Response style templates
├── models/
│   └── model_manager.py     # Model registry from Models.json (aliases, providers)
├── mcp_server/
│   └── server.py            # MCP server for Claude Code integration
└── utils/
    ├── security_utils.py    # Input validation, path sanitization, safe SQL
    ├── text_utils.py        # Text cleaning, entity preservation
    ├── logging_config.py    # Centralized KB-specific logging
    ├── logging_utils.py     # Logging utility helpers
    ├── optimization_manager.py  # Memory tier auto-optimization
    ├── performance_analyzer.py  # System profiling and recommendations
    ├── exceptions.py        # Custom exception hierarchy
    ├── faiss_loader.py      # FAISS loading with GPU/CPU fallback
    ├── gpu_utils.py         # GPU detection and memory management
    ├── resource_manager.py  # Thread pool and resource lifecycle
    ├── language_detector.py # Language detection with caching
    ├── encoding_converter.py # File encoding conversion to UTF-8
    ├── enums.py             # Shared enumerations
    └── context_managers.py  # Database and resource context managers
```

### Data Flow

1. **Database**: Files → text extraction → language-aware chunking → SQLite storage with metadata
2. **Embed**: Chunks → vector embeddings (via LiteLLM) → FAISS index (auto-selected type) → two-tier cache
3. **Query**: Question → embedding → FAISS search → optional BM25 hybrid → optional reranking → context assembly → LLM response

### Storage Structure

```
$VECTORDBS/<kb_name>/
├── <kb_name>.cfg      # Configuration (required)
├── <kb_name>.db       # SQLite database (chunks, metadata, categories)
├── <kb_name>.faiss    # FAISS vector index
└── <kb_name>.bm25     # BM25 index (optional, for hybrid search)
```

### Design Principles

- **Lazy imports**: Heavy modules loaded only when their command is invoked (fast CLI startup)
- **Config-driven**: All thresholds, batch sizes, and behaviors configurable via `.cfg` with env var overrides
- **Provider-agnostic**: LiteLLM abstraction layer for embeddings and LLM calls
- **Safe by default**: JSON serialization only, parameterized SQL, validated file paths

## Advanced Features

### Supported Models

#### Embedding Models

| Model | Provider | Dimensions | Notes |
|-------|----------|-----------|-------|
| `text-embedding-3-small` | OpenAI | 1536 | Cost-effective default |
| `text-embedding-3-large` | OpenAI | 3072 | Best quality |
| `text-embedding-ada-002` | OpenAI | 1536 | Legacy |
| `gemini-embedding-001` | Google | 768/1536/3072 | 30k token context, Matryoshka dimensions via `vector_dimensions` config |

#### LLM Models (via LiteLLM)

| Provider | Models |
|----------|--------|
| **OpenAI** | GPT-5.x, GPT-4.1/mini/nano, GPT-4o/mini, o3/o3-pro, o4-mini, Codex |
| **Anthropic** | Claude Opus 4.6, Sonnet 4.6, Haiku 4.5 |
| **Google** | Gemini 3.x Pro, 2.5 Pro/Flash, 2.0 Flash, 1.5 Pro/Flash |
| **xAI** | Grok 4, Grok 4-fast, Grok 4-fast-non-reasoning |
| **Ollama** | Llama 3.3, Gemma 3, DeepSeek R1, Qwen 2.5, Mistral, Phi-4 |

Model aliases are resolved via `Models.json`. Run `customkb query <kb> --model <alias>` to use any supported model.

### Prompt Templates

```bash
customkb query myproject "question" --prompt-template <template>
```

Available: `default`, `instructive`, `scholarly`, `concise`, `analytical`, `conversational`, `technical`

### Output Formats

```bash
customkb query myproject "search" --format json     # Structured API output
customkb query myproject "search" --format xml      # XML with references
customkb query myproject "search" --format markdown # Documentation-friendly
customkb query myproject "search" --format plain    # Plain text
```

### KB Name Resolution

All of these resolve to the same knowledgebase:

```bash
customkb query myproject "test"
customkb query myproject.cfg "test"
customkb query $VECTORDBS/myproject "test"
customkb query $VECTORDBS/myproject/myproject.cfg "test"
```

## Security

### Implemented Protections

- **Safe serialization**: JSON-only format for all caches, checkpoints, and indexes (zero pickle)
- **SQL injection prevention**: Parameterized queries and table name validation throughout
- **Path traversal protection**: All file paths validated before access
- **API key security**: Environment variable storage, never logged or exposed in errors
- **Atomic operations**: Database writes with rollback support
- **Input validation**: All user-provided parameters sanitized at system boundaries

### Best Practices

1. Store API keys in environment variables, never in config files
2. Restrict `$VECTORDBS` directory permissions to the application user/group
3. Run behind authentication proxy for network-exposed deployments
4. Use local Ollama models for maximum data privacy
5. For multi-user setups, ensure `$VECTORDBS` and cache directories use setgid permissions (`chmod 2775`) so all users in the group can read/write caches

## Performance Optimization

### Auto-Optimization

```bash
customkb optimize myproject           # Apply optimizations
customkb optimize --analyze           # Show recommendations
customkb optimize myproject --dry-run # Preview changes
```

### Memory Tiers

CustomKB auto-configures based on available system memory:

| Memory | Tier | Batch Size | Cache Size | Hybrid Search |
|--------|------|-----------|------------|---------------|
| <16GB | Low | 50 | 5,000 | Disabled |
| 16–64GB | Medium | 100 | 10,000 | Available |
| 64–128GB | High | 200 | 20,000 | Enabled |
| >128GB | Very High | 300 | 50,000 | Enabled |

### GPU Acceleration

CUDA-enabled NVIDIA GPUs accelerate cross-encoder reranking (20-40% faster) and FAISS index operations:

```bash
uv sync --extra faiss-gpu-cu12   # Install GPU FAISS
./scripts/benchmark_gpu.py       # Benchmark GPU vs CPU
```

### Database Indexes

```bash
customkb verify-indexes myproject  # Check index health
customkb bm25 myproject           # Build hybrid search index
```

## MCP Server Integration

CustomKB includes a Model Context Protocol server for integration with Claude Code and other MCP-compatible tools:

```bash
uv sync --extra mcp  # Install MCP dependencies
```

The MCP server exposes knowledgebase search as tools, allowing AI assistants to query your knowledgebases directly. See `mcp_server/server.py` for configuration.

## Troubleshooting

### Common Issues

**"Knowledgebase not found"**
```bash
ls -la $VECTORDBS/                              # Verify KB exists
ls -la $VECTORDBS/myproject/myproject.cfg       # Check config file
```

**"API rate limit exceeded"** — Rate limit errors are automatically retried with exponential backoff (up to `api_max_retries` attempts). To reduce rate limit pressure, adjust:
```ini
api_call_delay_seconds = 0.1   # Delay between batches (seconds)
api_max_concurrency = 4        # Concurrent batch tasks (lower = less pressure)
api_max_retries = 20           # Max retry attempts per batch
```

**"Out of memory during embedding"**
```bash
customkb optimize myproject  # Auto-adjust for your system
# Or manually: embedding_batch_size = 50
```

**"AssertionError" or "inhomogeneous shape" during embed/query** — Dimension mismatch between cached embeddings and FAISS index. Clear caches and rebuild:
```bash
# Clear embedding caches for the model
find $VECTORDBS/.embedding_cache -name "gemini-embedding-001_*" -delete
rm -rf $VECTORDBS/.query_embedding_cache
# Delete FAISS index and re-embed
rm -f $VECTORDBS/<kb>/<kb>.faiss
customkb embed <kb>
```

**Low similarity scores** — Adjust search parameters:
```ini
similarity_threshold = 0.5
enable_hybrid_search = true
# Or use stronger model: vector_model = text-embedding-3-large
```

### Debug Mode

```bash
customkb query myproject "test" -v -d   # Verbose + debug
./scripts/diagnose_crashes.py myproject # Run diagnostics
```

## Utility Scripts

| Script | Purpose |
|--------|---------|
| `scripts/benchmark_gpu.py` | GPU vs CPU performance comparison |
| `scripts/benchmark_vectordb.py` | Vector database benchmarking |
| `scripts/show_optimization_tiers.py` | Display memory tier settings |
| `scripts/emergency_optimize.py` | Conservative recovery settings |
| `scripts/clean_corrupted_cache.py` | Clean corrupted cache files |
| `scripts/rebuild_bm25_filtered.py` | Rebuild BM25 indexes with filters |
| `scripts/upgrade_bm25_tokens.py` | Upgrade database for BM25 tokens |
| `scripts/diagnose_crashes.py` | Analyze crash logs and system state |
| `scripts/security-check.sh` | Run security validation checks |
| `scripts/emergency_cleanup.sh` | Emergency cleanup of stale resources |
| `scripts/gpu_env.sh` | GPU environment variable setup |
| `scripts/gpu_monitor.sh` | Real-time GPU monitoring |
| `scripts/test_cuda.sh` | CUDA installation verification |

## Testing

```bash
uv sync --extra test              # Install test dependencies

python run_tests.py               # All tests
python run_tests.py --unit        # Unit tests only
python run_tests.py --integration # Integration tests only
python run_tests.py --safe --memory-limit 2048  # With safety limits
python run_tests.py --coverage    # Generate coverage report
```

Test markers: `@pytest.mark.unit`, `@pytest.mark.integration`, `@pytest.mark.slow`, `@pytest.mark.requires_api`, `@pytest.mark.performance`, `@pytest.mark.requires_data`, `@pytest.mark.resource_intensive`

## Frequently Asked Questions

**Can I use CustomKB without API keys?**
Yes. Use local Ollama models for both embeddings and queries. No external API calls required.

**How much does embedding cost?**
OpenAI `text-embedding-3-small`: ~$0.02/1M tokens. A 500-page manual (~250k tokens) costs ~$0.005. Ollama is free.

**Is my data private?**
Documents stay local. Only text chunks are sent to API providers during embedding/query. For maximum privacy, use Ollama.

**Can I use multiple embedding models in one KB?**
No. Each KB uses one embedding model. To switch, create a new KB or regenerate with `--reset-database`.

**How do I update a KB when documents change?**
```bash
customkb database myproject docs/*.md --force
customkb embed myproject
```

**Maximum KB size?**
Tested to 10M+ chunks (~4GB database). FAISS scales to billions of vectors.

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `VECTORDBS` | Knowledgebase base directory (default: `/var/lib/vectordbs`) |
| `OPENAI_API_KEY` | OpenAI API key |
| `ANTHROPIC_API_KEY` | Anthropic API key |
| `GOOGLE_API_KEY` | Google/Gemini API key |
| `XAI_API_KEY` | xAI API key |
| `NLTK_DATA` | NLTK data location (optional) |

## Contributing

1. Fork and clone the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Set up dev environment: `uv sync --extra test --extra dev`
4. Make changes, add tests, run `python run_tests.py`
5. Submit a pull request

See [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) for detailed development guidelines.

## License

GPL-3.0 License — see [LICENSE](LICENSE) file.

**Copyright © 2025-2026 Indonesian Open Technology Foundation**

---

**Maintained by the [Indonesian Open Technology Foundation](https://github.com/Open-Technology-Foundation)**
