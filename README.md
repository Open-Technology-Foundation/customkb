# CustomKB: Production-Ready AI Knowledgebase System

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Version](https://img.shields.io/badge/version-0.8.0-green.svg)](https://github.com/Open-Technology-Foundation/customkb)

CustomKB transforms your documents into AI-powered, searchable knowledgebases with state-of-the-art embedding models, vector search, and language models to deliver contextually relevant answers from your data.

## Key Features

### Core Capabilities
- **Semantic Search**: Find information by meaning, not just keywords
- **Multi-Provider AI**: OpenAI, Anthropic, Google, xAI, and local models via Ollama
- **Universal Document Support**: Process Markdown, HTML, code, PDFs, and plain text
- **27+ Language Support**: Multi-language processing with automatic detection
- **Hybrid Search**: Combines vector similarity with BM25 keyword matching
- **Cross-Encoder Reranking**: Boosts accuracy by 20-40% with advanced models
- **Enterprise Security**: Input validation, path protection, API key security

### Performance & Scale
- **Memory-Optimized Tiers**: Automatically adapts from 4GB to 128GB+ systems
- **GPU Acceleration**: CUDA support for faster reranking
- **Concurrent Processing**: Batch operations with configurable thread pools
- **Smart Caching**: Two-tier cache system with LRU eviction
- **Production Ready**: Checkpoint saving, automatic retries, graceful error handling

## Prerequisites

- Python 3.12 or higher
- SQLite 3.45+
- 4GB+ RAM (8GB+ recommended)
- NVIDIA GPU with CUDA (optional, for acceleration)
- API keys for chosen providers

## Installation

### 1. Clone Repository
```bash
git clone https://github.com/Open-Technology-Foundation/customkb.git
cd customkb
```

### 2. Setup Environment
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Install NLTK Data
```bash
sudo ./setup/nltk_setup.py download cleanup
```

### 4. Configure API Keys
```bash
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
export GOOGLE_API_KEY="your-google-key"      # Optional
export XAI_API_KEY="your-xai-key"            # Optional
export VECTORDBS="/var/lib/vectordbs"        # KB storage location
```

## Quick Start

### Create Your First Knowledgebase

```bash
# 1. Create knowledgebase directory
mkdir -p /var/lib/vectordbs/myproject

# 2. Create configuration
cat > /var/lib/vectordbs/myproject/myproject.cfg << 'EOF'
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

# 5. Query your knowledgebase
customkb query myproject "What are the main features?"
```

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

**Example:**
```bash
customkb database myproject ~/docs/**/*.md --detect-language
```

### `embed` - Generate Embeddings
```bash
customkb embed <kb_name> [options]
```
Create vector embeddings for all text chunks.

**Options:**
- `-r, --reset-database`: Reset embedding status
- `-v, --verbose`: Show progress

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

# Advanced query with options
customkb query myproject "Explain the architecture" \
  --model claude-3-5-sonnet-latest \
  --format json \
  --prompt-template technical
```

## Performance Optimization

### Auto-Optimization
```bash
# Analyze and show recommendations
customkb optimize --analyze

# Apply optimizations
customkb optimize myproject

# Preview changes
customkb optimize myproject --dry-run
```

### Memory Tiers
CustomKB automatically configures based on available memory:

| Memory | Tier | Features |
|--------|------|----------|
| <16GB | Low | Conservative settings, no hybrid search |
| 16-64GB | Medium | Balanced performance, moderate caching |
| 64-128GB | High | Large batches, hybrid search enabled |
| >128GB | Very High | Maximum performance, all features |

### Database Indexes
```bash
# Verify performance indexes
customkb verify-indexes myproject

# Build BM25 hybrid search index
customkb bm25 myproject
```

## Supported Models

### Language Models (LLMs)

**OpenAI**
- GPT-4o, GPT-4o-mini (128k context)
- o3, o3-mini, o3-pro (reasoning models)
- o4-mini (multimodal reasoning)

**Anthropic**
- Claude 4.0 Opus/Sonnet (200k context)
- Claude 3.7 Sonnet (extended thinking)
- Claude 3.5 Sonnet/Haiku

**Google**
- Gemini 2.5 Pro/Flash/Lite (thinking models)
- Gemini 2.0 Pro/Flash (2M context)

**xAI**
- Grok 4.0, Grok 4.0-heavy (PhD-level reasoning)

**Local (Ollama)**
- Llama 3.3 (8B-70B)
- Gemma 3 (4B-27B)
- DeepSeek R1
- Qwen 2.5, Mistral, Phi-4

### Embedding Models

**OpenAI**
- text-embedding-3-large (3072 dims, best quality)
- text-embedding-3-small (1536 dims, cost-effective)
- text-embedding-ada-002 (1536 dims, legacy)

**Google**
- gemini-embedding-001 (768/1536/3072 dims)
  - 68% MTEB score vs 64.6% for OpenAI
  - 30k token context vs 8k
  - Matryoshka Representation Learning

## Configuration

CustomKB uses INI-style configuration with environment variable overrides.

### Priority Order
1. Environment variables (highest)
2. Configuration file
3. Default values (lowest)

### Example Configuration
```ini
[DEFAULT]
# Models
vector_model = text-embedding-3-small
query_model = gpt-4o-mini

# Text Processing
db_min_tokens = 200
db_max_tokens = 400

# Query Settings
query_max_tokens = 4096
query_top_k = 30
query_temperature = 0.1
query_role = You are a helpful expert assistant.

# Output Format
reference_format = json  # xml, json, markdown, plain
query_prompt_template = technical  # default, scholarly, concise, etc.

[ALGORITHMS]
# Search Configuration
similarity_threshold = 0.6
enable_hybrid_search = true
bm25_weight = 0.5
bm25_max_results = 1000

# Reranking
enable_reranking = true
reranking_model = cross-encoder/ms-marco-MiniLM-L-6-v2
reranking_top_k = 30

[PERFORMANCE]
# Optimization
embedding_batch_size = 100
cache_thread_pool_size = 4
memory_cache_size = 10000
checkpoint_interval = 10

[API]
# Rate Limiting
api_call_delay_seconds = 0.05
api_max_concurrency = 8
api_max_retries = 20
```

## Advanced Features

### Prompt Templates
Customize response styles:
```bash
customkb query myproject "question" --prompt-template <template>
```
Templates: `default`, `instructive`, `scholarly`, `concise`, `analytical`, `conversational`, `technical`

### Output Formats
Control how results are formatted:
```bash
# JSON for APIs
customkb query myproject "search" --format json

# Markdown for documentation
customkb query myproject "search" --format markdown
```

### Category Filtering
Filter results by categories:
```bash
# Categorize documents
customkb categorize myproject --import

# Query with filters
customkb query myproject "query" --categories "Technical,Legal"
```

### Multi-Language Support
```bash
# Process with specific language
customkb database myproject docs/*.txt --language french

# Auto-detect languages
customkb database myproject docs/ --detect-language
```

## Knowledgebase Structure

All knowledgebases live in `$VECTORDBS` (default: `/var/lib/vectordbs`):

```
/var/lib/vectordbs/
├── myproject/
│   ├── myproject.cfg       # Configuration (required)
│   ├── myproject.db        # SQLite database
│   ├── myproject.faiss     # Vector index
│   ├── myproject.bm25      # BM25 index (optional)
│   └── logs/               # Runtime logs
```

### Name Resolution
The system intelligently resolves KB names:
```bash
# All resolve to the same KB:
customkb query myproject "test"
customkb query myproject.cfg "test"
customkb query /path/to/myproject "test"
# → Uses /var/lib/vectordbs/myproject/myproject.cfg
```

## Utility Scripts

Located in `scripts/`:

### Performance
- `optimize_kb_performance.py` - Apply memory tiers
- `performance_analyzer.py` - Analyze metrics
- `emergency_optimize.py` - Conservative recovery

### GPU
- `benchmark_gpu.py` - GPU vs CPU benchmarks
- `gpu_monitor.sh` - Real-time monitoring

### Maintenance
- `rebuild_bm25_filtered.py` - Filtered BM25 indexes
- `upgrade_bm25_tokens.py` - Database upgrades
- `diagnose_crashes.py` - Crash diagnostics

## Testing

```bash
# Install test dependencies
pip install -r requirements-test.txt

# Run tests
python run_tests.py              # All tests
python run_tests.py --unit       # Unit only
python run_tests.py --safe       # With memory limits
python run_tests.py --coverage   # Coverage report
```

## Troubleshooting

### Common Issues

**"Knowledgebase not found"**
- Verify KB exists in `$VECTORDBS`
- Check error message for available KBs

**"API rate limit"**
- Increase `api_call_delay_seconds`
- Reduce `api_max_concurrency`

**"Out of memory"**
- Run `customkb optimize --analyze`
- Reduce `embedding_batch_size`

**"Low similarity scores"**
- Check language match
- Try stronger embedding model
- Adjust `similarity_threshold`

### Debug Mode
```bash
# Enable debug logging
customkb query myproject "test" --debug

# Check logs
tail -f /var/lib/vectordbs/myproject/logs/myproject.log
```

## Integration

### With Dejavu2-CLI
```bash
# Use as dv2 knowledgebase
dv2 -k /var/lib/vectordbs/myproject/myproject.cfg "question"

# Create custom agents
dv2 --edit-templates
```

### Production Deployment
```bash
# From development server
yes | ./push-to-okusi 3 -N

# Container deployment
customkb optimize --memory-gb 8  # Override for containers
```

## Complete Example

### Building a Production Knowledgebase

```bash
# 1. Prepare data
mkdir -p /var/lib/vectordbs/techbase
cd /var/lib/vectordbs/techbase

# 2. Create configuration
cat > techbase.cfg << 'EOF'
[DEFAULT]
vector_model = gemini-embedding-001
embedding_dimensions = 1536
query_model = claude-3-5-sonnet-latest
db_min_tokens = 250
db_max_tokens = 500

[ALGORITHMS]
enable_hybrid_search = true
enable_reranking = true
similarity_threshold = 0.65

[PERFORMANCE]
embedding_batch_size = 150
memory_cache_size = 20000
EOF

# 3. Process documents
customkb database techbase ~/docs/**/*.md --detect-language

# 4. Generate embeddings
customkb embed techbase --verbose

# 5. Build indexes
customkb bm25 techbase
customkb optimize techbase

# 6. Verify setup
customkb verify-indexes techbase

# 7. Test queries
customkb query techbase "What are the best practices?" \
  --prompt-template technical \
  --format markdown
```

## Quick Reference

### Environment Variables
```bash
OPENAI_API_KEY       # OpenAI API key
ANTHROPIC_API_KEY    # Anthropic API key
GOOGLE_API_KEY       # Google/Gemini API key
XAI_API_KEY          # xAI API key
VECTORDBS            # KB base directory
NLTK_DATA            # NLTK data location
```

### Command Aliases
```bash
# Model shortcuts
-m gpt4o     → gpt-4o
-m sonnet    → claude-3-5-sonnet-latest
-m gemini2   → gemini-2.0-flash
-m list      → Show all models
```

### Performance Tips
- Use batch processing for large datasets
- Enable hybrid search for technical content
- Configure GPU acceleration when available
- Monitor cache hit rates in logs
- Run optimize after major changes

## License

GPL-3.0 License - see [LICENSE](LICENSE) file.

*Actively maintained by the Indonesian Open Technology Foundation*
