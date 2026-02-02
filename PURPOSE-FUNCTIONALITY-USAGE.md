# CustomKB - Purpose, Functionality & Usage

## Purpose

**CustomKB** is a production-ready, AI-powered knowledgebase system (v1.0.0) that transforms documents into searchable, intelligent systems with semantic understanding.

### What Problem Does It Solve?

CustomKB addresses the challenge of creating meaningful, context-aware search over document collections. Unlike traditional keyword search, it:

- **Understands meaning** - Semantic search finds relevant content even when exact words differ
- **Generates AI responses** - Returns synthesized answers, not just document fragments
- **Scales efficiently** - From 4GB laptops to 128GB+ servers with automatic optimization
- **Supports multiple providers** - OpenAI, Anthropic, Google, xAI, and local Ollama models

### Who Is It For?

- Developers building knowledge-intensive applications
- Teams needing searchable documentation systems
- Researchers managing large document collections
- Anyone requiring RAG (Retrieval-Augmented Generation) capabilities

---

## Functionality

### Core Features

| Feature | Description |
|---------|-------------|
| **Document Processing** | Ingests Markdown, HTML, PDF, code, and text files with intelligent chunking |
| **Vector Embeddings** | Creates semantic representations using OpenAI or Google embedding models |
| **Hybrid Search** | Combines vector similarity + BM25 keyword matching for optimal results |
| **Cross-Encoder Reranking** | Optional neural reranking for 20-40% accuracy improvement |
| **Multi-Model Responses** | Generates answers using GPT-4, Claude, Gemini, Grok, or local models |
| **Language Detection** | Supports 27+ languages with automatic detection |
| **Memory Optimization** | Auto-configures based on available system resources |

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         CustomKB                                │
├─────────────────────────────────────────────────────────────────┤
│  config/           │  Configuration management, KB settings    │
│  database/         │  Document ingestion, chunking, SQLite     │
│  embedding/        │  Vector generation, FAISS indexing, BM25  │
│  query/            │  Search, reranking, AI response generation│
│  utils/            │  Logging, security, optimization, GPU     │
│  models/           │  Model specifications and aliases         │
└─────────────────────────────────────────────────────────────────┘
```

### Data Pipeline

```
INPUT (documents)
    ↓
DATABASE Stage
├── Language detection
├── Format-specific extraction
├── Intelligent chunking (200-400 tokens)
├── Duplicate detection
└── SQLite storage

    ↓
EMBEDDING Stage
├── Batch vector generation
├── Checkpoint recovery
├── FAISS index creation
└── Optional BM25 tokenization

    ↓
QUERY Stage
├── Query enhancement
├── Vector + hybrid search
├── Optional reranking
└── LLM response generation
```

### Supported Models

**Embedding Models:**
- OpenAI: `text-embedding-3-small` (1536 dims), `text-embedding-3-large` (3072 dims)
- Google: `gemini-embedding-001`

**Language Models:**
- OpenAI: GPT-4o, GPT-4o-mini, o3, o4-mini
- Anthropic: Claude Sonnet 4.5, Haiku 4.5, Opus 4.1
- Google: Gemini 2.5 Pro/Flash/Lite
- xAI: Grok 4.0
- Local: Ollama (Llama, Gemma, DeepSeek, Mistral)

---

## Usage

### Quick Start

```bash
# Set environment
export VECTORDBS=/path/to/knowledgebases
export OPENAI_API_KEY=your-key

# Create and configure a knowledgebase
customkb edit myproject

# Import documents
customkb database myproject ~/docs/*.md

# Generate embeddings
customkb embed myproject

# Query the knowledgebase
customkb query myproject "What are the main features?"
```

### Commands Reference

| Command | Description | Example |
|---------|-------------|---------|
| `database` | Import and chunk documents | `customkb database myproject docs/*.md` |
| `embed` | Generate vector embeddings | `customkb embed myproject --verbose` |
| `query` | Search with AI response | `customkb query myproject "question"` |
| `edit` | Edit KB configuration | `customkb edit myproject` |
| `optimize` | Auto-tune performance | `customkb optimize myproject` |
| `verify-indexes` | Check index health | `customkb verify-indexes myproject` |
| `bm25` | Build hybrid search index | `customkb bm25 myproject` |
| `convert-encoding` | Convert files to UTF-8 | `customkb convert-encoding *.txt` |
| `version` | Show version | `customkb version` |

### Query Options

```bash
customkb query myproject "question" \
  --model claude-sonnet-4-5 \      # LLM model
  --format json \                   # Output: json, xml, markdown, plain
  --prompt-template technical \     # Style: default, technical, scholarly, concise
  --top-k 30 \                      # Number of results
  --temperature 0.1 \               # Response creativity
  --context-only                    # Return context without AI response
```

### Common Workflows

**1. Standard Knowledge Base Setup**
```bash
customkb database techbase ~/docs/**/*.md
customkb embed techbase
customkb bm25 techbase
customkb optimize techbase
```

**2. Production Deployment**
```bash
customkb database prodkb /data/docs/ --detect-language --verbose
customkb embed prodkb --verbose
customkb bm25 prodkb
customkb optimize prodkb
customkb verify-indexes prodkb
```

**3. Query with Different Models**
```bash
# OpenAI
customkb query myproject "explain this" --model gpt-4o

# Anthropic Claude
customkb query myproject "explain this" --model claude-sonnet-4-5

# Google Gemini
customkb query myproject "explain this" --model gemini-2.5-pro

# Local Ollama
customkb query myproject "explain this" --model llama3.2
```

### Configuration

Configuration files use INI format with these sections:

```ini
[DEFAULT]
vector_model = text-embedding-3-small
query_model = gpt-4o-mini
db_min_tokens = 200
db_max_tokens = 400

[ALGORITHMS]
enable_hybrid_search = true
enable_reranking = true
similarity_threshold = 0.6
bm25_weight = 0.5

[PERFORMANCE]
embedding_batch_size = 100
memory_cache_size = 10000

[API]
api_call_delay_seconds = 0.05
api_max_concurrency = 8
```

### Environment Variables

| Variable | Purpose |
|----------|---------|
| `VECTORDBS` | Knowledgebase directory (required) |
| `OPENAI_API_KEY` | OpenAI API access |
| `ANTHROPIC_API_KEY` | Anthropic Claude access |
| `GOOGLE_API_KEY` | Google Gemini access |
| `GROK_API_KEY` | xAI Grok access |
| `EDITOR` | Default editor for `edit` command |

### Knowledgebase Directory Structure

```
$VECTORDBS/
└── myproject/
    ├── myproject.cfg       # Configuration
    ├── myproject.db        # SQLite database
    ├── myproject.faiss     # Vector index
    ├── myproject.bm25      # BM25 index (optional)
    └── logs/               # Runtime logs
```

---

## Dependencies

**Core Requirements:**
- Python 3.12+
- FAISS (CPU or GPU)
- SQLite 3

**Key Libraries:**
- `sentence_transformers` - Embedding models
- `torch` - PyTorch for ML
- `anthropic`, `openai` - API clients
- `rank_bm25` - Keyword search
- `langchain_text_splitters` - Document chunking
- `numpy<2` - Required for FAISS compatibility

**Installation:**
```bash
pip install -r requirements.txt
./setup/install_faiss.sh  # Auto-detects GPU
```

---

## Performance

### Memory Tiers (Auto-configured)

| System Memory | Batch Size | Cache Size |
|---------------|-----------|-----------|
| <16GB | 50 | 5,000 |
| 16-64GB | 100 | 10,000 |
| 64-128GB | 200 | 20,000 |
| >128GB | 300 | 50,000 |

### FAISS Index Selection

| Document Count | Index Type | Characteristics |
|----------------|-----------|-----------------|
| <10k | Flat | Exact search |
| 10k-100k | IVF | Approximate, fast |
| >100k | HNSW | Fastest approximate |

---

## Testing

```bash
# Run all tests
python run_tests.py

# Unit tests only
python run_tests.py --unit

# Safe mode with memory limits
python run_tests.py --safe --memory-limit 1024

# With coverage
python run_tests.py --coverage
```

---

## Version

- **Version:** 1.0.0
- **Release Date:** 2025-11-07
- **Python:** 3.12+
- **License:** See LICENSE file
