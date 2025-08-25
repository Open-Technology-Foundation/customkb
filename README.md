# CustomKB: AI-Powered Knowledgebase System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Version](https://img.shields.io/badge/version-0.8.0-green.svg)](https://github.com/Open-Technology-Foundation/customkb)

CustomKB is a production-ready knowledgebase system that transforms your document collections into AI-powered, searchable knowledge repositories. It supports multiple embedding providers (OpenAI, Google) and combines state-of-the-art models with efficient vector search to deliver contextually relevant answers from your data.

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
  - OpenAI GPT (4.1, 4o, 4o-mini, o3 series)
  - Anthropic Claude (3.5, 3.7, 4.0 Opus/Sonnet)
  - Google Gemini (2.0, 2.5 Flash/Pro)
  - xAI Grok (4.0)
  - Meta Llama (via Ollama integration)
  - Local models via Ollama (llama3.1, gemma3, mistral, qwen2.5)
- **Context Management**: XML/JSON/Markdown formatted reference contexts for precise prompting
- **Customizable Behavior**: Fine-tune temperature, token limits, and system roles
- **Prompt Templates**: Pre-configured prompt styles (instructive, scholarly, technical, conversational)

### Enterprise-Ready Features
- **Security First**: Input validation, path traversal prevention, API key protection
- **Performance Optimized**: Batch processing, caching, concurrent API calls
- **Resilient Design**: Checkpoint saving, automatic retries, graceful error handling
- **Comprehensive Logging**: Per-knowledge-base logs with performance metrics
- **GPU Acceleration**: Optional CUDA support for reranking models

## 🚀 Quick Start

### Prerequisites

- Python 3.12 or higher
- SQLite 3.45+
- API keys for your chosen embedding/LLM providers
- 4GB+ RAM recommended for large datasets
- NVIDIA GPU with CUDA (optional, for accelerated reranking)

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

### Basic Usage Example

1. **Create a knowledgebase directory**:
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

5. **Query your knowledgebase**:
   ```bash
   customkb query myproject "How do I configure the vector model?"
   ```

## 📖 Detailed Documentation

### Command Reference

```bash
customkb <command> <knowledge_base> [options]
```

#### Core Commands

##### `database` - Import Documents
```bash
customkb database <knowledge_base> [files...] [options]
```
Processes text files and stores them in the knowledgebase.

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

##### `embed` - Generate Embeddings
```bash
customkb embed <knowledge_base> [options]
```
Creates vector embeddings for all text chunks in the database.

**Options:**
- `-r, --reset-database`: Reset embedding status flags
- `-v, --verbose`: Show progress information

**Example:**
```bash
customkb embed myproject --verbose
```

##### `query` - Search Knowledgebase
```bash
customkb query <knowledge_base> "<query>" [options]
```
Performs semantic search and generates AI responses.

**Options:**
- `-c, --context-only`: Return only context without AI response
- `-m, --model MODEL`: AI model to use (e.g., gpt-4.1, claude-3-5-sonnet)
- `-k, --top-k N`: Number of search results to return
- `-s, --context-scope N`: Context segments per result
- `-t, --temperature T`: Model temperature (0.0-2.0)
- `-M, --max-tokens N`: Maximum response tokens
- `-R, --role "ROLE"`: Custom system role
- `-f, --format FORMAT`: Output format: xml, json, markdown, plain
- `-p, --prompt-template TEMPLATE`: Prompt style: default, instructive, scholarly, concise, analytical, conversational, technical

**Examples:**
```bash
# Simple query
customkb query myproject "What are the main features?"

# Context-only search with custom parameters
customkb query myproject "authentication" -c -k 10

# Custom model with specific format
customkb query myproject "Explain the architecture" \
  --model claude-3-5-sonnet-latest \
  --temperature 0.7 \
  --format json \
  --prompt-template technical
```

##### `optimize` - Performance Optimization
```bash
customkb optimize [knowledge_base] [options]
```
Automatically optimizes knowledgebase performance based on system resources.

**Options:**
- `--dry-run`: Preview changes without applying
- `--analyze`: Analyze and show recommendations
- `--show-tiers`: Display all memory tier settings
- `--memory-gb N`: Override detected memory (GB)

**Examples:**
```bash
# Show optimization tiers
customkb optimize --show-tiers

# Analyze all knowledgebases
customkb optimize --analyze

# Optimize specific KB
customkb optimize myproject

# Container deployment with memory override
customkb optimize --memory-gb 8
```

##### `verify-indexes` - Check Database Health
```bash
customkb verify-indexes <knowledge_base>
```
Verifies that all performance-critical indexes exist in the database.

**Expected Indexes:**
- `idx_embedded`: Filters embedded vs non-embedded documents
- `idx_embedded_embedtext`: Speeds up embedded text queries
- `idx_keyphrase_processed`: Enables fast keyphrase searches
- `idx_sourcedoc`: Filters by source document
- `idx_sourcedoc_sid`: Compound queries on source and section

##### `bm25` - Build Hybrid Search Index
```bash
customkb bm25 <knowledge_base> [options]
```
Builds BM25 index for keyword-based hybrid search.

**Options:**
- `--force`: Force rebuild existing index

**Requirements:**
- Set `enable_hybrid_search=true` in configuration
- For older databases, run `scripts/upgrade_bm25_tokens.py` first

##### `edit` - Modify Configuration
```bash
customkb edit <knowledge_base>
```
Opens the configuration file in your default editor.

##### `version` - Display Version
```bash
customkb version [--build]
```
Shows version information with optional build number.

##### `help` - Show Usage
```bash
customkb help
```
Displays comprehensive usage information.

### Creating a Knowledgebase

A complete guide to creating a production-ready knowledgebase from raw data to deployment.

#### Overview

A knowledgebase consists of:
- Configuration file (`.cfg`) - Settings and system prompts
- SQLite database (`.db`) - Structured text storage
- FAISS index (`.faiss`) - Vector embeddings
- BM25 index (`.bm25`) - Keyword search (optional)

All knowledgebases are stored in `$VECTORDBS` (default: `/var/lib/vectordbs/`)

#### Step 1: Data Acquisition

Centralize your source documents in a workshop directory:

```bash
# Create workshop directory
mkdir -p ~/workshop/myproject

# Collect source materials
cp /path/to/documents/*.pdf ~/workshop/myproject/
cp /path/to/emails/*.eml ~/workshop/myproject/
cp /path/to/archives/*.zip ~/workshop/myproject/
```

Supported formats for acquisition:
- PDFs, Word docs, emails
- HTML pages, web archives
- Database exports, CSV files
- Any text-containing format

#### Step 2: Data Preprocessing

Convert source materials to markdown or text files:

```bash
cd ~/workshop/myproject

# Example: Convert PDFs to text
for pdf in *.pdf; do
    pdftotext "$pdf" "${pdf%.pdf}.txt"
done

# Example: Extract emails to markdown
for eml in *.eml; do
    # Custom script to extract email content
    extract_email.py "$eml" > "${eml%.eml}.md"
done

# Example: Process HTML files
for html in *.html; do
    pandoc -f html -t markdown "$html" -o "${html%.html}.md"
done
```

Best practices:
- Preserve document structure with markdown headers
- Include metadata (date, author, source) at the top
- Clean formatting inconsistencies
- Remove duplicate content

#### Step 3: Data Staging

Prepare documents for ingestion:

```bash
# Create staging directory in KB location
mkdir -p /var/lib/vectordbs/myproject/staging.text

# Copy processed documents
cp ~/workshop/myproject/*.{md,txt} /var/lib/vectordbs/myproject/staging.text/

# Post-processing in staging directory
cd /var/lib/vectordbs/myproject/staging.text

# Example: Add citations to documents
~/customkb/utils/citations/gen-citations.sh .
~/customkb/utils/citations/append-citations.sh .

# Example: Remove email signatures
for file in *.txt; do
    sed -i '/^--$/,$d' "$file"  # Remove everything after signature delimiter
done

# Example: Add document headers
for file in *.md; do
    # Add YAML frontmatter if missing
    if ! grep -q "^---" "$file"; then
        echo -e "---\nsource: $file\ndate: $(date -r "$file" +%Y-%m-%d)\n---\n" | cat - "$file" > temp && mv temp "$file"
    fi
done
```

#### Step 4: Create Configuration

Create the knowledgebase configuration file:

```bash
cat > /var/lib/vectordbs/myproject/myproject.cfg << 'EOF'
[DEFAULT]
# Embedding model configuration
vector_model = text-embedding-3-small
vector_dimensions = 1536
vector_chunks = 500

# Text processing
db_min_tokens = 200
db_max_tokens = 400

# Query settings
query_model = gpt-4o-mini
query_max_tokens = 4096
query_top_k = 30
query_context_scope = 4
query_temperature = 0.1

# System role - customize for your use case
query_role = |
  You are an expert assistant with deep knowledge of the documents in this knowledgebase.
  Provide accurate, detailed answers based on the provided context.
  Always cite specific sources when making claims.
  If information is not available in the context, clearly state this limitation.

# Optional: Additional context files
query_context_files = /var/lib/vectordbs/myproject/glossary.md

[ALGORITHMS]
enable_hybrid_search = true
enable_reranking = true

[PERFORMANCE]
embedding_batch_size = 100
checkpoint_interval = 10
EOF
```

#### Step 5: Data Ingestion

Process staged documents into the database:

```bash
# Create the knowledgebase directory structure
cd /var/lib/vectordbs/myproject

# Ingest all documents from staging directory
customkb database myproject staging.text/*.md staging.text/*.txt

# Or ingest with specific language processing
customkb database myproject staging.text/* --language english

# For multilingual content
customkb database myproject staging.text/* --detect-language
```

The ingestion process:
- Chunks documents into semantic segments
- Extracts metadata and structure
- Stores in SQLite database
- Creates full-text search tokens

#### Step 6: Data Embedding

Generate vector embeddings for semantic search:

```bash
# Generate embeddings for all chunks
customkb embed myproject --verbose

# Monitor progress
tail -f /var/lib/vectordbs/myproject/logs/myproject.log
```

This process:
- Calls the embedding API for each chunk
- Stores vectors in memory-efficient format
- Builds FAISS index for fast retrieval
- Saves checkpoints for resumability

#### Step 7: Build Search Indexes

Create additional indexes for optimal performance:

```bash
# Create BM25 index for hybrid search
customkb bm25 myproject

# Optimize for system resources
customkb optimize myproject

# Verify all indexes are created
customkb verify-indexes myproject
```

#### Step 8: Testing and Validation

Test your knowledgebase before deployment:

```bash
# Test basic query
customkb query myproject "What are the main topics covered?"

# Test with context only
customkb query myproject "List all document types" --context-only

# Test with different models
customkb query myproject "Summarize the key findings" --model claude-3-5-sonnet-latest

# Analyze performance
customkb optimize myproject --analyze
```

#### Step 9: Production Deployment

Configure for production use:

1. **Set production parameters**:
   ```ini
   [API]
   api_max_retries = 30
   api_max_concurrency = 16
   
   [PERFORMANCE]
   memory_cache_size = 100000
   query_cache_ttl_days = 30
   ```

2. **Enable monitoring**:
   ```bash
   # Set up log rotation
   cat > /etc/logrotate.d/customkb << EOF
   /var/lib/vectordbs/*/logs/*.log {
       daily
       rotate 7
       compress
       missingok
       notifempty
   }
   EOF
   ```

3. **Create backup script**:
   ```bash
   #!/bin/bash
   # backup_kb.sh
   KB_NAME="myproject"
   BACKUP_DIR="/backup/vectordbs"
   DATE=$(date +%Y%m%d)
   
   mkdir -p "$BACKUP_DIR"
   tar -czf "$BACKUP_DIR/${KB_NAME}_${DATE}.tar.gz" \
       -C /var/lib/vectordbs \
       "$KB_NAME"
   ```

#### Complete Example: Email Archive KB

Here's a real-world example of creating an email archive knowledgebase:

```bash
# 1. Acquisition - Extract emails from mail server
mkdir -p ~/workshop/email_archive
fetchmail --folder "Archive" --output ~/workshop/email_archive/

# 2. Preprocessing - Convert to markdown
cd ~/workshop/email_archive
for eml in *.eml; do
    python3 << EOF
import email
from email import policy
from email.parser import BytesParser

with open("$eml", 'rb') as f:
    msg = BytesParser(policy=policy.default).parse(f)
    
with open("${eml%.eml}.md", 'w') as out:
    out.write(f"---\n")
    out.write(f"from: {msg['From']}\n")
    out.write(f"to: {msg['To']}\n")
    out.write(f"date: {msg['Date']}\n")
    out.write(f"subject: {msg['Subject']}\n")
    out.write(f"---\n\n")
    out.write(f"# {msg['Subject']}\n\n")
    
    body = msg.get_body(preferencelist=('plain', 'html'))
    if body:
        out.write(body.get_content())
EOF
done

# 3. Staging
mkdir -p /var/lib/vectordbs/email_archive/staging.text
cp *.md /var/lib/vectordbs/email_archive/staging.text/

# 4. Post-process - Remove signatures and quotes
cd /var/lib/vectordbs/email_archive/staging.text
for file in *.md; do
    # Remove email signatures
    sed -i '/^-- $/,$d' "$file"
    # Remove excessive quoting
    sed -i 's/^>>>>*/> /g' "$file"
done

# 5. Configure
cat > /var/lib/vectordbs/email_archive/email_archive.cfg << 'EOF'
[DEFAULT]
vector_model = text-embedding-3-small
query_model = gpt-4o-mini
db_min_tokens = 150
db_max_tokens = 350

query_role = |
  You are an email archive assistant with access to years of correspondence.
  Help users find specific emails, summarize conversations, and extract information.
  Always mention the date and participants when referencing emails.
  Respect privacy and confidentiality in your responses.

[ALGORITHMS]
enable_hybrid_search = true
similarity_threshold = 0.7
EOF

# 6. Ingest and embed
customkb database email_archive staging.text/*.md
customkb embed email_archive

# 7. Create indexes
customkb bm25 email_archive
customkb optimize email_archive

# 8. Test
customkb query email_archive "Show me emails about project deadlines"
```

#### Tips for Success

1. **Data Quality**: Clean, well-structured documents produce better results
2. **Chunking Strategy**: Adjust `db_min_tokens` and `db_max_tokens` based on document types
3. **System Prompts**: Craft specific prompts that guide the AI's expertise
4. **Testing**: Always test with real queries before production use
5. **Monitoring**: Regular index verification and performance analysis
6. **Backup**: Regular backups of both configuration and data files

### Knowledgebase Organization

#### Directory Structure

All knowledgebases MUST be organized as subdirectories within the VECTORDBS directory:

```
$VECTORDBS/                          # Default: /var/lib/vectordbs
├── myproject/                       # Knowledgebase directory
│   ├── myproject.cfg               # Configuration file (required)
│   ├── myproject.db                # SQLite database
│   ├── myproject.faiss             # FAISS vector index
│   ├── myproject.bm25              # BM25 index (if hybrid search enabled)
│   ├── logs/                       # Runtime logs
│   │   └── myproject.log
│   ├── embed_data/                 # Original source documents (optional)
│   └── embed_data.text/            # Processed text files (optional)
```

#### KB Name Resolution

The system automatically handles various input formats:
```bash
# All of these resolve to the same KB:
customkb query myproject "search"           # Simple name
customkb query myproject.cfg "search"       # With .cfg extension
customkb query /path/to/myproject "search"  # With path (stripped)

# Result: Uses $VECTORDBS/myproject/myproject.cfg
```

#### Real-World Example: Okusi Associates KB

```
/var/lib/vectordbs/okusiassociates/
├── okusiassociates.cfg             # Configuration
├── okusiassociates.db              # 1.14M+ document segments
├── okusiassociates.faiss           # Vector index
├── okusiassociates.bm25            # Keyword search index
├── embed_data.text/                # Processed documents
│   ├── corporate_law/              # Organized by topic
│   ├── immigration/
│   ├── taxation/
│   └── ...
├── glossary.md                     # Reference documents
├── quick_price_list.txt
└── logs/
    └── okusiassociates.log
```

### Configuration Guide

CustomKB uses INI-style configuration files with environment variable overrides.

#### Configuration Sections

##### [DEFAULT] - Core Settings
```ini
[DEFAULT]
# Embedding model configuration
vector_model = text-embedding-3-small  # Options: ada-002, 3-small, 3-large, gemini-embedding-001
vector_dimensions = 1536               # Must match model output (auto-detected for most models)
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

##### [API] - External Service Settings
```ini
[API]
api_call_delay_seconds = 0.05          # Rate limiting delay
api_max_retries = 20                   # Retry attempts
api_max_concurrency = 8                # Parallel API calls
api_min_concurrency = 3                # Minimum parallel calls
backoff_exponent = 2                   # Exponential backoff factor
backoff_jitter = 0.1                   # Randomization factor
```

##### [LIMITS] - Resource Constraints
```ini
[LIMITS]
max_file_size_mb = 100                 # Maximum file size
max_query_file_size_mb = 1             # Max query file size
memory_cache_size = 10000              # Cache entries
cache_thread_pool_size = 4             # Cache operation threads
api_key_min_length = 20                # Security validation
max_query_length = 10000               # Query text limit
```

##### [PERFORMANCE] - Optimization
```ini
[PERFORMANCE]
embedding_batch_size = 100             # Embeddings per batch
checkpoint_interval = 10               # Save progress frequency
commit_frequency = 1000                # Database commit interval
io_thread_pool_size = 4                # Parallel I/O threads
query_cache_ttl_days = 7               # Cache expiration
reference_batch_size = 20              # References per query batch
```

##### [ALGORITHMS] - Processing Settings
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
```

### Model Support

#### Available Models (via Models.json)

##### Language Models (LLMs)
**OpenAI:**
- GPT-4.1 / 4.1-mini / 4.1-nano (1M context window)
- GPT-4o / 4o-mini (flagship multimodal)
- o3 / o3-mini / o3-pro (advanced reasoning)
- o4-mini (multimodal reasoning with tools)

**Anthropic:**
- Claude 4.0 Opus/Sonnet (world's best coding model)
- Claude 3.7 Sonnet (extended thinking capability)
- Claude 3.5 Sonnet/Haiku (latest generation)

**Google:**
- Gemini 2.5 Pro/Flash/Lite (thinking models)
- Gemini 2.0 Pro/Flash (2M context window)

**xAI:**
- Grok 4.0 / 4.0-heavy (PhD-level reasoning)

**Local Models (via Ollama):**
- Llama 3.1/3.2/3.3 (8B to 70B)
- Gemma 3 (4B/12B/27B)
- Mistral 7B
- Qwen 2.5
- DeepSeek R1
- CodeLlama 13B
- Phi-4 14B

##### Embedding Models

**OpenAI Models:**
- text-embedding-3-large (3072 dimensions)
- text-embedding-3-small (1536 dimensions)
- text-embedding-ada-002 (1536 dimensions)

**Google Models:**
- gemini-embedding-001 (configurable: 768/1536/3072 dimensions)

#### Model Aliases

Use convenient aliases instead of full model names:
```bash
# Using aliases
customkb query myproject "test" -m gpt4o      # → gpt-4o
customkb query myproject "test" -m sonnet     # → claude-sonnet-4-0
customkb query myproject "test" -m gemini2    # → gemini-2.0-flash

# List all models and aliases
customkb query myproject "test" -m list
```

#### Using Google Embeddings

CustomKB now supports Google's Gemini embedding models with superior performance:

**Setup:**
```bash
# Set your Google API key
export GOOGLE_API_KEY="your-google-api-key"
# or
export GEMINI_API_KEY="your-gemini-api-key"
```

**Configuration:**
```ini
[DEFAULT]
vector_model = gemini-embedding-001
# Optional: specify dimensions (768, 1536, or 3072)
# If not specified, defaults to 3072
vector_dimensions = 1536
```

**Key Advantages:**
- **Superior Performance**: 68% MTEB benchmark score vs 64.6% for OpenAI
- **Configurable Dimensions**: Choose 768, 1536, or 3072 based on your needs
- **Longer Context**: Supports up to 30k tokens vs 8k for OpenAI
- **Matryoshka Learning**: Maintains quality even at lower dimensions

**Note**: Google API limits batch size to 100 embeddings per request (automatically handled).

### Integration with Dejavu2-CLI (dv2)

CustomKB integrates seamlessly with [dejavu2-cli](https://github.com/Open-Technology-Foundation/dejavu2-cli) for enhanced AI interactions.

#### Using CustomKB as dv2 Knowledgebase

```bash
# Query with dv2 using CustomKB knowledgebase
dv2 -k /var/lib/vectordbs/myproject/myproject.cfg "What are the key features?"

# Use pre-configured templates (Agents)
dv2 -T askOkusi "How to establish a PMA company in Indonesia?"
```

#### Available dv2 Templates with CustomKB

```bash
# List all templates
dv2 --list-template-names

# Example templates using CustomKB:
- askOkusi          # Okusi Associates business advisor
- OkusiMail         # Okusi email archive advisor  
- JP                # Jakarta Post archive 1994-2005
- DrAA              # Applied Anthropology KB
- Ollama            # Ollama documentation expert
- ProSocial         # ProSocial psychology/philosophy
```

#### Creating Custom dv2 Agents

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

### Performance Optimization

#### Memory-Based Optimization Tiers

CustomKB automatically optimizes based on available system memory:

##### Low Memory Systems (<16GB)
```ini
memory_cache_size = 50000
embedding_batch_size = 375
reference_batch_size = 15
enable_hybrid_search = false
```

##### Medium Memory Systems (16-64GB)
```ini
memory_cache_size = 100000
embedding_batch_size = 562
reference_batch_size = 22
enable_hybrid_search = false
```

##### High Memory Systems (64-128GB)
```ini
memory_cache_size = 150000
embedding_batch_size = 750
reference_batch_size = 30
enable_hybrid_search = true
```

##### Very High Memory Systems (>128GB)
```ini
memory_cache_size = 200000
embedding_batch_size = 1125
reference_batch_size = 45
enable_hybrid_search = true
enable_reranking = true
reranking_device = cuda
```

#### GPU Acceleration

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

#### Best Practices

1. **Initial Setup**:
   ```bash
   # Analyze system and apply optimizations
   customkb optimize --analyze
   customkb optimize myproject
   
   # Verify database health
   customkb verify-indexes myproject
   ```

2. **Large Datasets** (100k+ documents):
   - Use `embedding_batch_size = 200+`
   - Enable hybrid search for better performance
   - Consider GPU acceleration for reranking

3. **Container Deployments**:
   ```bash
   # Override memory detection
   customkb optimize --memory-gb 8
   ```

4. **Production Deployments**:
   - Enable comprehensive logging
   - Set up regular index verification
   - Monitor embedding cache hit rates
   - Use dedicated vector model API keys

### Advanced Features

#### Prompt Templates

Customize how queries are presented to LLMs:

```bash
# Available templates
customkb query myproject "question" --prompt-template <template>

# Templates:
- default         # Simple format with minimal instructions
- instructive     # Clear instructions with explicit guidelines
- scholarly       # Academic style with emphasis on citations
- concise        # Minimal, direct responses
- analytical     # Structured analytical approach
- conversational # Friendly, conversational tone
- technical      # Technical depth with precise terminology
```

#### Reference Output Formats

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

#### Multi-Language Processing

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

#### Batch Processing

```python
#!/usr/bin/env python
"""Batch process multiple knowledgebases"""

import subprocess
import glob

# Find all KB configs
kb_configs = glob.glob("/var/lib/vectordbs/*/[!.]*.cfg")

for config in kb_configs:
    kb_name = config.split('/')[-1].replace('.cfg', '')
    print(f"Processing {kb_name}...")
    
    # Run optimization
    subprocess.run(['customkb', 'optimize', kb_name])
    
    # Verify indexes
    subprocess.run(['customkb', 'verify-indexes', kb_name])
```

### Utility Scripts

CustomKB includes utility scripts in the `scripts/` directory:

#### Performance Scripts
- `optimize_kb_performance.py` - Apply memory-based optimization tiers
- `show_optimization_tiers.py` - Display settings for different memory tiers
- `emergency_optimize.py` - Apply conservative settings after crashes
- `performance_analyzer.py` - Analyze KB performance metrics

#### GPU Scripts
- `benchmark_gpu.py` - Benchmark GPU vs CPU performance
- `gpu_monitor.sh` - Real-time GPU usage monitoring
- `gpu_env.sh` - GPU environment setup

#### Maintenance Scripts
- `rebuild_bm25_filtered.py` - Create filtered BM25 indexes
- `upgrade_bm25_tokens.py` - Upgrade older databases for BM25
- `diagnose_crashes.py` - Diagnose crash issues
- `emergency_cleanup.sh` - Emergency cleanup operations

#### Security
- `security-check.sh` - Run security scans on dependencies

### Citations Extraction System

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

### Testing

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

### Troubleshooting

#### Common Issues

**"Knowledgebase 'name' not found"**
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

#### Debug Mode

Enable comprehensive logging:
```bash
# Debug mode for commands
customkb query myproject "test query" --debug

# Check logs
tail -f /var/lib/vectordbs/myproject/logs/myproject.log
```

## 🤝 Contributing

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

## 📄 License

CustomKB is released under the MIT License. See [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

CustomKB is built on excellent open-source projects:
- [FAISS](https://github.com/facebookresearch/faiss) - Efficient similarity search
- [LangChain](https://github.com/langchain-ai/langchain) - Text splitting utilities
- [NLTK](https://www.nltk.org/) - Natural language processing
- [spaCy](https://spacy.io/) - Advanced NLP features
- [Sentence Transformers](https://www.sbert.net/) - Cross-encoder reranking

## 📞 Support

- **Documentation**: [docs/](docs/) directory
- **Issues**: [GitHub Issues](https://github.com/Open-Technology-Foundation/customkb/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Open-Technology-Foundation/customkb/discussions)

---

*CustomKB is actively maintained by the Open Technology Foundation*