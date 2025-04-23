# CustomKB: AI-Powered Knowledge Base System

CustomKB is a powerful, flexible tool for creating, managing, and querying custom knowledge bases using vector embeddings and large language models. It enables semantic search on your document corpus with intelligent, context-aware AI responses.

## Features

- **Vector-Based Knowledge Storage**
  - SQLite database for text storage with metadata
  - FAISS vector indices for fast semantic search
  - Document chunking with configurable token limits
  - Automatic metadata extraction and tracking

- **Advanced Embedding Generation**
  - Multiple embedding model support (OpenAI, Anthropic)
  - Batch processing with automatic checkpointing
  - Efficient caching to reduce API costs
  - Optimized vector indexing based on dataset size

- **Intelligent Querying**
  - Semantic search with relevance scoring
  - Context-aware retrieval with adjustable scope
  - Integration with multiple LLMs:
    - OpenAI GPT models (4o, 4.1, o1, etc.)
    - Anthropic Claude models (3, 3.5, 3.7)
    - Meta Llama models via Ollama
  - XML-formatted context for precise AI prompting

- **Text Processing & NLP**
  - Entity recognition and preservation
  - Multi-language stopword filtering
  - Lemmatization and text normalization
  - Intelligent document splitting

- **Flexible Configuration**
  - INI-style configuration files
  - Environment variable overrides
  - Command-line parameter customization
  - Model aliasing through Models.json

## Installation

### Requirements

- Python 3.12+
- Required packages: see `requirements.txt`

### Setup

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Set required environment variables:
   ```bash
   export OPENAI_API_KEY="your-api-key"
   export ANTHROPIC_API_KEY="your-api-key"
   export VECTORDBS="/path/to/vectordbs" # Default: /var/lib/vectordbs
   export NLTK_DATA="/path/to/nltk_data" # Required for text processing
   ```

## Usage

CustomKB is used through a command-line interface with subcommands:

```bash
customkb <command> <config_file> [options]
```

### Basic Workflow

1. **Create Configuration**:
   - Create a `.cfg` file for your knowledge base (see example below)

2. **Process Documents**:
   ```bash
   customkb database myknowledge.cfg *.txt *.md
   ```

3. **Generate Embeddings**:
   ```bash
   customkb embed myknowledge.cfg
   ```

4. **Query the Knowledge Base**:
   ```bash
   customkb query myknowledge.cfg "What are the key features?"
   ```

### Configuration Example

Create a file named `myknowledge.cfg`:

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
```

### Commands

#### Database Command
Process files and store them in the knowledge base.

```bash
customkb database <config_file> [files...]
```

Options:
- `-l, --language`: Language for stopwords (default: english)
- `-v/q, --verbose/--quiet`: Control output verbosity
- `-d, --debug`: Enable debug output

#### Embedding Command
Generate vector embeddings for stored text chunks.

```bash
customkb embed <config_file> [options]
```

Options:
- `-r, --reset-database`: Reset the 'embedded' flag to reprocess all chunks

#### Query Command
Search the knowledge base and generate AI responses.

```bash
customkb query <config_file> <query_text> [options]
```

Options:
- `-Q, --query_file`: Load additional query text from file
- `-c, --context, --context-only`: Return only context without AI response
- `-R, --role`: Set custom LLM system role
- `-m, --model`: Specify LLM model to use
- `-k, --top-k`: Number of results to return
- `-s, --context-scope`: Context segments per result
- `-t, --temperature`: Model temperature
- `-M, --max-tokens`: Maximum output tokens

#### Edit Command
Edit the knowledge base configuration file.

```bash
customkb edit <config_file>
```

#### Help Command
Display usage information.

```bash
customkb help
```

## Advanced Configuration

Settings are resolved in this order:
1. Command-line arguments
2. Environment variables
3. Configuration file values
4. Default values

### Environment Variables

- `OPENAI_API_KEY`: OpenAI API key
- `ANTHROPIC_API_KEY`: Anthropic API key
- `VECTORDBS`: Base directory for vector databases
- `NLTK_DATA`: NLTK data directory
- `VECTOR_MODEL`: Default embedding model
- `QUERY_MODEL`: Default query model
- `QUERY_ROLE`: Default system role for AI
- And many more (see configuration manager)

## License

CustomKB is licensed under MIT License.