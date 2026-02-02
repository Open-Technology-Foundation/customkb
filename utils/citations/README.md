# Citations Extraction System

An automated system for extracting bibliographic citations from document collections using OpenAI's API. This tool processes markdown and text files to identify title, author, and publication year information, storing results in a SQLite database for later use.

## Features

- **Automated Citation Extraction**: Uses AI to intelligently extract bibliographic information from document headers
- **Parallel Processing**: Process large document collections efficiently with configurable worker pools (1-20+ workers)
- **Smart File Name Hints**: Uses file names as context to improve citation accuracy
- **Database Storage**: SQLite database with proper indexing and concurrent access support
- **YAML Frontmatter Generation**: Automatically adds citation metadata to source documents
- **Incremental Processing**: Skip already-processed files unless forced
- **Comprehensive Error Handling**: Retry logic, rate limiting, and graceful failure recovery
- **Progress Tracking**: Real-time progress updates with performance metrics

## Requirements

- Bash 5.0+
- SQLite3
- jq (JSON processor)
- curl
- OpenAI API key

## Installation

1. Clone or copy the citations directory to your system
2. Ensure all scripts are executable:
   ```bash
   chmod +x gen-citations.sh append-citations.sh
   ```
3. Set your OpenAI API key:
   ```bash
   export OPENAI_API_KEY="sk-..."
   ```

## Quick Start

### Extract Citations

```bash
# Process current directory
./gen-citations.sh .

# Process specific directory
./gen-citations.sh /path/to/documents

# Process with parallel workers for speed
./gen-citations.sh -p 10 /path/to/documents
```

### Apply Citations to Documents

```bash
# Apply citations as YAML frontmatter
./append-citations.sh /path/to/documents

# Preview changes first
./append-citations.sh --dry-run /path/to/documents
```

## Usage

### gen-citations.sh - Extract Citations

```bash
./gen-citations.sh [OPTIONS] [SOURCE_DIRECTORY...]
```

#### Options

| Option | Description | Default |
|--------|-------------|---------|
| `-d, --database PATH` | Database file path | `./citations.db` |
| `-f, --force` | Force update existing citations | Off |
| `-r, --reprocess-blank` | Reprocess files where title and author are blank | Off |
| `-n, --dry-run` | Preview what would be processed | Off |
| `-m, --model MODEL` | OpenAI model to use | `gpt-4o-mini` |
| `-c, --chunk-size SIZE` | Characters to read from each file | 7420 |
| `-M, --max-tokens N` | Maximum tokens in API response | 128 |
| `-t, --temperature T` | AI temperature (0-1) | 0.1 |
| `-x, --exclude PATTERN` | Exclude files matching pattern | None |
| `-p, --parallel N` | Number of parallel workers | 5 |
| `--no-parallel` | Disable parallel processing | Off |
| `-q, --quiet` | Suppress verbose output | Off |
| `-v, --verbose` | Increase verbosity (can stack) | On |
| `-D, --debug` | Enable debug output | Off |

#### Examples

```bash
# Basic usage
./gen-citations.sh /var/lib/vectordbs/myproject/

# High-performance parallel processing
./gen-citations.sh -p 20 /large/document/collection/

# Force update with different model
./gen-citations.sh --force --model gpt-4 /path/to/docs/

# Exclude backup directories
./gen-citations.sh -x "*/backup/*" -x "*/archive/*" /documents/

# Reprocess documents with missing data
./gen-citations.sh --reprocess-blank /path/to/docs/

# Debug mode for troubleshooting
./gen-citations.sh -D -v -v /path/to/docs/
```

### append-citations.sh - Apply Citations

```bash
./append-citations.sh [OPTIONS] [SOURCE_DIRECTORY...]
```

#### Options

| Option | Description | Default |
|--------|-------------|---------|
| `-d, --database PATH` | Database file path | `./citations.db` |
| `-c, --context DOMAINS` | Broad context domains (comma-delimited) | None |
| `-n, --dry-run` | Preview changes without applying | Off |
| `-b, --no-backup` | Don't create backup files | Off |
| `-f, --force` | Overwrite existing frontmatter | Off |
| `-q, --quiet` | Suppress verbose output | Off |
| `-v, --verbose` | Increase verbosity | On |

#### Examples

```bash
# Basic usage
./append-citations.sh /path/to/documents

# Preview changes
./append-citations.sh --dry-run /documents/

# Force overwrite with context
./append-citations.sh --force --context "history,anthropology" /docs/

# Apply without backups (careful!)
./append-citations.sh --no-backup /documents/
```

## Performance

### Sequential vs Parallel Processing

| Mode | Files/Minute | Best For |
|------|--------------|----------|
| Sequential | ~60 | Small collections (<100 files) |
| 5 workers | ~250-300 | Medium collections (100-1000 files) |
| 10 workers | ~450-550 | Large collections (1000-10000 files) |
| 20 workers | ~800-1000 | Very large collections (10000+ files) |

### Example: Processing 13,384 files
- Sequential: ~3.7 hours
- 5 workers: ~45-55 minutes
- 10 workers: ~25-30 minutes
- 20 workers: ~15-20 minutes

## Database Schema

The SQLite database stores extracted citations with the following schema:

```sql
CREATE TABLE citations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sourcefile TEXT UNIQUE NOT NULL,
    title TEXT,
    author TEXT,
    year TEXT,
    raw_citation TEXT,
    processed_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_modified TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

Indexes are created on `sourcefile` and `processed_date` for optimal query performance.

## Citation Format

### Extracted Data
The system extracts citations in the format:
```
"Title of Document", Author Name, Year
```

Fields may be empty if not found. The special value "NF" (Not Found) is used internally but cleaned from final output.

### YAML Frontmatter Output
Citations are applied to documents as:
```yaml
---
title: "Document Title"
author: "Author Name"
date: "YYYY"
broad_context: "optional,domains"
---
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | **Required**: OpenAI API key | None |
| `CITATION_MODEL` | Default model to use | `gpt-4o-mini` |
| `CITATION_DATABASE` | Default database path | `./citations.db` |
| `CITATION_CHUNK_SIZE` | Characters to read from files | 7420 |
| `CITATION_MAX_TOKENS` | Max response tokens | 128 |
| `CITATION_TEMPERATURE` | AI temperature (0-1) | 0.1 |
| `CITATION_EXCLUDES` | Space-separated exclude patterns | None |
| `VECTORDBS` | Base directory for knowledgebases | None |
| `BROAD_CONTEXT` | Default context domains | None |

## Architecture

### Component Overview

1. **gen-citations.sh**: Main extraction script
   - Orchestrates the citation extraction process
   - Manages parallel workers and progress tracking
   - Handles command-line arguments and configuration

2. **append-citations.sh**: Citation application script
   - Reads citations from database
   - Applies YAML frontmatter to documents
   - Manages backups and updates

3. **lib/db_functions.sh**: Database operations
   - SQLite initialization and queries
   - Citation parsing and storage
   - UPSERT operations for idempotency

4. **lib/api_functions.sh**: OpenAI API integration
   - Rate limiting and retry logic
   - JSON request/response handling
   - API key validation

5. **lib/parallel_functions.sh**: Parallel processing framework
   - Worker process management
   - Queue-based work distribution
   - Atomic progress tracking
   - Batch result processing

### Parallel Processing Architecture

The parallel processing system uses:
- **Work Queue**: Files are added to a queue file with atomic operations
- **Worker Processes**: Independent bash processes with own rate limiting
- **Result Collection**: Workers write results to separate files
- **Batch Processing**: Results are collected and inserted in a single transaction
- **Progress Tracking**: Real-time updates with performance metrics

## Troubleshooting

### Common Issues

#### API Key Not Set
```bash
export OPENAI_API_KEY="sk-..."
```

#### Rate Limit Errors
The system automatically handles rate limits with exponential backoff. For persistent issues, reduce the number of parallel workers.

#### Database Locked
- Ensure no other process is using the database
- The system uses SQLite WAL mode for better concurrency
- Built-in retry logic handles transient locks

#### No Citations Found
- Document may lack clear bibliographic information
- Try increasing `--chunk-size` for more context
- Check the `raw_citation` field in the database

#### Worker Failures
If parallel workers fail to start:
- Check system resource limits (`ulimit -n`)
- Reduce worker count with `-p`
- Use `--no-parallel` for sequential processing
- Enable debug mode with `-D` for detailed logs

### Debug Commands

```bash
# View all citations
sqlite3 citations.db "SELECT * FROM citations;"

# Check specific file
sqlite3 citations.db "SELECT * FROM citations WHERE sourcefile LIKE '%filename%';"

# Export to CSV
sqlite3 -header -csv citations.db "SELECT * FROM citations;" > citations.csv

# Count statistics
sqlite3 citations.db "SELECT COUNT(*) as total, 
  COUNT(CASE WHEN title != '' THEN 1 END) as with_title,
  COUNT(CASE WHEN author != '' THEN 1 END) as with_author,
  COUNT(CASE WHEN year != '' THEN 1 END) as with_year
FROM citations;"

# Find problematic entries
sqlite3 citations.db "SELECT sourcefile FROM citations 
WHERE title = '' AND author = '' AND year = '';"
```

## Best Practices

1. **Start with Dry Run**: Always use `--dry-run` first to preview changes
2. **Backup Important Data**: The system creates `.bak` files by default
3. **Monitor Progress**: Use verbose mode to track processing
4. **Optimize Worker Count**: Start with 5-10 workers, increase if needed
5. **Handle Failures Gracefully**: The system is designed to resume after interruption
6. **Check Results**: Review extracted citations before applying to documents

## Contributing

When contributing to this project:
1. Follow the existing bash style (2-space indentation)
2. Add comprehensive error handling
3. Document new functions thoroughly
4. Test with both small and large datasets
5. Ensure compatibility with parallel processing

## License

This tool is part of the CustomKB project. See the main project license for details.