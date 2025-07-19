# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Citations Module Overview

The citations module automatically extracts bibliographic information from source documents in the CustomKB system. It uses OpenAI's API to analyze document headers and generate structured citation data.

## Module Architecture

### Current Implementation
- **gen-citations.sh**: Main script that processes markdown/text files and extracts citations using GPT-4o-mini
  - Supports sequential processing with progress tracking
  - Parallel processing framework with configurable workers (1-20+)
  - Advanced progress monitoring with adaptive sleep intervals
  - Comprehensive error handling and cleanup mechanisms
- **append-citations.sh**: Companion script that adds citations as YAML frontmatter to documents
- **lib/db_functions.sh**: Database helper functions for SQLite operations
- **lib/api_functions.sh**: API interaction helpers with rate limiting and retry logic
- **lib/parallel_functions.sh**: Parallel processing functions for multi-worker citation extraction
- **citations.db**: SQLite database for storing extracted citations with full schema support

### Key Components

1. **Citation Extraction** (gen-citations.sh)
   - Processes first 7KB of each document (configurable via CITATION_CHUNK_SIZE)
   - Uses OpenAI API to identify title, author, and year
   - Stores results in SQLite database with full citation details
   - Supports batch processing with real-time progress tracking
   - Includes dry-run mode and force update options
   - Advanced file title hint processing for better accuracy
   - Intelligent citation parsing with NF (Not Found) handling
   - Cleanup functions with proper signal handling (SIGINT, EXIT)

2. **Citation Application** (append-citations.sh)
   - Reads citations from database
   - Adds YAML frontmatter to source documents
   - Creates backups before modification
   - Handles existing frontmatter appropriately
   - Supports dry-run mode for preview

3. **Database Operations**
   - Automatic database initialization
   - UPSERT operations for citation storage
   - Indexed queries for performance
   - Transaction support for reliability

## Commands and Usage

### Citation Extraction

```bash
# Process default directory
./gen-citations.sh

# Process specific directory
./gen-citations.sh /path/to/documents

# Process with options
./gen-citations.sh -d ./my-citations.db  # Specify database path
./gen-citations.sh -f                     # Force update existing entries
./gen-citations.sh -n                     # Dry run (preview only)
./gen-citations.sh -m gpt-4              # Use different model
./gen-citations.sh -v                     # Verbose output (can use multiple times)
./gen-citations.sh -q                     # Quiet mode (suppress verbose output)
./gen-citations.sh -D                     # Debug mode
./gen-citations.sh -h                     # Show help

# Advanced options
./gen-citations.sh -c 10000               # Set chunk size to 10KB
./gen-citations.sh -M 256                 # Set max tokens to 256
./gen-citations.sh -t 0.2                 # Set temperature to 0.2
./gen-citations.sh -x "*/backup/*"        # Exclude backup directories

# Parallel processing options
./gen-citations.sh -p 10                  # Use 10 parallel workers
./gen-citations.sh --no-parallel          # Disable parallel processing
./gen-citations.sh --sequential           # Same as --no-parallel

# Process multiple directories
./gen-citations.sh /path/to/docs1 /path/to/docs2

# Parallel processing with high worker count for large datasets
./gen-citations.sh -p 20 /var/lib/vectordbs/large-project/
```

### Citation Application

```bash
# Apply citations to default directory
./append-citations.sh

# Apply to specific directory
./append-citations.sh /path/to/documents

# Apply with options
./append-citations.sh -d ./my-citations.db  # Specify database path
./append-citations.sh -n                     # Dry run (preview changes)
./append-citations.sh -f                     # Force overwrite frontmatter
./append-citations.sh -b                     # No backup files
./append-citations.sh -v                     # Verbose output
./append-citations.sh -h                     # Show help
```

### Complete Workflow

```bash
# Step 1: Extract citations from documents
./gen-citations.sh /var/lib/vectordbs/myproject/embed_data.text/

# Step 2: Review extracted citations (optional)
sqlite3 citations.db "SELECT * FROM citations;"

# Step 3: Apply citations to documents (dry run first)
./append-citations.sh -n /var/lib/vectordbs/myproject/embed_data.text/

# Step 4: Apply citations for real
./append-citations.sh /var/lib/vectordbs/myproject/embed_data.text/
```

## Environment Requirements

```bash
# Required environment variables
export OPENAI_API_KEY="your-api-key"
export VECTORDBS="/var/lib/vectordbs"  # CustomKB vector database directory

# Optional environment variables
export CITATION_MODEL="gpt-4o-mini"      # Override default model
export CITATION_DATABASE="./citations.db" # Override database path
export CITATION_CHUNK_SIZE="7420"        # Override chunk size (chars)
export CITATION_MAX_TOKENS="128"         # Override max response tokens
export CITATION_TEMPERATURE="0"          # Override temperature (0-1)
export CITATION_EXCLUDES="*/backup/*"    # Space-separated exclude patterns
```

## Parallel Processing Performance

### Optimal Worker Count
- **Default**: 5 workers (safe for most API rate limits)
- **Small datasets (<1000 files)**: 3-5 workers
- **Medium datasets (1000-10000 files)**: 5-10 workers
- **Large datasets (>10000 files)**: 10-20 workers

### Performance Expectations
- **Sequential mode**: ~60 files/minute (1 file/second with rate limiting)
- **5 workers**: ~250-300 files/minute (4-5x speedup)
- **10 workers**: ~450-550 files/minute (7-9x speedup)
- **20 workers**: ~800-1000 files/minute (13-16x speedup)

### Example Performance
Processing 13,384 files:
- **Sequential**: ~3.7 hours
- **5 workers**: ~45-55 minutes
- **10 workers**: ~25-30 minutes
- **20 workers**: ~15-20 minutes

## Database Schema

```sql
CREATE TABLE citations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sourcefile VARCHAR(255) UNIQUE,
    title TEXT,
    author TEXT,
    year TEXT,
    raw_citation TEXT
);

CREATE INDEX idx_sourcefile ON citations(sourcefile);
```

## Citation Format

The module aims to generate YAML frontmatter for documents:

```yaml
---
title: "Document Title"
author: "Author Name"
date: "YYYY"
---
```

## API Configuration

- Default Model: `gpt-4o-mini` (configurable via `-m` option or `CITATION_MODEL` env var)
- Max tokens: 128 (configurable via `CITATION_MAX_TOKENS`)
- Temperature: 0.1 (near-deterministic, configurable via `CITATION_TEMPERATURE`)
- Chunk size: 7420 characters (configurable via `CITATION_CHUNK_SIZE`)
- Rate limiting: 1 second between API calls
- Retry logic: 3 attempts with exponential backoff

## Features

### gen-citations.sh
- **Parallel Processing**: Process multiple files concurrently with configurable workers
- **Progress Tracking**: Real-time progress indicator with file counts and processing rate
- **Error Handling**: Comprehensive error handling with retry logic
- **Rate Limiting**: Per-worker rate limiting to maximize throughput while respecting API limits
- **Duplicate Detection**: Skip already processed files (override with `-f`)
- **Dry Run Mode**: Preview what would be processed without making changes
- **Multiple Sources**: Process multiple directories in one run
- **Statistics**: Final summary of processed, skipped, and error counts
- **Debug Mode**: Detailed logging for troubleshooting
- **Performance Metrics**: Shows processing rate (files/minute) and ETA in parallel mode

### append-citations.sh
- **Backup Creation**: Automatic `.bak` files before modification
- **Frontmatter Handling**: Preserve or overwrite existing frontmatter
- **Dry Run Mode**: Preview changes before applying
- **Batch Processing**: Apply all citations from database
- **Error Recovery**: Restore from backup if update fails
- **Progress Display**: Real-time progress with file counts

### Database Features
- **Automatic Initialization**: Database and tables created on first run
- **Indexed Queries**: Performance optimized with proper indexes
- **UPSERT Operations**: Insert or update citations as needed
- **Transaction Safety**: Atomic operations for data integrity
- **Flexible Schema**: Stores parsed and raw citation data

## Technical Details

### System Prompt
The script uses a sophisticated system prompt that:
- Extracts bibliographic info with specific formatting rules
- Handles file-title hints for improved accuracy
- Uses "NF" (Not Found) for missing fields
- Validates year format (4-digit numbers only)
- Prevents single numeric values from being treated as titles

### Progress Monitoring
In parallel mode, the script implements:
- Real-time progress updates with worker status
- Processing rate calculation (files/minute)
- ETA estimation based on current throughput
- Adaptive sleep intervals (0.1s to 1.0s) to reduce CPU usage
- Comprehensive statistics tracking (processed, skipped, errors, not found)

### File Processing Logic
- Normalizes file paths for consistent database storage
- Creates file title hints from filenames (replacing hyphens/underscores)
- Validates file existence and content before processing
- Implements intelligent duplicate detection (unless --force is used)

### Citation Parsing
The `parse_citation` function (from lib/api_functions.sh) handles:
- Comma-separated citation format: "title", author, year
- Proper handling of quoted titles with commas
- NF (Not Found) field detection and replacement
- Clean result formatting by removing NF values from raw citations

### Cleanup and Signal Handling
- Comprehensive cleanup function (`xcleanup`) with duplicate execution prevention
- Proper cursor visibility restoration for terminal
- Parallel processing cleanup when applicable
- Final statistics display on exit
- Signal trapping for SIGINT and EXIT

## Known Issues

### Parallel Processing
The parallel processing implementation has been enhanced with better worker management and progress tracking. If workers fail to complete, use `--no-parallel` or `--sequential` to force sequential processing.

## Troubleshooting

### Common Issues

1. **API Key Not Set**
   ```bash
   export OPENAI_API_KEY="sk-..."
   ```

2. **Rate Limit Errors**
   - The script automatically handles rate limits with retry logic
   - Increase delay with custom API_DELAY_MS if needed

3. **Database Locked**
   - Ensure no other process is accessing the database
   - Check file permissions on citations.db

4. **Citation Not Found (NF)**
   - Document may not contain clear bibliographic information
   - Try adjusting the chunk size for more context

5. **Parsing Errors**
   - Check raw_citation field in database for original response
   - Some citations may need manual correction

### Debug Commands

```bash
# View all citations in database
sqlite3 citations.db "SELECT * FROM citations;"

# Check specific file
sqlite3 citations.db "SELECT * FROM citations WHERE sourcefile LIKE '%filename%';"

# Export citations to CSV
sqlite3 -header -csv citations.db "SELECT * FROM citations;" > citations.csv

# Check database schema
sqlite3 citations.db ".schema"
```

## Coding Principles
  - K.I.S.S.
  - "The best process is no process"
  - "Everything should be made as simple as possible, but not simpler."

## Code Style
- Python: 
  - Import order: standard lib, third-party, local modules
  - Constants: Define at top of files, use UPPER_CASE
  - Use descriptive function and variable names
  - Docstrings for functions; comment complex logic sections
  - Always end scripts with '\n#fin\n' to indicate the end of script

- Shell scripts:
  - Shebang '#!/bin/bash'
  - Use `set -Eeuo pipefail` for enhanced error handling (capital E for ERR trap inheritance)
  - 2-space indentation !!important
  - Always declare variables before use; use local within functions
  - Use descriptive variable names with `declare` or `local` statements
  - Prefer `[[` over `[` for conditionals
  - Prefer `((...)) && ...` or `[[...]] && ...` for simple conditionals over `if...then`
  - Use integer values where appropriate, and always declare with `-i`
  - Always end scripts with line '#fin' to indicate end of script

- PHP:
  - Always use 2-space indent
  - Always use <?=...?> where possible for simple output; never <?php echo ...?>
  - Follow PSR-12 coding standards
  - Use prepared statements for all database queries
  - Always check array keys with isset() before accessing
  - Filter user inputs with filter_input() functions
  - Sanitize output with htmlspecialchars() or similar
  - Always check file operations for errors
  - Use proper HTTP status codes for errors

- JavaScript:
  - Use ES6+ syntax and features
  - Avoid jQuery where possible, use modern DOM APIs
  - Follow Bootstrap patterns for UI components, unless there is a good reason to do otherwise
  - Always sanitize dynamic content before insertion
  - Use strict mode ('use strict'), unless there is a good reason to do otherwise

- Error handling:
  - Python: Use try/except with logging
  - Shell: Use proper exit codes and error messages

- Environment:
  - Python venvs: Activate with `source <dir>/.venv/bin/activate`
  - Use MySQL or Sqlite3 database for data storage, as appropriate

## Developer Tech Stack
- Ubuntu 24.04.2
- Bash 5.2.21
- Python 3.12.3
- Apache2 2.4.58
- PHP 8.3.6
- MySQL 8.0.42
- sqlite3 3.45.1
- Bootstrap 5.3
- FontAwesome

## Hardware
- Development Machine (hostname 'okusi'):
  - model: Lenovo Legion i9
  - gpu: GEForce RTX
  - system memory: 32GB

- Production Machine (hostname 'okusi3'):
  - model: Intel Xeon Silver 4410Y, 2 cpu
  - gpu: NVIDIA L4
  - system memory: 256GiB

## Backups and Code Checkpoints
- Use `checkpoint -q` for checkpoint backups. 
- checkpoint backups are located in /var/backups/{codebase_dir}/{YYYYMMDD_hhmmss}/
- .gudang directories should normally be ignored.

#fin