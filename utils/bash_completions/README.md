# CustomKB -- Custom Knowledgebases

## customkb help

CustomKB 0.8.0


CustomKB: AI-Powered Knowledgebase System

Create and query AI knowledgebases with semantic search and LLM integration.

Usage:
  customkb <command> <knowledge_base> [options]

Commands:
  database     Process text files into knowledgebase
  embed        Generate embeddings for stored text
  query        Search knowledgebase with AI responses
  edit         Edit knowledgebase configuration
  optimize     Optimize performance and create indexes
  verify-indexes   Check database index health
  bm25         Build BM25 index for hybrid search
  version      Show version information
  help         Show this help message

Examples:
  customkb database knowledgebase docs/*.txt
  customkb embed knowledgebase
  customkb query knowledgebase "What are the key features?"
  customkb optimize knowledgebase --analyze

Run 'customkb <command> -h' for detailed help on each command.


---

## customkb database --help

usage: customkb database [-h] [-l LANGUAGE] [--detect-language] [-f] [-q] [-v]
                         [-d]
                         config_file [files ...]

Process text files into knowledgebase

positional arguments:
  config_file           Knowledgebase name
  files                 Files or patterns to process (e.g., *.txt docs/)

options:
  -h, --help            show this help message and exit
  -l LANGUAGE, --language LANGUAGE
                        Language for stopwords (en, fr, de, etc.)
  --detect-language     Auto-detect language per file
  -f, --force           Force reprocess existing files
  -q, --quiet           Disable verbose output
  -v, --verbose         Enable verbose output (default)
  -d, --debug           Enable debug output

---

## customkb embed --help

usage: customkb embed [-h] [-r] [-q] [-v] [-d] config_file

Generate vector embeddings for text in knowledgebase

positional arguments:
  config_file           Knowledgebase name

options:
  -h, --help            show this help message and exit
  -r, --reset-database  Reset embedding status flags
  -q, --quiet           Disable verbose output
  -v, --verbose         Enable verbose output (default)
  -d, --debug           Enable debug output

---

## customkb query --help

usage: customkb query [-h] [-Q QUERY_FILE] [-c] [-R ROLE] [-m MODEL]
                      [-k TOP_K] [-s CONTEXT_SCOPE] [-t TEMPERATURE]
                      [-M MAX_TOKENS] [-f FORMAT] [-p PROMPT_TEMPLATE]
                      [--category CATEGORY] [--categories CATEGORIES] [-q]
                      [-v] [-d]
                      config_file query_text

Search knowledgebase and get AI-powered answers

positional arguments:
  config_file           Knowledgebase name
  query_text            Search query text

options:
  -h, --help            show this help message and exit
  -Q QUERY_FILE, --query_file QUERY_FILE
                        Read query from file
  -c, --context, --context-only
                        Return only context without AI response
  -R ROLE, --role ROLE  Custom system role for AI
  -m MODEL, --model MODEL
                        AI model to use (e.g., gpt-4, claude-3-sonnet)
  -k TOP_K, --top-k TOP_K
                        Number of search results to return
  -s CONTEXT_SCOPE, --context-scope CONTEXT_SCOPE
                        Context segments per result
  -t TEMPERATURE, --temperature TEMPERATURE
                        Model temperature (0.0-2.0)
  -M MAX_TOKENS, --max-tokens MAX_TOKENS
                        Maximum response tokens
  -f FORMAT, --format FORMAT
                        Output format: xml, json, markdown, plain
  -p PROMPT_TEMPLATE, --prompt-template PROMPT_TEMPLATE
                        Prompt style: default, instructive, scholarly,
                        concise, analytical, conversational, technical
  --category CATEGORY   Filter results by category (exact match)
  --categories CATEGORIES
                        Filter results by multiple categories (comma-
                        separated)
  -q, --quiet           Disable verbose output
  -v, --verbose         Enable verbose output (default)
  -d, --debug           Enable debug output

---

## customkb edit --help

usage: customkb edit [-h] [-q] [-v] [-d] config_file

Edit knowledgebase configuration file

positional arguments:
  config_file    Knowledgebase name

options:
  -h, --help     show this help message and exit
  -q, --quiet    Disable verbose output
  -v, --verbose  Enable verbose output (default)
  -d, --debug    Enable debug output

---

## customkb optimize --help

usage: customkb optimize [-h] [-n] [-a] [-s] [-m MEMORY_GB] [-q] [-v] [-d]
                         [config_file]

Optimize performance and create database indexes

positional arguments:
  config_file           Knowledgebase name (default: all KBs)

options:
  -h, --help            show this help message and exit
  -n, --dry-run         Preview changes without applying
  -a, --analyze         Analyze and show recommendations
  -s, --show-tiers      Display all memory tier settings
  -m MEMORY_GB, --memory-gb MEMORY_GB
                        Override detected memory (GB)
  -q, --quiet           Disable verbose output
  -v, --verbose         Enable verbose output (default)
  -d, --debug           Enable debug output

---

## customkb verify-indexes --help

usage: customkb verify-indexes [-h] [-q] [-v] [-d] config_file

Check database indexes for performance issues

positional arguments:
  config_file    Knowledgebase name

options:
  -h, --help     show this help message and exit
  -q, --quiet    Disable verbose output
  -v, --verbose  Enable verbose output (default)
  -d, --debug    Enable debug output

---

## customkb bm25 --help

usage: customkb bm25 [-h] [--force] config_file

Build BM25 index for keyword + semantic hybrid search

positional arguments:
  config_file  Knowledgebase name

options:
  -h, --help   show this help message and exit
  --force      Force rebuild existing index

---

