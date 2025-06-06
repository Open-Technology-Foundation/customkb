#!/usr/bin/env python
"""
# CustomKB: AI-Powered Knowledge Base System

CustomKB enables the creation, management, and querying of custom knowledge bases from text datasets using vector embeddings and large language models. It provides semantic search capabilities with context-aware AI responses.

## Key Features

- **Vector-Based Knowledge Storage**:
  - SQLite database for structured text storage with metadata
  - FAISS vector indices for efficient similarity search
  - Intelligent chunking with configurable token limits

- **Advanced Embedding Generation**:
  - Integration with modern embedding models (OpenAI, Anthropic)
  - Batch processing with checkpoint saving
  - Caching to reduce redundant API calls
  - Optimized vector indexing based on dataset size

- **Context-Aware Querying**:
  - Semantic similarity search using vector embeddings
  - Scope-aware context retrieval with relevance ranking
  - Integration with multiple LLMs (OpenAI, Claude, Llama)
  - XML-formatted reference context for precise AI prompting

- **Preprocessing & NLP Features**:
  - Named entity recognition and preservation
  - Multi-language stopword filtering
  - Lemmatization and text normalization
  - Metadata extraction and tracking

- **Flexible Configuration**:
  - `.cfg` files with environment variable overrides
  - Model aliasing through Models.json
  - Domain-style knowledge base naming

## Usage

```
customkb <command> <config_file> [options]
```

### Commands

1. **database**: Process text files into the knowledge base
   ```
   customkb database <config_file> [files...]
   ```
   - **Config**: Knowledge base configuration file
   - **Files**: File paths or glob patterns to process
   - **Options**:
     - `-l, --language`: Stopwords language (default: english)
     - `-v/q, --verbose/--quiet`: Control output verbosity
     - `-d, --debug`: Enable debug mode

2. **embed**: Generate embeddings for stored text
   ```
   customkb embed <config_file> [options]
   ```
   - **Config**: Knowledge base configuration file
   - **Options**:
     - `-r, --reset-database`: Reset embedding status flags

3. **query**: Search the knowledge base
   ```
   customkb query <config_file> <query_text> [options]
   ```
   - **Config**: Knowledge base configuration file
   - **Query**: The search text
   - **Options**:
     - `-Q, --query_file`: Load additional query text from file
     - `-c, --context-only`: Return context without AI response
     - `-R, --role`: Set custom system role
     - `-m, --model`: Specify LLM model to use
     - `-k, --top-k`: Number of results to return
     - `-s, --context-scope`: Context segments per result
     - `-t, --temperature`: Model temperature
     - `-M, --max-tokens`: Maximum output tokens

4. **edit**: Edit configuration file
   ```
   customkb edit <config_file>
   ```
   - Opens the configuration file in system editor

5. **help**: Display usage information
   ```
   customkb help
   ```

## Configuration

CustomKB uses INI-style configuration files with settings for:
- Embedding model and dimensions
- Database token limits
- Query model and parameters
- Additional context files

Settings are resolved in this order:
1. Command-line arguments
2. Environment variables
3. Configuration file values
4. Default values
"""

import sys
import os
import argparse
import textwrap
import signal
import time
from pathlib import Path
from typing import List, Optional, Dict, Any

# Ensure the project root is in the Python path
project_root = Path(__file__).parent.absolute()
if str(project_root) not in sys.path:
  sys.path.insert(0, str(project_root))

# Import modules
from utils.logging_utils import setup_logging, dashes, elapsed_time
from config.config_manager import get_fq_cfg_filename
from database.db_manager import process_database
from embedding.embed_manager import process_embeddings
from query.query_manager import process_query
from models.model_manager import get_canonical_model
from utils.text_utils import get_env

# Initialize module logger
logger = None

def customkb_usage() -> str:
  """Return the usage information for the CustomKB script."""
  from version import VERSION
  import __main__
  helpstr = f'''CustomKB {VERSION}: AI-Powered Knowledge Base System

{__main__.__doc__}
{dashes(0, '=')}'''
  return helpstr

def edit_config(args: argparse.Namespace, logger) -> int:
  """
  Edit the knowledge base configuration file.
  
  Opens the knowledge base configuration file in the system's default editor
  (or falls back to 'joe' if no default is set). The configuration file
  controls various aspects of the knowledge base behavior.
  
  Args:
      args: Command-line arguments containing:
          config_file: Path to knowledge base configuration
          verbose: Enable verbose output
          debug: Enable debug output
      logger: Initialized logger instance
          
  Returns:
      0 on success, 1 on failure.
  """
  # Get config file path
  config_file = get_fq_cfg_filename(args.config_file)
  if not config_file:
    sys.exit(1)
    
  logger.debug(f"{config_file=}")

  editor = os.getenv('EDITOR')
  if not editor:
    logger.warning("EDITOR envvar not defined; defaulting to 'joe'")
    editor = 'joe'

  try:
    import subprocess
    from utils.security_utils import validate_file_path, safe_log_error
    
    # Validate the config file path
    try:
      validated_config = validate_file_path(config_file, ['.cfg'])
    except ValueError as e:
      logger.error(f'Invalid config file path: {e}')
      return 1
    
    # Check if file exists
    if not os.path.exists(validated_config):
      logger.error(f'Config file does not exist: {validated_config}')
      return 1
    
    # Use safe subprocess call without shell=True
    try:
      subprocess.run([editor, validated_config], check=True)
      return 0
    except subprocess.CalledProcessError as e:
      safe_log_error(f'Editor process failed: {e}')
      return 1
    except FileNotFoundError:
      logger.error(f'Editor not found: {editor}. Please check EDITOR environment variable.')
      return 1
      
  except Exception as e:
    safe_log_error(f'Edit error: {e}')
    return 1

def main() -> None:
  """
  Entry point for the CustomKB application.
  This function sets up command-line argument parsing, handles user commands,
  and dispatches to the appropriate subcommands.
  """
  global logger
  
  # Set up signal handler for graceful exit
  def signal_handler(sig, frame):
    if logger:
      logger.info("Interrupted by user (Ctrl+C)")
    else:
      print('^C', file=sys.stderr)
    sys.exit(1)
  signal.signal(signal.SIGINT, signal_handler)

  # Display usage if no arguments provided
  if len(sys.argv) < 2:
    print(customkb_usage())
    sys.exit(0)

  # Set up argument parser
  prog = 'customkb'
  main_parser = argparse.ArgumentParser(prog=prog)
  subparsers = main_parser.add_subparsers(dest='command', required=True)

  # QUERY
  query_parser = subparsers.add_parser(
    "query",
    description=textwrap.dedent(process_query.__doc__),
    formatter_class=argparse.RawDescriptionHelpFormatter,
  )
  query_parser.add_argument('config_file', help='Knowledgebase Configuration file')
  query_parser.add_argument('query_text', help='Query text')
  query_parser.add_argument('-Q', '--query_file', default='', help='Query text from file')
  query_parser.add_argument('-c', '--context', '--context-only', dest='context_only', action='store_true', help='Return only context')
  query_parser.add_argument('-R', '--role', default='', help='LLM System Role')
  query_parser.add_argument('-m', '--model', default='', help='LLM Model')
  query_parser.add_argument('-k', '--top-k', default=None, type=int, help='Return top TOP_K results')
  query_parser.add_argument('-s', '--context-scope', default=None, type=int, help='For each result segment, return SCOPE segments')
  query_parser.add_argument('-t', '--temperature', default='', help='Temperature')
  query_parser.add_argument('-M', '--max-tokens', default='', help='Max Output Tokens')
  query_parser.add_argument('-q', '--quiet', dest='verbose', action='store_false', help='Disable verbose output')
  query_parser.add_argument('-v', '--verbose', dest='verbose', action='store_true', help='Enable verbose output (default)')
  query_parser.add_argument('-d', '--debug', action='store_true', help='Enable debug output')
  query_parser.set_defaults(verbose=True)

  # DATABASE
  database_parser = subparsers.add_parser(
    'database',
    description=textwrap.dedent(process_database.__doc__),
    formatter_class=argparse.RawDescriptionHelpFormatter,
  )
  database_parser.add_argument('config_file', help='Knowledgebase Configuration file')
  database_parser.add_argument('files', nargs='*', help='List of file paths or patterns to process into the database.')
  database_parser.add_argument('-l', '--language', default='english', help='Language for stopwords')
  database_parser.add_argument('-f', '--force', action='store_true', help='Force reprocessing of files already in the database')
  database_parser.add_argument('-q', '--quiet', dest='verbose', action='store_false', help='Disable verbose output')
  database_parser.add_argument('-v', '--verbose', dest='verbose', action='store_true', help='Enable verbose output (default)')
  database_parser.add_argument('-d', '--debug', action='store_true', help='Enable debug output')
  database_parser.set_defaults(verbose=True)

  # EMBED
  embed_parser = subparsers.add_parser(
    'embed',
    description=textwrap.dedent(process_embeddings.__doc__),
    formatter_class=argparse.RawDescriptionHelpFormatter,
  )
  embed_parser.add_argument('config_file', help='Knowledgebase Configuration file')
  embed_parser.add_argument('-r', '--reset-database', action='store_true', help='Reset already-embedded flag in knowledgebase database file')
  embed_parser.add_argument('-q', '--quiet', dest='verbose', action='store_false', help='Disable verbose output')
  embed_parser.add_argument('-v', '--verbose', dest='verbose', action='store_true', help='Enable verbose output (default)')
  embed_parser.add_argument('-d', '--debug', action='store_true', help='Enable debug output')
  embed_parser.set_defaults(verbose=True)

  # EDIT
  edit_parser = subparsers.add_parser(
    'edit',
    description=textwrap.dedent(edit_config.__doc__),
    formatter_class=argparse.RawDescriptionHelpFormatter,
  )
  edit_parser.add_argument('config_file', help='Knowledgebase Configuration file')
  edit_parser.add_argument('-q', '--quiet', dest='verbose', action='store_false', help='Disable verbose output')
  edit_parser.add_argument('-v', '--verbose', dest='verbose', action='store_true', help='Enable verbose output (default)')
  edit_parser.add_argument('-d', '--debug', action='store_true', help='Enable debug output')
  edit_parser.set_defaults(verbose=True)

  # HELP
  help_parser = subparsers.add_parser(
    'help',
    description=textwrap.dedent(customkb_usage()),
    formatter_class=argparse.RawDescriptionHelpFormatter,
  )
  help_parser.set_defaults(verbose=True)
  
  # VERSION
  version_parser = subparsers.add_parser(
    'version',
    description='Display version information',
  )
  version_parser.add_argument('--build', action='store_true', help='Include build number')
  version_parser.set_defaults(verbose=True)

  # Parse arguments
  args = main_parser.parse_args()
  process_start_time = int(time.time())

  # Handle help/version commands without any logging
  if args.command in ['help', 'version']:
    if args.command == 'help':
      print(customkb_usage())
    elif args.command == 'version':
      from version import get_version
      version_info = f"CustomKB {get_version(args.build)}"
      print(version_info)
    sys.exit(0)

  # For all other commands, setup KB-specific logging with fail-fast behavior
  verbose = getattr(args, 'verbose', True)
  debug = getattr(args, 'debug', False)
  
  # Extract KB info from config file for logging setup
  from config.config_manager import get_fq_cfg_filename
  from utils.logging_utils import get_kb_info_from_config
  
  config_file_fq = get_fq_cfg_filename(args.config_file)
  if not config_file_fq:
    print("Error: Configuration file not found.", file=sys.stderr)
    sys.exit(1)
    
  try:
    kb_directory, kb_name = get_kb_info_from_config(config_file_fq)
  except Exception as e:
    print(f"Error: Failed to extract KB info from config file: {e}", file=sys.stderr)
    sys.exit(1)
  
  # Setup KB-specific logging - FAIL FAST if logging cannot be initialized
  logger = setup_logging(verbose, debug, 
                        config_file=config_file_fq,
                        kb_directory=kb_directory, 
                        kb_name=kb_name)
  
  if logger is None:
    print("Error: Failed to initialize logging system. Application cannot continue.", file=sys.stderr)
    sys.exit(1)

  # Execute the appropriate command
  try:
    if args.command == 'database':
      result = process_database(args, logger)
      print(result)  # Keep print for user output
      logger.debug(f"Database command completed: {result}")
    elif args.command == 'embed':
      result = process_embeddings(args, logger)
      print(result)  # Keep print for user output
      logger.debug(f"Embed command completed: {result}")
    elif args.command == 'query':
      # Set environment variables from command line arguments
      if args.model:
        args.model = get_canonical_model(args.model)['model']
        os.environ['QUERY_MODEL'] = args.model
      if args.top_k:
        os.environ['QUERY_TOP_K'] = str(args.top_k)
      if args.context_scope:
        os.environ['QUERY_CONTEXT_SCOPE'] = str(args.context_scope)
      if args.temperature:
        os.environ['QUERY_TEMPERATURE'] = args.temperature
      if args.max_tokens:
        os.environ['QUERY_MAX_TOKENS'] = args.max_tokens
      if args.role:
        os.environ['QUERY_ROLE'] = args.role
      result = process_query(args, logger)
      print(result)  # Keep print for user output
      logger.debug(f"Query command completed")
    elif args.command == 'edit':
      result = edit_config(args, logger)
      if result == 0:
        logger.info("Configuration file edited successfully")
      else:
        logger.error("Failed to edit configuration file")
      sys.exit(result)
    else:
      logger.error(f'Unknown command: {args.command}')
      sys.exit(1)
  except Exception as e:
    logger.error(f"Application error: {e}")
    sys.exit(1)
  finally:
    elapsed = elapsed_time(process_start_time)
    logger.info(f'Elapsed Time: {elapsed}')
    if not verbose:  # Only print to stderr if not verbose (to avoid duplicate with log)
      print(f'Elapsed Time: {elapsed}', file=sys.stderr)

if __name__ == "__main__":
  main()

#fin
