#!/usr/bin/env python
"""
CustomKB: AI-Powered Knowledgebase System

Create and query AI knowledgebases with semantic search and LLM integration.

Usage:
  customkb <command> <knowledge_base> [options]

Commands:
  database         Process text files into knowledgebase
  embed            Generate embeddings for stored text
  query            Search knowledgebase with AI responses
  edit             Edit knowledgebase configuration
  optimize         Optimize performance and create indexes
  verify-indexes   Check database index health
  bm25             Build BM25 index for hybrid search
  convert-encoding Convert text files to UTF-8 encoding
  version          Show version information
  help             Show this help message

Examples:
  customkb database knowledgebase docs/*.txt
  customkb embed knowledgebase
  customkb query knowledgebase "What are the key features?"
  customkb optimize knowledgebase --analyze

Run 'customkb <command> -h' for detailed help on each command.
"""

import argparse
import os
import signal
import sqlite3
import sys
import time
import warnings
from pathlib import Path

# Type hints are used inline, no need to import at module level

# Check Python version requirement

# Suppress PyTorch CUDA initialization warnings
warnings.filterwarnings("ignore", message="CUDA initialization: CUDA unknown error")

# Ensure the project root is in the Python path
project_root = Path(__file__).parent.absolute()
if str(project_root) not in sys.path:
  sys.path.insert(0, str(project_root))

# Import core modules only (performance optimization - other modules loaded conditionally)
from config.config_manager import KnowledgeBase, get_fq_cfg_filename
from utils.logging_config import elapsed_time, setup_logging

# Heavy modules (database, embedding, query) imported lazily when needed

# Initialize module logger
logger = None

def customkb_usage() -> str:
  """Return the usage information for the CustomKB script."""
  import __main__
  from version import VERSION
  return f'CustomKB {VERSION}\n\n{__main__.__doc__}'

def edit_config(args: argparse.Namespace, logger) -> int:
  """
  Edit the knowledgebase configuration file in system editor.

  Args:
      args: Command-line arguments with config_file
      logger: Logger instance

  Returns:
      0 on success, 1 on failure.
  """
  # Get config file path
  config_file = get_fq_cfg_filename(args.config_file)
  if not config_file:
    logger.error(f"Knowledgebase '{args.config_file}' not found")
    sys.exit(1)

  logger.debug(f"{config_file=}")

  editor = os.getenv('EDITOR')
  if not editor:
    # Load config to get default editor preference
    try:
      kb = KnowledgeBase(config_file)
      default_editor = getattr(kb, 'default_editor', 'joe')
      logger.warning(f"EDITOR envvar not defined; defaulting to '{default_editor}'")
      editor = default_editor
    except (FileNotFoundError, ValueError, KeyError, OSError):
      # Fallback if config loading fails
      logger.warning("EDITOR envvar not defined; defaulting to 'joe'")
      editor = 'joe'

  try:
    import subprocess

    from utils.security_utils import safe_log_error, validate_file_path

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

  except (PermissionError, OSError, ImportError) as e:
    safe_log_error(f'Edit error: {e}')
    return 1

def rebuild_bm25_index(args: argparse.Namespace, logger) -> str:
  """
  Build BM25 index for keyword-based hybrid search.

  Args:
      args: Command-line arguments with config_file and force flag
      logger: Logger instance

  Returns:
      Status message string.
  """
  try:
    from config.config_manager import KnowledgeBase, get_fq_cfg_filename
    from database.connection import close_database, connect_to_database
    from embedding.bm25_manager import build_bm25_index, load_bm25_index

    # Get configuration file
    cfgfile = get_fq_cfg_filename(args.config_file)
    if not cfgfile:
      return f"Error: Knowledgebase '{args.config_file}' not found."

    logger.info(f"Building BM25 index for: {cfgfile}")

    # Initialize knowledgebase
    kb = KnowledgeBase(cfgfile)

    # Check if hybrid search is enabled
    if not kb.enable_hybrid_search:
      return "Error: Hybrid search is disabled in configuration. Set enable_hybrid_search=true in [ALGORITHMS] section."

    # Connect to database
    connect_to_database(kb)

    try:
      # Check if index already exists
      if not args.force:
        existing_index = load_bm25_index(kb)
        if existing_index:
          close_database(kb)
          return "BM25 index already exists. Use --force to rebuild."

      # Build the index
      bm25_index = build_bm25_index(kb)

      if bm25_index:
        return f"BM25 index built successfully for {kb.knowledge_base_name}"
      else:
        return "Failed to build BM25 index. Check logs for details."

    finally:
      close_database(kb)

  except (FileNotFoundError, ValueError, OSError, RuntimeError, sqlite3.DatabaseError) as e:
    logger.error(f"Error building BM25 index: {e}")
    return f"Error: {e}"

def main() -> None:
  """
  CustomKB application entry point.
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
    description="Search knowledgebase and get AI-powered answers",
    formatter_class=argparse.RawDescriptionHelpFormatter,
  )
  query_parser.add_argument('config_file', metavar='kb_name', help='Knowledgebase name')
  query_parser.add_argument('query_text', help='Search query text')
  query_parser.add_argument('-Q', '--query_file', default='', help='Read query from file')
  query_parser.add_argument('-c', '--context', '--context-only', dest='context_only', action='store_true', help='Return only context without AI response')
  query_parser.add_argument('-R', '--role', default='', help='Custom system role for AI')
  query_parser.add_argument('-m', '--model', default='', help='AI model to use (e.g., gpt-4, claude-3-sonnet)')
  query_parser.add_argument('-k', '--top-k', default=None, type=int, help='Number of search results to return')
  query_parser.add_argument('-s', '--context-scope', default=None, type=int, help='Context segments per result')
  query_parser.add_argument('-t', '--temperature', default='', help='Model temperature (0.0-2.0)')
  query_parser.add_argument('-M', '--max-tokens', default='', help='Maximum response tokens')
  query_parser.add_argument('-f', '--format', default='', help='Output format: xml, json, markdown, plain')
  query_parser.add_argument('-p', '--prompt-template', default='', help='Prompt style: default, instructive, scholarly, concise, analytical, conversational, technical')
  query_parser.add_argument('--category', default='', help='Filter results by category (exact match)')
  query_parser.add_argument('--categories', default='', help='Filter results by multiple categories (comma-separated)')
  query_parser.add_argument('--context-files', nargs='+', help='Additional context files to include in the query')
  query_parser.add_argument('-q', '--quiet', dest='verbose', action='store_false', help='Disable verbose output')
  query_parser.add_argument('-v', '--verbose', dest='verbose', action='store_true', help='Enable verbose output (default)')
  query_parser.add_argument('-d', '--debug', action='store_true', help='Enable debug output')
  query_parser.set_defaults(verbose=True)

  # DATABASE
  database_parser = subparsers.add_parser(
    'database',
    description="Process text files into knowledgebase",
    formatter_class=argparse.RawDescriptionHelpFormatter,
  )
  database_parser.add_argument('config_file', metavar='kb_name', help='Knowledgebase name')
  database_parser.add_argument('files', nargs='*', help='Files or patterns to process (e.g., *.txt docs/)')
  database_parser.add_argument('-l', '--language', default='en', help='Language for stopwords (en, fr, de, etc.)')
  database_parser.add_argument('--detect-language', action='store_true', help='Auto-detect language per file')
  database_parser.add_argument('-f', '--force', action='store_true', help='Force reprocess existing files')
  database_parser.add_argument('-q', '--quiet', dest='verbose', action='store_false', help='Disable verbose output')
  database_parser.add_argument('-v', '--verbose', dest='verbose', action='store_true', help='Enable verbose output (default)')
  database_parser.add_argument('-d', '--debug', action='store_true', help='Enable debug output')
  database_parser.set_defaults(verbose=True)

  # EMBED
  embed_parser = subparsers.add_parser(
    'embed',
    description="Generate vector embeddings for text in knowledgebase",
    formatter_class=argparse.RawDescriptionHelpFormatter,
  )
  embed_parser.add_argument('config_file', metavar='kb_name', help='Knowledgebase name')
  embed_parser.add_argument('-r', '--reset-database', action='store_true', help='Reset embedding status flags')
  embed_parser.add_argument('-q', '--quiet', dest='verbose', action='store_false', help='Disable verbose output')
  embed_parser.add_argument('-v', '--verbose', dest='verbose', action='store_true', help='Enable verbose output (default)')
  embed_parser.add_argument('-d', '--debug', action='store_true', help='Enable debug output')
  embed_parser.set_defaults(verbose=True)

  # EDIT
  edit_parser = subparsers.add_parser(
    'edit',
    description="Edit knowledgebase configuration file",
    formatter_class=argparse.RawDescriptionHelpFormatter,
  )
  edit_parser.add_argument('config_file', metavar='kb_name', help='Knowledgebase name')
  edit_parser.add_argument('-q', '--quiet', dest='verbose', action='store_false', help='Disable verbose output')
  edit_parser.add_argument('-v', '--verbose', dest='verbose', action='store_true', help='Enable verbose output (default)')
  edit_parser.add_argument('-d', '--debug', action='store_true', help='Enable debug output')
  edit_parser.set_defaults(verbose=True)

  # BM25 REBUILD
  bm25_parser = subparsers.add_parser(
    'bm25',
    description='Build BM25 index for keyword + semantic hybrid search',
  )
  bm25_parser.add_argument('config_file', metavar='kb_name', help='Knowledgebase name')
  bm25_parser.add_argument('--force', action='store_true', help='Force rebuild existing index')

  # OPTIMIZE
  optimize_parser = subparsers.add_parser(
    'optimize',
    description='Optimize performance and create database indexes',
  )
  optimize_parser.add_argument('config_file', metavar='kb_name', nargs='?', help='Knowledgebase name (default: all KBs)')
  optimize_parser.add_argument('-n', '--dry-run', action='store_true', help='Preview changes without applying')
  optimize_parser.add_argument('-a', '--analyze', action='store_true', help='Analyze and show recommendations')
  optimize_parser.add_argument('-s', '--show-tiers', action='store_true', help='Display all memory tier settings')
  optimize_parser.add_argument('-m', '--memory-gb', type=float, help='Override detected memory (GB)')
  optimize_parser.add_argument('-q', '--quiet', dest='verbose', action='store_false', help='Disable verbose output')
  optimize_parser.add_argument('-v', '--verbose', dest='verbose', action='store_true', help='Enable verbose output (default)')
  optimize_parser.add_argument('-d', '--debug', action='store_true', help='Enable debug output')
  optimize_parser.set_defaults(verbose=True)

  # VERIFY-INDEXES
  verify_parser = subparsers.add_parser(
    'verify-indexes',
    description='Check database indexes for performance issues',
  )
  verify_parser.add_argument('config_file', metavar='kb_name', help='Knowledgebase name')
  verify_parser.add_argument('-q', '--quiet', dest='verbose', action='store_false', help='Disable verbose output')
  verify_parser.add_argument('-v', '--verbose', dest='verbose', action='store_true', help='Enable verbose output (default)')
  verify_parser.add_argument('-d', '--debug', action='store_true', help='Enable debug output')
  verify_parser.set_defaults(verbose=True)

  # CATEGORIZE
  categorize_parser = subparsers.add_parser(
    'categorize',
    description='Auto-categorize articles in knowledgebase using AI',
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
Examples:
  customkb categorize knowledgebase -S 10        # Process 10 sample articles
  customkb categorize knowledgebase --full       # Process all articles
  customkb categorize knowledgebase --fresh      # Ignore checkpoint, start fresh
  customkb categorize knowledgebase --list       # List existing categories
    """
  )
  categorize_parser.add_argument('config_file', metavar='kb_name', help='Knowledgebase name')
  categorize_parser.add_argument('-S', '--sample', type=int, metavar='N',
                                help='Process only N sample articles')
  categorize_parser.add_argument('-f', '--full', action='store_true',
                                help='Process all articles (default if neither --sample nor --full)')
  categorize_parser.add_argument('--fresh', action='store_true',
                                help='Ignore checkpoint, reprocess all articles')
  categorize_parser.add_argument('--import', dest='import_to_db', action='store_true',
                                help='Import categories to database after processing')
  categorize_parser.add_argument('--list', dest='list_categories', action='store_true',
                                help='List existing categories and counts')
  categorize_parser.add_argument('-m', '--model', default='claude-haiku-4-5',
                                help='AI model to use (default: claude-haiku-4-5)')
  categorize_parser.add_argument('-s', '--sampling', type=str, metavar='T-M-B',
                                help='Chunk sampling config (e.g., 5-10-5 for top-middle-bottom)')
  categorize_parser.add_argument('-M', '--max-concurrent', type=int, default=5,
                                help='Maximum concurrent API requests (default: 5)')
  categorize_parser.add_argument('-c', '--confidence-threshold', type=float, default=0.5,
                                help='Minimum confidence for category assignment (default: 0.5)')
  categorize_parser.add_argument('-D', '--no-dedup', dest='enable_deduplication',
                                action='store_false', default=True,
                                help='Disable category deduplication')
  categorize_parser.add_argument('--dedup-threshold', type=float, default=85.0,
                                help='Similarity threshold for deduplication (default: 85.0)')
  categorize_parser.add_argument('--fixed-categories', dest='use_variable_categories',
                                action='store_false', default=True,
                                help='Use fixed number of categories per article')
  categorize_parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output')
  categorize_parser.add_argument('-q', '--quiet', action='store_true', help='Minimal output')
  categorize_parser.add_argument('-d', '--debug', action='store_true', help='Enable debug output')
  categorize_parser.set_defaults(verbose=True)

  # CONVERT-ENCODING
  convert_encoding_parser = subparsers.add_parser(
    'convert-encoding',
    description='Convert text files to UTF-8 encoding in-place',
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
Examples:
  customkb convert-encoding *.txt              # Convert all .txt files
  customkb convert-encoding docs/ --recursive  # Convert all files in docs/
  customkb convert-encoding *.txt --dry-run    # Preview without converting
  customkb convert-encoding *.txt --no-backup  # Convert without backups
    """
  )
  convert_encoding_parser.add_argument('files', nargs='+', help='Files or patterns to convert')
  convert_encoding_parser.add_argument('--backup', dest='backup', action='store_true', default=True,
                                      help='Create backups of original files (default)')
  convert_encoding_parser.add_argument('--no-backup', dest='backup', action='store_false',
                                      help='Do not create backups')
  convert_encoding_parser.add_argument('--dry-run', action='store_true',
                                      help='Preview changes without converting files')
  convert_encoding_parser.add_argument('-r', '--recursive', action='store_true',
                                      help='Process directories recursively')
  convert_encoding_parser.add_argument('-v', '--verbose', action='store_true', default=True, help='Enable verbose output')
  convert_encoding_parser.add_argument('-q', '--quiet', dest='verbose', action='store_false', help='Minimal output')
  convert_encoding_parser.add_argument('-d', '--debug', action='store_true', help='Enable debug output')

  # HELP
  help_parser = subparsers.add_parser(
    'help',
    description='Show detailed help information',
  )
  help_parser.set_defaults(verbose=True)

  # VERSION
  version_parser = subparsers.add_parser(
    'version',
    description='Show version information',
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

  # Handle convert-encoding command without KB requirement
  if args.command == 'convert-encoding':
    # Setup basic console logging
    import logging
    logging.basicConfig(
      level=logging.DEBUG if args.debug else (logging.INFO if args.verbose else logging.WARNING),
      format='customkb:%(levelname)s: %(message)s'
    )
    logger = logging.getLogger(__name__)

    from utils.encoding_converter import convert_files_to_utf8, format_conversion_summary

    results = convert_files_to_utf8(
      file_patterns=args.files,
      backup=args.backup,
      dry_run=args.dry_run,
      recursive=args.recursive
    )

    summary = format_conversion_summary(results)
    print(summary)

    # Exit with error code if any conversions failed
    sys.exit(1 if results['failed'] > 0 else 0)

  # Handle optimize command when no config_file is provided
  if args.command == 'optimize' and not args.config_file:
    # Setup basic console logging for optimize command
    import logging
    logging.basicConfig(
      level=logging.DEBUG if args.debug else (logging.INFO if args.verbose else logging.WARNING),
      format='%(asctime)s - %(levelname)s - %(message)s',
      datefmt='%Y-%m-%d %H:%M:%S'
    )
    logger = logging.getLogger(__name__)

    logger.debug("Optimize command with no config_file")
    logger.debug(f"Args: {args}")

    # Set target from config_file for the optimization manager
    args.target = args.config_file
    from utils.optimization_manager import process_optimize
    result = process_optimize(args, logger)
    print(result)
    sys.exit(0)

  # For all other commands, setup KB-specific logging with fail-fast behavior
  verbose = getattr(args, 'verbose', True)
  debug = getattr(args, 'debug', False)

  # For optimize command with a config file, handle it specially
  if args.command == 'optimize' and args.config_file:
    # Setup basic console logging for optimize command
    import logging
    logging.basicConfig(
      level=logging.DEBUG if debug else (logging.INFO if verbose else logging.WARNING),
      format='%(asctime)s - %(levelname)s - %(message)s',
      datefmt='%Y-%m-%d %H:%M:%S'
    )
    logger = logging.getLogger(__name__)

    # Set target from config_file for the optimization manager
    args.target = args.config_file
    from utils.optimization_manager import process_optimize
    result = process_optimize(args, logger)
    print(result)
    sys.exit(0)

  # Extract KB info from config file for logging setup
  from config.config_manager import get_fq_cfg_filename
  from utils.logging_config import get_kb_info_from_config

  config_file_fq = get_fq_cfg_filename(args.config_file)
  if not config_file_fq:
    print(f"Error: Knowledgebase '{args.config_file}' not found in {os.getenv('VECTORDBS', '/var/lib/vectordbs')}", file=sys.stderr)
    sys.exit(1)

  try:
    kb_directory, kb_name = get_kb_info_from_config(config_file_fq)
  except (ValueError, KeyError, FileNotFoundError, OSError) as e:
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

  # Execute the appropriate command using pattern matching (Python 3.10+)
  try:
    match args.command:
      case 'database':
        from database.db_manager import process_database
        result = process_database(args, logger)
        print(result)  # Keep print for user output
        logger.debug(f"Database command completed: {result}")

      case 'embed':
        from embedding.embed_manager import process_embeddings
        result = process_embeddings(args, logger)
        print(result)  # Keep print for user output
        logger.debug(f"Embed command completed: {result}")

      case 'query':
        from models.model_manager import get_canonical_model
        from query.processing import process_query
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
        if args.prompt_template:
          # Validate prompt template
          from query.prompt_templates import validate_template_name
          if not validate_template_name(args.prompt_template):
            logger.error(f"Invalid prompt template: {args.prompt_template}")
            from query.prompt_templates import list_templates
            templates = list_templates()
            logger.info("Available templates:")
            for name, desc in templates.items():
              logger.info(f"  {name}: {desc}")
            sys.exit(1)
        result = process_query(args, logger)
        print(result)  # Keep print for user output
        logger.debug("Query command completed")

      case 'edit':
        result = edit_config(args, logger)
        if result == 0:
          logger.info("Configuration file edited successfully")
        else:
          logger.error("Failed to edit configuration file")
        sys.exit(result)

      case 'bm25':
        result = rebuild_bm25_index(args, logger)
        print(result)  # Keep print for user output
        logger.debug("BM25 command completed")

      case 'verify-indexes':
        from database.index_manager import process_verify_indexes
        result = process_verify_indexes(args, logger)
        print(result)  # Keep print for user output
        logger.debug("Verify-indexes command completed")

      case 'categorize':
        from categorize.categorize_manager import process_categorize
        result = process_categorize(args, logger)
        print(result)  # Keep print for user output
        logger.debug("Categorize command completed")

      case _:
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
