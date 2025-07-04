"""
Database index management for CustomKB.

This module provides functionality to verify and manage SQLite indexes
for optimal query performance.
"""

import sqlite3
from typing import List, Tuple, Dict, Optional
from pathlib import Path

from utils.logging_utils import get_logger
from config.config_manager import KnowledgeBase
from database.db_manager import sqlite_connection

logger = get_logger(__name__)


# Expected indexes for CustomKB databases
# Note: Current implementation uses 'docs' table
EXPECTED_INDEXES = [
  'idx_embedded',
  'idx_embedded_embedtext', 
  'idx_keyphrase_processed',
  'idx_sourcedoc',
  'idx_sourcedoc_sid'
]


def get_database_indexes(db_path: str) -> List[Tuple[str, Optional[str]]]:
  """
  Get all indexes from a SQLite database.
  
  Args:
      db_path: Path to the SQLite database
      
  Returns:
      List of tuples containing (index_name, index_sql)
  """
  with sqlite_connection(db_path) as (conn, cursor):
    cursor.execute("""
      SELECT name, sql 
      FROM sqlite_master 
      WHERE type='index' 
      ORDER BY name
    """)
    return cursor.fetchall()


def verify_indexes(db_path: str) -> Dict[str, bool]:
  """
  Verify that all expected indexes exist in the database.
  
  Args:
      db_path: Path to the SQLite database
      
  Returns:
      Dictionary mapping index names to their presence status
  """
  logger.info(f"Verifying indexes in: {db_path}")
  
  try:
    indexes = get_database_indexes(db_path)
    
    # Extract index names, excluding SQLite internal indexes
    found_indexes = [
      idx_name for idx_name, _ in indexes 
      if not idx_name.startswith('sqlite_')
    ]
    
    # Check each expected index
    results = {}
    for expected in EXPECTED_INDEXES:
      results[expected] = expected in found_indexes
      if results[expected]:
        logger.debug(f"Index {expected} is present")
      else:
        logger.warning(f"Index {expected} is MISSING")
    
    return results
    
  except Exception as e:
    logger.error(f"Error verifying indexes: {e}")
    raise


def get_table_name(db_path: str) -> str:
  """
  Determine the table name used in the database (chunks or docs).
  
  Args:
      db_path: Path to the SQLite database
      
  Returns:
      Table name ('chunks' or 'docs')
  """
  with sqlite_connection(db_path) as (conn, cursor):
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name IN ('chunks', 'docs')")
    result = cursor.fetchone()
    return result[0] if result else 'chunks'  # Default to 'chunks' for new databases


def create_missing_indexes(db_path: str, dry_run: bool = False) -> List[str]:
  """
  Create any missing indexes in the database.
  
  Args:
      db_path: Path to the SQLite database
      dry_run: If True, only report what would be done
      
  Returns:
      List of index names that were created (or would be created)
  """
  verification = verify_indexes(db_path)
  missing = [idx for idx, present in verification.items() if not present]
  
  if not missing:
    logger.info("All indexes are present")
    return []
  
  logger.info(f"Found {len(missing)} missing indexes")
  
  if dry_run:
    logger.info("DRY RUN - Would create the following indexes:")
    for idx in missing:
      logger.info(f"  - {idx}")
    return missing
  
  # Get the table name used in this database
  table_name = get_table_name(db_path)
  logger.debug(f"Database uses table: {table_name}")
  
  # Check if we have keyphrase column (newer schema) or just keyphrase_processed (older schema)
  with sqlite_connection(db_path) as (conn, cursor):
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = [row[1] for row in cursor.fetchall()]
    has_keyphrase = 'keyphrase' in columns
    has_processed = 'processed' in columns
  
  # Index creation SQL statements
  index_sql = {
    'idx_embedded': f'CREATE INDEX idx_embedded ON {table_name}(embedded)',
    'idx_embedded_embedtext': f'CREATE INDEX idx_embedded_embedtext ON {table_name}(embedded, embedtext)',
    'idx_sourcedoc': f'CREATE INDEX idx_sourcedoc ON {table_name}(sourcedoc)',
    'idx_sourcedoc_sid': f'CREATE INDEX idx_sourcedoc_sid ON {table_name}(sourcedoc, sid)'
  }
  
  # Add keyphrase index based on schema
  if has_keyphrase and has_processed:
    # Newer schema with both columns
    index_sql['idx_keyphrase_processed'] = f'CREATE INDEX idx_keyphrase_processed ON {table_name}(keyphrase, processed)'
  elif 'keyphrase_processed' in columns:
    # Older schema with combined column - create index on the single column
    index_sql['idx_keyphrase_processed'] = f'CREATE INDEX idx_keyphrase_processed ON {table_name}(keyphrase_processed)'
  # else: no keyphrase-related columns, skip this index
  
  created = []
  with sqlite_connection(db_path) as (conn, cursor):
    for idx in missing:
      if idx in index_sql:
        logger.info(f"Creating index: {idx}")
        try:
          cursor.execute(index_sql[idx])
          created.append(idx)
        except sqlite3.Error as e:
          logger.error(f"Failed to create index {idx}: {e}")
    
    conn.commit()
    logger.info(f"Successfully created {len(created)} indexes")
  
  return created


def process_verify_indexes(args, logger) -> str:
  """
  Process the verify-indexes command to check database index health.
  
  This command verifies that all expected performance indexes exist in the
  knowledge base SQLite database. Missing indexes can significantly impact
  query performance, especially for large databases.
  
  Expected indexes:
  - idx_embedded: For filtering embedded vs non-embedded documents
  - idx_embedded_embedtext: For efficient embedded text queries
  - idx_keyphrase_processed: For keyphrase-based searches
  - idx_sourcedoc: For filtering by source document
  - idx_sourcedoc_sid: For compound queries on source and section ID
  
  Args:
      args: Command line arguments containing:
          - config_file: Path to knowledge base configuration
      logger: Logger instance
      
  Returns:
      Formatted result message showing index status and recommendations
  """
  try:
    # Load configuration
    kb = KnowledgeBase(args.config_file)
    db_path = kb.knowledge_base_db
    
    if not Path(db_path).exists():
      return f"Error: Database not found at {db_path}"
    
    # Verify indexes
    verification = verify_indexes(db_path)
    
    # Build output
    output = [f"Database: {db_path}", "", "Index verification:", "-" * 60]
    
    missing = []
    for idx, present in sorted(verification.items()):
      if present:
        output.append(f"✓ {idx} - Present")
      else:
        output.append(f"✗ {idx} - MISSING")
        missing.append(idx)
    
    if missing:
      output.extend([
        "",
        f"⚠️  Missing {len(missing)} indexes: {', '.join(missing)}",
        "",
        "To create missing indexes, run:",
        f"  customkb optimize {args.config_file}"
      ])
    else:
      output.extend(["", "✅ All expected indexes are present!"])
    
    # Check for any unexpected indexes
    all_indexes = get_database_indexes(db_path)
    unexpected = []
    for idx_name, _ in all_indexes:
      if (not idx_name.startswith('sqlite_') and 
          idx_name not in EXPECTED_INDEXES):
        unexpected.append(idx_name)
    
    if unexpected:
      output.extend([
        "",
        "Additional indexes found:",
        *[f"  - {idx}" for idx in unexpected]
      ])
    
    return '\n'.join(output)
    
  except Exception as e:
    logger.error(f"Error verifying indexes: {e}")
    return f"Error: {e}"


#fin