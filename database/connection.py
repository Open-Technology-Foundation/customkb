#!/usr/bin/env python
"""
Database connection management for CustomKB.

This module handles SQLite database connections, initialization,
and connection lifecycle management.
"""

import sqlite3
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Any

from utils.exceptions import ConnectionError as CustomConnectionError
from utils.exceptions import DatabaseError
from utils.logging_config import get_logger
from utils.security_utils import validate_table_name

logger = get_logger(__name__)


def connect_to_database(kb: Any) -> None:
  """
  Initialize database connection for a KnowledgeBase.

  Sets up SQLite connection with optimized pragmas and creates
  necessary tables if they don't exist.

  Args:
      kb: KnowledgeBase instance to connect

  Raises:
      ConnectionError: If connection fails
      DatabaseError: If table creation fails
  """
  try:
    # Get database path
    db_path = kb.knowledge_base_db
    logger.info(f"Connecting to database: {db_path}")

    # Create connection
    kb.sql_connection = sqlite3.connect(
      db_path,
      timeout=30.0,
      check_same_thread=False
    )
    kb.sql_cursor = kb.sql_connection.cursor()

    # Set optimized pragmas
    kb.sql_cursor.execute("PRAGMA foreign_keys = ON")
    kb.sql_cursor.execute("PRAGMA journal_mode = WAL")
    kb.sql_cursor.execute("PRAGMA synchronous = NORMAL")
    kb.sql_cursor.execute("PRAGMA cache_size = -64000")  # 64MB cache
    kb.sql_cursor.execute("PRAGMA temp_store = MEMORY")
    kb.sql_cursor.execute("PRAGMA mmap_size = 268435456")  # 256MB mmap

    # Create tables if needed
    create_tables(kb)

    # Check for migrations
    if hasattr(kb, 'enable_hybrid_search') and kb.enable_hybrid_search:
      from .migrations import migrate_for_bm25
      migrate_for_bm25(kb)

    kb.sql_connection.commit()
    logger.info("Database connection established successfully")

  except sqlite3.Error as e:
    logger.error(f"Database connection failed: {e}")
    raise CustomConnectionError(f"Failed to connect to database: {e}") from e
  except (FileNotFoundError, PermissionError, OSError, AttributeError) as e:
    logger.error(f"Unexpected error during database connection: {e}")
    raise DatabaseError(f"Database initialization failed: {e}") from e


def create_tables(kb: Any) -> None:
  """
  Create necessary database tables if they don't exist.

  Args:
      kb: KnowledgeBase instance with database connection

  Raises:
      DatabaseError: If table creation fails
  """
  try:
    # Determine table name
    table_name = getattr(kb, 'table_name', 'docs')

    # Validate table name to prevent SQL injection
    if not validate_table_name(table_name):
      raise ValueError(f"Invalid table name: {table_name}")

    # Main documents table
    kb.sql_cursor.execute(f'''
      CREATE TABLE IF NOT EXISTS {table_name} (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        sid INTEGER,
        sourcedoc TEXT,
        keyphrase TEXT,
        processed TEXT,
        embedtext TEXT,
        token_count INTEGER DEFAULT 0,
        originaltext TEXT,
        language TEXT DEFAULT 'en',
        metadata TEXT,
        embedded INTEGER DEFAULT 0,
        file_hash TEXT,
        bm25_tokens TEXT,
        doc_length INTEGER DEFAULT 0,
        keyphrase_processed INTEGER DEFAULT 0,
        primary_category TEXT,
        categories TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
      )
    ''')

    # Metadata table
    kb.sql_cursor.execute('''
      CREATE TABLE IF NOT EXISTS file_metadata (
        file_hash TEXT PRIMARY KEY,
        file_path TEXT NOT NULL,
        file_size INTEGER,
        modified_time TIMESTAMP,
        processed_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        chunk_count INTEGER DEFAULT 0,
        status TEXT DEFAULT 'processed'
      )
    ''')

    # Categories table (if categorization is enabled)
    if hasattr(kb, 'enable_categorization') and kb.enable_categorization:
      kb.sql_cursor.execute('''
        CREATE TABLE IF NOT EXISTS categories (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          name TEXT UNIQUE NOT NULL,
          description TEXT,
          parent_id INTEGER,
          created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
          FOREIGN KEY (parent_id) REFERENCES categories(id)
        )
      ''')

    logger.debug(f"Tables created/verified for {table_name}")

  except sqlite3.Error as e:
    logger.error(f"Table creation failed: {e}")
    raise DatabaseError(f"Failed to create tables: {e}") from e


def close_database(kb: Any) -> None:
  """
  Close database connection for a KnowledgeBase.

  Commits any pending transactions and closes the connection.

  Args:
      kb: KnowledgeBase instance with open connection
  """
  try:
    if hasattr(kb, 'sql_connection') and kb.sql_connection:
      # Commit any pending transactions
      try:
        kb.sql_connection.commit()
      except sqlite3.Error as e:
        logger.warning(f"Error committing final transaction: {e}")

      # Close cursor
      if hasattr(kb, 'sql_cursor') and kb.sql_cursor:
        kb.sql_cursor.close()
        kb.sql_cursor = None

      # Close connection
      kb.sql_connection.close()
      kb.sql_connection = None

      logger.info("Database connection closed")
  except (AttributeError, sqlite3.Error) as e:
    logger.error(f"Error closing database connection: {e}")


@contextmanager
def database_connection(kb: Any):
  """
  Context manager for database operations.

  Ensures proper connection handling and cleanup.

  Args:
      kb: KnowledgeBase instance

  Yields:
      KnowledgeBase instance with active connection

  Example:
      with database_connection(kb) as connected_kb:
          connected_kb.sql_cursor.execute("SELECT * FROM docs")
  """
  # Track if we opened the connection
  opened_connection = False

  try:
    # Connect if not already connected
    if not hasattr(kb, 'sql_connection') or kb.sql_connection is None:
      connect_to_database(kb)
      opened_connection = True

    yield kb

    # Commit on successful completion
    if kb.sql_connection:
      kb.sql_connection.commit()

  except (sqlite3.Error, OSError):
    # Rollback on error
    if hasattr(kb, 'sql_connection') and kb.sql_connection:
      with suppress(sqlite3.Error):
        kb.sql_connection.rollback()
    raise
  finally:
    # Only close if we opened it
    if opened_connection:
      close_database(kb)


@contextmanager
def sqlite_connection(db_path: str):
  """
  Context manager for standalone SQLite connections.

  Provides a simple way to work with SQLite databases without
  a KnowledgeBase instance.

  Args:
      db_path: Path to SQLite database file

  Yields:
      Tuple of (connection, cursor)

  Example:
      with sqlite_connection('/path/to/db.sqlite') as (conn, cursor):
          cursor.execute("SELECT COUNT(*) FROM docs")
          count = cursor.fetchone()[0]
  """
  conn = None
  cursor = None

  try:
    # Validate path
    if not Path(db_path).exists():
      raise CustomConnectionError(f"Database not found: {db_path}")

    # Create connection
    conn = sqlite3.connect(db_path, timeout=30.0)
    cursor = conn.cursor()

    # Set basic pragmas
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA synchronous=NORMAL")

    yield conn, cursor

    # Commit on success
    conn.commit()

  except sqlite3.Error as e:
    if conn:
      conn.rollback()
    raise DatabaseError(f"SQLite operation failed: {e}") from e
  finally:
    # Clean up
    if cursor:
      cursor.close()
    if conn:
      conn.close()


def get_connection_info(kb: Any) -> dict:
  """
  Get information about the current database connection.

  Args:
      kb: KnowledgeBase instance

  Returns:
      Dictionary with connection information
  """
  info = {
    'connected': False,
    'database_path': None,
    'table_count': 0,
    'row_count': 0
  }

  if hasattr(kb, 'sql_connection') and kb.sql_connection:
    try:
      info['connected'] = True
      info['database_path'] = kb.knowledge_base_db

      # Get table count
      kb.sql_cursor.execute(
        "SELECT COUNT(*) FROM sqlite_master WHERE type='table'"
      )
      info['table_count'] = kb.sql_cursor.fetchone()[0]

      # Get row count from main table
      table_name = getattr(kb, 'table_name', 'docs')
      if validate_table_name(table_name):
        kb.sql_cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        info['row_count'] = kb.sql_cursor.fetchone()[0]
      else:
        logger.warning(f"Invalid table name in connection info: {table_name}")
        info['row_count'] = 0

    except sqlite3.Error as e:
      logger.warning(f"Error getting connection info: {e}")

  return info


#fin
