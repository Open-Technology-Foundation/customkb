#!/usr/bin/env python
"""
Database migration utilities for CustomKB.

This module handles schema migrations and upgrades for the database.
"""

import sqlite3
from typing import List, Dict, Any, Optional
from datetime import datetime

from utils.logging_config import get_logger
from utils.exceptions import DatabaseError

logger = get_logger(__name__)


def get_current_schema_version(kb: Any) -> int:
  """
  Get the current schema version from the database.
  
  Args:
      kb: KnowledgeBase instance with database connection
      
  Returns:
      Current schema version (0 if not tracked)
  """
  try:
    # Check if migration table exists
    kb.sql_cursor.execute("""
      SELECT name FROM sqlite_master 
      WHERE type='table' AND name='schema_migrations'
    """)
    
    if not kb.sql_cursor.fetchone():
      # Migration table doesn't exist, assume version 0
      return 0
    
    # Get current version
    kb.sql_cursor.execute("""
      SELECT MAX(version) FROM schema_migrations 
      WHERE applied_at IS NOT NULL
    """)
    
    result = kb.sql_cursor.fetchone()
    return result[0] if result[0] else 0
    
  except sqlite3.Error as e:
    logger.warning(f"Error getting schema version: {e}")
    return 0


def create_migration_table(kb: Any) -> None:
  """
  Create the schema migrations tracking table.
  
  Args:
      kb: KnowledgeBase instance with database connection
  """
  try:
    kb.sql_cursor.execute("""
      CREATE TABLE IF NOT EXISTS schema_migrations (
        version INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        applied_at TIMESTAMP,
        rollback_at TIMESTAMP,
        description TEXT
      )
    """)
    kb.sql_connection.commit()
    logger.info("Migration tracking table created")
    
  except sqlite3.Error as e:
    logger.error(f"Failed to create migration table: {e}")
    raise DatabaseError(f"Migration table creation failed: {e}") from e


def record_migration(kb: Any, version: int, name: str, description: str = "") -> None:
  """
  Record a completed migration.
  
  Args:
      kb: KnowledgeBase instance
      version: Migration version number
      name: Migration name
      description: Optional description
  """
  try:
    kb.sql_cursor.execute("""
      INSERT OR REPLACE INTO schema_migrations 
      (version, name, applied_at, description)
      VALUES (?, ?, ?, ?)
    """, (version, name, datetime.now(), description))
    kb.sql_connection.commit()
    logger.info(f"Recorded migration {version}: {name}")
    
  except sqlite3.Error as e:
    logger.error(f"Failed to record migration: {e}")
    raise DatabaseError(f"Migration recording failed: {e}") from e


def migrate_for_bm25(kb: Any) -> bool:
  """
  Add BM25 token column for hybrid search support.
  
  Args:
      kb: KnowledgeBase instance with database connection
      
  Returns:
      True if migration was applied, False if already exists
  """
  try:
    table_name = getattr(kb, 'table_name', 'docs')
    
    # Check if bm25_tokens column exists
    kb.sql_cursor.execute(f"PRAGMA table_info({table_name})")
    columns = [row[1] for row in kb.sql_cursor.fetchall()]
    
    if 'bm25_tokens' in columns:
      logger.debug("BM25 tokens column already exists")
      return False
    
    logger.info("Adding bm25_tokens column for hybrid search...")
    
    # Add the column
    kb.sql_cursor.execute(f"""
      ALTER TABLE {table_name} 
      ADD COLUMN bm25_tokens TEXT
    """)
    
    # Create index for better performance
    kb.sql_cursor.execute(f"""
      CREATE INDEX IF NOT EXISTS idx_bm25_tokens 
      ON {table_name}(bm25_tokens)
    """)
    
    kb.sql_connection.commit()
    
    # Record migration
    create_migration_table(kb)
    record_migration(kb, 1, "add_bm25_tokens", "Added BM25 tokens column for hybrid search")
    
    logger.info("BM25 migration completed successfully")
    return True
    
  except sqlite3.Error as e:
    logger.error(f"BM25 migration failed: {e}")
    kb.sql_connection.rollback()
    raise DatabaseError(f"Failed to migrate for BM25: {e}") from e


def migrate_add_categories(kb: Any) -> bool:
  """
  Add category columns for document categorization.
  
  Args:
      kb: KnowledgeBase instance
      
  Returns:
      True if migration was applied, False if already exists
  """
  try:
    table_name = getattr(kb, 'table_name', 'docs')
    
    # Check if category columns exist
    kb.sql_cursor.execute(f"PRAGMA table_info({table_name})")
    columns = [row[1] for row in kb.sql_cursor.fetchall()]
    
    if 'primary_category' in columns:
      logger.debug("Category columns already exist")
      return False
    
    logger.info("Adding category columns...")
    
    # Add columns
    kb.sql_cursor.execute(f"""
      ALTER TABLE {table_name} 
      ADD COLUMN primary_category TEXT
    """)
    
    kb.sql_cursor.execute(f"""
      ALTER TABLE {table_name} 
      ADD COLUMN categories TEXT
    """)
    
    # Create indexes
    kb.sql_cursor.execute(f"""
      CREATE INDEX IF NOT EXISTS idx_primary_category 
      ON {table_name}(primary_category)
    """)
    
    kb.sql_connection.commit()
    
    # Record migration (create table if needed)
    create_migration_table(kb)
    record_migration(kb, 2, "add_categories", "Added category columns for document classification")
    
    logger.info("Category migration completed successfully")
    return True
    
  except sqlite3.Error as e:
    logger.error(f"Category migration failed: {e}")
    kb.sql_connection.rollback()
    raise DatabaseError(f"Failed to add category columns: {e}") from e


def migrate_add_timestamps(kb: Any) -> bool:
  """
  Add timestamp columns for tracking.
  
  Args:
      kb: KnowledgeBase instance
      
  Returns:
      True if migration was applied, False if already exists
  """
  try:
    table_name = getattr(kb, 'table_name', 'docs')
    
    # Check if timestamp columns exist
    kb.sql_cursor.execute(f"PRAGMA table_info({table_name})")
    columns = [row[1] for row in kb.sql_cursor.fetchall()]
    
    if 'created_at' in columns:
      logger.debug("Timestamp columns already exist")
      return False
    
    logger.info("Adding timestamp columns...")
    
    # Add columns with defaults
    kb.sql_cursor.execute(f"""
      ALTER TABLE {table_name} 
      ADD COLUMN created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    """)
    
    kb.sql_cursor.execute(f"""
      ALTER TABLE {table_name} 
      ADD COLUMN updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    """)
    
    # Create trigger to update updated_at
    kb.sql_cursor.execute(f"""
      CREATE TRIGGER IF NOT EXISTS update_{table_name}_timestamp
      AFTER UPDATE ON {table_name}
      BEGIN
        UPDATE {table_name} 
        SET updated_at = CURRENT_TIMESTAMP 
        WHERE id = NEW.id;
      END
    """)
    
    kb.sql_connection.commit()
    
    # Record migration (create table if needed)
    create_migration_table(kb)
    record_migration(kb, 3, "add_timestamps", "Added timestamp columns for tracking")
    
    logger.info("Timestamp migration completed successfully")
    return True
    
  except sqlite3.Error as e:
    logger.error(f"Timestamp migration failed: {e}")
    kb.sql_connection.rollback()
    raise DatabaseError(f"Failed to add timestamp columns: {e}") from e


def run_all_migrations(kb: Any) -> int:
  """
  Run all pending migrations in order.
  
  Args:
      kb: KnowledgeBase instance
      
  Returns:
      Number of migrations applied
  """
  # Define migrations in order
  migrations = [
    (1, "add_bm25_tokens", migrate_for_bm25),
    (2, "add_categories", migrate_add_categories),
    (3, "add_timestamps", migrate_add_timestamps),
  ]
  
  # Create migration table if needed
  create_migration_table(kb)
  
  # Get current version
  current_version = get_current_schema_version(kb)
  logger.info(f"Current schema version: {current_version}")
  
  applied = 0
  
  for version, name, migration_func in migrations:
    if version > current_version:
      logger.info(f"Applying migration {version}: {name}")
      try:
        if migration_func(kb):
          applied += 1
      except Exception as e:
        logger.error(f"Migration {version} failed: {e}")
        raise
  
  if applied > 0:
    logger.info(f"Applied {applied} migrations")
  else:
    logger.info("Database is up to date")
  
  return applied


def check_migration_status(kb: Any) -> Dict[str, Any]:
  """
  Check the status of database migrations.
  
  Args:
      kb: KnowledgeBase instance
      
  Returns:
      Dictionary with migration status information
  """
  status = {
    'current_version': 0,
    'latest_version': 3,  # Update when adding new migrations
    'applied_migrations': [],
    'pending_migrations': [],
    'is_up_to_date': False
  }
  
  try:
    # Get current version
    current = get_current_schema_version(kb)
    status['current_version'] = current
    
    # Check if migration table exists
    kb.sql_cursor.execute("""
      SELECT name FROM sqlite_master 
      WHERE type='table' AND name='schema_migrations'
    """)
    
    if kb.sql_cursor.fetchone():
      # Get applied migrations
      kb.sql_cursor.execute("""
        SELECT version, name, applied_at 
        FROM schema_migrations 
        WHERE applied_at IS NOT NULL
        ORDER BY version
      """)
      
      for row in kb.sql_cursor.fetchall():
        status['applied_migrations'].append({
          'version': row[0],
          'name': row[1],
          'applied_at': row[2]
        })
    
    # Calculate pending migrations
    all_migrations = [
      (1, "add_bm25_tokens"),
      (2, "add_categories"),
      (3, "add_timestamps"),
    ]
    
    for version, name in all_migrations:
      if version > current:
        status['pending_migrations'].append({
          'version': version,
          'name': name
        })
    
    status['is_up_to_date'] = current >= status['latest_version']
    
  except Exception as e:
    logger.error(f"Error checking migration status: {e}")
    status['error'] = str(e)
  
  return status


#fin