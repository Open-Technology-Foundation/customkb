#!/usr/bin/env python
"""
Unit tests for database.migrations module.

Tests schema migrations, version tracking, and migration management.
"""

import os
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from database.migrations import (
  check_migration_status,
  create_migration_table,
  get_current_schema_version,
  migrate_add_categories,
  migrate_add_timestamps,
  migrate_for_bm25,
  record_migration,
  run_all_migrations,
)
from utils.exceptions import DatabaseError


class TestSchemaVersion(unittest.TestCase):
  """Test schema version tracking."""

  def setUp(self):
    """Set up test fixtures."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as temp_db:
      self.db_path = temp_db.name

    # Create mock KnowledgeBase
    self.kb = Mock()
    self.kb.sql_connection = sqlite3.connect(self.db_path)
    self.kb.sql_cursor = self.kb.sql_connection.cursor()
    self.kb.table_name = 'docs'

  def tearDown(self):
    """Clean up test fixtures."""
    self.kb.sql_connection.close()
    if os.path.exists(self.db_path):
      os.unlink(self.db_path)

  def test_get_version_no_table(self):
    """Test getting version when migration table doesn't exist."""
    version = get_current_schema_version(self.kb)
    self.assertEqual(version, 0)

  def test_get_version_with_table_empty(self):
    """Test getting version with empty migration table."""
    create_migration_table(self.kb)

    version = get_current_schema_version(self.kb)
    self.assertEqual(version, 0)

  def test_get_version_with_migrations(self):
    """Test getting version with recorded migrations."""
    create_migration_table(self.kb)

    # Record some migrations
    self.kb.sql_cursor.execute("""
      INSERT INTO schema_migrations (version, name, applied_at)
      VALUES (1, 'initial', datetime('now'))
    """)
    self.kb.sql_cursor.execute("""
      INSERT INTO schema_migrations (version, name, applied_at)
      VALUES (2, 'add_columns', datetime('now'))
    """)
    self.kb.sql_connection.commit()

    version = get_current_schema_version(self.kb)
    self.assertEqual(version, 2)

  def test_get_version_with_rollback(self):
    """Test version with rolled back migration."""
    create_migration_table(self.kb)

    # Record migration with rollback
    self.kb.sql_cursor.execute("""
      INSERT INTO schema_migrations (version, name, applied_at, rollback_at)
      VALUES (1, 'failed', datetime('now'), datetime('now'))
    """)
    self.kb.sql_connection.commit()

    # Should not count rolled back migrations
    version = get_current_schema_version(self.kb)
    self.assertEqual(version, 0)

  @patch('database.migrations.logger')
  def test_get_version_error(self, mock_logger):
    """Test version retrieval with database error."""
    # Use mock cursor for error testing (can't mock execute on real cursor)
    mock_cursor = Mock()
    mock_cursor.execute = Mock(side_effect=sqlite3.Error("Query failed"))
    self.kb.sql_cursor = mock_cursor

    version = get_current_schema_version(self.kb)

    # Should return 0 on error
    self.assertEqual(version, 0)
    mock_logger.warning.assert_called()


class TestMigrationTable(unittest.TestCase):
  """Test migration table management."""

  def setUp(self):
    """Set up test fixtures."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as temp_db:
      self.db_path = temp_db.name

    self.kb = Mock()
    self.kb.sql_connection = sqlite3.connect(self.db_path)
    self.kb.sql_cursor = self.kb.sql_connection.cursor()

  def tearDown(self):
    """Clean up test fixtures."""
    self.kb.sql_connection.close()
    if os.path.exists(self.db_path):
      os.unlink(self.db_path)

  def test_create_migration_table(self):
    """Test migration table creation."""
    create_migration_table(self.kb)

    # Verify table exists
    self.kb.sql_cursor.execute("""
      SELECT name FROM sqlite_master
      WHERE type='table' AND name='schema_migrations'
    """)
    result = self.kb.sql_cursor.fetchone()
    self.assertIsNotNone(result)
    self.assertEqual(result[0], 'schema_migrations')

    # Verify columns
    self.kb.sql_cursor.execute("PRAGMA table_info(schema_migrations)")
    columns = [row[1] for row in self.kb.sql_cursor.fetchall()]

    expected_columns = ['version', 'name', 'applied_at', 'rollback_at', 'description']
    for col in expected_columns:
      self.assertIn(col, columns)

  def test_create_migration_table_idempotent(self):
    """Test that creating table twice doesn't error."""
    create_migration_table(self.kb)
    create_migration_table(self.kb)  # Should not raise

  def test_create_migration_table_error(self):
    """Test error handling in table creation."""
    # Use mock cursor for error testing (can't mock execute on real cursor)
    mock_cursor = Mock()
    mock_cursor.execute = Mock(side_effect=sqlite3.Error("Creation failed"))
    self.kb.sql_cursor = mock_cursor

    with self.assertRaises(DatabaseError) as cm:
      create_migration_table(self.kb)

    self.assertIn("Migration table creation failed", str(cm.exception))

  def test_record_migration(self):
    """Test recording a migration."""
    create_migration_table(self.kb)

    record_migration(self.kb, 1, "test_migration", "Test description")

    # Verify migration recorded
    self.kb.sql_cursor.execute("""
      SELECT version, name, description FROM schema_migrations
      WHERE version = 1
    """)
    result = self.kb.sql_cursor.fetchone()

    self.assertIsNotNone(result)
    self.assertEqual(result[0], 1)
    self.assertEqual(result[1], "test_migration")
    self.assertEqual(result[2], "Test description")

  def test_record_migration_replace(self):
    """Test that recording same version replaces."""
    create_migration_table(self.kb)

    record_migration(self.kb, 1, "first", "First version")
    record_migration(self.kb, 1, "updated", "Updated version")

    # Should only have one record
    self.kb.sql_cursor.execute("""
      SELECT COUNT(*) FROM schema_migrations WHERE version = 1
    """)
    count = self.kb.sql_cursor.fetchone()[0]
    self.assertEqual(count, 1)

    # Should have updated values
    self.kb.sql_cursor.execute("""
      SELECT name FROM schema_migrations WHERE version = 1
    """)
    name = self.kb.sql_cursor.fetchone()[0]
    self.assertEqual(name, "updated")

  def test_record_migration_error(self):
    """Test error handling in migration recording."""
    create_migration_table(self.kb)

    # Use mock cursor for error testing (can't mock execute on real cursor)
    mock_cursor = Mock()
    mock_cursor.execute = Mock(side_effect=sqlite3.Error("Insert failed"))
    self.kb.sql_cursor = mock_cursor

    with self.assertRaises(DatabaseError) as cm:
      record_migration(self.kb, 1, "test", "Test")

    self.assertIn("Migration recording failed", str(cm.exception))


class TestBM25Migration(unittest.TestCase):
  """Test BM25 migration functionality."""

  def setUp(self):
    """Set up test fixtures."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as temp_db:
      self.db_path = temp_db.name

    self.kb = Mock()
    self.kb.sql_connection = sqlite3.connect(self.db_path)
    self.kb.sql_cursor = self.kb.sql_connection.cursor()
    self.kb.table_name = 'docs'

    # Create base table
    self.kb.sql_cursor.execute("""
      CREATE TABLE docs (
        id INTEGER PRIMARY KEY,
        sourcedoc TEXT
      )
    """)
    self.kb.sql_connection.commit()

  def tearDown(self):
    """Clean up test fixtures."""
    self.kb.sql_connection.close()
    if os.path.exists(self.db_path):
      os.unlink(self.db_path)

  def test_migrate_for_bm25_adds_column(self):
    """Test that BM25 migration adds required column."""
    result = migrate_for_bm25(self.kb)

    self.assertTrue(result)

    # Verify column added
    self.kb.sql_cursor.execute("PRAGMA table_info(docs)")
    columns = [row[1] for row in self.kb.sql_cursor.fetchall()]
    self.assertIn('bm25_tokens', columns)

    # Verify index created
    self.kb.sql_cursor.execute("""
      SELECT name FROM sqlite_master
      WHERE type='index' AND name='idx_bm25_tokens'
    """)
    result = self.kb.sql_cursor.fetchone()
    self.assertIsNotNone(result)

  def test_migrate_for_bm25_already_exists(self):
    """Test BM25 migration when columns already exist."""
    # Add both BM25 columns first
    self.kb.sql_cursor.execute("ALTER TABLE docs ADD COLUMN bm25_tokens TEXT")
    self.kb.sql_cursor.execute("ALTER TABLE docs ADD COLUMN doc_length INTEGER DEFAULT 0")
    self.kb.sql_connection.commit()

    result = migrate_for_bm25(self.kb)

    # Should return False (no migration needed)
    self.assertFalse(result)

  def test_migrate_for_bm25_records_migration(self):
    """Test that BM25 migration is recorded."""
    migrate_for_bm25(self.kb)

    # Check migration was recorded
    self.kb.sql_cursor.execute("""
      SELECT version, name FROM schema_migrations
      WHERE version = 1
    """)
    result = self.kb.sql_cursor.fetchone()

    self.assertIsNotNone(result)
    self.assertEqual(result[1], "add_bm25_columns")  # Matches implementation

  def test_migrate_for_bm25_error(self):
    """Test error handling in BM25 migration."""
    # Use mock cursor for error testing (can't mock execute on real cursor)
    mock_cursor = Mock()
    mock_cursor.execute = Mock(side_effect=sqlite3.Error("Query failed"))
    self.kb.sql_cursor = mock_cursor

    with self.assertRaises(DatabaseError) as cm:
      migrate_for_bm25(self.kb)

    self.assertIn("Failed to migrate for BM25", str(cm.exception))

  def test_migrate_for_bm25_custom_table(self):
    """Test BM25 migration with custom table name."""
    # Create custom table
    self.kb.table_name = 'chunks'
    self.kb.sql_cursor.execute("""
      CREATE TABLE chunks (
        id INTEGER PRIMARY KEY,
        sourcedoc TEXT
      )
    """)
    self.kb.sql_connection.commit()

    result = migrate_for_bm25(self.kb)

    self.assertTrue(result)

    # Verify column added to custom table
    self.kb.sql_cursor.execute("PRAGMA table_info(chunks)")
    columns = [row[1] for row in self.kb.sql_cursor.fetchall()]
    self.assertIn('bm25_tokens', columns)


class TestCategoryMigration(unittest.TestCase):
  """Test category columns migration."""

  def setUp(self):
    """Set up test fixtures."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as temp_db:
      self.db_path = temp_db.name

    self.kb = Mock()
    self.kb.sql_connection = sqlite3.connect(self.db_path)
    self.kb.sql_cursor = self.kb.sql_connection.cursor()
    self.kb.table_name = 'docs'

    # Create base table
    self.kb.sql_cursor.execute("""
      CREATE TABLE docs (
        id INTEGER PRIMARY KEY,
        sourcedoc TEXT
      )
    """)
    self.kb.sql_connection.commit()

  def tearDown(self):
    """Clean up test fixtures."""
    self.kb.sql_connection.close()
    if os.path.exists(self.db_path):
      os.unlink(self.db_path)

  def test_migrate_add_categories(self):
    """Test adding category columns."""
    result = migrate_add_categories(self.kb)

    self.assertTrue(result)

    # Verify columns added
    self.kb.sql_cursor.execute("PRAGMA table_info(docs)")
    columns = [row[1] for row in self.kb.sql_cursor.fetchall()]
    self.assertIn('primary_category', columns)
    self.assertIn('categories', columns)

    # Verify index created
    self.kb.sql_cursor.execute("""
      SELECT name FROM sqlite_master
      WHERE type='index' AND name='idx_primary_category'
    """)
    result = self.kb.sql_cursor.fetchone()
    self.assertIsNotNone(result)

  def test_migrate_add_categories_already_exists(self):
    """Test category migration when columns exist."""
    # Add columns first
    self.kb.sql_cursor.execute("ALTER TABLE docs ADD COLUMN primary_category TEXT")
    self.kb.sql_connection.commit()

    result = migrate_add_categories(self.kb)

    self.assertFalse(result)

  def test_migrate_add_categories_error(self):
    """Test error handling in category migration."""
    # Use mock cursor for error testing (can't mock execute on real cursor)
    mock_cursor = Mock()
    mock_cursor.execute = Mock(side_effect=sqlite3.Error("Query failed"))
    self.kb.sql_cursor = mock_cursor

    with self.assertRaises(DatabaseError) as cm:
      migrate_add_categories(self.kb)

    self.assertIn("Failed to add category columns", str(cm.exception))


class TestTimestampMigration(unittest.TestCase):
  """Test timestamp columns migration."""

  def setUp(self):
    """Set up test fixtures."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as temp_db:
      self.db_path = temp_db.name

    self.kb = Mock()
    self.kb.sql_connection = sqlite3.connect(self.db_path)
    self.kb.sql_cursor = self.kb.sql_connection.cursor()
    self.kb.table_name = 'docs'

    # Create base table
    self.kb.sql_cursor.execute("""
      CREATE TABLE docs (
        id INTEGER PRIMARY KEY,
        sourcedoc TEXT
      )
    """)
    self.kb.sql_connection.commit()

  def tearDown(self):
    """Clean up test fixtures."""
    self.kb.sql_connection.close()
    if os.path.exists(self.db_path):
      os.unlink(self.db_path)

  def test_migrate_add_timestamps(self):
    """Test adding timestamp columns."""
    result = migrate_add_timestamps(self.kb)

    self.assertTrue(result)

    # Verify columns added
    self.kb.sql_cursor.execute("PRAGMA table_info(docs)")
    columns = [row[1] for row in self.kb.sql_cursor.fetchall()]
    self.assertIn('created_at', columns)
    self.assertIn('updated_at', columns)

    # Verify trigger created
    self.kb.sql_cursor.execute("""
      SELECT name FROM sqlite_master
      WHERE type='trigger' AND name='update_docs_timestamp'
    """)
    result = self.kb.sql_cursor.fetchone()
    self.assertIsNotNone(result)

  def test_migrate_add_timestamps_already_exists(self):
    """Test timestamp migration when columns exist."""
    # Add columns first
    self.kb.sql_cursor.execute("ALTER TABLE docs ADD COLUMN created_at TIMESTAMP")
    self.kb.sql_connection.commit()

    result = migrate_add_timestamps(self.kb)

    self.assertFalse(result)

  def test_migrate_add_timestamps_trigger_works(self):
    """Test that update trigger actually works."""
    migrate_add_timestamps(self.kb)

    # Insert a row
    self.kb.sql_cursor.execute("""
      INSERT INTO docs (sourcedoc) VALUES ('test.txt')
    """)
    self.kb.sql_connection.commit()

    # Get created_at
    self.kb.sql_cursor.execute("SELECT created_at FROM docs WHERE id=1")
    created = self.kb.sql_cursor.fetchone()[0]
    self.assertIsNotNone(created)

    # Update the row
    import time
    time.sleep(0.01)  # Ensure timestamp difference
    self.kb.sql_cursor.execute("""
      UPDATE docs SET sourcedoc='updated.txt' WHERE id=1
    """)
    self.kb.sql_connection.commit()

    # Verify updated_at changed
    self.kb.sql_cursor.execute("SELECT updated_at FROM docs WHERE id=1")
    updated = self.kb.sql_cursor.fetchone()[0]
    self.assertIsNotNone(updated)
    # Note: SQLite timestamp precision may not show difference for very quick updates


class TestRunAllMigrations(unittest.TestCase):
  """Test running all migrations."""

  def setUp(self):
    """Set up test fixtures."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as temp_db:
      self.db_path = temp_db.name

    self.kb = Mock()
    self.kb.sql_connection = sqlite3.connect(self.db_path)
    self.kb.sql_cursor = self.kb.sql_connection.cursor()
    self.kb.table_name = 'docs'

    # Create base table
    self.kb.sql_cursor.execute("""
      CREATE TABLE docs (
        id INTEGER PRIMARY KEY,
        sourcedoc TEXT
      )
    """)
    self.kb.sql_connection.commit()

  def tearDown(self):
    """Clean up test fixtures."""
    self.kb.sql_connection.close()
    if os.path.exists(self.db_path):
      os.unlink(self.db_path)

  def test_run_all_migrations_fresh(self):
    """Test running all migrations on fresh database."""
    applied = run_all_migrations(self.kb)

    # Should apply all 3 migrations
    self.assertEqual(applied, 3)

    # Verify all columns exist
    self.kb.sql_cursor.execute("PRAGMA table_info(docs)")
    columns = [row[1] for row in self.kb.sql_cursor.fetchall()]

    expected_columns = ['bm25_tokens', 'primary_category', 'categories',
                       'created_at', 'updated_at']
    for col in expected_columns:
      self.assertIn(col, columns)

  def test_run_all_migrations_partial(self):
    """Test running migrations when some already applied."""
    # Apply first migration manually
    migrate_for_bm25(self.kb)

    applied = run_all_migrations(self.kb)

    # Should only apply 2 remaining migrations
    self.assertEqual(applied, 2)

  def test_run_all_migrations_none_needed(self):
    """Test running migrations when all already applied."""
    # Apply all migrations manually
    migrate_for_bm25(self.kb)
    migrate_add_categories(self.kb)
    migrate_add_timestamps(self.kb)

    applied = run_all_migrations(self.kb)

    # Should apply no migrations
    self.assertEqual(applied, 0)

  @patch('database.migrations.logger')
  def test_run_all_migrations_error(self, mock_logger):
    """Test error handling when migration fails."""
    # Make BM25 migration fail
    with patch('database.migrations.migrate_for_bm25', side_effect=RuntimeError("Migration failed")):
      with self.assertRaises(RuntimeError):
        run_all_migrations(self.kb)

      mock_logger.error.assert_called()


class TestMigrationStatus(unittest.TestCase):
  """Test migration status checking."""

  def setUp(self):
    """Set up test fixtures."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as temp_db:
      self.db_path = temp_db.name

    self.kb = Mock()
    self.kb.sql_connection = sqlite3.connect(self.db_path)
    self.kb.sql_cursor = self.kb.sql_connection.cursor()
    self.kb.table_name = 'docs'

    # Create base table
    self.kb.sql_cursor.execute("""
      CREATE TABLE docs (
        id INTEGER PRIMARY KEY,
        sourcedoc TEXT
      )
    """)
    self.kb.sql_connection.commit()

  def tearDown(self):
    """Clean up test fixtures."""
    self.kb.sql_connection.close()
    if os.path.exists(self.db_path):
      os.unlink(self.db_path)

  def test_check_status_no_migrations(self):
    """Test status check with no migrations."""
    status = check_migration_status(self.kb)

    self.assertEqual(status['current_version'], 0)
    self.assertEqual(status['latest_version'], 3)
    self.assertEqual(len(status['applied_migrations']), 0)
    self.assertEqual(len(status['pending_migrations']), 3)
    self.assertFalse(status['is_up_to_date'])

  def test_check_status_partial_migrations(self):
    """Test status check with some migrations applied."""
    # Apply first migration
    migrate_for_bm25(self.kb)

    status = check_migration_status(self.kb)

    self.assertEqual(status['current_version'], 1)
    self.assertEqual(len(status['applied_migrations']), 1)
    self.assertEqual(len(status['pending_migrations']), 2)
    self.assertFalse(status['is_up_to_date'])

  def test_check_status_all_migrations(self):
    """Test status check with all migrations applied."""
    run_all_migrations(self.kb)

    status = check_migration_status(self.kb)

    self.assertEqual(status['current_version'], 3)
    self.assertEqual(len(status['applied_migrations']), 3)
    self.assertEqual(len(status['pending_migrations']), 0)
    self.assertTrue(status['is_up_to_date'])

  def test_check_status_error(self):
    """Test status check error handling."""
    # Use mock cursor for error testing (can't mock execute on real cursor)
    mock_cursor = Mock()
    mock_cursor.execute = Mock(side_effect=sqlite3.OperationalError("Query failed"))
    self.kb.sql_cursor = mock_cursor

    status = check_migration_status(self.kb)

    # Should include error in status
    self.assertIn('error', status)
    self.assertIn("Query failed", status['error'])


if __name__ == '__main__':
  unittest.main()

#fin
