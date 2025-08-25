#!/usr/bin/env python
"""
Unit tests for database.connection module.

Tests database connection lifecycle management, table creation,
and context managers.
"""

import os
import sqlite3
import tempfile
import unittest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from database.connection import (
  connect_to_database,
  close_database,
  create_tables,
  database_connection,
  sqlite_connection,
  get_connection_info
)
from utils.exceptions import ConnectionError as CustomConnectionError, DatabaseError


class TestDatabaseConnection(unittest.TestCase):
  """Test database connection functions."""
  
  def setUp(self):
    """Set up test fixtures."""
    self.temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
    self.db_path = self.temp_db.name
    self.temp_db.close()
    
    # Create mock KnowledgeBase
    self.kb = Mock()
    self.kb.knowledge_base_db = self.db_path
    self.kb.table_name = 'test_docs'
    self.kb.enable_hybrid_search = False
    self.kb.enable_categorization = False
    
  def tearDown(self):
    """Clean up test fixtures."""
    if os.path.exists(self.db_path):
      os.unlink(self.db_path)
  
  def test_connect_to_database_success(self):
    """Test successful database connection."""
    connect_to_database(self.kb)
    
    # Verify connection established
    self.assertIsNotNone(self.kb.sql_connection)
    self.assertIsNotNone(self.kb.sql_cursor)
    
    # Verify pragmas set
    self.kb.sql_cursor.execute("PRAGMA foreign_keys")
    result = self.kb.sql_cursor.fetchone()
    self.assertEqual(result[0], 1)  # Foreign keys enabled
    
    # Clean up
    close_database(self.kb)
  
  def test_connect_to_database_error(self):
    """Test database connection error handling."""
    self.kb.knowledge_base_db = "/invalid/path/database.db"
    
    with self.assertRaises(CustomConnectionError) as cm:
      connect_to_database(self.kb)
    
    self.assertIn("Failed to connect to database", str(cm.exception))
  
  def test_create_tables_basic(self):
    """Test basic table creation."""
    # Establish connection
    self.kb.sql_connection = sqlite3.connect(self.db_path)
    self.kb.sql_cursor = self.kb.sql_connection.cursor()
    
    # Create tables
    create_tables(self.kb)
    
    # Verify main table exists
    self.kb.sql_cursor.execute(
      "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
      (self.kb.table_name,)
    )
    result = self.kb.sql_cursor.fetchone()
    self.assertIsNotNone(result)
    self.assertEqual(result[0], self.kb.table_name)
    
    # Verify file_metadata table exists
    self.kb.sql_cursor.execute(
      "SELECT name FROM sqlite_master WHERE type='table' AND name='file_metadata'"
    )
    result = self.kb.sql_cursor.fetchone()
    self.assertIsNotNone(result)
    
    # Clean up
    self.kb.sql_connection.close()
  
  def test_create_tables_with_categories(self):
    """Test table creation with categorization enabled."""
    self.kb.enable_categorization = True
    self.kb.sql_connection = sqlite3.connect(self.db_path)
    self.kb.sql_cursor = self.kb.sql_connection.cursor()
    
    create_tables(self.kb)
    
    # Verify categories table exists
    self.kb.sql_cursor.execute(
      "SELECT name FROM sqlite_master WHERE type='table' AND name='categories'"
    )
    result = self.kb.sql_cursor.fetchone()
    self.assertIsNotNone(result)
    
    # Clean up
    self.kb.sql_connection.close()
  
  def test_create_tables_error(self):
    """Test table creation error handling."""
    self.kb.sql_connection = sqlite3.connect(self.db_path)
    self.kb.sql_cursor = Mock()
    self.kb.sql_cursor.execute.side_effect = sqlite3.Error("Table creation failed")
    
    with self.assertRaises(DatabaseError) as cm:
      create_tables(self.kb)
    
    self.assertIn("Failed to create tables", str(cm.exception))
    
    # Clean up
    self.kb.sql_connection.close()
  
  def test_close_database(self):
    """Test database connection closure."""
    # Establish connection
    connect_to_database(self.kb)
    
    # Verify connection is open
    self.assertIsNotNone(self.kb.sql_connection)
    self.assertIsNotNone(self.kb.sql_cursor)
    
    # Close connection
    close_database(self.kb)
    
    # Verify connection closed
    self.assertIsNone(self.kb.sql_connection)
    self.assertIsNone(self.kb.sql_cursor)
  
  def test_close_database_no_connection(self):
    """Test closing when no connection exists."""
    # Should not raise error
    close_database(self.kb)
  
  def test_close_database_with_pending_transaction(self):
    """Test closing with pending transaction."""
    connect_to_database(self.kb)
    
    # Start a transaction
    self.kb.sql_cursor.execute(f"INSERT INTO {self.kb.table_name} (sid, sourcedoc) VALUES (1, 'test.txt')")
    
    # Close should commit pending transaction
    close_database(self.kb)
    
    # Verify data was committed
    conn = sqlite3.connect(self.db_path)
    cursor = conn.cursor()
    cursor.execute(f"SELECT COUNT(*) FROM {self.kb.table_name}")
    count = cursor.fetchone()[0]
    conn.close()
    
    self.assertEqual(count, 1)
  
  def test_database_connection_context_manager(self):
    """Test database_connection context manager."""
    # Ensure no existing connection
    self.kb.sql_connection = None
    self.kb.sql_cursor = None
    
    with database_connection(self.kb) as kb:
      # Verify connection established
      self.assertIsNotNone(kb.sql_connection)
      self.assertIsNotNone(kb.sql_cursor)
      
      # Perform operation
      kb.sql_cursor.execute(f"SELECT COUNT(*) FROM {kb.table_name}")
      result = kb.sql_cursor.fetchone()
      self.assertIsNotNone(result)
    
    # Verify connection closed after context (since we opened it)
    self.assertIsNone(self.kb.sql_connection)
    self.assertIsNone(self.kb.sql_cursor)
  
  def test_database_connection_context_manager_error(self):
    """Test context manager error handling."""
    # Ensure no existing connection
    self.kb.sql_connection = None
    self.kb.sql_cursor = None
    
    with self.assertRaises(ValueError):
      with database_connection(self.kb) as kb:
        # Simulate error during operation
        raise ValueError("Test error")
    
    # Verify connection still closed on error (since we opened it)
    self.assertIsNone(self.kb.sql_connection)
  
  def test_database_connection_existing_connection(self):
    """Test context manager with existing connection."""
    # Pre-establish connection
    connect_to_database(self.kb)
    original_conn = self.kb.sql_connection
    
    with database_connection(self.kb) as kb:
      # Should use existing connection
      self.assertIs(kb.sql_connection, original_conn)
    
    # Should not close connection that was already open
    self.assertIsNotNone(self.kb.sql_connection)
    self.assertIs(self.kb.sql_connection, original_conn)
    
    # Clean up
    close_database(self.kb)
  
  def test_sqlite_connection_context_manager(self):
    """Test sqlite_connection context manager."""
    with sqlite_connection(self.db_path) as (conn, cursor):
      # Verify connection established
      self.assertIsNotNone(conn)
      self.assertIsNotNone(cursor)
      
      # Create test table
      cursor.execute("CREATE TABLE test (id INTEGER PRIMARY KEY)")
      cursor.execute("INSERT INTO test (id) VALUES (1)")
      
    # Verify changes committed and connection closed
    conn2 = sqlite3.connect(self.db_path)
    cursor2 = conn2.cursor()
    cursor2.execute("SELECT COUNT(*) FROM test")
    count = cursor2.fetchone()[0]
    conn2.close()
    
    self.assertEqual(count, 1)
  
  def test_sqlite_connection_invalid_path(self):
    """Test sqlite_connection with invalid path."""
    with self.assertRaises(CustomConnectionError) as cm:
      with sqlite_connection("/nonexistent/database.db") as (conn, cursor):
        pass
    
    self.assertIn("Database not found", str(cm.exception))
  
  def test_sqlite_connection_error_rollback(self):
    """Test sqlite_connection rollback on error."""
    with sqlite_connection(self.db_path) as (conn, cursor):
      cursor.execute("CREATE TABLE test (id INTEGER PRIMARY KEY)")
    
    # Try operation that should fail and rollback
    with self.assertRaises(DatabaseError):
      with sqlite_connection(self.db_path) as (conn, cursor):
        cursor.execute("INSERT INTO test (id) VALUES (1)")
        # Force error
        cursor.execute("INVALID SQL")
    
    # Verify transaction was rolled back
    conn2 = sqlite3.connect(self.db_path)
    cursor2 = conn2.cursor()
    cursor2.execute("SELECT COUNT(*) FROM test")
    count = cursor2.fetchone()[0]
    conn2.close()
    
    self.assertEqual(count, 0)  # No rows due to rollback
  
  def test_get_connection_info(self):
    """Test getting connection information."""
    # No connection - initially the Mock kb has no sql_connection
    self.kb.sql_connection = None
    info = get_connection_info(self.kb)
    self.assertFalse(info['connected'])
    self.assertIsNone(info['database_path'])
    self.assertEqual(info['table_count'], 0)
    self.assertEqual(info['row_count'], 0)
    
    # With connection
    connect_to_database(self.kb)
    
    # Add some test data
    self.kb.sql_cursor.execute(
      f"INSERT INTO {self.kb.table_name} (sid, sourcedoc) VALUES (1, 'test.txt')"
    )
    self.kb.sql_connection.commit()
    
    info = get_connection_info(self.kb)
    self.assertTrue(info['connected'])
    self.assertEqual(info['database_path'], self.db_path)
    self.assertGreater(info['table_count'], 0)
    self.assertEqual(info['row_count'], 1)
    
    # Clean up
    close_database(self.kb)
  
  def test_get_connection_info_error(self):
    """Test connection info with database error."""
    self.kb.sql_connection = Mock()
    self.kb.sql_cursor = Mock()
    self.kb.sql_cursor.execute.side_effect = sqlite3.Error("Query failed")
    
    info = get_connection_info(self.kb)
    
    # Should return partial info on error
    self.assertTrue(info['connected'])
    self.assertEqual(info['database_path'], self.db_path)
    self.assertEqual(info['table_count'], 0)
    self.assertEqual(info['row_count'], 0)
  
  @patch('database.connection.logger')
  def test_logging(self, mock_logger):
    """Test logging throughout operations."""
    connect_to_database(self.kb)
    
    # Verify connection logged
    mock_logger.info.assert_called_with("Database connection established successfully")
    
    close_database(self.kb)
    
    # Verify closure logged
    mock_logger.info.assert_called_with("Database connection closed")
  
  def test_pragma_settings(self):
    """Test that all pragmas are properly set."""
    connect_to_database(self.kb)
    
    # Check all expected pragmas
    pragmas = {
      'foreign_keys': 1,
      'journal_mode': 'wal',
      'synchronous': 1,  # NORMAL
      'cache_size': -64000,
      'temp_store': 2,  # MEMORY
      'mmap_size': 268435456
    }
    
    for pragma, expected in pragmas.items():
      self.kb.sql_cursor.execute(f"PRAGMA {pragma}")
      result = self.kb.sql_cursor.fetchone()
      if isinstance(expected, str):
        self.assertEqual(result[0].lower(), expected.lower())
      else:
        self.assertEqual(result[0], expected)
    
    close_database(self.kb)


if __name__ == '__main__':
  unittest.main()

#fin