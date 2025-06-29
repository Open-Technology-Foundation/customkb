#!/usr/bin/env python
"""
Add index on embedded column to improve performance for large databases.
"""

import sqlite3
import sys
import time

def add_embedded_index(db_path):
  """Add index on embedded column if it doesn't exist."""
  
  print(f"Connecting to database: {db_path}")
  conn = sqlite3.connect(db_path)
  cursor = conn.cursor()
  
  try:
    # Check current row count
    cursor.execute("SELECT COUNT(*) FROM docs")
    total_rows = cursor.fetchone()[0]
    print(f"Total rows in database: {total_rows:,}")
    
    # Check how many need embedding
    cursor.execute("SELECT COUNT(*) FROM docs WHERE embedded=0 AND embedtext != ''")
    unembedded_rows = cursor.fetchone()[0]
    print(f"Rows needing embedding: {unembedded_rows:,}")
    
    # Check if index already exists
    cursor.execute("""
      SELECT name FROM sqlite_master 
      WHERE type='index' AND name='idx_embedded'
    """)
    index_exists = cursor.fetchone()
    
    if index_exists:
      print("Index 'idx_embedded' already exists")
    else:
      print("Creating index on embedded column...")
      start_time = time.time()
      cursor.execute("CREATE INDEX idx_embedded ON docs(embedded)")
      conn.commit()
      elapsed = time.time() - start_time
      print(f"Index created successfully in {elapsed:.2f} seconds")
    
    # Also create composite index for the SELECT query
    cursor.execute("""
      SELECT name FROM sqlite_master 
      WHERE type='index' AND name='idx_embedded_embedtext'
    """)
    composite_exists = cursor.fetchone()
    
    if composite_exists:
      print("Composite index 'idx_embedded_embedtext' already exists")
    else:
      print("Creating composite index for embedded and embedtext columns...")
      start_time = time.time()
      cursor.execute("CREATE INDEX idx_embedded_embedtext ON docs(embedded, embedtext)")
      conn.commit()
      elapsed = time.time() - start_time
      print(f"Composite index created successfully in {elapsed:.2f} seconds")
    
    # Analyze database to update statistics
    print("Analyzing database...")
    cursor.execute("ANALYZE")
    conn.commit()
    
    print("Database optimization complete!")
    
  except Exception as e:
    print(f"Error: {e}")
    conn.rollback()
  finally:
    conn.close()

if __name__ == "__main__":
  if len(sys.argv) != 2:
    print("Usage: python add_embedded_index.py /path/to/database.db")
    sys.exit(1)
  
  db_path = sys.argv[1]
  add_embedded_index(db_path)

#fin