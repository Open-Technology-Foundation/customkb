#!/usr/bin/env python
"""
Verify that all expected indexes exist in a CustomKB database.
"""

import sqlite3
import sys

def verify_indexes(db_path):
  """Check which indexes exist in the database."""
  
  print(f"Checking indexes in: {db_path}\n")
  
  try:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get all indexes
    cursor.execute("""
      SELECT name, sql 
      FROM sqlite_master 
      WHERE type='index' 
      ORDER BY name
    """)
    
    indexes = cursor.fetchall()
    
    # Expected indexes
    expected_indexes = [
      'idx_embedded',
      'idx_embedded_embedtext', 
      'idx_keyphrase_processed',
      'idx_sourcedoc',
      'idx_sourcedoc_sid'
    ]
    
    print("Found indexes:")
    print("-" * 60)
    found_indexes = []
    for idx_name, idx_sql in indexes:
      if idx_name.startswith('sqlite_'):  # Skip SQLite internal indexes
        continue
      found_indexes.append(idx_name)
      print(f"{idx_name}:")
      if idx_sql:
        print(f"  {idx_sql}")
      print()
    
    print("\nIndex verification:")
    print("-" * 60)
    missing = []
    for expected in expected_indexes:
      if expected in found_indexes:
        print(f"✓ {expected} - Present")
      else:
        print(f"✗ {expected} - MISSING")
        missing.append(expected)
    
    if missing:
      print(f"\n⚠️  Missing {len(missing)} indexes: {', '.join(missing)}")
      return False
    else:
      print("\n✅ All expected indexes are present!")
      return True
      
  except Exception as e:
    print(f"Error: {e}")
    return False
  finally:
    conn.close()

if __name__ == "__main__":
  if len(sys.argv) != 2:
    print("Usage: python verify_indexes.py /path/to/database.db")
    sys.exit(1)
  
  db_path = sys.argv[1]
  success = verify_indexes(db_path)
  sys.exit(0 if success else 1)

#fin