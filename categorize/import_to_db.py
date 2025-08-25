"""
Import categorization results into the knowledgebase database.
"""

import sqlite3
import json
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass

from config.config_manager import KnowledgeBase
from utils.logging_config import get_logger

logger = get_logger(__name__)

def import_categories(kb: KnowledgeBase, results: List[Any]) -> str:
  """
  Import categorization results into the knowledgebase database.
  
  Args:
      kb: Knowledgebase configuration
      results: List of categorization results
      
  Returns:
      Status message describing the import results
  """
  try:
    # Connect to database
    db_path = kb.knowledge_base_db
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Detect the actual table name (docs or chunks)
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name IN ('chunks', 'docs')")
    result = cursor.fetchone()
    table_name = result[0] if result else 'docs'
    logger.debug(f"Using table name: {table_name}")
    
    # Check if categories column exists in the table
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = [col[1] for col in cursor.fetchall()]
    
    if 'categories' not in columns:
      logger.info(f"Adding categories column to {table_name} table")
      cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN categories TEXT")
      conn.commit()
    
    if 'primary_category' not in columns:
      logger.info(f"Adding primary_category column to {table_name} table")
      cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN primary_category TEXT")
      conn.commit()
    
    # Import categories for each article
    updated_count = 0
    error_count = 0
    
    for result in results:
      if result.error:
        error_count += 1
        continue
      
      try:
        # Extract the source document name from the article path
        article_path = Path(result.article_path)
        source_doc_name = article_path.name
        
        # Get all categories as comma-separated string
        categories = ', '.join([cat.name for cat in result.categories])
        primary_category = result.primary_category if result.primary_category else None
        
        # Update all records from this source document
        # Use the correct column name based on table schema
        source_col = 'sourcedoc' if table_name == 'docs' else 'source_doc'
        
        # Try to match by filename at the end of the path (using LIKE pattern)
        cursor.execute(f"""
          UPDATE {table_name} 
          SET categories = ?, primary_category = ?
          WHERE {source_col} LIKE ?
        """, (categories, primary_category, f'%/{source_doc_name}'))
        
        # If no rows affected, try exact match with just the filename
        if cursor.rowcount == 0:
          cursor.execute(f"""
            UPDATE {table_name} 
            SET categories = ?, primary_category = ?
            WHERE {source_col} = ?
          """, (categories, primary_category, source_doc_name))
        
        rows_affected = cursor.rowcount
        if rows_affected > 0:
          updated_count += 1
          logger.debug(f"Updated {rows_affected} records for {source_doc_name}")
        else:
          logger.warning(f"No records found for source document: {source_doc_name}")
          
      except Exception as e:
        logger.error(f"Error importing categories for {result.article_path}: {e}")
        error_count += 1
    
    # Commit all changes
    conn.commit()
    
    # Create indexes for category columns if they don't exist
    try:
      cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_primary_category ON {table_name}(primary_category)")
      cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_categories ON {table_name}(categories)")
      conn.commit()
      logger.info("Created category indexes")
    except Exception as e:
      logger.warning(f"Could not create category indexes: {e}")
    
    # Close connection
    conn.close()
    
    # Generate summary
    summary = f"""
Database Import Complete
========================
Articles updated: {updated_count}
Errors: {error_count}
Total processed: {len(results)}
Table used: {table_name}

Categories have been imported into the database.
You can now use --category and --categories filters in queries.
"""
    
    return summary
    
  except Exception as e:
    logger.error(f"Failed to import categories to database: {e}")
    return f"Error importing categories: {e}"

#fin