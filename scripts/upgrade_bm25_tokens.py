#!/usr/bin/env python3
"""
Upgrade existing CustomKB database to add BM25 tokens without reprocessing files.
This script processes existing embedtext content to generate BM25 tokens.
"""

import sys
import sqlite3
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config.config_manager import KnowledgeBase, get_fq_cfg_filename
from utils.text_utils import tokenize_for_bm25
from utils.logging_config import get_logger

def upgrade_database_bm25(kb_config: str, batch_size: int = 1000) -> bool:
    """
    Upgrade existing database to include BM25 tokens.
    
    Args:
        kb_config: Path to knowledgebase configuration
        batch_size: Number of records to process in each batch
        
    Returns:
        True if successful, False otherwise
    """
    logger = get_logger(__name__)
    
    try:
        # Resolve configuration file path
        config_path = get_fq_cfg_filename(kb_config)
        if not config_path:
            logger.error(f"Configuration file not found: {kb_config}")
            return False
        
        logger.info(f"Using configuration file: {config_path}")
        
        # Load configuration
        kb = KnowledgeBase(config_path)
        
        # Check if BM25 is enabled
        if not getattr(kb, 'enable_hybrid_search', False):
            logger.error("BM25/hybrid search is not enabled in configuration. Please enable it first.")
            return False
        
        # Connect to database
        kb.sql_connection = sqlite3.connect(kb.knowledge_base_db)
        kb.sql_cursor = kb.sql_connection.cursor()
        
        # Check if BM25 columns exist
        kb.sql_cursor.execute("PRAGMA table_info(docs)")
        columns = [col[1] for col in kb.sql_cursor.fetchall()]
        
        if 'bm25_tokens' not in columns or 'doc_length' not in columns:
            logger.error("BM25 columns not found. Run migration first: customkb bm25 <config>")
            return False
        
        # Get total count of documents to process
        kb.sql_cursor.execute("""
            SELECT COUNT(*) FROM docs 
            WHERE (bm25_tokens IS NULL OR bm25_tokens = '') 
            AND embedtext IS NOT NULL AND embedtext != ''
        """)
        total_docs = kb.sql_cursor.fetchone()[0]
        
        if total_docs == 0:
            logger.info("No documents need BM25 token generation.")
            return True
        
        logger.info(f"Processing {total_docs} documents to generate BM25 tokens...")
        
        processed = 0
        
        # Process in batches
        while processed < total_docs:
            # Get batch of documents without BM25 tokens
            kb.sql_cursor.execute("""
                SELECT id, embedtext, language FROM docs 
                WHERE (bm25_tokens IS NULL OR bm25_tokens = '') 
                AND embedtext IS NOT NULL AND embedtext != ''
                LIMIT ?
            """, (batch_size,))
            
            batch = kb.sql_cursor.fetchall()
            
            if not batch:
                break
            
            # Process each document in the batch
            updates = []
            for doc_id, embedtext, language in batch:
                try:
                    # Generate BM25 tokens from existing embedtext
                    bm25_tokens, doc_length = tokenize_for_bm25(embedtext, language or 'en')
                    updates.append((bm25_tokens, doc_length, 1, doc_id))
                except Exception as e:
                    logger.warning(f"Failed to tokenize document {doc_id}: {e}")
                    # Still mark as processed to avoid infinite loop
                    updates.append(("", 0, 1, doc_id))
            
            # Update database with BM25 tokens
            kb.sql_cursor.executemany("""
                UPDATE docs 
                SET bm25_tokens = ?, doc_length = ?, keyphrase_processed = ?
                WHERE id = ?
            """, updates)
            
            kb.sql_connection.commit()
            processed += len(batch)
            
            # Progress report
            progress = (processed / total_docs) * 100
            logger.info(f"Progress: {processed}/{total_docs} ({progress:.1f}%)")
        
        # Final verification
        kb.sql_cursor.execute("""
            SELECT COUNT(*) FROM docs 
            WHERE bm25_tokens IS NOT NULL AND bm25_tokens != ''
        """)
        final_count = kb.sql_cursor.fetchone()[0]
        
        logger.info(f"BM25 upgrade completed. {final_count} documents now have BM25 tokens.")
        
        kb.sql_connection.close()
        return True
        
    except Exception as e:
        logger.error(f"BM25 upgrade failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Upgrade existing CustomKB database with BM25 tokens"
    )
    parser.add_argument(
        'config',
        help='Knowledgebase configuration file or name'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1000,
        help='Number of documents to process in each batch (default: 1000)'
    )
    
    args = parser.parse_args()
    
    # Upgrade the database
    success = upgrade_database_bm25(args.config, args.batch_size)
    
    if success:
        print(f"✓ BM25 upgrade completed for {args.config}")
        print("Now run: customkb bm25 <config> to build the search index")
        sys.exit(0)
    else:
        print(f"✗ BM25 upgrade failed for {args.config}")
        print("Check the logs for details")
        sys.exit(1)

if __name__ == "__main__":
    main()

#fin