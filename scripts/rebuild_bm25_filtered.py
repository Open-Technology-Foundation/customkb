#!/usr/bin/env python
"""
Rebuild BM25 index with filtering for specific knowledge base content.
This creates a focused BM25 index while using symlinked db/faiss files.
"""

import os
import sys
import sqlite3
import pickle
import argparse
import logging
from typing import List, Set, Dict, Any
from rank_bm25 import BM25Okapi

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config_manager import KnowledgeBase, get_fq_cfg_filename
from database.db_manager import connect_to_database, close_database
from utils.logging_utils import get_logger

logger = get_logger(__name__)

def rebuild_bm25_with_filter(kb: KnowledgeBase, 
                           keywords: List[str] = None,
                           include_patterns: List[str] = None,
                           exclude_patterns: List[str] = None,
                           save_path: str = None) -> bool:
    """
    Rebuild BM25 index with filtering criteria.
    
    Args:
        kb: KnowledgeBase instance
        keywords: Keywords to filter documents
        include_patterns: Path patterns to include
        exclude_patterns: Path patterns to exclude
        save_path: Where to save the index
        
    Returns:
        True if successful, False otherwise
    """
    try:
        cursor = kb.sql_cursor
        
        # Build filter query
        query_parts = ["""
            SELECT id, bm25_tokens, doc_length, sourcedoc, originaltext
            FROM docs 
            WHERE keyphrase_processed = 1 
            AND bm25_tokens IS NOT NULL 
            AND bm25_tokens != ''
        """]
        params = []
        
        # Add keyword filters
        if keywords:
            keyword_conditions = []
            for keyword in keywords:
                keyword_conditions.append("(originaltext LIKE ? OR sourcedoc LIKE ?)")
                params.extend([f'%{keyword}%', f'%{keyword}%'])
            query_parts.append(f" AND ({' OR '.join(keyword_conditions)})")
        
        # Add path include patterns
        if include_patterns:
            include_conditions = []
            for pattern in include_patterns:
                include_conditions.append("sourcedoc LIKE ?")
                params.append(f'%{pattern}%')
            query_parts.append(f" AND ({' OR '.join(include_conditions)})")
        
        # Add path exclude patterns
        if exclude_patterns:
            for pattern in exclude_patterns:
                query_parts.append(" AND sourcedoc NOT LIKE ?")
                params.append(f'%{pattern}%')
        
        query_parts.append(" ORDER BY id")
        query = ''.join(query_parts)
        
        logger.info("Building filtered BM25 index...")
        if keywords:
            logger.info(f"  Keywords: {keywords}")
        if include_patterns:
            logger.info(f"  Include patterns: {include_patterns}")
        if exclude_patterns:
            logger.info(f"  Exclude patterns: {exclude_patterns}")
        
        cursor.execute(query, params)
        
        corpus = []
        doc_ids = []
        total_length = 0
        sample_docs = []
        
        for doc_id, tokens_str, doc_length, sourcedoc, text in cursor.fetchall():
            if tokens_str and tokens_str.strip():
                tokens = tokens_str.split()
                if tokens:  # Only add non-empty token lists
                    corpus.append(tokens)
                    doc_ids.append(doc_id)
                    total_length += doc_length or len(tokens)
                    
                    # Collect sample documents
                    if len(sample_docs) < 5:
                        preview = text[:100] + "..." if len(text) > 100 else text
                        sample_docs.append(f"  - {sourcedoc}: {preview}")
        
        if not corpus:
            logger.error("No documents matched the filtering criteria")
            return False
        
        logger.info(f"Found {len(corpus)} documents matching criteria")
        logger.info("Sample documents:")
        for doc in sample_docs:
            logger.info(doc)
        
        # Build BM25 index
        logger.info("Building BM25 index...")
        bm25 = BM25Okapi(corpus, k1=kb.bm25_k1, b=kb.bm25_b)
        
        # Calculate average document length
        avg_doc_length = total_length / len(corpus) if corpus else 0
        
        # Prepare data for serialization
        bm25_data = {
            'bm25': bm25,
            'doc_ids': doc_ids,
            'total_docs': len(doc_ids),
            'avg_doc_length': avg_doc_length,
            'filters': {
                'keywords': keywords,
                'include_patterns': include_patterns,
                'exclude_patterns': exclude_patterns
            }
        }
        
        # Determine save path
        if not save_path:
            save_path = kb.knowledge_base_vector.replace('.faiss', '.bm25')
        
        # Create backup if file exists
        if os.path.exists(save_path):
            backup_path = save_path + '.backup'
            os.rename(save_path, backup_path)
            logger.info(f"Backed up existing index to {backup_path}")
        
        # Save the index
        with open(save_path, 'wb') as f:
            pickle.dump(bm25_data, f)
        
        logger.info(f"BM25 index saved to {save_path}")
        logger.info(f"Index contains {len(doc_ids)} documents")
        logger.info(f"Average document length: {avg_doc_length:.2f} tokens")
        
        return True
        
    except Exception as e:
        logger.error(f"Error building BM25 index: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description='Rebuild BM25 index with filtering for specific content'
    )
    parser.add_argument('config_file', help='Path to KB configuration file')
    parser.add_argument(
        '--keywords', 
        nargs='+',
        help='Keywords to filter documents (e.g., dharma secular samin ethics)'
    )
    parser.add_argument(
        '--include-paths',
        nargs='+',
        help='Path patterns to include (e.g., secular_dharma faqs CoT)'
    )
    parser.add_argument(
        '--exclude-paths',
        nargs='+', 
        help='Path patterns to exclude (e.g., zenquotes test)'
    )
    parser.add_argument(
        '--output',
        help='Output path for BM25 index (defaults to KB directory)'
    )
    parser.add_argument(
        '--restore-backup',
        action='store_true',
        help='Restore the backup BM25 index if it exists'
    )
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Load configuration
    cfgfile = get_fq_cfg_filename(args.config_file)
    if not cfgfile:
        logger.error(f"Configuration file not found: {args.config_file}")
        return 1
    
    kb = KnowledgeBase(cfgfile)
    
    # Handle backup restoration
    if args.restore_backup:
        bm25_path = kb.knowledge_base_vector.replace('.faiss', '.bm25')
        backup_path = bm25_path + '.backup'
        
        if os.path.exists(backup_path):
            os.rename(backup_path, bm25_path)
            logger.info(f"Restored backup from {backup_path}")
            return 0
        else:
            logger.error(f"No backup found at {backup_path}")
            return 1
    
    # Connect to database
    connect_to_database(kb)
    
    try:
        # Rebuild with filters
        success = rebuild_bm25_with_filter(
            kb,
            keywords=args.keywords,
            include_patterns=args.include_paths,
            exclude_patterns=args.exclude_paths,
            save_path=args.output
        )
        
        return 0 if success else 1
        
    finally:
        close_database(kb)

if __name__ == '__main__':
    sys.exit(main())

#fin