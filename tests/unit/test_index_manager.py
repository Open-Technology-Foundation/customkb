"""Unit tests for index_manager module."""

import os
import sqlite3
import tempfile
from unittest.mock import Mock, patch

from database.index_manager import (
    EXPECTED_INDEXES,
    create_missing_indexes,
    get_database_indexes,
    get_table_name,
    process_verify_indexes,
    verify_indexes,
)


class TestIndexManager:
    """Test index manager functionality."""

    def setup_method(self):
        """Create a temporary database for testing."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as temp_db:
            self.db_path = temp_db.name

        # Create test table with all columns needed for indexes
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE docs (
                id INTEGER PRIMARY KEY,
                embedded INTEGER,
                embedtext TEXT,
                keyphrase TEXT,
                processed INTEGER,
                sourcedoc TEXT,
                sid INTEGER,
                language TEXT,
                originaltext TEXT,
                metadata TEXT
            )
        """)
        conn.commit()
        conn.close()

    def teardown_method(self):
        """Clean up temporary database."""
        if os.path.exists(self.db_path):
            os.unlink(self.db_path)

    def test_get_database_indexes(self):
        """Test retrieving database indexes."""
        # Create an index
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("CREATE INDEX idx_test ON docs(id)")
        conn.commit()
        conn.close()

        indexes = get_database_indexes(self.db_path)
        index_names = [idx[0] for idx in indexes]
        assert 'idx_test' in index_names

    def test_get_table_name_docs(self):
        """Test table name detection for docs table."""
        table_name = get_table_name(self.db_path)
        assert table_name == 'docs'

    def test_get_table_name_chunks(self):
        """Test table name detection for chunks table."""
        # Create a new db with chunks table
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            chunks_db = f.name

        conn = sqlite3.connect(chunks_db)
        cursor = conn.cursor()
        cursor.execute("CREATE TABLE chunks (id INTEGER PRIMARY KEY)")
        conn.commit()
        conn.close()

        try:
            table_name = get_table_name(chunks_db)
            assert table_name == 'chunks'
        finally:
            os.unlink(chunks_db)

    def test_verify_indexes_all_present(self):
        """Test verify_indexes when all indexes exist."""
        # Create all expected indexes
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        # Create indexes using parameterized queries where possible
        # Note: Index names and column names can't be parameterized in SQLite,
        # but these come from our controlled EXPECTED_INDEXES constant
        index_definitions = {
            'idx_embedded': "CREATE INDEX idx_embedded ON docs(embedded)",
            'idx_embedded_embedtext': "CREATE INDEX idx_embedded_embedtext ON docs(embedded, embedtext)",
            'idx_keyphrase_processed': "CREATE INDEX idx_keyphrase_processed ON docs(keyphrase, processed)",
            'idx_sourcedoc': "CREATE INDEX idx_sourcedoc ON docs(sourcedoc)",
            'idx_sourcedoc_sid': "CREATE INDEX idx_sourcedoc_sid ON docs(sourcedoc, sid)",
            'idx_id': "CREATE UNIQUE INDEX idx_id ON docs(id)",
            'idx_language_embedded': "CREATE INDEX idx_language_embedded ON docs(language, embedded)",
            'idx_metadata': "CREATE INDEX idx_metadata ON docs(metadata)",
            'idx_sourcedoc_sid_covering': "CREATE INDEX idx_sourcedoc_sid_covering ON docs(sourcedoc, sid, id, originaltext, metadata)"
        }
        for idx in EXPECTED_INDEXES:
            if idx in index_definitions:
                cursor.execute(index_definitions[idx])
        conn.commit()
        conn.close()

        results = verify_indexes(self.db_path)
        assert all(results.values())

    def test_verify_indexes_some_missing(self):
        """Test verify_indexes when some indexes are missing."""
        # Create only some indexes
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("CREATE INDEX idx_embedded ON docs(embedded)")
        conn.commit()
        conn.close()

        results = verify_indexes(self.db_path)
        assert results['idx_embedded'] is True
        assert results['idx_sourcedoc'] is False

    def test_create_missing_indexes_dry_run(self):
        """Test creating missing indexes in dry-run mode."""
        created = create_missing_indexes(self.db_path, dry_run=True)
        assert len(created) == len(EXPECTED_INDEXES)

        # Verify no indexes were actually created
        indexes = get_database_indexes(self.db_path)
        assert len(indexes) == 0

    def test_create_missing_indexes_actual(self):
        """Test actually creating missing indexes."""
        created = create_missing_indexes(self.db_path, dry_run=False)
        # Some might fail due to column mismatch, but embedded should work
        assert 'idx_embedded' in created

        # Verify index was created
        indexes = get_database_indexes(self.db_path)
        index_names = [idx[0] for idx in indexes]
        assert 'idx_embedded' in index_names

    @patch('database.index_manager.KnowledgeBase')
    def test_process_verify_indexes(self, mock_kb):
        """Test process_verify_indexes command handler."""
        # Mock KB instance
        mock_kb_instance = Mock()
        mock_kb_instance.knowledge_base_db = self.db_path
        mock_kb.return_value = mock_kb_instance

        args = Mock(config_file='test.cfg')
        logger = Mock()

        result = process_verify_indexes(args, logger)
        assert 'Database:' in result
        assert 'Index verification:' in result
        assert any('MISSING' in result for idx in EXPECTED_INDEXES)


class TestOldSchemaSupport:
    """Test support for old database schema."""

    def test_create_indexes_old_schema(self):
        """Test index creation on old schema with keyphrase_processed column."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name

        try:
            # Create old schema table
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE docs (
                    id INTEGER PRIMARY KEY,
                    embedded INTEGER,
                    embedtext TEXT,
                    keyphrase_processed INTEGER,
                    sourcedoc TEXT,
                    sid INTEGER
                )
            """)
            conn.commit()
            conn.close()

            # Try to create indexes
            create_missing_indexes(db_path, dry_run=False)

            # Should create keyphrase index on single column
            indexes = get_database_indexes(db_path)
            index_names = [idx[0] for idx in indexes]
            if 'idx_keyphrase_processed' in index_names:
                # Check the SQL for the index
                idx_sql = next(sql for name, sql in indexes
                              if name == 'idx_keyphrase_processed')
                assert 'keyphrase_processed)' in idx_sql  # Single column

        finally:
            os.unlink(db_path)


#fin
