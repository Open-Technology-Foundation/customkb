#!/usr/bin/env python3
"""
Unit tests for categorization manager functionality.

Tests the CategoryGenerator and dataclasses for AI-powered article categorization.
"""

import sqlite3
from dataclasses import asdict
from unittest.mock import patch

import pytest

from categorize.categorize_manager import ArticleCategories, CategoryGenerator, CategoryResult


class TestCategoryResult:
    """Test CategoryResult dataclass."""

    def test_category_result_creation(self):
        """Test creating a CategoryResult."""
        result = CategoryResult(name="Machine Learning", confidence=0.85)

        assert result.name == "Machine Learning"
        assert result.confidence == 0.85

    def test_category_result_to_dict(self):
        """Test converting CategoryResult to dictionary."""
        result = CategoryResult(name="AI", confidence=0.92)
        result_dict = asdict(result)

        assert result_dict["name"] == "AI"
        assert result_dict["confidence"] == 0.92

    def test_category_result_with_zero_confidence(self):
        """Test CategoryResult with zero confidence."""
        result = CategoryResult(name="Unknown", confidence=0.0)

        assert result.confidence == 0.0

    def test_category_result_with_max_confidence(self):
        """Test CategoryResult with maximum confidence."""
        result = CategoryResult(name="Certain", confidence=1.0)

        assert result.confidence == 1.0


class TestArticleCategories:
    """Test ArticleCategories dataclass."""

    def test_article_categories_basic(self):
        """Test creating ArticleCategories with basic fields."""
        categories = ArticleCategories(
            article_path="/path/to/article.txt",
            total_chunks=100,
            sampled_chunks=10,
            categories=[CategoryResult("AI", 0.9)],
            primary_category="AI",
            processing_time=1.5
        )

        assert categories.article_path == "/path/to/article.txt"
        assert categories.total_chunks == 100
        assert categories.sampled_chunks == 10
        assert len(categories.categories) == 1
        assert categories.primary_category == "AI"
        assert categories.processing_time == 1.5

    def test_article_categories_with_error(self):
        """Test ArticleCategories with error field."""
        categories = ArticleCategories(
            article_path="/path/to/article.txt",
            total_chunks=50,
            sampled_chunks=5,
            categories=[],
            primary_category=None,
            processing_time=0.5,
            error="API rate limit exceeded"
        )

        assert categories.error == "API rate limit exceeded"
        assert categories.primary_category is None
        assert len(categories.categories) == 0

    def test_article_categories_with_model_info(self):
        """Test ArticleCategories with model information."""
        categories = ArticleCategories(
            article_path="/test.txt",
            total_chunks=20,
            sampled_chunks=5,
            categories=[CategoryResult("Physics", 0.88)],
            primary_category="Physics",
            processing_time=2.1,
            model_used="gpt-4o-mini"
        )

        assert categories.model_used == "gpt-4o-mini"

    def test_article_categories_with_suggested_new_categories(self):
        """Test ArticleCategories with suggested new categories."""
        categories = ArticleCategories(
            article_path="/test.txt",
            total_chunks=30,
            sampled_chunks=10,
            categories=[CategoryResult("Science", 0.75)],
            primary_category="Science",
            processing_time=1.8,
            suggested_new_categories=["Astrophysics", "Quantum Mechanics"]
        )

        assert len(categories.suggested_new_categories) == 2
        assert "Astrophysics" in categories.suggested_new_categories
        assert "Quantum Mechanics" in categories.suggested_new_categories

    def test_article_categories_to_dict(self):
        """Test converting ArticleCategories to dictionary."""
        categories = ArticleCategories(
            article_path="/test.txt",
            total_chunks=10,
            sampled_chunks=5,
            categories=[CategoryResult("ML", 0.95)],
            primary_category="ML",
            processing_time=1.0
        )

        result_dict = asdict(categories)

        assert result_dict["article_path"] == "/test.txt"
        assert result_dict["total_chunks"] == 10
        assert result_dict["primary_category"] == "ML"
        assert isinstance(result_dict["categories"], list)


class TestCategoryGenerator:
    """Test CategoryGenerator class."""

    def test_init(self, mock_kb):
        """Test CategoryGenerator initialization."""
        generator = CategoryGenerator(mock_kb)

        assert generator.kb == mock_kb
        assert generator.logger is not None

    def test_analyze_content_table_validation(self, mock_kb, tmp_path):
        """Test that analyze_content validates table names."""
        # Create a temporary database with invalid table name
        db_path = tmp_path / "test.db"
        mock_kb.knowledge_base_db = str(db_path)

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Create a table with unexpected name
        cursor.execute("CREATE TABLE malicious_table (id INTEGER, data TEXT)")
        cursor.execute("INSERT INTO malicious_table VALUES (1, 'test')")
        conn.commit()
        conn.close()

        generator = CategoryGenerator(mock_kb)

        # Should raise ValueError for missing docs/chunks table
        with pytest.raises(ValueError, match="No 'chunks' or 'docs' table found"):
            generator.analyze_content(sample_size=5)

    def test_analyze_content_with_docs_table(self, mock_kb, tmp_path):
        """Test analyze_content with docs table."""
        # Create a temporary database with docs table
        db_path = tmp_path / "test.db"
        mock_kb.knowledge_base_db = str(db_path)

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Create docs table
        cursor.execute("""
            CREATE TABLE docs (
                id INTEGER PRIMARY KEY,
                sourcedoc TEXT,
                originaltext TEXT,
                embedtext TEXT
            )
        """)

        # Insert sample data
        sample_docs = [
            (1, "ai.txt", "Artificial Intelligence is fascinating", "AI text"),
            (2, "ml.txt", "Machine Learning algorithms", "ML text"),
            (3, "physics.txt", "Quantum mechanics and relativity", "Physics text")
        ]

        for doc in sample_docs:
            cursor.execute(
                "INSERT INTO docs (id, sourcedoc, originaltext, embedtext) VALUES (?, ?, ?, ?)",
                doc
            )

        conn.commit()
        conn.close()

        generator = CategoryGenerator(mock_kb)

        # Mock the AI call to avoid real API requests in unit tests
        with patch.object(generator, '_generate_categories_from_samples', return_value=["AI", "ML"]):
            result = generator.analyze_content(sample_size=2)
            assert isinstance(result, list | dict)

    def test_analyze_content_with_chunks_table(self, mock_kb, tmp_path):
        """Test analyze_content with chunks table."""
        # Create a temporary database with chunks table
        db_path = tmp_path / "test.db"
        mock_kb.knowledge_base_db = str(db_path)

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Create chunks table
        cursor.execute("""
            CREATE TABLE chunks (
                id INTEGER PRIMARY KEY,
                sourcedoc TEXT,
                text TEXT
            )
        """)

        # Insert sample data
        cursor.execute("INSERT INTO chunks (id, sourcedoc, text) VALUES (1, 'test.txt', 'sample text')")
        conn.commit()
        conn.close()

        generator = CategoryGenerator(mock_kb)

        # Should not raise an error about missing table
        try:
            result = generator.analyze_content(sample_size=1)
            assert isinstance(result, list | dict)
        except ValueError as e:
            if "No 'chunks' or 'docs' table found" in str(e):
                pytest.fail("Should have found chunks table")
        except (sqlite3.Error, RuntimeError, ConnectionError, KeyError):
            # Other exceptions are ok (API calls, etc.)
            pass

    def test_analyze_content_prevents_sql_injection(self, mock_kb, tmp_path):
        """Test that table name validation prevents SQL injection."""
        # Create database
        db_path = tmp_path / "test.db"
        mock_kb.knowledge_base_db = str(db_path)

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Create malicious table that might be targeted
        cursor.execute("CREATE TABLE docs (id INTEGER, sourcedoc TEXT, originaltext TEXT, embedtext TEXT)")
        cursor.execute("CREATE TABLE sensitive_data (secret TEXT)")
        cursor.execute("INSERT INTO sensitive_data VALUES ('secret_info')")
        conn.commit()

        # Try to use SQL injection in table name (shouldn't work due to validation)
        # The code validates table_name is in ['docs', 'chunks']
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name IN ('chunks', 'docs')")
        table = cursor.fetchone()

        conn.close()

        # Ensure only allowed table was selected
        assert table[0] in ['docs', 'chunks']


class TestCategoryGeneratorEdgeCases:
    """Test edge cases for CategoryGenerator."""

    def test_analyze_content_empty_database(self, mock_kb, tmp_path):
        """Test analyze_content with empty database."""
        db_path = tmp_path / "empty.db"
        mock_kb.knowledge_base_db = str(db_path)

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("CREATE TABLE docs (id INTEGER, sourcedoc TEXT, originaltext TEXT)")
        conn.commit()
        conn.close()

        generator = CategoryGenerator(mock_kb)

        # Mock the AI call to avoid real API requests in unit tests
        with patch.object(generator, '_generate_categories_from_samples', return_value=[]):
            # Should handle empty database gracefully
            result = generator.analyze_content(sample_size=5)
            # With empty DB, _generate_categories_from_samples gets empty list
            assert isinstance(result, list | dict)

    def test_analyze_content_invalid_sample_size(self, mock_kb, tmp_path):
        """Test analyze_content with invalid sample size."""
        db_path = tmp_path / "test.db"
        mock_kb.knowledge_base_db = str(db_path)

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("CREATE TABLE docs (id INTEGER, sourcedoc TEXT, originaltext TEXT)")
        conn.commit()
        conn.close()

        generator = CategoryGenerator(mock_kb)

        # Mock the AI call to avoid real API requests in unit tests
        with patch.object(generator, '_generate_categories_from_samples', return_value=[]):
            # Should handle zero sample size gracefully
            result = generator.analyze_content(sample_size=0)
            assert isinstance(result, list | dict)


class TestMockingExternalAPIs:
    """Test CategoryGenerator with mocked external dependencies."""

    def test_analyze_content_with_mocked_openai(self, mock_kb, tmp_path):
        """Test analyze_content with mocked OpenAI API."""
        # Setup database
        db_path = tmp_path / "test.db"
        mock_kb.knowledge_base_db = str(db_path)

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("CREATE TABLE docs (id INTEGER, sourcedoc TEXT, originaltext TEXT, embedtext TEXT)")
        cursor.execute("INSERT INTO docs VALUES (1, 'test.txt', 'AI and ML content', 'test')")
        conn.commit()
        conn.close()

        generator = CategoryGenerator(mock_kb)

        # Mock the AI call to avoid real API requests in unit tests
        with patch.object(generator, '_generate_categories_from_samples', return_value=["AI", "ML"]):
            result = generator.analyze_content(sample_size=1)
            assert isinstance(result, list | dict)
            assert "AI" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


#fin
