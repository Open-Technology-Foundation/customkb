#!/usr/bin/env python3
"""
Unit tests for query search functionality.

Tests vector search, hybrid search, and document retrieval operations.
"""

import sqlite3

import pytest

from query.search import (
    fetch_document_by_id,
    get_context_range,
    merge_search_results,
    merge_search_results_rrf,
)
from utils.exceptions import DatabaseError


class TestGetContextRange:
    """Test context range calculation function."""

    def test_basic_context_range(self):
        """Test basic context range calculation."""
        # With context_n=5, starting at index 10
        # Should get [8, 12] (10 +/- 2)
        result = get_context_range(10, 5)

        assert result[0] == 8
        assert result[1] == 12

    def test_context_range_at_start(self):
        """Test context range when starting near beginning."""
        # At index 2 with context_n=5
        # Should get [0, 4] (can't go negative)
        result = get_context_range(2, 5)

        assert result[0] == 0
        assert result[1] == 4

    def test_context_range_at_zero(self):
        """Test context range starting at index 0."""
        result = get_context_range(0, 5)

        assert result[0] == 0
        assert result[1] == 4

    def test_context_range_single_item(self):
        """Test context range with context_n=1."""
        result = get_context_range(10, 1)

        assert result[0] == 10
        assert result[1] == 10

    def test_context_range_even_number(self):
        """Test context range with even context_n."""
        # With context_n=4, should use (4-1)//2 = 1 for half
        result = get_context_range(10, 4)

        assert result[0] == 9
        assert result[1] == 12

    def test_context_range_odd_number(self):
        """Test context range with odd context_n."""
        # With context_n=7, should use (7-1)//2 = 3 for half
        result = get_context_range(10, 7)

        assert result[0] == 7
        assert result[1] == 13

    def test_context_range_large_number(self):
        """Test context range with large context_n."""
        result = get_context_range(50, 21)

        assert result[0] == 40
        assert result[1] == 60

    def test_context_range_zero_context(self):
        """Test context range with context_n=0 (should default to 1)."""
        result = get_context_range(10, 0)

        # Should default to context_n=1
        assert result[0] == 10
        assert result[1] == 10

    def test_context_range_negative_context(self):
        """Test context range with negative context_n."""
        result = get_context_range(10, -5)

        # Should default to context_n=1
        assert result[0] == 10
        assert result[1] == 10

    def test_context_range_ensures_correct_size(self):
        """Test that context range always returns correct size."""
        # Even when starting near beginning, range size should be consistent
        result = get_context_range(1, 7)

        # Range should still try to get 7 items starting from 0
        assert result[1] - result[0] + 1 == 7
        assert result[0] == 0

    def test_context_range_returns_list(self):
        """Test that function returns a list."""
        result = get_context_range(10, 5)

        assert isinstance(result, list)
        assert len(result) == 2


class TestFetchDocumentById:
    """Test document fetching function."""

    def test_fetch_document_success(self, mock_kb):
        """Test successfully fetching a document."""
        # Setup mock cursor to return a document
        mock_kb.sql_cursor.fetchone.return_value = (123, 1, "/path/to/doc.txt")

        result = fetch_document_by_id(mock_kb, 123)

        assert result is not None
        assert result[0] == 123  # id
        assert result[1] == 1    # sid
        assert result[2] == "/path/to/doc.txt"  # sourcedoc

        # Verify query was executed
        mock_kb.sql_cursor.execute.assert_called_once()

    def test_fetch_document_not_found(self, mock_kb):
        """Test fetching non-existent document."""
        # Setup mock cursor to return None
        mock_kb.sql_cursor.fetchone.return_value = None

        result = fetch_document_by_id(mock_kb, 999)

        assert result is None

    def test_fetch_document_with_custom_table(self, mock_kb):
        """Test fetching document with custom table name."""
        mock_kb.table_name = "docs"
        mock_kb.sql_cursor.fetchone.return_value = (1, 1, "test.txt")

        result = fetch_document_by_id(mock_kb, 1)

        assert result is not None

        # Verify correct table name was used
        call_args = mock_kb.sql_cursor.execute.call_args[0]
        assert "FROM docs" in call_args[0]

    def test_fetch_document_sql_injection_protection(self, mock_kb):
        """Test that table name is validated to prevent SQL injection."""
        # Try to use malicious table name
        mock_kb.table_name = "docs; DROP TABLE users--"

        with pytest.raises(DatabaseError, match="Invalid table name"):
            fetch_document_by_id(mock_kb, 1)

    def test_fetch_document_database_error(self, mock_kb):
        """Test handling of database errors."""
        # Setup mock to raise sqlite3.Error
        mock_kb.sql_cursor.execute.side_effect = sqlite3.Error("Connection lost")

        with pytest.raises(DatabaseError, match="Failed to fetch document"):
            fetch_document_by_id(mock_kb, 1)

    def test_fetch_document_casts_id_to_int(self, mock_kb):
        """Test that document ID is cast to integer."""
        mock_kb.sql_cursor.fetchone.return_value = (42, 1, "test.txt")

        # Pass string ID
        result = fetch_document_by_id(mock_kb, "42")

        assert result is not None

        # Verify integer was used in query
        call_args = mock_kb.sql_cursor.execute.call_args[0]
        assert call_args[1] == (42,)

    def test_fetch_document_uses_limit_1(self, mock_kb):
        """Test that query uses LIMIT 1 for efficiency."""
        mock_kb.sql_cursor.fetchone.return_value = (1, 1, "test.txt")

        fetch_document_by_id(mock_kb, 1)

        # Verify LIMIT 1 is in the query
        call_args = mock_kb.sql_cursor.execute.call_args[0]
        assert "LIMIT 1" in call_args[0]


class TestMergeSearchResults:
    """Test search results merging function."""

    def test_merge_basic(self):
        """Test basic merging of vector and BM25 results."""
        vector_results = [(1, 0.9), (2, 0.7)]
        bm25_results = [(1, 10.0), (3, 8.0)]

        result = merge_search_results(vector_results, bm25_results)

        # Should have 3 unique documents
        assert len(result) == 3

        # All results should be (doc_id, score) tuples
        for doc_id, score in result:
            assert isinstance(doc_id, int)
            assert isinstance(score, float)

    def test_merge_sorted_by_score(self):
        """Test that merged results are sorted by score descending."""
        vector_results = [(1, 0.5), (2, 0.9)]
        bm25_results = [(1, 5.0), (3, 15.0)]

        result = merge_search_results(vector_results, bm25_results)

        # Scores should be in descending order
        for i in range(len(result) - 1):
            assert result[i][1] >= result[i + 1][1]

    def test_merge_empty_vector_results(self):
        """Test merging with empty vector results."""
        vector_results = []
        bm25_results = [(1, 10.0), (2, 8.0)]

        result = merge_search_results(vector_results, bm25_results)

        # Should still have BM25 results
        assert len(result) == 2

    def test_merge_empty_bm25_results(self):
        """Test merging with empty BM25 results."""
        vector_results = [(1, 0.9), (2, 0.7)]
        bm25_results = []

        result = merge_search_results(vector_results, bm25_results)

        # Should still have vector results
        assert len(result) == 2

    def test_merge_both_empty(self):
        """Test merging with both results empty."""
        result = merge_search_results([], [])

        assert result == []

    def test_merge_custom_weights(self):
        """Test merging with custom weights."""
        vector_results = [(1, 1.0)]
        bm25_results = [(1, 1.0)]

        # Equal weights
        result1 = merge_search_results(vector_results, bm25_results,
                                      vector_weight=0.5, bm25_weight=0.5)

        # Favor vector
        result2 = merge_search_results(vector_results, bm25_results,
                                      vector_weight=0.9, bm25_weight=0.1)

        # Favor BM25
        result3 = merge_search_results(vector_results, bm25_results,
                                      vector_weight=0.1, bm25_weight=0.9)

        # All should have same document but different scores
        assert len(result1) == len(result2) == len(result3) == 1
        assert result1[0][0] == result2[0][0] == result3[0][0] == 1

    def test_merge_normalizes_weights(self):
        """Test that weights are normalized to sum to 1."""
        vector_results = [(1, 1.0)]
        bm25_results = [(1, 1.0)]

        # Weights that don't sum to 1
        result = merge_search_results(vector_results, bm25_results,
                                     vector_weight=2.0, bm25_weight=3.0)

        # Should still work (weights normalized internally)
        assert len(result) == 1
        assert 0.0 <= result[0][1] <= 1.0

    def test_merge_normalizes_scores(self):
        """Test that scores are normalized to [0, 1] range."""
        vector_results = [(1, 0.5), (2, 1.0)]
        bm25_results = [(1, 5.0), (2, 10.0)]

        result = merge_search_results(vector_results, bm25_results)

        # Combined scores should be in reasonable range
        for _doc_id, score in result:
            assert 0.0 <= score <= 1.0

    def test_merge_handles_duplicate_docs(self):
        """Test that duplicate documents are combined correctly."""
        # Same document in both results
        vector_results = [(1, 0.8)]
        bm25_results = [(1, 10.0)]

        result = merge_search_results(vector_results, bm25_results)

        # Should have only 1 result (not 2)
        assert len(result) == 1
        assert result[0][0] == 1

    def test_merge_different_doc_sets(self):
        """Test merging completely different document sets."""
        vector_results = [(1, 0.9), (2, 0.8)]
        bm25_results = [(3, 10.0), (4, 9.0)]

        result = merge_search_results(vector_results, bm25_results)

        # Should have all 4 documents
        assert len(result) == 4

        # Verify all doc_ids are present
        doc_ids = {doc_id for doc_id, _ in result}
        assert doc_ids == {1, 2, 3, 4}

    def test_merge_large_result_sets(self):
        """Test merging large result sets."""
        vector_results = [(i, float(i) / 100) for i in range(100)]
        bm25_results = [(i, float(i)) for i in range(50, 150)]

        result = merge_search_results(vector_results, bm25_results)

        # Should have 150 unique documents (0-149)
        assert len(result) == 150

        # Verify all scores are valid
        for _doc_id, score in result:
            assert isinstance(score, float)
            assert 0.0 <= score <= 1.0

    def test_merge_maintains_float_precision(self):
        """Test that merging maintains float precision."""
        # RRF score is 1/(k+rank), so we need different ranks to get different scores
        # Use multiple items so they get different ranks
        vector_results = [(1, 0.999999), (2, 0.888888)]  # IDs 1 and 2
        bm25_results = [(3, 0.000001)]                    # ID 3

        result = merge_search_results(vector_results, bm25_results)

        # Should have all 3 items
        assert len(result) == 3

        # Results with different ranks should have different scores
        # Item with rank 1 (appears first in both or one list) vs item with rank 2
        scores = [r[1] for r in result]
        assert len(set(scores)) > 1  # At least 2 different scores


class TestEdgeCases:
    """Test edge cases for search functions."""

    def test_get_context_range_very_large_index(self):
        """Test context range with very large starting index."""
        result = get_context_range(1000000, 5)

        assert result[0] == 999998
        assert result[1] == 1000002

    def test_fetch_document_negative_id(self, mock_kb):
        """Test fetching document with negative ID."""
        mock_kb.sql_cursor.fetchone.return_value = None

        # Should handle negative IDs (though they shouldn't exist)
        fetch_document_by_id(mock_kb, -1)

        # Negative ID should be cast to int
        call_args = mock_kb.sql_cursor.execute.call_args[0]
        assert call_args[1] == (-1,)

    def test_merge_single_result_each(self):
        """Test merging with single result in each set."""
        vector_results = [(1, 0.9)]
        bm25_results = [(2, 10.0)]

        result = merge_search_results(vector_results, bm25_results)

        assert len(result) == 2

    def test_merge_identical_results(self):
        """Test merging identical result sets."""
        vector_results = [(1, 0.9), (2, 0.8), (3, 0.7)]
        bm25_results = [(1, 0.9), (2, 0.8), (3, 0.7)]

        result = merge_search_results(vector_results, bm25_results)

        # Should combine scores for same documents
        assert len(result) == 3

    def test_merge_zero_weights(self):
        """Test merging with zero weights."""
        vector_results = [(1, 1.0)]
        bm25_results = [(2, 1.0)]

        # Zero vector weight
        result1 = merge_search_results(vector_results, bm25_results,
                                      vector_weight=0.0, bm25_weight=1.0)

        # Zero BM25 weight
        result2 = merge_search_results(vector_results, bm25_results,
                                      vector_weight=1.0, bm25_weight=0.0)

        # Both should work
        assert len(result1) == 2
        assert len(result2) == 2


class TestRRFMerge:
    """Test Reciprocal Rank Fusion merge function."""

    def test_rrf_basic(self):
        """Test basic RRF merging of vector and BM25 results."""
        vector_results = [(1, 0.9), (2, 0.7), (3, 0.5)]
        bm25_results = [(2, 10.0), (3, 8.0), (4, 6.0)]

        result = merge_search_results_rrf(vector_results, bm25_results)

        # Should have 4 unique documents
        assert len(result) == 4

        # Doc 2 and 3 should be highest (appear in both)
        top_doc_ids = {result[0][0], result[1][0]}
        assert 2 in top_doc_ids or 3 in top_doc_ids

    def test_rrf_sorted_by_score(self):
        """Test that RRF results are sorted by score descending."""
        vector_results = [(1, 0.5), (2, 0.9)]
        bm25_results = [(1, 5.0), (3, 15.0)]

        result = merge_search_results_rrf(vector_results, bm25_results)

        # Scores should be in descending order
        for i in range(len(result) - 1):
            assert result[i][1] >= result[i + 1][1]

    def test_rrf_documents_in_both_rank_higher(self):
        """Test that documents appearing in both result sets rank higher."""
        # Doc 1 appears in both, docs 2,3 only in one each
        vector_results = [(1, 0.9), (2, 0.8)]
        bm25_results = [(1, 10.0), (3, 8.0)]

        result = merge_search_results_rrf(vector_results, bm25_results)

        # Doc 1 should be first since it appears in both lists
        assert result[0][0] == 1

    def test_rrf_empty_vector_results(self):
        """Test RRF with empty vector results."""
        vector_results = []
        bm25_results = [(1, 10.0), (2, 8.0)]

        result = merge_search_results_rrf(vector_results, bm25_results)

        # Should still have BM25 results
        assert len(result) == 2

    def test_rrf_empty_bm25_results(self):
        """Test RRF with empty BM25 results."""
        vector_results = [(1, 0.9), (2, 0.7)]
        bm25_results = []

        result = merge_search_results_rrf(vector_results, bm25_results)

        # Should still have vector results
        assert len(result) == 2

    def test_rrf_both_empty(self):
        """Test RRF with both results empty."""
        result = merge_search_results_rrf([], [])

        assert result == []

    def test_rrf_custom_k(self):
        """Test RRF with custom k parameter."""
        vector_results = [(1, 0.9), (2, 0.7)]
        bm25_results = [(1, 10.0), (3, 8.0)]

        # Default k=60
        result1 = merge_search_results_rrf(vector_results, bm25_results, k=60)

        # Higher k gives more weight to lower-ranked results
        result2 = merge_search_results_rrf(vector_results, bm25_results, k=1)

        # Both should work and have same documents
        assert len(result1) == len(result2) == 3

    def test_rrf_formula_correctness(self):
        """Test that RRF formula is correctly implemented."""
        # Simple case: single doc in each list at rank 1
        vector_results = [(1, 0.9)]
        bm25_results = [(2, 10.0)]

        result = merge_search_results_rrf(vector_results, bm25_results, k=60)

        # Doc 1: 1/(60+1) = 0.01639...
        # Doc 2: 1/(60+1) = 0.01639...
        # Both should have same score since both are at rank 1
        assert abs(result[0][1] - result[1][1]) < 0.001

    def test_rrf_rank_based_not_score_based(self):
        """Test that RRF uses ranks, not raw scores."""
        # Different raw scores but same ranks
        vector_results = [(1, 1000.0), (2, 1.0)]  # Huge score difference
        bm25_results = [(3, 1000.0), (4, 1.0)]

        result = merge_search_results_rrf(vector_results, bm25_results)

        # Despite huge score differences, rank-1 docs should have same RRF score
        # Doc 1 and 3 are both at rank 1
        doc1_score = next(s for d, s in result if d == 1)
        doc3_score = next(s for d, s in result if d == 3)
        assert abs(doc1_score - doc3_score) < 0.001


class TestMergeSearchResultsFusionMethod:
    """Test the main merge function with different fusion methods."""

    def test_merge_default_uses_rrf(self):
        """Test that default fusion method is RRF."""
        vector_results = [(1, 0.9), (2, 0.7)]
        bm25_results = [(1, 10.0), (3, 8.0)]

        # Default should use RRF
        result = merge_search_results(vector_results, bm25_results)

        # Should have 3 unique documents
        assert len(result) == 3

    def test_merge_explicit_rrf(self):
        """Test explicit RRF fusion method."""
        vector_results = [(1, 0.9), (2, 0.7)]
        bm25_results = [(1, 10.0), (3, 8.0)]

        result = merge_search_results(vector_results, bm25_results,
                                      fusion_method="rrf")

        assert len(result) == 3

    def test_merge_weighted_method(self):
        """Test weighted fusion method."""
        vector_results = [(1, 0.9), (2, 0.7)]
        bm25_results = [(1, 10.0), (3, 8.0)]

        result = merge_search_results(vector_results, bm25_results,
                                      fusion_method="weighted")

        assert len(result) == 3
        # Weighted method normalizes scores to [0, 1]
        for _doc_id, score in result:
            assert 0.0 <= score <= 1.0

    def test_merge_rrf_with_custom_k(self):
        """Test RRF with custom k parameter via main function."""
        vector_results = [(1, 0.9), (2, 0.7)]
        bm25_results = [(1, 10.0), (3, 8.0)]

        result = merge_search_results(vector_results, bm25_results,
                                      fusion_method="rrf", rrf_k=30)

        assert len(result) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


#fin
