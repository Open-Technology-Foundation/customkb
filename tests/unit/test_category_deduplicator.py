#!/usr/bin/env python3
"""
Unit tests for category deduplication functionality.

Tests the CategoryDeduplicator class and related functions for merging
similar category names using fuzzy string matching.
"""

import pytest

from categorize.category_deduplicator import CategoryDeduplicator, CategoryMergeGroup, deduplicate_categories


class TestCategoryDeduplicator:
    """Test CategoryDeduplicator class."""

    def test_init_default_values(self):
        """Test initialization with default values."""
        dedup = CategoryDeduplicator()

        assert dedup.similarity_threshold == 85.0
        assert dedup.case_sensitive is False
        assert 'and' in dedup.ignore_words
        assert 'the' in dedup.ignore_words
        assert dedup.merge_groups == []

    def test_init_custom_values(self):
        """Test initialization with custom values."""
        custom_ignore = {'custom', 'words'}
        dedup = CategoryDeduplicator(
            similarity_threshold=90.0,
            case_sensitive=True,
            ignore_words=custom_ignore
        )

        assert dedup.similarity_threshold == 90.0
        assert dedup.case_sensitive is True
        assert dedup.ignore_words == custom_ignore

    def test_normalize_category_case_insensitive(self):
        """Test category normalization with case insensitivity."""
        dedup = CategoryDeduplicator(case_sensitive=False)

        assert dedup._normalize_category("Machine Learning") == "machine learning"
        assert dedup._normalize_category("MACHINE LEARNING") == "machine learning"

    def test_normalize_category_case_sensitive(self):
        """Test category normalization with case sensitivity."""
        dedup = CategoryDeduplicator(case_sensitive=True)

        assert dedup._normalize_category("Machine Learning") == "Machine Learning"
        assert dedup._normalize_category("MACHINE LEARNING") == "MACHINE LEARNING"

    def test_normalize_category_removes_ignore_words(self):
        """Test that ignored words are removed during normalization."""
        dedup = CategoryDeduplicator()

        # "and", "of", "the" should be removed
        result = dedup._normalize_category("Machine Learning and AI")
        assert result == "machine learning ai"

        result = dedup._normalize_category("History of the World")
        assert result == "history world"

    def test_normalize_category_preserves_if_only_ignore_words(self):
        """Test normalization preserves original if only ignore words remain."""
        dedup = CategoryDeduplicator()

        # If all words are ignored, keep original
        result = dedup._normalize_category("and the of")
        assert result == "and the of"

    def test_find_duplicates_empty_list(self):
        """Test finding duplicates with empty list."""
        dedup = CategoryDeduplicator()
        result = dedup.find_duplicates([])

        assert result == []
        assert dedup.merge_groups == []

    def test_find_duplicates_no_duplicates(self):
        """Test finding duplicates when none exist."""
        dedup = CategoryDeduplicator(similarity_threshold=85.0)
        categories = ["Machine Learning", "Quantum Physics", "Biology"]

        result = dedup.find_duplicates(categories)

        assert result == []
        assert dedup.merge_groups == []

    def test_find_duplicates_exact_match(self):
        """Test finding exact duplicate categories."""
        dedup = CategoryDeduplicator()
        categories = ["Machine Learning", "machine learning"]

        result = dedup.find_duplicates(categories)

        assert len(result) == 1
        assert result[0].primary == "Machine Learning"
        assert "machine learning" in result[0].aliases
        assert result[0].similarity_scores["machine learning"] >= 95.0

    def test_find_duplicates_similar_categories(self):
        """Test finding similar categories."""
        dedup = CategoryDeduplicator(similarity_threshold=80.0)
        categories = [
            "Machine Learning",
            "Machine learning",
            "ML",
            "Artificial Intelligence",
            "AI"
        ]

        result = dedup.find_duplicates(categories)

        # Should find at least one merge group
        assert len(result) > 0

        # Check that similar categories are grouped
        ml_group = next((g for g in result if "Machine Learning" in g.primary or "machine learning" in g.primary), None)
        assert ml_group is not None

    def test_find_duplicates_prefers_shorter_names(self):
        """Test that shorter category names are preferred as primary."""
        dedup = CategoryDeduplicator(similarity_threshold=85.0)
        categories = [
            "Machine Learning and Artificial Intelligence",
            "Machine Learning"
        ]

        result = dedup.find_duplicates(categories)

        if len(result) > 0:
            # Shorter name should be primary
            assert len(result[0].primary) <= len(list(result[0].aliases)[0])

    def test_merge_category_counts_empty_groups(self):
        """Test merging counts with no merge groups."""
        dedup = CategoryDeduplicator()
        counts = {"AI": 10, "ML": 5, "Physics": 3}

        result = dedup.merge_category_counts(counts)

        assert result == counts

    def test_merge_category_counts_with_groups(self):
        """Test merging category counts based on merge groups."""
        dedup = CategoryDeduplicator()

        # Manually create merge groups
        group = CategoryMergeGroup(
            primary="AI",
            aliases={"Artificial Intelligence", "A.I."},
            similarity_scores={"Artificial Intelligence": 90.0, "A.I.": 95.0}
        )
        dedup.merge_groups = [group]

        counts = {
            "AI": 10,
            "Artificial Intelligence": 20,
            "A.I.": 5,
            "Physics": 3
        }

        result = dedup.merge_category_counts(counts)

        assert result["AI"] == 35  # 10 + 20 + 5
        assert result["Physics"] == 3
        assert "Artificial Intelligence" not in result
        assert "A.I." not in result

    def test_merge_category_counts_preserves_unmerged(self):
        """Test that unmerged categories are preserved."""
        dedup = CategoryDeduplicator()

        group = CategoryMergeGroup(
            primary="ML",
            aliases={"Machine Learning"},
            similarity_scores={"Machine Learning": 92.0}
        )
        dedup.merge_groups = [group]

        counts = {
            "ML": 5,
            "Machine Learning": 10,
            "Physics": 8,
            "Chemistry": 12
        }

        result = dedup.merge_category_counts(counts)

        assert result["ML"] == 15
        assert result["Physics"] == 8
        assert result["Chemistry"] == 12

    def test_apply_to_results_no_groups(self):
        """Test applying deduplication with no merge groups."""
        dedup = CategoryDeduplicator()
        categories = ["AI", "ML", "Physics"]

        result = dedup.apply_to_results(categories)

        assert result == categories

    def test_apply_to_results_with_groups(self):
        """Test applying deduplication with merge groups."""
        dedup = CategoryDeduplicator()

        group = CategoryMergeGroup(
            primary="AI",
            aliases={"Artificial Intelligence", "A.I."},
            similarity_scores={"Artificial Intelligence": 90.0, "A.I.": 95.0}
        )
        dedup.merge_groups = [group]

        categories = ["AI", "Artificial Intelligence", "Physics", "A.I.", "AI"]

        result = dedup.apply_to_results(categories)

        # Should replace aliases with primary and remove duplicates
        assert "AI" in result
        assert "Physics" in result
        assert "Artificial Intelligence" not in result
        assert "A.I." not in result
        assert len(result) == 2  # Only AI and Physics

    def test_apply_to_results_preserves_order(self):
        """Test that deduplication preserves first occurrence order."""
        dedup = CategoryDeduplicator()

        group = CategoryMergeGroup(
            primary="ML",
            aliases={"Machine Learning"},
            similarity_scores={"Machine Learning": 90.0}
        )
        dedup.merge_groups = [group]

        categories = ["Physics", "Machine Learning", "Chemistry", "ML"]

        result = dedup.apply_to_results(categories)

        # First occurrence of ML category should be preserved in position
        assert result.index("Physics") == 0
        assert result.index("ML") == 1
        assert result.index("Chemistry") == 2

    def test_get_merge_report_no_groups(self):
        """Test merge report with no groups."""
        dedup = CategoryDeduplicator()

        report = dedup.get_merge_report()

        assert "No duplicate categories found" in report

    def test_get_merge_report_with_groups(self):
        """Test merge report generation."""
        dedup = CategoryDeduplicator()

        group = CategoryMergeGroup(
            primary="AI",
            aliases={"Artificial Intelligence", "A.I."},
            similarity_scores={"Artificial Intelligence": 88.5, "A.I.": 92.3}
        )
        dedup.merge_groups = [group]

        report = dedup.get_merge_report()

        assert "AI" in report
        assert "Artificial Intelligence" in report
        assert "A.I." in report
        assert "88.5%" in report
        assert "92.3%" in report
        assert "Group 1" in report

    def test_suggest_manual_review_empty_list(self):
        """Test manual review suggestions with empty list."""
        dedup = CategoryDeduplicator()

        result = dedup.suggest_manual_review([])

        assert result == []

    def test_suggest_manual_review_finds_near_duplicates(self):
        """Test that manual review finds categories in threshold range."""
        dedup = CategoryDeduplicator()
        categories = [
            "Machine Learning",
            "Machine learning applications",
            "Deep Learning"
        ]

        # Look for categories with 70-85% similarity
        suggestions = dedup.suggest_manual_review(categories, threshold_range=(70.0, 85.0))

        # Should find some suggestions (may vary based on fuzzy matching)
        assert isinstance(suggestions, list)

        # Each suggestion should be a tuple of (cat1, cat2, score)
        for cat1, cat2, score in suggestions:
            assert isinstance(cat1, str)
            assert isinstance(cat2, str)
            assert 70.0 <= score < 85.0

    def test_suggest_manual_review_sorted_by_score(self):
        """Test that manual review suggestions are sorted by similarity."""
        dedup = CategoryDeduplicator()
        categories = [
            "AI",
            "Artificial Intelligence",
            "A.I. Research",
            "Machine Learning",
            "ML Algorithms"
        ]

        suggestions = dedup.suggest_manual_review(categories, threshold_range=(60.0, 90.0))

        # Verify sorting (higher scores first)
        if len(suggestions) > 1:
            for i in range(len(suggestions) - 1):
                assert suggestions[i][2] >= suggestions[i + 1][2]

    def test_suggest_manual_review_respects_threshold_range(self):
        """Test that manual review respects the threshold range."""
        dedup = CategoryDeduplicator()
        categories = ["AI", "Artificial Intelligence"]

        # These are very similar, so shouldn't appear in low threshold range
        suggestions = dedup.suggest_manual_review(categories, threshold_range=(10.0, 20.0))

        # Should be empty or have low scores only
        for _cat1, _cat2, score in suggestions:
            assert 10.0 <= score < 20.0


class TestDeduplicateCategoriesFunction:
    """Test the convenience function deduplicate_categories."""

    def test_deduplicate_categories_basic(self):
        """Test basic deduplication."""
        categories = ["AI", "Artificial Intelligence", "Machine Learning"]

        deduplicated, report = deduplicate_categories(categories, verbose=False)

        assert isinstance(deduplicated, list)
        assert report is None

    def test_deduplicate_categories_with_report(self):
        """Test deduplication with verbose report."""
        categories = ["AI", "Artificial Intelligence", "Machine Learning", "ML"]

        deduplicated, report = deduplicate_categories(categories, verbose=True)

        assert isinstance(deduplicated, list)
        assert isinstance(report, str)

    def test_deduplicate_categories_custom_threshold(self):
        """Test deduplication with custom threshold."""
        categories = ["Machine Learning", "machine learning", "ML"]

        # High threshold - fewer merges
        dedup_high, _ = deduplicate_categories(categories, similarity_threshold=95.0)

        # Low threshold - more merges
        dedup_low, _ = deduplicate_categories(categories, similarity_threshold=70.0)

        # Lower threshold should result in fewer unique categories (more merging)
        assert len(dedup_low) <= len(dedup_high)

    def test_deduplicate_categories_preserves_unique(self):
        """Test that unique categories are preserved."""
        categories = ["Physics", "Chemistry", "Biology"]

        deduplicated, _ = deduplicate_categories(categories)

        # Should preserve all unique categories
        assert set(deduplicated) == set(categories)


class TestCategoryMergeGroup:
    """Test the CategoryMergeGroup dataclass."""

    def test_category_merge_group_creation(self):
        """Test creating a CategoryMergeGroup."""
        group = CategoryMergeGroup(
            primary="AI",
            aliases={"Artificial Intelligence"},
            similarity_scores={"Artificial Intelligence": 90.0}
        )

        assert group.primary == "AI"
        assert "Artificial Intelligence" in group.aliases
        assert group.similarity_scores["Artificial Intelligence"] == 90.0

    def test_category_merge_group_empty_aliases(self):
        """Test CategoryMergeGroup with no aliases."""
        group = CategoryMergeGroup(
            primary="Physics",
            aliases=set(),
            similarity_scores={}
        )

        assert group.primary == "Physics"
        assert len(group.aliases) == 0
        assert len(group.similarity_scores) == 0


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_single_category(self):
        """Test deduplication with single category."""
        dedup = CategoryDeduplicator()
        result = dedup.find_duplicates(["AI"])

        assert result == []

    def test_special_characters(self):
        """Test categories with special characters."""
        dedup = CategoryDeduplicator()
        categories = ["AI & ML", "AI/ML", "AI-ML"]

        # Should handle special characters without error
        result = dedup.find_duplicates(categories)
        assert isinstance(result, list)

    def test_unicode_categories(self):
        """Test categories with unicode characters."""
        dedup = CategoryDeduplicator()
        categories = ["机器学习", "人工智能", "深度学习"]

        # Should handle unicode without error
        result = dedup.find_duplicates(categories)
        assert isinstance(result, list)

    def test_very_long_category_names(self):
        """Test with very long category names."""
        dedup = CategoryDeduplicator()
        long_name = "A" * 500 + " Machine Learning"
        categories = [long_name, "Machine Learning"]

        # Should handle long names without error
        result = dedup.find_duplicates(categories)
        assert isinstance(result, list)

    def test_empty_string_categories(self):
        """Test handling of empty string categories."""
        dedup = CategoryDeduplicator()
        categories = ["AI", "", "ML", ""]

        # Should handle empty strings gracefully
        result = dedup.find_duplicates(categories)
        assert isinstance(result, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


#fin
