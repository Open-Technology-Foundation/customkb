#!/usr/bin/env python3
"""
Category Deduplication Module
Merges similar categories using fuzzy string matching
"""


import logging
from dataclasses import dataclass

from rapidfuzz import fuzz

logger = logging.getLogger(__name__)

@dataclass
class CategoryMergeGroup:
  """Represents a group of similar categories that should be merged"""
  primary: str  # The category name to keep
  aliases: set[str]  # Similar category names to merge
  similarity_scores: dict[str, float]  # Similarity scores for each alias

class CategoryDeduplicator:
  """Handles deduplication of similar category names"""

  def __init__(self, similarity_threshold: float = 85.0,
               case_sensitive: bool = False,
               ignore_words: set[str] | None = None):
    """
    Initialize the deduplicator.

    Args:
        similarity_threshold: Minimum similarity score (0-100) to consider categories as duplicates
        case_sensitive: Whether to consider case when matching
        ignore_words: Set of words to ignore when comparing (e.g., 'and', 'of', 'the')
    """
    self.similarity_threshold = similarity_threshold
    self.case_sensitive = case_sensitive
    self.ignore_words = ignore_words or {'and', 'of', 'the', 'in', 'on', 'for', 'with', 'a', 'an'}
    self.merge_groups: list[CategoryMergeGroup] = []

  def _normalize_category(self, category: str) -> str:
    """Normalize category name for comparison"""
    normalized = category

    if not self.case_sensitive:
      normalized = normalized.lower()

    # Remove ignored words for comparison
    words = normalized.split()
    filtered_words = [w for w in words if w.lower() not in self.ignore_words]

    return ' '.join(filtered_words) if filtered_words else normalized

  def find_duplicates(self, categories: list[str]) -> list[CategoryMergeGroup]:
    """
    Find groups of similar categories.

    Args:
        categories: List of category names to check for duplicates

    Returns:
        List of CategoryMergeGroup objects representing merge groups
    """
    if not categories:
      return []

    # Clear previous results
    self.merge_groups = []
    processed = set()

    # Sort categories by length (prefer shorter names as primary)
    sorted_categories = sorted(categories, key=len)

    for i, cat1 in enumerate(sorted_categories):
      if cat1 in processed:
        continue

      # Normalize for comparison
      norm_cat1 = self._normalize_category(cat1)
      merge_group = CategoryMergeGroup(
        primary=cat1,
        aliases=set(),
        similarity_scores={}
      )

      # Find similar categories
      for cat2 in sorted_categories[i+1:]:
        if cat2 in processed:
          continue

        norm_cat2 = self._normalize_category(cat2)

        # Calculate similarity using multiple metrics
        ratio_score = fuzz.ratio(norm_cat1, norm_cat2)
        partial_score = fuzz.partial_ratio(norm_cat1, norm_cat2)
        token_sort_score = fuzz.token_sort_ratio(norm_cat1, norm_cat2)
        token_set_score = fuzz.token_set_ratio(norm_cat1, norm_cat2)

        # Use the maximum score
        max_score = max(ratio_score, partial_score, token_sort_score, token_set_score)

        if max_score >= self.similarity_threshold:
          merge_group.aliases.add(cat2)
          merge_group.similarity_scores[cat2] = max_score
          processed.add(cat2)

      # Only add groups with duplicates
      if merge_group.aliases:
        self.merge_groups.append(merge_group)
        processed.add(cat1)

    return self.merge_groups

  def merge_category_counts(self, category_counts: dict[str, int]) -> dict[str, int]:
    """
    Merge category counts based on identified duplicates.

    Args:
        category_counts: Dictionary of category names to their counts

    Returns:
        New dictionary with merged counts
    """
    if not self.merge_groups:
      return category_counts.copy()

    merged_counts = {}
    processed = set()

    # Process merge groups
    for group in self.merge_groups:
      total_count = category_counts.get(group.primary, 0)
      processed.add(group.primary)

      for alias in group.aliases:
        total_count += category_counts.get(alias, 0)
        processed.add(alias)

      merged_counts[group.primary] = total_count

    # Add categories that weren't merged
    for category, count in category_counts.items():
      if category not in processed:
        merged_counts[category] = count

    return merged_counts

  def apply_to_results(self, categories: list[str]) -> list[str]:
    """
    Apply deduplication to a list of categories.

    Args:
        categories: List of category names

    Returns:
        Deduplicated list with aliases replaced by primary categories
    """
    if not self.merge_groups:
      return categories

    # Create alias to primary mapping
    alias_map = {}
    for group in self.merge_groups:
      for alias in group.aliases:
        alias_map[alias] = group.primary

    # Apply mapping and remove duplicates
    deduplicated = []
    seen = set()

    for category in categories:
      # Map to primary if it's an alias
      primary = alias_map.get(category, category)

      # Add if not already present
      if primary not in seen:
        deduplicated.append(primary)
        seen.add(primary)

    return deduplicated

  def get_merge_report(self) -> str:
    """Generate a human-readable report of the merge groups"""
    if not self.merge_groups:
      return "No duplicate categories found."

    report = f"Found {len(self.merge_groups)} groups of similar categories:\n"
    report += "-" * 50 + "\n"

    for i, group in enumerate(self.merge_groups, 1):
      report += f"\nGroup {i}: '{group.primary}' (primary)\n"
      for alias in sorted(group.aliases):
        score = group.similarity_scores[alias]
        report += f"  - '{alias}' (similarity: {score:.1f}%)\n"

    return report

  def suggest_manual_review(self, categories: list[str],
                          threshold_range: tuple[float, float] = (70.0, 85.0)) -> list[tuple[str, str, float]]:
    """
    Find category pairs that might be duplicates but fall below auto-merge threshold.

    Args:
        categories: List of category names
        threshold_range: (min, max) similarity scores to flag for review

    Returns:
        List of (category1, category2, similarity_score) tuples for manual review
    """
    suggestions = []
    min_threshold, max_threshold = threshold_range

    for i, cat1 in enumerate(categories):
      norm_cat1 = self._normalize_category(cat1)

      for cat2 in categories[i+1:]:
        norm_cat2 = self._normalize_category(cat2)

        # Calculate similarity
        ratio_score = fuzz.ratio(norm_cat1, norm_cat2)
        partial_score = fuzz.partial_ratio(norm_cat1, norm_cat2)
        token_sort_score = fuzz.token_sort_ratio(norm_cat1, norm_cat2)
        token_set_score = fuzz.token_set_ratio(norm_cat1, norm_cat2)

        max_score = max(ratio_score, partial_score, token_sort_score, token_set_score)

        # Check if in manual review range
        if min_threshold <= max_score < max_threshold:
          suggestions.append((cat1, cat2, max_score))

    # Sort by similarity score (highest first)
    suggestions.sort(key=lambda x: x[2], reverse=True)

    return suggestions

def deduplicate_categories(categories: list[str],
                          similarity_threshold: float = 85.0,
                          verbose: bool = False) -> tuple[list[str], str | None]:
  """
  Convenience function to deduplicate a list of categories.

  Args:
      categories: List of category names
      similarity_threshold: Minimum similarity score for auto-merging
      verbose: Whether to return a merge report

  Returns:
      Tuple of (deduplicated_categories, merge_report or None)
  """
  deduplicator = CategoryDeduplicator(similarity_threshold=similarity_threshold)
  deduplicator.find_duplicates(categories)

  deduplicated = deduplicator.apply_to_results(categories)
  report = deduplicator.get_merge_report() if verbose else None

  return deduplicated, report

#fin
