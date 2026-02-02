#!/usr/bin/env python3
"""
Adaptive performance thresholds for tests.

This module provides system-aware performance thresholds that adjust
based on available CPU, memory, and system load to make tests more
reliable across different hardware configurations.
"""

import os
from typing import Any

import psutil


class AdaptiveThresholds:
    """
    Compute adaptive performance thresholds based on system capabilities.

    This class measures system resources and provides scaled thresholds
    for performance tests, allowing tests to pass on slower hardware while
    still catching real performance regressions on capable systems.
    """

    def __init__(self):
        """Initialize and measure system capabilities."""
        self.cpu_count = os.cpu_count() or 1
        self.memory_gb = psutil.virtual_memory().total / (1024**3)
        self.cpu_freq_mhz = self._get_cpu_frequency()
        self.load_factor = self._compute_load_factor()

    def _get_cpu_frequency(self) -> float:
        """Get CPU frequency in MHz."""
        try:
            freq = psutil.cpu_freq()
            if freq:
                return freq.current
        except (AttributeError, OSError):
            pass
        # Default to a baseline frequency if unavailable
        return 2000.0  # 2 GHz baseline

    def _compute_load_factor(self) -> float:
        """
        Compute a load factor based on system characteristics.

        Returns:
            Float multiplier (1.0 = baseline, >1.0 = slower system)
        """
        # Baseline: 4 cores, 8GB RAM, 2GHz CPU
        baseline_cores = 4
        baseline_memory_gb = 8
        baseline_freq_mhz = 2000.0

        # Compute relative performance factors
        core_factor = baseline_cores / max(self.cpu_count, 1)
        memory_factor = baseline_memory_gb / max(self.memory_gb, 1)
        freq_factor = baseline_freq_mhz / max(self.cpu_freq_mhz, 100)

        # Weight the factors (CPU is most important for performance tests)
        load_factor = (
            0.5 * core_factor +     # 50% weight on cores
            0.3 * freq_factor +      # 30% weight on frequency
            0.2 * memory_factor      # 20% weight on memory
        )

        # Clamp between 0.5x (twice as fast) and 3.0x (three times slower)
        return max(0.5, min(3.0, load_factor))

    def adjust(self, baseline_threshold: float,
               category: str = 'default') -> float:
        """
        Adjust a baseline threshold based on system capabilities.

        Args:
            baseline_threshold: The threshold for a baseline system
            category: Category of threshold ('fast', 'medium', 'slow', 'default')

        Returns:
            Adjusted threshold appropriate for this system
        """
        # Apply category-specific scaling
        category_multipliers = {
            'fast': 0.8,      # Strict for fast operations
            'medium': 1.0,    # Standard scaling
            'slow': 1.3,      # More lenient for slow operations
            'default': 1.0
        }

        multiplier = category_multipliers.get(category, 1.0)
        adjusted = baseline_threshold * self.load_factor * multiplier

        return adjusted

    def get_thresholds(self) -> dict[str, float]:
        """
        Get a dictionary of common performance thresholds.

        Returns:
            Dictionary of threshold names to values (in seconds)
        """
        return {
            # Build/initialization thresholds
            'bm25_build_small': self.adjust(2.0, 'medium'),      # Small dataset build
            'bm25_build_medium': self.adjust(5.0, 'medium'),     # Medium dataset build
            'bm25_build_large': self.adjust(15.0, 'slow'),       # Large dataset build

            # Load/I|O thresholds
            'avg_load_time': self.adjust(0.5, 'fast'),           # Average load time
            'max_load_time': self.adjust(1.0, 'medium'),         # Maximum load time

            # Search thresholds
            'single_search': self.adjust(0.01, 'fast'),          # Single search operation
            'avg_search_time': self.adjust(0.1, 'medium'),       # Average search time
            'batch_search': self.adjust(10.0, 'slow'),           # Batch search operations

            # Tokenization thresholds
            'tokenization': self.adjust(0.001, 'fast'),          # Per-document tokenization

            # Memory/cache thresholds
            'cache_lookup': self.adjust(0.005, 'fast'),          # Cache lookup time
            'cache_save': self.adjust(0.010, 'fast'),            # Cache save time

            # Scoring thresholds
            'scoring_single': self.adjust(0.005, 'fast'),        # Single document scoring
            'scoring_batch': self.adjust(0.1, 'medium'),         # Batch scoring
        }

    def get_system_info(self) -> dict[str, Any]:
        """
        Get system information for debugging.

        Returns:
            Dictionary of system characteristics
        """
        return {
            'cpu_count': self.cpu_count,
            'memory_gb': round(self.memory_gb, 2),
            'cpu_freq_mhz': round(self.cpu_freq_mhz, 2),
            'load_factor': round(self.load_factor, 3),
            'system_category': self._get_system_category()
        }

    def _get_system_category(self) -> str:
        """Categorize system performance tier."""
        if self.load_factor < 0.7:
            return 'high_performance'
        elif self.load_factor < 1.3:
            return 'baseline'
        else:
            return 'resource_constrained'


# Global instance for convenience
_adaptive_thresholds = None


def get_adaptive_thresholds() -> AdaptiveThresholds:
    """
    Get or create the global adaptive thresholds instance.

    Returns:
        AdaptiveThresholds instance
    """
    global _adaptive_thresholds
    if _adaptive_thresholds is None:
        _adaptive_thresholds = AdaptiveThresholds()
    return _adaptive_thresholds


def get_threshold(name: str, default: float = 1.0) -> float:
    """
    Convenience function to get a specific threshold.

    Args:
        name: Threshold name (e.g., 'avg_search_time')
        default: Default value if threshold not found

    Returns:
        Threshold value in seconds
    """
    thresholds = get_adaptive_thresholds()
    return thresholds.get_thresholds().get(name, default)


def adjust_threshold(baseline: float, category: str = 'default') -> float:
    """
    Convenience function to adjust a baseline threshold.

    Args:
        baseline: Baseline threshold value
        category: Performance category ('fast', 'medium', 'slow', 'default')

    Returns:
        Adjusted threshold
    """
    thresholds = get_adaptive_thresholds()
    return thresholds.adjust(baseline, category)


#fin
