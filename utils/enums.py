"""
Enumerations for CustomKB type safety and code clarity.

This module provides type-safe enums for common constants used throughout
the CustomKB codebase, improving code maintainability and preventing typos.
"""

from enum import Enum


class ReferenceFormat(Enum):
  """
  Output formats for search result references.

  These formats determine how search results are presented to LLMs.
  """
  XML = 'xml'
  JSON = 'json'
  MARKDOWN = 'markdown'
  PLAIN = 'plain'

  @property
  def value_str(self) -> str:
    """Return the string value of the enum."""
    return self.value

  @classmethod
  def from_string(cls, value: str) -> 'ReferenceFormat':
    """
    Convert string to ReferenceFormat enum.

    Args:
        value: Format string (case-insensitive)

    Returns:
        ReferenceFormat enum member

    Raises:
        ValueError: If value is not a valid format
    """
    # Handle aliases
    aliases = {
      'md': cls.MARKDOWN,
      'text': cls.PLAIN,
    }

    value_lower = value.lower()
    if value_lower in aliases:
      return aliases[value_lower]

    try:
      return cls(value_lower)
    except ValueError:
      valid = ', '.join([f.value for f in cls])
      raise ValueError(f"Invalid reference format: '{value}'. Must be one of: {valid}") from None


class OptimizationTier(Enum):
  """
  Memory-based optimization tiers for system resource management.

  These tiers determine batch sizes, cache limits, and concurrency settings
  based on available system memory.
  """
  LOW = 'low'           # < 16GB RAM
  MEDIUM = 'medium'     # 16-64GB RAM
  HIGH = 'high'         # 64-128GB RAM
  VERY_HIGH = 'very_high'  # > 128GB RAM

  @property
  def value_str(self) -> str:
    """Return the string value of the enum."""
    return self.value

  @classmethod
  def from_memory(cls, memory_gb: float) -> 'OptimizationTier':
    """
    Determine optimization tier from system memory.

    Args:
        memory_gb: System memory in gigabytes

    Returns:
        OptimizationTier enum member
    """
    if memory_gb < 16:
      return cls.LOW
    elif memory_gb < 64:
      return cls.MEDIUM
    elif memory_gb < 128:
      return cls.HIGH
    else:
      return cls.VERY_HIGH

  @classmethod
  def from_string(cls, value: str) -> 'OptimizationTier':
    """
    Convert string to OptimizationTier enum.

    Args:
        value: Tier string (case-insensitive)

    Returns:
        OptimizationTier enum member

    Raises:
        ValueError: If value is not a valid tier
    """
    try:
      return cls(value.lower())
    except ValueError:
      valid = ', '.join([t.value for t in cls])
      raise ValueError(f"Invalid optimization tier: '{value}'. Must be one of: {valid}") from None

#fin
