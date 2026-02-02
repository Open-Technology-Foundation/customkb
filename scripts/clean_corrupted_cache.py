#!/usr/bin/env python
"""
Clean corrupted embedding cache files.

This script scans the embedding cache directory and removes corrupted
or invalid cache files, including those with dimension mismatches.
"""

import argparse
import json
import os
import sys
from typing import Any

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from embedding.cache import CACHE_DIR, MODEL_DIMENSIONS
from utils.logging_config import get_logger

logger = get_logger(__name__)


def scan_cache_directory() -> tuple[int, int, int]:
  """
  Scan cache directory for corrupted files.

  Returns:
      Tuple of (total_files, corrupted_files, cleaned_files)
  """
  total_files = 0
  corrupted_files = 0
  cleaned_files = 0

  if not os.path.exists(CACHE_DIR):
    logger.info(f"Cache directory does not exist: {CACHE_DIR}")
    return 0, 0, 0

  logger.info(f"Scanning cache directory: {CACHE_DIR}")

  # Walk through all subdirectories
  for root, _dirs, files in os.walk(CACHE_DIR):
    for file in files:
      if not file.endswith('.json'):
        continue

      total_files += 1
      file_path = os.path.join(root, file)

      try:
        # Try to load and validate the cache file
        with open(file_path) as f:
          cache_data = json.load(f)

        # Check required fields
        if not all(key in cache_data for key in ['model', 'text_hash', 'embedding']):
          logger.warning(f"Missing required fields in {file_path}")
          corrupted_files += 1
          if remove_file(file_path):
            cleaned_files += 1
          continue

        # Validate embedding dimensions
        model = cache_data['model']
        embedding = cache_data['embedding']

        if not isinstance(embedding, list):
          logger.warning(f"Invalid embedding type in {file_path}")
          corrupted_files += 1
          if remove_file(file_path):
            cleaned_files += 1
          continue

        # Check known model dimensions
        expected_dims = MODEL_DIMENSIONS.get(model)
        if expected_dims and len(embedding) != expected_dims:
          logger.warning(f"Dimension mismatch in {file_path}: got {len(embedding)}, expected {expected_dims} for model {model}")
          corrupted_files += 1
          if remove_file(file_path):
            cleaned_files += 1
          continue

        # Check for unusual dimensions (likely corrupted)
        if len(embedding) not in [768, 1536, 3072, 1024, 2048]:
          logger.warning(f"Unusual dimension {len(embedding)} in {file_path} for model {model}")
          corrupted_files += 1
          if remove_file(file_path):
            cleaned_files += 1
          continue

      except (OSError, json.JSONDecodeError, KeyError) as e:
        logger.warning(f"Cannot read/parse {file_path}: {e}")
        corrupted_files += 1
        if remove_file(file_path):
          cleaned_files += 1
      except (TypeError, ValueError) as e:
        logger.error(f"Unexpected error checking {file_path}: {e}")

  return total_files, corrupted_files, cleaned_files


def remove_file(file_path: str) -> bool:
  """
  Remove a file safely.

  Args:
      file_path: Path to file to remove

  Returns:
      True if removed successfully
  """
  try:
    os.remove(file_path)
    logger.info(f"Removed: {file_path}")
    return True
  except OSError as e:
    logger.error(f"Failed to remove {file_path}: {e}")
    return False


def get_cache_stats() -> dict[str, Any]:
  """
  Get statistics about the cache directory.

  Returns:
      Dictionary with cache statistics
  """
  stats = {
    'total_files': 0,
    'total_size_mb': 0.0,
    'subdirectories': 0,
    'by_model': {}
  }

  if not os.path.exists(CACHE_DIR):
    return stats

  # Count files and size
  for root, dirs, files in os.walk(CACHE_DIR):
    stats['subdirectories'] += len(dirs)

    for file in files:
      if not file.endswith('.json'):
        continue

      stats['total_files'] += 1
      file_path = os.path.join(root, file)

      # Get file size
      try:
        size_bytes = os.path.getsize(file_path)
        stats['total_size_mb'] += size_bytes / (1024 * 1024)

        # Try to get model info
        with open(file_path) as f:
          cache_data = json.load(f)
          model = cache_data.get('model', 'unknown')
          if model not in stats['by_model']:
            stats['by_model'][model] = {'count': 0, 'size_mb': 0}
          stats['by_model'][model]['count'] += 1
          stats['by_model'][model]['size_mb'] += size_bytes / (1024 * 1024)
      except (OSError, json.JSONDecodeError, KeyError) as e:
        logger.debug(f"Could not process cache file {file_path}: {e}")
        pass  # Skip corrupted or inaccessible cache files

  return stats


def main():
  """Main entry point."""
  parser = argparse.ArgumentParser(
    description='Clean corrupted embedding cache files',
    formatter_class=argparse.RawDescriptionHelpFormatter
  )

  parser.add_argument(
    '--dry-run',
    action='store_true',
    help='Show what would be cleaned without removing files'
  )

  parser.add_argument(
    '--stats',
    action='store_true',
    help='Show cache statistics'
  )

  args = parser.parse_args()

  if args.stats:
    print("\nCache Statistics:")
    print("=" * 60)
    stats = get_cache_stats()
    print(f"Total files: {stats['total_files']:,}")
    print(f"Total size: {stats['total_size_mb']:.2f} MB")
    print(f"Subdirectories: {stats['subdirectories']:,}")

    if stats['by_model']:
      print("\nBy Model:")
      for model, info in sorted(stats['by_model'].items()):
        print(f"  {model}: {info['count']:,} files, {info['size_mb']:.2f} MB")
    return

  if args.dry_run:
    print("\nDRY RUN - No files will be removed")
    print("=" * 60)

  # Store original remove function
  global remove_file

  if args.dry_run:
    # Replace with no-op for dry run
    def remove_file(path):
      logger.info(f"Would remove: {path}")
      return True

  # Scan and clean
  total, corrupted, cleaned = scan_cache_directory()

  # Print summary
  print("\nSummary:")
  print("=" * 60)
  print(f"Total cache files: {total:,}")
  print(f"Corrupted files found: {corrupted:,}")
  if not args.dry_run:
    print(f"Files cleaned: {cleaned:,}")
  else:
    print(f"Files that would be cleaned: {corrupted:,}")

  if corrupted > 0:
    percentage = (corrupted / total * 100) if total > 0 else 0
    print(f"Corruption rate: {percentage:.2f}%")


if __name__ == "__main__":
  main()

#fin
