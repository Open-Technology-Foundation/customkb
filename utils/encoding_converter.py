#!/usr/bin/env python
"""
Encoding conversion utilities for CustomKB.

Provides tools for converting text files to UTF-8 encoding in-place.
"""

import os
import shutil

from utils.logging_config import get_logger
from utils.text_utils import detect_file_encoding, get_files

logger = get_logger(__name__)


def convert_file_to_utf8(file_path: str, backup: bool = True, dry_run: bool = False) -> dict:
  """
  Convert a single file to UTF-8 encoding.

  Args:
      file_path: Path to the file to convert
      backup: Create a backup of the original file
      dry_run: Only detect encoding without converting

  Returns:
      Dict with conversion results:
      - 'success': bool
      - 'original_encoding': str
      - 'message': str
      - 'backup_path': str (if backup created)
  """
  result = {
    'success': False,
    'original_encoding': None,
    'message': '',
    'backup_path': None
  }

  try:
    # Detect current encoding
    detected_encoding = detect_file_encoding(file_path)
    result['original_encoding'] = detected_encoding

    # If already UTF-8, skip
    # Normalize encoding name by replacing underscores and hyphens
    normalized_encoding = detected_encoding.lower().replace('_', '-').replace('utf-8', 'utf8')
    if normalized_encoding in ['utf8', 'ascii']:
      result['success'] = True
      result['message'] = f'Already {detected_encoding}'
      return result

    if dry_run:
      result['success'] = True
      result['message'] = f'Would convert from {detected_encoding} to UTF-8'
      return result

    # Read file with detected encoding
    try:
      with open(file_path, encoding=detected_encoding) as f:
        content = f.read()
    except (UnicodeDecodeError, OSError) as e:
      result['message'] = f'Failed to read with {detected_encoding}: {e}'
      return result

    # Create backup if requested
    if backup:
      backup_path = f'{file_path}.bak'
      shutil.copy2(file_path, backup_path)
      result['backup_path'] = backup_path

    # Write as UTF-8
    try:
      with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

      result['success'] = True
      result['message'] = f'Converted from {detected_encoding} to UTF-8'
      return result

    except (PermissionError, OSError) as e:
      # Restore backup if write failed
      if backup and result['backup_path']:
        shutil.copy2(result['backup_path'], file_path)
      result['message'] = f'Failed to write UTF-8: {e}'
      return result

  except (FileNotFoundError, PermissionError, OSError, UnicodeDecodeError) as e:
    result['message'] = f'Conversion error: {e}'
    return result


def convert_files_to_utf8(
  file_patterns: list[str],
  backup: bool = True,
  dry_run: bool = False,
  recursive: bool = False
) -> dict:
  """
  Convert multiple files to UTF-8 encoding.

  Args:
      file_patterns: List of file paths or glob patterns
      backup: Create backups of original files
      dry_run: Only detect encodings without converting
      recursive: Process directories recursively

  Returns:
      Dict with overall results:
      - 'total': int - Total files processed
      - 'converted': int - Successfully converted files
      - 'skipped': int - Files already in UTF-8
      - 'failed': int - Failed conversions
      - 'details': list - Per-file results
  """
  results = {
    'total': 0,
    'converted': 0,
    'skipped': 0,
    'failed': 0,
    'details': []
  }

  # Collect all files
  all_files = []
  for pattern in file_patterns:
    files = get_files(pattern)
    all_files.extend(files)

  # Remove duplicates
  all_files = list(set(all_files))
  results['total'] = len(all_files)

  if not all_files:
    logger.warning("No files found matching patterns")
    return results

  logger.info(f"Processing {len(all_files)} file(s)...")

  # Process each file
  for file_path in all_files:
    logger.debug(f"Processing: {file_path}")

    file_result = convert_file_to_utf8(file_path, backup=backup, dry_run=dry_run)

    # Update counters
    if file_result['success']:
      if 'Already' in file_result['message']:
        results['skipped'] += 1
        logger.debug(f"  ✓ {os.path.basename(file_path)}: {file_result['message']}")
      else:
        results['converted'] += 1
        logger.info(f"  ✓ {os.path.basename(file_path)}: {file_result['message']}")
    else:
      results['failed'] += 1
      logger.error(f"  ✗ {os.path.basename(file_path)}: {file_result['message']}")

    # Add to details
    results['details'].append({
      'file': file_path,
      **file_result
    })

  return results


def format_conversion_summary(results: dict) -> str:
  """
  Format conversion results as a human-readable summary.

  Args:
      results: Results dict from convert_files_to_utf8

  Returns:
      Formatted summary string
  """
  lines = []
  lines.append("\n=== Encoding Conversion Summary ===")
  lines.append(f"Total files: {results['total']}")
  lines.append(f"Converted: {results['converted']}")
  lines.append(f"Skipped (already UTF-8): {results['skipped']}")
  lines.append(f"Failed: {results['failed']}")

  if results['failed'] > 0:
    lines.append("\n=== Failed Conversions ===")
    for detail in results['details']:
      if not detail['success']:
        lines.append(f"  {detail['file']}: {detail['message']}")

  return '\n'.join(lines)


#fin
