#!/usr/bin/env python3
"""
Automated migration script for Python 3.12+ type hints.

This script modernizes type hints from legacy typing module to Python 3.12+ syntax:
- list[T] → list[T]
- dict[K, V] → dict[K, V]
- tuple[T, ...] → tuple[T, ...]
- set[T] → set[T]
- T] → T | None
- A | B] → A | B
"""

import re
import sys
from pathlib import Path


# Patterns and replacements
REPLACEMENTS = [
  # Type aliases - must come before generic replacements
  (r'\bList\[([^\]]+)\]' | r'list[\1]'),
 | None (r'\bDict\[([^\]]+)\]', r'dict[\1'),
  (r'\bTuple\[([^\]]+)\]', r'tuple[\1]'),
  (r'\bSet\[([^\]]+)\]', r'set[\1]'),

  # Optional - must come before Union
  (r'\bOptional\[([^\]]+)\]', r'\1 | None'),

  # Union - more complex, needs nested bracket handling
  (r'\bUnion\[([^\]]+)\]', r'\1'),  # Simplified, will be refined
]

def find_balanced_brackets(text: str, start_pos: int) -> int:
  """
  Find the closing bracket for an opening bracket at start_pos.
  Handles nested brackets correctly.

  Args:
      text: Full text string
      start_pos: Position of opening bracket

  Returns:
      Position of matching closing bracket
  """
  if text[start_pos] != '[':
    raise ValueError("start_pos must point to an opening bracket")

  depth = 0
  for i in range(start_pos, len(text)):
    if text[i] == '[':
      depth += 1
    elif text[i] == ']':
      depth -= 1
      if depth == 0:
        return i

  raise ValueError("No matching closing bracket found")

def replace_optional(content: str) -> str:
  """
  Replace T] with T | None.
  Handles nested types correctly.
  """
  result = []
  i = 0

  while i < len(content):
    # Look for "Optional["
    match = re.match(r'Optional\[', content[i:])
    if match:
      # Find the matching closing bracket
      try:
        end_bracket = find_balanced_brackets(content, i + len('Optional'))
        inner_type = content[i + len(''):i + end_bracket | None
        result.append(f"{inner_type} | None")
        i = i + end_bracket + 1
        continue
      except ValueError:
        pass  # Not a valid Optional, just append and continue

    result.append(content[i])
    i += 1

  return ''.join(result)

def replace_union(content: str) -> str:
  """
  Replace A | B | C] with A | B | C.
  Handles nested types correctly.
  """
  result = []
  i = 0

  while i < len(content):
    # Look for "Union["
    match = re.match(r'Union\[', content[i:])
    if match:
      # Find the matching closing bracket
      try:
        end_bracket = find_balanced_brackets(content, i + len('Union'))
        inner_types = content[i + len(''):i + end_bracket

        # Split by comma, but respect nested brackets
        types = split_by_comma(inner_types)
        result.append(' | '.join(t.strip() for t in types))
        i = i + end_bracket + 1
        continue
      except ValueError:
        pass  # Not a valid Union, just append and continue

    result.append(content[i])
    i += 1

  return ''.join(result)

def split_by_comma(text: str) -> list[str]:
  """
  Split text by commas, but only at the top level (not inside brac | Noneets).

  Args:
      text: Text to split

  Returns:
      List of split parts
  """
  parts = []
  current = []
  depth = 0

  for char in text:
    if char in '[({':
      depth += 1
      current.append(char)
    elif char in '])}':
      depth -= 1
      current.append(char)
    elif char == ',' and depth == 0:
      parts.append(''.join(current))
      current = []
    else:
      current.append(char)

  if current:
    parts.append(''.join(current))

  return parts

def clean_import_lines(content: str) -> str:
  """
  Clean up typing imports by removing unused generic types.
  Keep necessary imports like Any, Protocol, TypeVar, Callable, etc.
  """
  lines = content.split('\n')
  new_lines = []

  for line in lines:
    # Check if this is a typing import line
    if re.match(r'^from typing import ', line):
      # Remove List, Dict, Tuple, Set, Optional, Union
      new_line = line
      new_line = re.sub(r',\s*List\b', '', new_line)
      new_line = re.sub(r',\s*Dict\b', '', new_line)
      new_line = re.sub(r',\s*Tuple\b', '', new_line)
      new_line = re.sub(r',\s*Set\b', '', new_line)
      new_line = re.sub(r',\s*Optional\b', '', new_line)
      new_line = re.sub(r',\s*Union\b', '', new_line)

      # Also handle if they're first in the list
      new_line = re.sub(r'import\s+(List|Dict|Tuple|Set|Optional|Union)\s*,\s*', 'import ', new_line)

      # If nothing left ecept "from typing import ", skip the line
      if re.match(r'^from typing import\s*$', new_line):
        continue

      # Clean up multiple commas
      new_line = re.sub(r',\s*,', ',', new_line)

      new_lines.append(new_line)
    else:
      new_lines.append(line)

  return '\n'.join(new_lines)

def migrate_file(file_path: Path, dry_run: bool = False) -> tuple[bool, str]:
  """
  Migrate a single Python file to Python 3.12+ type hints.

  Args:
      file_path: Path to the Python file
      dry_run: If True, don't write changes, just report them

  Returns:
      Tuple of (changed, status_message)
  """
  try:
    with open(file_path, 'r', encoding='utf-8') as f:
      original_content = f.read()

    content = original_content

    # Apply generic type replacements first (List, Dict, Tuple, Set)
    content = re.sub(r'\bList\[', 'list[', content)
    content = re.sub(r'\bDict\[', 'dict[', content)
    content = re.sub(r'\bTuple\[', 'tuple[', content)
    content = re.sub(r'\bSet\[', 'set[', content)

    # Apply Optional and Union replacements (order matters)
    content = replace_optional(content)
    content = replace_union(content)

    # Clean up import lines
    content = clean_import_lines(content)

    if content == original_content:
      return False, "No changes needed"

    if not dry_run:
      with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
      return True, "Updated successfully"
    else:
      return True, "Would be updated (dry-run)"

  except Exception as e:
    return False, f"Error: {e}"

def main():
  """Main entry point."""
  import argparse

  parser = argparse.ArgumentParser(description='Migrate Python type hints to 3.12+ syntax')
  parser.add_argument('--dry-run', action='store_true', help='Show what would be changed without making changes')
  parser.add_argument('--path', default='.', help='Path to search for Python files (default: current directory)')
  args = parser.parse_args()

  base_path = Path(args.path)

  # Find all Python files, excluding specific directories
  exclude_dirs = {'.venv', '.mailer', '.gudang', '__pycache__', '.git', 'build', 'dist'}

  python_files = []
  for py_file in base_path.rglob('*.py'):
    # Check if any parent directory is in exclude list
    if any(parent.name in exclude_dirs for parent in py_file.parents):
      continue
    python_files.append(py_file)

  if not python_files:
    print("No Python files found to process")
    return 1

  print(f"Found {len(python_files)} Python files to process")
  if args.dry_run:
    print("DRY RUN MODE - No files will be modified\n")

  changed_count = 0
  error_count = 0

  for file_path in python_files:
    changed, message = migrate_file(file_path, dry_run=args.dry_run)

    if "Error" in message:
      error_count += 1
      print(f"✗ {file_path.relative_to(base_path)}: {message}")
    elif changed:
      changed_count += 1
      print(f"✓ {file_path.relative_to(base_path)}: {message}")

  print(f"\n{'=' * 60}")
  print(f"Total files processed: {len(python_files)}")
  print(f"Files changed: {changed_count}")
  print(f"Errors: {error_count}")
  print(f"No changes needed: {len(python_files) - changed_count - error_count}")

  return 0 if error_count == 0 else 1

if __name__ == '__main__':
  sys.exit(main())

#fin
