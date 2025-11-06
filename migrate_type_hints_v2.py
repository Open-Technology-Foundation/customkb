#!/usr/bin/env python3
"""
Automated migration script for Python 3.12+ type hints - Version 2.

This script modernizes type hints from legacy typing module to Python 3.12+ syntax:
- list[T] → list[T]
- dict[K, V] → dict[K, V]
- tuple[T, ...] → tuple[T, ...]
- set[T] → set[T]
- T | None → T | None
- A | B → A | B
"""

import re
import sys
from pathlib import Path

def migrate_generic_types(content: str) -> str:
  """Replace List, Dict, Tuple, Set with lowercase versions."""
  # Use word boundaries to avoid replacing in comments
  content = re.sub(r'\bList\[', 'list[', content)
  content = re.sub(r'\bDict\[', 'dict[', content)
  content = re.sub(r'\bTuple\[', 'tuple[', content)
  content = re.sub(r'\bSet\[', 'set[', content)
  return content

def migrate_optional(content: str) -> str:
  """Replace T | None with T | None using proper parsing."""
  def replace_match(match):
    inner = match.group(1)
    return f"{inner} | None"

  # Match ... | None with proper bracket counting
  # This regex matches Optional followed by balanced brackets
  pattern = r'Optional\[([^\[\]]*(?:\[[^\[\]]*\])?[^\[\]]*)\]'

  # Keep replacing until no more matches (handles nested cases)
  while 'Optional[' in content:
    new_content = re.sub(pattern, replace_match, content)
    if new_content == content:
      break
    content = new_content

  return content

def migrate_union(content: str) -> str:
  """Replace A | B | C with A | B | C using proper parsing."""
  def replace_match(match):
    inner = match.group(1)
    # Split by comma at top level only
    parts = split_at_top_level(inner, ',')
    return ' | '.join(part.strip() for part in parts)

  # Match ... with proper bracket counting
  pattern = r'Union\[([^\[\]]*(?:\[[^\[\]]*\])?[^\[\]]*)\]'

  # Keep replacing until no more matches
  while 'Union[' in content:
    new_content = re.sub(pattern, replace_match, content)
    if new_content == content:
      break
    content = new_content

  return content

def split_at_top_level(text: str, delimiter: str) -> list:
  """Split text by delimiter, but only at bracket depth 0."""
  parts = []
  current = []
  depth = 0

  for char in text:
    if char in '[{(':
      depth += 1
      current.append(char)
    elif char in ']})':
      depth -= 1
      current.append(char)
    elif char == delimiter and depth == 0:
      parts.append(''.join(current))
      current = []
    else:
      current.append(char)

  if current:
    parts.append(''.join(current))

  return parts

def clean_typing_imports(content: str) -> str:
  """Remove legacy typing imports that are no longer needed."""
  lines = content.split('\n')
  new_lines = []

  for line in lines:
    if not line.strip().startswith('from typing import'):
      new_lines.append(line)
      continue

    # This is a typing import line
    new_line = line

    # Remove generic types
    for type_name in ['List', 'Dict', 'Tuple', 'Set', 'Optional', 'Union']:
      # Remove if it's in the middle with comma
      new_line = re.sub(rf',\s*{type_name}\b', '', new_line)
      # Remove if it's at the start
      new_line = re.sub(rf'import\s+{type_name}\s*,\s*', 'import ', new_line)
      # Remove if it's the only import
      new_line = re.sub(rf'from typing import\s+{type_name}\s*$', '', new_line)

    # Clean up multiple commas and spaces
    new_line = re.sub(r',\s*,', ',', new_line)
    new_line = re.sub(r'import\s*,', 'import', new_line)

    # Skip if nothing left to import
    if re.match(r'from typing import\s*$', new_line.strip()):
      continue

    new_lines.append(new_line)

  return '\n'.join(new_lines)

def migrate_file(file_path: Path, dry_run: bool = False):
  """Migrate a single file."""
  try:
    with open(file_path, 'r', encoding='utf-8') as f:
      original = f.read()

    content = original

    # Apply migrations in order
    content = migrate_generic_types(content)
    content = migrate_optional(content)
    content = migrate_union(content)
    content = clean_typing_imports(content)

    if content == original:
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
  parser.add_argument('--dry-run', action='store_true', help='Show what would be changed')
  parser.add_argument('--path', default='.', help='Path to search for Python files')
  args = parser.parse_args()

  base_path = Path(args.path)
  exclude_dirs = {'.venv', '.mailer', '.gudang', '__pycache__', '.git', 'build', 'dist'}

  python_files = []
  for py_file in base_path.rglob('*.py'):
    if any(parent.name in exclude_dirs for parent in py_file.parents):
      continue
    python_files.append(py_file)

  if not python_files:
    print("No Python files found")
    return 1

  print(f"Found {len(python_files)} Python files")
  if args.dry_run:
    print("DRY RUN MODE - No files will be modified\n")

  changed = 0
  errors = 0

  for file_path in sorted(python_files):
    was_changed, message = migrate_file(file_path, dry_run=args.dry_run)

    if "Error" in message:
      errors += 1
      print(f"✗ {file_path.relative_to(base_path)}: {message}")
    elif was_changed:
      changed += 1
      print(f"✓ {file_path.relative_to(base_path)}: {message}")

  print(f"\n{'=' * 60}")
  print(f"Total: {len(python_files)} | Changed: {changed} | Errors: {errors}")

  return 0 if errors == 0 else 1

if __name__ == '__main__':
  sys.exit(main())

#fin
