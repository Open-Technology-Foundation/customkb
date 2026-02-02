#!/usr/bin/env python
"""
NLTK setup utility for CustomKB - download and manage NLTK data.

This script consolidates the functionality of:
- download_nltk_stopwords.py
- cleanup_nltk_stopwords.py
"""

import argparse
import os
import sys
from pathlib import Path

import nltk

# Set NLTK data directory
NLTK_DATA_DIR = '/usr/share/nltk_data'

# Languages required by CustomKB (ISO code: full name)
REQUIRED_LANGUAGES = {
  'zh': 'chinese',
  'da': 'danish',
  'nl': 'dutch',
  'en': 'english',
  'fi': 'finnish',
  'fr': 'french',
  'de': 'german',
  'id': 'indonesian',
  'it': 'italian',
  'pt': 'portuguese',
  'es': 'spanish',
  'sv': 'swedish'
}


def check_permissions():
  """Check if we have write permissions to NLTK data directory."""
  if not os.access(NLTK_DATA_DIR, os.W_OK):
    print(f"❌ Error: No write permission to {NLTK_DATA_DIR}")
    print(f"Please run with sudo: sudo python {sys.argv[0]}")
    return False
  return True


def download_nltk_data():
  """Download required NLTK data for CustomKB."""
  print("NLTK Data Download")
  print(f"Target directory: {NLTK_DATA_DIR}")
  print("=" * 50)

  if not check_permissions():
    return 1

  # Set NLTK data path
  nltk.data.path = [NLTK_DATA_DIR]

  print(f"\nDownloading data for {len(REQUIRED_LANGUAGES)} languages:")
  for iso, lang in sorted(REQUIRED_LANGUAGES.items()):
    print(f"  {iso}: {lang}")

  print("\nStarting download...")

  try:
    # Download the stopwords corpus (includes all languages)
    nltk.download('stopwords', download_dir=NLTK_DATA_DIR, quiet=False)
    print("\n✅ Successfully downloaded stopwords corpus")

    # Also download punkt tokenizers which are needed
    print("\nDownloading required tokenizers...")
    nltk.download('punkt', download_dir=NLTK_DATA_DIR, quiet=False)
    nltk.download('punkt_tab', download_dir=NLTK_DATA_DIR, quiet=False)
    nltk.download('wordnet', download_dir=NLTK_DATA_DIR, quiet=False)
    print("✅ Successfully downloaded tokenizers")

    # Verify the languages are available
    print("\nVerifying stopwords availability...")
    from nltk.corpus import stopwords

    available = stopwords.fileids()
    print(f"Total languages available: {len(available)}")

    print("\nChecking required languages:")
    all_found = True
    for iso, lang in sorted(REQUIRED_LANGUAGES.items()):
      if lang in available:
        print(f"  ✅ {iso}: {lang}")
      else:
        print(f"  ❌ {iso}: {lang} NOT FOUND")
        all_found = False

    if all_found:
      print("\n✅ All required languages are available!")
    else:
      print("\n⚠️  Some required languages are missing")

  except (LookupError, OSError) as e:
    print(f"\n❌ Error downloading: {e}")
    return 1

  return 0


def cleanup_nltk_data():
  """Remove stopwords for languages not required by CustomKB."""
  stopwords_dir = Path('/usr/share/nltk_data/corpora/stopwords')

  if not stopwords_dir.exists():
    print(f"❌ Stopwords directory not found: {stopwords_dir}")
    return 1

  if not check_permissions():
    return 1

  print("NLTK Stopwords Cleanup")
  print("=" * 50)
  print(f"Directory: {stopwords_dir}")
  print(f"Keeping {len(REQUIRED_LANGUAGES)} languages:")
  for iso, lang in sorted(REQUIRED_LANGUAGES.items()):
    print(f"  {iso}: {lang}")

  # Get all files in directory
  all_files = list(stopwords_dir.iterdir())
  keep_files = set(REQUIRED_LANGUAGES.values())
  keep_files.add('README')  # Keep the README file

  removed_count = 0
  kept_count = 0

  print("\nProcessing files...")
  for file_path in all_files:
    if file_path.is_file():
      filename = file_path.name
      if filename in keep_files:
        print(f"  ✅ Keeping: {filename}")
        kept_count += 1
      else:
        print(f"  🗑️  Removing: {filename}")
        try:
          file_path.unlink()
          removed_count += 1
        except (PermissionError, OSError) as e:
          print(f"     ❌ Error removing {filename}: {e}")

  print("\n" + "=" * 50)
  print("Summary:")
  print(f"  Files kept: {kept_count}")
  print(f"  Files removed: {removed_count}")

  # Show final state
  print("\nFinal stopwords directory contents:")
  remaining_files = sorted([f.name for f in stopwords_dir.iterdir() if f.is_file()])
  for f in remaining_files:
    print(f"  {f}")

  return 0


def show_status():
  """Show current NLTK data status."""
  print("NLTK Data Status")
  print("=" * 50)
  print(f"NLTK data directory: {NLTK_DATA_DIR}")

  # Check stopwords
  stopwords_dir = Path(NLTK_DATA_DIR) / 'corpora' / 'stopwords'
  if stopwords_dir.exists():
    files = sorted([f.name for f in stopwords_dir.iterdir() if f.is_file() and f.name != 'README'])
    print(f"\nStopwords installed: {len(files)} languages")

    # Check which required languages are present
    required_present = []
    required_missing = []
    for iso, lang in sorted(REQUIRED_LANGUAGES.items()):
      if lang in files:
        required_present.append(f"{iso}:{lang}")
      else:
        required_missing.append(f"{iso}:{lang}")

    if required_present:
      print(f"\nRequired languages present ({len(required_present)}):")
      for lang in required_present:
        print(f"  ✅ {lang}")

    if required_missing:
      print(f"\nRequired languages missing ({len(required_missing)}):")
      for lang in required_missing:
        print(f"  ❌ {lang}")

    # Show extra languages
    extra_langs = [f for f in files if f not in REQUIRED_LANGUAGES.values()]
    if extra_langs:
      print(f"\nExtra languages installed ({len(extra_langs)}):")
      for lang in sorted(extra_langs):
        print(f"  - {lang}")
  else:
    print("\n❌ Stopwords directory not found!")

  # Check tokenizers
  print("\nTokenizers:")
  for tokenizer in ['punkt', 'punkt_tab', 'wordnet']:
    path = Path(NLTK_DATA_DIR) / 'tokenizers' / tokenizer
    if path.exists():
      print(f"  ✅ {tokenizer}")
    else:
      # Check in corpora for wordnet
      if tokenizer == 'wordnet':
        path = Path(NLTK_DATA_DIR) / 'corpora' / tokenizer
        if path.exists():
          print(f"  ✅ {tokenizer}")
        else:
          print(f"  ❌ {tokenizer}")
      else:
        print(f"  ❌ {tokenizer}")

  return 0


def main():
  parser = argparse.ArgumentParser(
    description='NLTK setup utility for CustomKB',
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
Examples:
  # Show current NLTK data status
  %(prog)s status

  # Download all required NLTK data
  sudo %(prog)s download

  # Remove unnecessary language files
  sudo %(prog)s cleanup

  # Download then cleanup (full setup)
  sudo %(prog)s download cleanup
"""
  )

  parser.add_argument(
    'commands',
    nargs='+',
    choices=['download', 'cleanup', 'status'],
    help='Commands to run (can specify multiple)'
  )

  args = parser.parse_args()

  # Execute commands in order
  for cmd in args.commands:
    if cmd == 'download':
      result = download_nltk_data()
      if result != 0:
        return result
    elif cmd == 'cleanup':
      if 'download' not in args.commands:
        print("\n⚠️  Warning: Running cleanup without download.")
        response = input("Continue? (y/N): ")
        if response.lower() != 'y':
          print("Cancelled.")
          return 0
      result = cleanup_nltk_data()
      if result != 0:
        return result
    elif cmd == 'status':
      result = show_status()
      if result != 0:
        return result

    if cmd != args.commands[-1]:
      print("\n" + "-" * 50 + "\n")

  return 0


if __name__ == "__main__":
  sys.exit(main())

#fin
