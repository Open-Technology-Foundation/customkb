#!/usr/bin/env python
"""Remove NLTK stopwords for languages not needed."""

import os
import sys
from pathlib import Path

# Languages to KEEP (ISO code: full name)
KEEP_LANGUAGES = {
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

def cleanup_stopwords():
    """Remove stopwords for languages not in the keep list."""
    stopwords_dir = Path('/usr/share/nltk_data/corpora/stopwords')
    
    if not stopwords_dir.exists():
        print(f"❌ Stopwords directory not found: {stopwords_dir}")
        return 1
        
    if not os.access(stopwords_dir, os.W_OK):
        print(f"❌ No write permission to {stopwords_dir}")
        print(f"Please run with sudo: sudo python {sys.argv[0]}")
        return 1
    
    print("NLTK Stopwords Cleanup")
    print("=" * 50)
    print(f"Directory: {stopwords_dir}")
    print(f"Keeping {len(KEEP_LANGUAGES)} languages:")
    for iso, lang in sorted(KEEP_LANGUAGES.items()):
        print(f"  {iso}: {lang}")
    
    # Get all files in directory
    all_files = list(stopwords_dir.iterdir())
    keep_files = set(KEEP_LANGUAGES.values())
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
                except Exception as e:
                    print(f"     ❌ Error removing {filename}: {e}")
    
    print("\n" + "=" * 50)
    print(f"Summary:")
    print(f"  Files kept: {kept_count}")
    print(f"  Files removed: {removed_count}")
    print(f"  Total files now: {kept_count}")
    
    # Show final state
    print("\nFinal stopwords directory contents:")
    remaining_files = sorted([f.name for f in stopwords_dir.iterdir() if f.is_file()])
    for f in remaining_files:
        print(f"  {f}")
    
    return 0

if __name__ == "__main__":
    print("This script will REMOVE stopword files for languages not in your list.")
    print("Make sure you have a backup if needed.")
    response = input("\nContinue? (y/N): ")
    
    if response.lower() == 'y':
        sys.exit(cleanup_stopwords())
    else:
        print("Cancelled.")
        sys.exit(0)

#fin