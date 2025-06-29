#!/usr/bin/env python
"""Download specific NLTK stopwords for required languages."""

import nltk
import os
import sys

# Set NLTK data directory
NLTK_DATA_DIR = '/usr/share/nltk_data'

# Languages you want (ISO code: full name)
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

def download_required_stopwords():
    """Download only the required stopwords."""
    print(f"NLTK stopwords download script")
    print(f"Target directory: {NLTK_DATA_DIR}")
    print("=" * 50)
    
    # Check if running with sufficient permissions
    if not os.access(NLTK_DATA_DIR, os.W_OK):
        print(f"❌ Error: No write permission to {NLTK_DATA_DIR}")
        print(f"Please run with sudo: sudo python {sys.argv[0]}")
        return 1
    
    # Set NLTK data path
    nltk.data.path = [NLTK_DATA_DIR]
    
    print(f"\nDownloading stopwords for {len(REQUIRED_LANGUAGES)} languages:")
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
            
        # List other available languages
        other_langs = [l for l in available if l not in REQUIRED_LANGUAGES.values() and l != 'README']
        if other_langs:
            print(f"\nOther available languages (not required): {len(other_langs)}")
            for lang in sorted(other_langs):
                print(f"  - {lang}")
                
    except Exception as e:
        print(f"\n❌ Error downloading: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(download_required_stopwords())

#fin