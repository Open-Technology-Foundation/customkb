#!/usr/bin/env python
"""
Script to safely update dependencies in requirements.txt
Focuses on security and compatibility updates.
"""

import subprocess
import sys
from pathlib import Path

# Critical packages to update for security/stability
PRIORITY_UPDATES = {
    # Security critical
    'charset-normalizer': '>=3.4.3',
    'filelock': '>=3.19.1',
    
    # Core functionality
    'google-genai': '>=1.31.0',
    'langchain-text-splitters': '>=0.3.9',
    'langchain-core': '>=0.3.74',
    'langsmith': '>=0.4.15',
    
    # Testing and development
    'coverage': '>=7.10.4',
    'Faker': '>=37.5.3',
    
    # Performance improvements
    'anyio': '>=4.10.0',
    'cachetools': '>=6.1.0',
    'huggingface-hub': '>=0.34.4',
}

# Packages to keep at specific versions for compatibility
PINNED_PACKAGES = {
    'numpy': '==1.26.4',  # Keep for compatibility with other packages
    'faiss-gpu-cu12': '==1.11.0',  # GPU version compatibility
    'spacy': '==3.7.5',  # NLP model compatibility
}

def update_requirements():
    """Update requirements.txt with priority updates."""
    req_file = Path('requirements.txt')
    
    if not req_file.exists():
        print("Error: requirements.txt not found")
        return 1
    
    # Read current requirements
    with open(req_file, 'r') as f:
        lines = f.readlines()
    
    # Process each line
    updated_lines = []
    updated_packages = []
    
    for line in lines:
        line = line.strip()
        
        # Skip empty lines and comments
        if not line or line.startswith('#'):
            updated_lines.append(line)
            continue
        
        # Parse package name
        package_name = line.split('>=')[0].split('==')[0].split('[')[0].strip()
        
        # Check if this package needs updating
        if package_name in PRIORITY_UPDATES:
            new_line = f"{package_name}{PRIORITY_UPDATES[package_name]}"
            updated_lines.append(new_line)
            updated_packages.append((package_name, line, new_line))
        elif package_name in PINNED_PACKAGES:
            new_line = f"{package_name}{PINNED_PACKAGES[package_name]}"
            updated_lines.append(new_line)
            if line != new_line:
                updated_packages.append((package_name, line, new_line))
        else:
            updated_lines.append(line)
    
    # Backup original file
    backup_file = req_file.with_suffix('.txt.backup')
    with open(backup_file, 'w') as f:
        f.writelines([line + '\n' for line in lines])
    print(f"Backed up original to {backup_file}")
    
    # Write updated requirements
    with open(req_file, 'w') as f:
        for line in updated_lines:
            f.write(line + '\n' if line else '\n')
    
    # Report changes
    if updated_packages:
        print("\nUpdated packages:")
        for name, old, new in updated_packages:
            print(f"  {name}: {old} -> {new}")
    else:
        print("\nNo packages needed updating")
    
    return 0

def verify_updates():
    """Verify that updates can be installed."""
    print("\nVerifying updates can be installed...")
    
    try:
        # Use virtual environment's pip
        venv_pip = Path('.venv/bin/pip')
        if not venv_pip.exists():
            print("Warning: Virtual environment not found, skipping verification")
            return True
        
        # Dry run pip install
        result = subprocess.run(
            [str(venv_pip), 'install', '--dry-run', '-r', 'requirements.txt'],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print("✅ All updates are compatible")
            return True
        else:
            print("❌ Some updates have conflicts:")
            print(result.stderr)
            return False
    except Exception as e:
        print(f"Error verifying updates: {e}")
        return False

def main():
    """Main function."""
    print("CustomKB Dependency Update Tool")
    print("=" * 40)
    
    # Update requirements
    if update_requirements() != 0:
        return 1
    
    # Verify updates
    if not verify_updates():
        print("\nRolling back changes...")
        subprocess.run(['mv', 'requirements.txt.backup', 'requirements.txt'])
        return 1
    
    print("\n" + "=" * 40)
    print("Updates complete! To apply changes, run:")
    print("  source .venv/bin/activate")
    print("  pip install --upgrade -r requirements.txt")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())

#fin