#!/usr/bin/env python
"""
Demonstration of per-knowledge-base logging functionality.
"""

import os
from utils.logging_utils import get_kb_info_from_config, get_log_file_path

def demo_per_kb_logging():
    """Demonstrate per-KB logging path generation."""
    print("🔧 CustomKB Per-Knowledge-Base Logging Demo")
    print("=" * 50)
    
    # Test cases
    test_cases = [
        "/path/to/company.cfg",
        "/projects/research.cfg", 
        "local_kb.cfg",
        "config/example_logging.cfg"
    ]
    
    print("\n📁 KB Directory and Name Extraction:")
    for config_file in test_cases:
        try:
            kb_dir, kb_name = get_kb_info_from_config(config_file)
            log_file = get_log_file_path('auto', kb_dir, kb_name)
            print(f"   Config: {config_file}")
            print(f"   KB Dir: {kb_dir}")
            print(f"   KB Name: {kb_name}")
            print(f"   Log File: {log_file}")
            print()
        except Exception as e:
            print(f"   Config: {config_file} - Error: {e}")
            print()
    
    print("📋 Configuration Options:")
    print("   log_file = auto                    # {kb_dir}/logs/{kb_name}.log")
    print("   log_file = custom/path.log         # {kb_dir}/custom/path.log")
    print("   log_file = /absolute/path.log      # /absolute/path.log")
    
    print("\n🎯 Expected Behavior:")
    print("   customkb help                      # No logging at all")
    print("   customkb version                   # No logging at all") 
    print("   customkb database company.cfg ...  # company.kb/logs/company.log")
    print("   customkb query research.cfg ...    # research.kb/logs/research.log")
    
    print("\n✅ Per-KB logging implementation complete!")

if __name__ == "__main__":
    demo_per_kb_logging()