#!/usr/bin/env python
"""
Diagnostic script to identify potential crash causes WITHOUT running tests.
This script is designed to be as safe as possible.
"""

import os
import sys
import subprocess
import psutil
import json
from pathlib import Path

def check_system_state():
    """Check current system state."""
    print("=== System State ===")
    
    # Memory
    mem = psutil.virtual_memory()
    swap = psutil.swap_memory()
    print(f"Memory: {mem.used/1024/1024/1024:.1f}GB/{mem.total/1024/1024/1024:.1f}GB ({mem.percent}%)")
    print(f"Available: {mem.available/1024/1024/1024:.1f}GB")
    print(f"Swap: {swap.used/1024/1024/1024:.1f}GB/{swap.total/1024/1024/1024:.1f}GB ({swap.percent}%)")
    
    # CPU
    print(f"CPU Count: {psutil.cpu_count()} cores")
    print(f"CPU Usage: {psutil.cpu_percent(interval=1)}%")
    
    # Disk
    disk = psutil.disk_usage('/')
    print(f"Disk: {disk.used/1024/1024/1024:.1f}GB/{disk.total/1024/1024/1024:.1f}GB ({disk.percent}%)")
    
    # Process count
    print(f"Process count: {len(psutil.pids())}")
    
    # Check for zombie processes
    zombies = []
    for proc in psutil.process_iter(['pid', 'name', 'status']):
        try:
            if proc.info['status'] == psutil.STATUS_ZOMBIE:
                zombies.append(proc.info)
        except:
            pass
    if zombies:
        print(f"WARNING: {len(zombies)} zombie processes found!")
    
    print()


def check_gpu_state():
    """Check GPU state if nvidia-smi is available."""
    print("=== GPU State ===")
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,memory.used,memory.free', 
                               '--format=csv,noheader'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print(result.stdout)
        else:
            print("No NVIDIA GPU detected or nvidia-smi not available")
    except Exception as e:
        print(f"Could not check GPU: {e}")
    print()


def check_test_files():
    """Identify potentially problematic test files."""
    print("=== Potentially Problematic Tests ===")
    
    problem_patterns = [
        ('multiprocessing', 'May spawn too many processes'),
        ('ThreadPoolExecutor', 'May create too many threads'),
        ('spawn', 'Process spawning'),
        ('fork', 'Process forking'),
        ('cuda', 'GPU operations'),
        ('torch.cuda', 'PyTorch GPU operations'),
        ('large_data', 'Large data generation'),
        ('np.random.rand(1000000', 'Large array creation'),
        ('range(1000000', 'Large loops'),
        ('@pytest.mark.parametrize', 'Parametrized tests can multiply'),
    ]
    
    test_dir = Path('tests')
    if not test_dir.exists():
        print("No tests directory found")
        return
    
    problematic = []
    
    for test_file in test_dir.rglob('test_*.py'):
        try:
            content = test_file.read_text()
            issues = []
            
            for pattern, reason in problem_patterns:
                if pattern in content:
                    issues.append((pattern, reason))
            
            if issues:
                problematic.append({
                    'file': str(test_file),
                    'issues': issues
                })
        except Exception as e:
            print(f"Error reading {test_file}: {e}")
    
    if problematic:
        print(f"Found {len(problematic)} files with potential issues:")
        for item in problematic[:10]:  # Show first 10
            print(f"\n{item['file']}:")
            for pattern, reason in item['issues']:
                print(f"  - {pattern}: {reason}")
    else:
        print("No obviously problematic patterns found")
    
    print()


def check_pytest_plugins():
    """Check installed pytest plugins."""
    print("=== Pytest Plugins ===")
    try:
        result = subprocess.run([sys.executable, '-m', 'pytest', '--version', '--verbose'], 
                              capture_output=True, text=True, timeout=10)
        print(result.stdout)
        
        # Check for problematic plugins
        if 'xdist' in result.stdout:
            print("WARNING: pytest-xdist found - can spawn many processes!")
        if 'timeout' not in result.stdout:
            print("WARNING: pytest-timeout not found - tests may run forever!")
        
    except Exception as e:
        print(f"Could not check pytest: {e}")
    print()


def check_system_limits():
    """Check system resource limits."""
    print("=== System Limits ===")
    try:
        # Check ulimits
        limits_to_check = [
            ('Max processes', '-u'),
            ('Max open files', '-n'),
            ('Max memory size', '-m'),
            ('Virtual memory', '-v'),
        ]
        
        for name, flag in limits_to_check:
            try:
                result = subprocess.run(['ulimit', flag], 
                                      capture_output=True, text=True, shell=True)
                print(f"{name}: {result.stdout.strip()}")
            except:
                pass
                
    except Exception as e:
        print(f"Could not check limits: {e}")
    print()


def suggest_safe_commands():
    """Suggest safe testing commands."""
    print("=== Safe Testing Recommendations ===")
    print("Based on your system, here are the safest ways to run tests:")
    print()
    
    mem_gb = psutil.virtual_memory().total / 1024 / 1024 / 1024
    
    print("1. Single test with strict limits:")
    print("   pytest --timeout=30 -v tests/unit/test_config_manager.py::TestKnowledgeBase::test_init")
    print()
    
    print("2. Small batch with monitoring:")
    if mem_gb < 8:
        print("   python tests/batch_runner.py --batch unit_core --memory-limit 0.5")
    else:
        print("   python tests/batch_runner.py --batch unit_core --memory-limit 1.0")
    print()
    
    print("3. Diagnostic test only:")
    print("   python test_fixes.py")
    print()
    
    print("4. List tests without running:")
    print("   pytest --collect-only tests/unit/")
    print()
    
    print("5. Dry run to check for issues:")
    print("   pytest --collect-only --quiet tests/ 2>&1 | grep -E 'error|Error|WARNING'")
    print()


def check_for_crash_logs():
    """Check for system crash logs."""
    print("=== Recent Crash Indicators ===")
    
    log_locations = [
        '/var/log/syslog',
        '/var/log/kern.log', 
        '/var/log/messages',
        'dmesg',
    ]
    
    keywords = ['panic', 'OOM', 'Out of memory', 'killed process', 'GPU has fallen']
    
    for log in log_locations:
        if log == 'dmesg':
            try:
                result = subprocess.run(['dmesg', '-T'], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    lines = result.stdout.split('\n')[-100:]  # Last 100 lines
                    for line in lines:
                        for keyword in keywords:
                            if keyword.lower() in line.lower():
                                print(f"Found in dmesg: {line[:120]}...")
                                break
            except:
                pass
        elif os.path.exists(log):
            try:
                result = subprocess.run(['tail', '-n', '100', log], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    for line in result.stdout.split('\n'):
                        for keyword in keywords:
                            if keyword.lower() in line.lower():
                                print(f"Found in {log}: {line[:120]}...")
                                break
            except:
                pass
    print()


def main():
    """Run all diagnostics."""
    print("=" * 70)
    print("CustomKB Crash Diagnostics")
    print("=" * 70)
    print("This script will analyze your system WITHOUT running any tests.")
    print()
    
    try:
        check_system_state()
        check_gpu_state()
        check_system_limits()
        check_pytest_plugins()
        check_test_files()
        check_for_crash_logs()
        suggest_safe_commands()
        
    except KeyboardInterrupt:
        print("\nDiagnostics interrupted by user.")
    except Exception as e:
        print(f"\nError during diagnostics: {e}")
        import traceback
        traceback.print_exc()
    
    print("=" * 70)
    print("Diagnostics complete. Please share this output for analysis.")
    print("=" * 70)


if __name__ == '__main__':
    main()

#fin