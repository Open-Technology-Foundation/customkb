# CustomKB Testing Safety Guide

## ⚠️ CRITICAL: System Hang Issue

### Problem Summary

**Date**: 2025-11-07
**Severity**: CRITICAL
**Impact**: System hangs requiring hard reboot

### Root Cause Analysis

**Log Evidence**:
```
Nov 07 09:12:34 okusi kernel: INFO: task txg_sync:1062 blocked for more than 122 seconds.
Nov 07 09:12:34 okusi kernel: INFO: task gnome-shell:5913 blocked for more than 122 seconds.
Nov 07 09:12:16 okusi systemd[1]: systemd-oomd.service: Watchdog timeout (limit 3min)!
Nov 07 09:12:40 okusi systemd[1]: systemd-oomd.service: Main process exited, code=dumped, status=6/ABRT
```

**Analysis**:
1. **Symptom**: Multiple system tasks blocked for 120+ seconds
2. **Trigger**: systemd-oomd (OOM daemon) watchdog timeout
3. **Cascading Failure**: ZFS tasks (txg_sync, zvol) hung, gnome-shell blocked
4. **Result**: System unresponsive, requiring hard reboot

**Contributing Factors**:
- Running full test suite without memory limits
- Tests using `pytest` directly instead of `./run_tests.py --safe`
- ZFS ARC configured to use up to 16.7GB (51% of 32GB RAM)
- Concurrent test execution without resource limits
- FAISS index operations consuming significant memory

### System Configuration

**Hardware**:
- **RAM**: 32GB total
- **Swap**: 8GB
- **Storage**: ZFS with ARC up to 16.7GB

**ZFS ARC Settings**:
```
c_max:  16,687,497,216 bytes (~16.7GB)
c:       1,402,630,144 bytes (~1.4GB current)
size:    1,323,119,696 bytes (~1.3GB actual)
```

## ✅ Safe Testing Practices

### 1. ALWAYS Use the Safe Mode Test Runner

**DO THIS**:
```bash
# Run with memory limits and timeouts
./run_tests.py --safe

# Or with custom limits
./run_tests.py --safe --memory-limit 2048 --timeout 60
```

**NOT THIS**:
```bash
# DANGEROUS: No resource limits!
pytest tests/

# DANGEROUS: Can consume unlimited memory!
timeout 300 pytest tests/
```

### 2. Memory Limits

The safe test runner sets:
- **Default memory limit**: 2048MB (2GB)
- **Default timeout**: 120 seconds per test
- **Max parallel workers**: 2 (in safe mode)

**Override if needed**:
```bash
# For larger tests, increase carefully
./run_tests.py --safe --memory-limit 4096  # 4GB max
```

### 3. Monitor Memory During Tests

**Before running tests**:
```bash
# Check available memory
free -h

# Check ZFS ARC usage
cat /proc/spl/kstat/zfs/arcstats | grep -E "^c |^size|^c_max"
```

**During tests** (another terminal):
```bash
# Watch memory usage
watch -n 1 'free -h && echo "---" && ps aux --sort=-%mem | head -20'
```

### 4. Test Subset Execution

Instead of running all 589 tests at once:

**Unit Tests Only** (safer):
```bash
./run_tests.py --unit --safe
```

**Specific Test Files**:
```bash
pytest tests/unit/test_config_manager.py -v --timeout=30
```

**Single Tests** (debugging):
```bash
pytest tests/unit/test_embed_manager.py::TestCacheThreadManager::test_configure_cache_manager_function -v
```

### 5. Integration Test Warnings

**Integration tests** are more resource-intensive because they:
- Create real SQLite databases
- Generate FAISS indexes
- Mock API clients with large response objects
- Process actual text documents

**Safe execution**:
```bash
# Run integration tests one at a time
pytest tests/integration/ -v --timeout=60 -x  # -x stops on first failure
```

## 🛡️ Resource Protection Mechanisms

### Built-in Safeguards

The `run_tests.py` script provides:

1. **Memory Limits** (Unix systems):
   ```python
   resource.setrlimit(resource.RLIMIT_AS, (memory_limit_bytes, memory_limit_bytes))
   ```

2. **Test Timeouts**:
   ```python
   --timeout 120 --timeout-method=thread
   ```

3. **Limited Parallelization**:
   ```python
   -n 2  # Max 2 parallel workers in safe mode
   ```

4. **Memory Monitoring** (with psutil):
   - Reports initial memory usage
   - Shows memory delta after tests

### System-Level Protection

**For ZFS users**, consider limiting ARC:

```bash
# Limit ZFS ARC to 8GB (example)
echo 8589934592 | sudo tee /sys/module/zfs/parameters/zfs_arc_max
```

Add to `/etc/modprobe.d/zfs.conf`:
```
options zfs zfs_arc_max=8589934592
```

## 📊 Test Execution Guidelines

### Small Test Runs

**Safe for any system**:
```bash
# Single test file
pytest tests/unit/test_config_manager.py -v

# Single test class
pytest tests/unit/test_embed_manager.py::TestCacheThreadManager -v

# Specific tests matching pattern
pytest -k "test_cache" -v
```

### Medium Test Runs

**Requires 4GB+ available memory**:
```bash
# All unit tests
./run_tests.py --unit --safe

# Specific integration test file
pytest tests/integration/test_end_to_end.py -v --timeout=60
```

### Large Test Runs

**Requires 8GB+ available memory, monitoring recommended**:
```bash
# Full test suite with safeguards
./run_tests.py --safe --memory-limit 4096 --timeout 180

# Or use timeout wrapper
timeout 300 ./run_tests.py --safe
```

## ⚠️ Warning Signs

**Stop tests immediately if you see**:

1. **System slowdown**: UI becomes sluggish
2. **Swap usage**: `free -h` shows swap being used
3. **High memory**: Process using >8GB
4. **Disk thrashing**: Constant disk I/O
5. **Long hangs**: Test takes >5 minutes

**Emergency stop**:
```bash
# Find pytest processes
ps aux | grep pytest

# Kill gracefully
pkill -TERM pytest

# Force kill if needed (last resort)
pkill -9 python
```

## 🔍 Troubleshooting

### Test Hangs

If a test hangs:

1. **Check process memory**:
   ```bash
   ps aux --sort=-%mem | head -20
   ```

2. **Check for blocked I/O**:
   ```bash
   sudo iotop -o
   ```

3. **Examine test logs**:
   ```bash
   tail -f /var/lib/vectordbs/*/logs/*.log
   ```

### System Recovery

If system becomes unresponsive:

1. **Try SysRq keys** (if enabled):
   - `Alt + SysRq + R` (take back keyboard)
   - `Alt + SysRq + E` (terminate processes)
   - `Alt + SysRq + I` (kill processes)
   - `Alt + SysRq + S` (sync disks)
   - `Alt + SysRq + U` (unmount)
   - `Alt + SysRq + B` (reboot)

2. **SSH from another machine**:
   ```bash
   ssh user@machine
   sudo pkill -9 python
   ```

3. **Hard reboot** (last resort)

## 📝 Reporting Issues

When reporting test-related hangs, include:

1. **System logs**:
   ```bash
   sudo journalctl --since "1 hour ago" --priority=0..3 > system-error.log
   sudo dmesg -T | tail -200 > kernel.log
   ```

2. **Memory state**:
   ```bash
   free -h > memory-state.txt
   cat /proc/spl/kstat/zfs/arcstats | grep -E "^c |^size" >> memory-state.txt
   ```

3. **Test command used**:
   - Exact command line
   - Any custom flags
   - Time when hang occurred

4. **Test output**:
   - Last test that ran
   - Any error messages
   - Timeout or manual stop?

## 🎯 Best Practices Summary

### ✅ DO

- Use `./run_tests.py --safe` for test runs
- Monitor memory during large test suites
- Run tests in batches (unit, then integration, then performance)
- Set reasonable timeouts
- Check available memory before starting

### ❌ DON'T

- Run `pytest tests/` without safeguards
- Run tests with unlimited parallelization (`-n auto`)
- Run performance tests during development
- Ignore memory warnings or slowdowns
- Run full test suite on systems with <16GB RAM without limits

---

**Last Updated**: 2025-11-07
**System**: Ubuntu 24.04, 32GB RAM, ZFS filesystem
**Test Suite**: 589 tests (442 passing, 75% pass rate)

#fin

---

## Known Issues

### safe-test.sh Collection Hang (2025-11-07)

**Symptom**: safe-test.sh hangs during test collection phase with ulimit memory restrictions.

**Workaround**: Use `./run_tests.py --safe` instead, which uses Python's resource.setrlimit() instead of bash ulimit.

**Example**:
```bash
# May hang during collection:
./safe-test.sh tests/unit/test_config_manager.py -v

# Works correctly:
./run_tests.py --safe --unit
source .venv/bin/activate && timeout 60 pytest tests/unit/test_config_manager.py -v
```

**Root cause**: Under investigation - appears related to ulimit -v interaction with test collection.

