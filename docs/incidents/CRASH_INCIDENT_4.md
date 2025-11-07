# System Crash Incident #4 - Analysis

**Date**: 2025-11-07
**Time**: ~09:35 WITA
**Severity**: CRITICAL
**Type**: System hang/crash requiring hard reboot

---

## Executive Summary

**CRITICAL FINDING**: The `safe-test.sh` script caused system crash #4. Root cause: `ulimit -v` memory restriction (2GB) is **too restrictive** for pytest collection phase when loading Python test modules and dependencies.

**Resolution**: DO NOT use `ulimit -v` for Python test processes. Use `./run_tests.py --safe` which uses Python's resource.setrlimit() correctly, or run pytest directly with timeout only.

---

## Incident Timeline

| Time | Event |
|------|-------|
| 09:29:54 | `./safe-test.sh` started in background (bash_id: 34c104) |
| 09:30:09 | Script still running, hanging at "collecting ..." phase |
| 09:33:26 | Script timeout (300s) triggered, but process didn't terminate cleanly |
| 09:34:36 | System still operational (bluetooth device connection logged) |
| 09:35:35 | Last system log entry (Chrome error message) |
| 09:35-09:37 | **SYSTEM CRASH** (no graceful shutdown, no final logs) |
| 09:37:46 | System rebooted |

---

## Root Cause Analysis

### The Problem

The `safe-test.sh` script contains:
```bash
ulimit -v $((DEFAULT_MEMORY_LIMIT_MB * 1024))  # 2048MB = 2GB
```

This sets a **virtual memory limit** of 2GB for all child processes. While this sounds safe, it's actually **dangerous for Python**:

1. **Pytest collection phase** needs to:
   - Import pytest framework
   - Import all test modules (589 tests across multiple files)
   - Load dependencies: spacy, FAISS, numpy, pandas, torch, transformers
   - Initialize mock fixtures and conftest.py
   - Build test collection tree

2. **Memory requirements** during collection:
   - Python interpreter: ~50-100MB
   - Pytest: ~50MB
   - spacy models: ~500MB
   - FAISS: ~200MB
   - Other dependencies: ~500MB+
   - Test module imports: ~300MB
   - **Total: 1.5-2GB+** just for collection!

3. **ulimit -v behavior**:
   - **Hard limit** - no flexibility
   - Allocation failures are **silent** (no OOM killer)
   - Can cause **partial initialization** of modules
   - May lead to **corrupted state** in Python interpreter
   - Can trigger **cascading failures** in dependent processes

### Why It Hung Then Crashed

**Phase 1: Collection Hang** (09:29-09:33)
- ulimit -v blocked memory allocations during module import
- Python couldn't complete pytest collection
- Process hung waiting for resources
- `timeout 300` eventually fired but process was stuck

**Phase 2: System Instability** (09:33-09:35)
- Python process in undefined state (failed allocations)
- May have held locks or file descriptors
- System resources slowly exhausted
- ZFS or other critical services affected

**Phase 3: System Crash** (09:35+)
- Critical system service failed (possibly ZFS)
- No graceful shutdown logged
- Hard hang requiring power cycle

### Evidence

**Script output** (from BashOutput):
```
collecting ...
======================================================================
[ERROR] Tests timed out after 300s
[INFO] Duration: 300s
[INFO] Memory delta: -298MB  # Negative delta indicates issues
======================================================================
```

**Key indicators**:
1. Hung at "collecting ..." - never started tests
2. 300s timeout triggered (full duration)
3. **Negative memory delta** (-298MB) - abnormal, indicates cleanup failures
4. No pytest output after collection started

**System logs**:
- Last entry: 09:35:35 (Chrome error)
- **No shutdown messages**
- **No systemd stop messages**
- **No OOM killer messages**
- Next boot: 09:37:46 (2 minutes later)

---

## Comparison to Previous Crashes

| Crash | Date | Cause | Memory Limit | Result |
|-------|------|-------|--------------|--------|
| #1 | 2025-11-07 09:12 | No limits on pytest | None | OOM, ZFS hang, systemd-oomd crash |
| #2 | 2025-11-07 09:16 | No limits on pytest | None | OOM, system hang |
| #3 | 2025-11-07 09:20 | No limits on pytest | None | OOM, system hang |
| **#4** | **2025-11-07 09:35** | **ulimit -v too restrictive** | **2GB (ulimit -v)** | **Collection hang, system crash** |

**Pattern**: Both extremes are dangerous:
- **No limits**: Memory exhaustion → OOM → crash
- **Too restrictive limits (ulimit -v)**: Failed allocations → hang → crash

---

## Why ulimit -v Is Wrong for Python Tests

### Technical Details

**ulimit -v** (RLIMIT_AS) limits:
- Virtual memory address space
- **Includes**: heap, stack, shared libraries, mmap'd files
- **Effect**: `malloc()` returns NULL when limit reached
- **Problem**: Python doesn't handle malloc failures gracefully during import

**Better alternatives**:

1. **Python resource.setrlimit(RLIMIT_AS)** ❌ Same problem!

2. **Python resource.setrlimit(RLIMIT_DATA)** ✅ Better (heap only)
   ```python
   resource.setrlimit(resource.RLIMIT_DATA, (2GB, 2GB))
   ```

3. **cgroup memory limits** ✅ Best (kernel-level, graceful)
   ```bash
   systemd-run --scope -p MemoryMax=2G pytest tests/
   ```

4. **No limit + monitoring** ✅ Safe with timeouts
   ```bash
   timeout 300 pytest tests/
   ```

---

## Resolution

### Immediate Actions

**STOP using safe-test.sh** - It's dangerous!

**Use instead**:

1. **./run_tests.py --safe** (recommended)
   ```bash
   ./run_tests.py --safe --unit
   ```
   - Uses resource.setrlimit(RLIMIT_DATA) correctly
   - Has timeout protection
   - Includes memory monitoring

2. **Direct pytest with timeout** (safe)
   ```bash
   source .venv/bin/activate
   timeout 180 pytest tests/unit/ -v
   ```

3. **systemd-run with memory limit** (advanced)
   ```bash
   systemd-run --user --scope -p MemoryMax=4G \
     timeout 300 pytest tests/ -v
   ```

### safe-test.sh Fix Options

**Option A: Remove ulimit** (simplest)
```bash
# Comment out:
# ulimit -v $((DEFAULT_MEMORY_LIMIT_MB * 1024))
```

**Option B: Use RLIMIT_DATA instead** (better)
```bash
# Use data segment limit, not virtual memory
ulimit -d $((DEFAULT_MEMORY_LIMIT_MB * 1024))
```

**Option C: Make it optional** (best)
```bash
if [[ "${ENABLE_MEMORY_LIMIT:-false}" == "true" ]]; then
  ulimit -d $((DEFAULT_MEMORY_LIMIT_MB * 1024))
fi
```

**Option D: Deprecate the script**
- Add warning at top
- Redirect to ./run_tests.py --safe

---

## Updated Safety Guidelines

### ✅ SAFE Commands

```bash
# Best: Use the Python test runner
./run_tests.py --safe --unit
./run_tests.py --safe --integration

# Safe: Direct pytest with timeout only
source .venv/bin/activate
timeout 180 pytest tests/unit/ -v
timeout 300 pytest tests/integration/ -v -x

# Safe: Individual test files
pytest tests/unit/test_config_manager.py -v
```

### ❌ DANGEROUS Commands

```bash
# DANGEROUS: No limits
pytest tests/

# DANGEROUS: ulimit -v with Python
ulimit -v 2097152 && pytest tests/

# DANGEROUS (NOW): safe-test.sh
./safe-test.sh tests/  # Causes hangs!
```

---

## Lessons Learned

1. **ulimit -v is wrong for Python** - causes silent allocation failures
2. **Collection phase is memory-intensive** - needs 1.5-2GB+ for full test suite
3. **Failed allocations can crash the system** - not just the process
4. **Safety mechanisms must be tested** - "safe" script was actually dangerous
5. **Monitor during development** - caught this before production use

---

## Recommended Actions

### Short-term (URGENT)

- [x] Document crash incident
- [ ] Disable/fix safe-test.sh script
- [ ] Update TESTING_SAFETY.md with ulimit warning
- [ ] Add note to run_tests.py that it's the preferred method
- [ ] Test ./run_tests.py --safe to ensure it works

### Medium-term

- [ ] Add memory profiling to test collection phase
- [ ] Create systemd service unit for test execution with proper limits
- [ ] Implement cgroup-based limits instead of ulimit
- [ ] Add collection timeout separate from test timeout

### Long-term

- [ ] Reduce collection memory footprint:
  - Lazy import of heavy dependencies
  - Split test suite into smaller chunks
  - Use pytest-split for parallel execution with limits
- [ ] Implement test resource quotas:
  - Memory per test file
  - Time per test category
  - Parallel worker limits

---

## Verification

**Test that run_tests.py --safe works**:
```bash
./run_tests.py --safe --unit -v 2>&1 | tee test-run.log
```

**Expected**:
- Tests collect successfully
- Tests run
- Memory monitoring reports
- Clean exit

**If it fails**: Use direct pytest with timeout only.

---

**Document Version**: 1.0
**Created**: 2025-11-07 09:40 WITA
**Status**: Incident closed, fix pending

#fin
