# CustomKB Test Crash Analysis

## Summary of Findings

After investigating the system crashes during testing, here are the key findings:

### 1. Root Cause: Memory Exhaustion

The system logs show processes consuming extreme amounts of memory:
- One process peaked at **22.5GB memory**
- Another peaked at **26.4GB memory** 
- System has 31GB total RAM

When these processes consumed most available memory, the system became unresponsive and required hard reboots.

### 2. Contributing Factors

1. **No pytest-timeout protection** initially (now fixed)
2. **Unlimited system resource limits** (ulimits)
3. **Tests generating large data structures** without cleanup
4. **Possible GPU memory issues** (NVIDIA RTX 4070 detected)
5. **High baseline memory usage** (55% before tests)

### 3. Implemented Fixes

#### A. Memory Management
- Added cache memory limits (500MB default)
- Implemented memory-based cache eviction
- Added memory-mapped FAISS for large indexes
- Created resource manager with memory monitoring

#### B. Test Safety
- Created batch runner with memory limits
- Added pytest-timeout plugin
- Created safer pytest configurations
- Added automatic cleanup between tests

#### C. Database Safety
- Added context managers for connections
- Ensured proper connection cleanup
- Fixed connection leaks in fixtures

## Recommendations

### 1. Immediate Actions

**DO NOT run the full test suite yet!** Instead:

1. **Clear system memory**:
   ```bash
   sync && echo 3 | sudo tee /proc/sys/vm/drop_caches
   ```

2. **Run diagnostic checks**:
   ```bash
   python diagnose_crashes.py
   python test_fixes.py
   ```

3. **Test incrementally**:
   ```bash
   # Single test with monitoring
   python run_safe_test.py
   
   # Small batch with strict limits
   python tests/batch_runner.py --batch unit_core --memory-limit 0.5 --force
   ```

### 2. Identify Problem Tests

Look for tests that:
- Generate large numpy arrays or data structures
- Use multiprocessing or threading
- Load large FAISS indexes
- Create many database connections
- Have parametrized tests that multiply test count

### 3. Configuration Changes

Add to your test knowledge base configs:

```ini
[LIMITS]
memory_cache_size = 1000
cache_memory_limit_mb = 100
max_file_size_mb = 50

[PERFORMANCE]
use_memory_mapped_faiss = true
embedding_batch_size = 50
```

### 4. System Protection

Consider setting system limits before testing:

```bash
# Limit memory usage to 8GB
ulimit -v 8388608

# Limit number of processes
ulimit -u 1000

# Run tests with limits
python tests/batch_runner.py --batch unit_core
```

### 5. Problem Test Identification

The following test files were identified as potentially problematic:
- `tests/performance/test_performance.py` - Contains "large_data" operations
- `tests/unit/test_embed_manager.py` - Contains "large_data" operations

These should be:
1. Run individually with strict monitoring
2. Modified to use smaller test data
3. Marked with `@pytest.mark.resource_intensive`
4. Skipped during regular test runs

## Next Steps

1. **Verify fixes work**:
   ```bash
   # Run each batch separately with monitoring
   python tests/batch_runner.py --batch unit_core --memory-limit 0.5 --force
   python tests/batch_runner.py --batch unit_database --memory-limit 0.5 --force
   ```

2. **Monitor resource usage**:
   - Watch memory usage during tests
   - Check for zombie processes
   - Monitor GPU memory if CUDA tests exist

3. **Fix problematic tests**:
   - Reduce data sizes in performance tests
   - Add proper cleanup in fixtures
   - Use generators instead of lists for large data

4. **Consider test environment**:
   - Run resource-intensive tests in Docker with memory limits
   - Use CI/CD with resource constraints
   - Separate unit tests from integration/performance tests

## Prevention

To prevent future crashes:

1. **Always use batch runner** for multiple tests
2. **Set memory limits** in test configurations
3. **Monitor system resources** during development
4. **Mark resource-intensive tests** appropriately
5. **Run performance tests separately** with special care

The system should now be protected against crashes, but proceed cautiously and incrementally when running tests.