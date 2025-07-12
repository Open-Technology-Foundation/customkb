# Safe Testing Guide for CustomKB

This guide explains how to run tests safely without causing system crashes or memory exhaustion.

## Quick Start

### Option 1: Use the Test Runner Safe Mode (Recommended)

```bash
# Run tests in safe mode with automatic memory limits
python run_tests.py --safe

# Quick shortcut
python run_tests.py safe

# Run specific test types safely
python run_tests.py --safe --unit
python run_tests.py --safe --integration

# Custom memory limit (in MB)
python run_tests.py --safe --memory-limit 1024  # 1GB limit

# Custom timeout (in seconds)
python run_tests.py --safe --timeout 120  # 2 minute timeout
```

### Option 2: Use the Batch Runner

```bash
# List available test batches
python tests/batch_runner.py --list

# Run a specific batch with memory limits
python tests/batch_runner.py --batch unit_core

# Run all batches with stop-on-failure
python tests/batch_runner.py --batch all --stop-on-failure

# Override memory limit for constrained systems
python tests/batch_runner.py --batch unit_core --memory-limit 1.5
```

### Option 3: Use Safe pytest Configuration

```bash
# Run with safe configuration
pytest -c pytest-safe.ini

# Run specific test file safely
pytest -c pytest-safe.ini tests/unit/test_config_manager.py
```

### Option 4: Manual pytest with Resource Limits

```bash
# Run with explicit timeouts and no parallelization
pytest --timeout=60 --timeout-method=thread -v tests/unit/

# Run with memory monitoring (requires pytest-monitor)
pytest --monitor-memory -v tests/unit/
```

## Test Batches

The batch runner divides tests into these categories:

1. **unit_core** (2GB memory limit, 5min timeout)
   - Core configuration and utility tests
   - Low resource usage
   - Safe to run on any system

2. **unit_database** (2GB memory limit, 5min timeout)
   - Database and index management tests
   - Moderate resource usage
   - May create temporary databases

3. **unit_processing** (4GB memory limit, 10min timeout)
   - Embedding and query processing tests
   - Higher memory usage
   - May use API mocks

4. **integration_small** (4GB memory limit, 10min timeout)
   - Small integration tests
   - Moderate resource usage
   - Tests component interactions

5. **integration_large** (8GB memory limit, 20min timeout)
   - Full end-to-end tests
   - High resource usage
   - Tests complete workflows

6. **performance** (8GB memory limit, 30min timeout)
   - Performance benchmarks
   - Very high resource usage
   - Should be run separately

## Configuration for Low-Memory Systems

### 1. Reduce Cache Limits

Edit your knowledge base configuration:

```ini
[LIMITS]
memory_cache_size = 1000  # Reduce from 10000
cache_memory_limit_mb = 100  # Reduce from 500

[PERFORMANCE]
embedding_batch_size = 50  # Reduce from 100
file_processing_batch_size = 100  # Reduce from 500
```

### 2. Enable Memory-Mapped FAISS

For large vector databases:

```ini
[PERFORMANCE]
use_memory_mapped_faiss = true
```

### 3. Use Resource Guards in Code

```python
from utils.resource_manager import ResourceGuard

# Create guard with 2GB limit
guard = ResourceGuard(memory_limit_gb=2.0)

# Run operation with monitoring
with guard.guarded_operation("heavy_processing"):
    process_large_dataset()
```

## Monitoring and Debugging

### Check System Resources

```bash
# Before running tests
free -h  # Check available memory
df -h    # Check disk space
```

### Monitor Test Execution

The batch runner provides real-time monitoring:

```
Running batch: unit_core
Memory limit: 2GB
Timeout: 300s
==================================================
Memory OK: 245.3MB (3.2%)
✅ Batch completed in 45.2s
   Tests: 50 run, 48 passed, 2 failed
   Memory: 312.4MB (+67.1MB from start)
   Peak memory: 325.8MB
```

### Analyze Results

Test results are saved to `test_results.json`:

```json
{
  "timestamp": "2024-01-10 15:30:45",
  "overall_success": true,
  "total_tests": 150,
  "total_passed": 148,
  "total_failed": 2,
  "total_duration": 234.5,
  "max_memory_mb": 825.3,
  "batches": [...]
}
```

## Troubleshooting

### System Still Crashing

1. Reduce batch memory limits:
   ```bash
   python tests/batch_runner.py --batch unit_core --memory-limit 0.5
   ```

2. Run tests individually:
   ```bash
   pytest -c pytest-safe.ini tests/unit/test_config_manager.py::TestKnowledgeBase::test_init
   ```

3. Check for memory leaks:
   ```bash
   python test_fixes.py  # Run diagnostic script
   ```

### Tests Timing Out

1. Increase timeout in pytest-safe.ini:
   ```ini
   --timeout=120  # Increase from 60
   ```

2. Skip slow tests:
   ```bash
   pytest -c pytest-safe.ini -m "not slow"
   ```

### Memory Errors

1. Clear caches before testing:
   ```python
   from utils.resource_manager import cleanup_caches
   cleanup_caches()
   ```

2. Reduce test data size in fixtures
3. Enable garbage collection debugging:
   ```bash
   PYTHONGC=debug pytest -v
   ```

## Best Practices

1. **Always use batch runner for full test suite**
2. **Monitor system resources during development**
3. **Configure appropriate limits for your system**
4. **Run performance tests separately**
5. **Use memory-mapped FAISS for large indexes**
6. **Clean up test data between runs**

## Emergency Recovery

If system becomes unresponsive:

1. Kill pytest processes:
   ```bash
   pkill -9 pytest
   ```

2. Clear temporary files:
   ```bash
   rm -rf /tmp/test_*
   rm -rf ~/.pytest_cache
   ```

3. Reset system caches:
   ```bash
   sync && echo 3 | sudo tee /proc/sys/vm/drop_caches
   ```

## Environment Variables

Control test behavior with these variables:

```bash
# Limit test memory usage
export TEST_MEMORY_LIMIT_GB=1.5

# Disable parallel execution
export PYTEST_XDIST_WORKER_COUNT=0

# Enable debug logging
export CUSTOMKB_TEST_DEBUG=1
```

## Continuous Integration

For CI/CD pipelines:

```yaml
# Example GitHub Actions configuration
- name: Run Safe Tests
  run: |
    python tests/batch_runner.py --batch unit_core --memory-limit 2
    python tests/batch_runner.py --batch unit_database --memory-limit 2
  env:
    TEST_MEMORY_LIMIT_GB: 2
```

Remember: **Safety first!** It's better to run tests slowly than to crash your system.