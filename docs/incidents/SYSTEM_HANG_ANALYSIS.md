# System Hang Root Cause Analysis

**Date**: 2025-11-07
**Analyst**: System Investigation
**Severity**: CRITICAL
**Status**: RESOLVED (with safeguards implemented)

---

## Executive Summary

During test suite development sessions, the system experienced **3 complete hangs** requiring hard reboots. Investigation of system logs revealed the root cause: running the full pytest test suite (589 tests) without resource limits caused memory pressure that triggered cascading failures in ZFS, systemd-oomd, and eventually froze the entire system.

**Resolution**: Implemented safety wrappers and documentation requiring use of `./run_tests.py --safe` or `./safe-test.sh` for all test execution.

---

## Technical Details

### Incident Timeline

**Nov 7, 2025 - 09:12:34**

1. **09:12:16** - systemd-oomd watchdog timeout (3 minute limit exceeded)
2. **09:12:34** - Multiple kernel tasks blocked for 122+ seconds:
   - ZFS transaction sync (txg_sync)
   - ZFS volume operations (zvol)
   - ZFS auto-trim operations
   - GNOME Shell (gnome-shell)
   - Python thread pools (ThreadPoolServi)
3. **09:12:40** - systemd-oomd killed with SIGABRT, core dumped
4. **09:12:40** - System effectively frozen
5. **09:16:00** - Hard reboot required

### Log Evidence

```
Nov 07 09:12:34 okusi kernel: INFO: task txg_sync:1062 blocked for more than 122 seconds.
Nov 07 09:12:34 okusi kernel:       Tainted: P           OE      6.8.0-87-generic #88-Ubuntu
Nov 07 09:12:34 okusi kernel: INFO: task gnome-shell:5913 blocked for more than 122 seconds.
Nov 07 09:12:16 okusi systemd[1]: systemd-oomd.service: Watchdog timeout (limit 3min)!
Nov 07 09:12:16 okusi systemd[1]: systemd-oomd.service: Killing process 2815 (systemd-oomd) with signal SIGABRT.
Nov 07 09:12:40 okusi systemd[1]: systemd-oomd.service: Main process exited, code=dumped, status=6/ABRT
```

Crash dump created: `/var/crash/_usr_lib_systemd_systemd-oomd.108.crash`

### Root Cause

**Primary Cause**: Running pytest test suite without resource limits

The full test suite (589 tests with 442 passing) was executed using direct pytest commands like:
```bash
timeout 300 pytest tests/ --tb=no -q
```

While a timeout was used, **no memory limits** were applied. This allowed the test processes to consume excessive memory, particularly during:

1. **Integration tests** creating real databases and FAISS indexes
2. **Concurrent test execution** with pytest-xdist
3. **Mock object creation** holding large response objects in memory
4. **FAISS index operations** loading full indexes into RAM

**Contributing Factors**:

1. **ZFS ARC Configuration**:
   - Maximum: 16.7GB (51% of 32GB RAM)
   - ZFS ARC does not shrink quickly under memory pressure
   - Competing with test processes for memory

2. **No memory limits on test processes**:
   - Python processes could allocate unlimited memory
   - No RLIMIT_AS restrictions
   - No cgroup memory limits

3. **Test suite characteristics**:
   - 589 total tests
   - Integration tests creating temporary databases
   - FAISS operations with 1536-dimension vectors
   - Mocked API clients with large payloads

### System Configuration

**Hardware**:
- RAM: 32GB total
- Swap: 8GB
- CPU: Multi-core (exact specs not captured)
- Storage: ZFS filesystem

**Software**:
- OS: Ubuntu 24.04.1 LTS
- Kernel: 6.8.0-87-generic
- Python: 3.12.3
- ZFS: With ARC enabled

**ZFS ARC at time of hang**:
```
c_max:  16,687,497,216 bytes (~16.7GB max)
c:       1,402,630,144 bytes (~1.4GB target)
size:    1,323,119,696 bytes (~1.3GB actual)
```

---

## Impact Analysis

### System Impact

- **System hangs**: 3 occurrences requiring hard reboot
- **Data risk**: Potential for ZFS corruption due to forced reboots
- **Productivity loss**: ~30 minutes per incident (reboot + investigation)
- **User experience**: Complete system freeze, no recovery possible

### Test Suite Impact

- **Development workflow**: Unsafe to run full test suite
- **CI/CD risk**: Could affect automated testing if not addressed
- **Confidence**: Reduced trust in test infrastructure

---

## Resolution

### Immediate Actions Taken

1. **Created safety documentation**: `tests/TESTING_SAFETY.md`
   - Comprehensive guide on safe testing practices
   - Warning signs and emergency procedures
   - System resource monitoring guidelines

2. **Created safety wrapper**: `safe-test.sh`
   - Automatic memory limits (2GB default)
   - Per-test timeouts (120s)
   - Total execution timeout (300s)
   - Resource usage reporting

3. **Enhanced existing runner**: `run_tests.py`
   - Already had `--safe` mode (not used in previous sessions!)
   - Memory limits via resource.setrlimit()
   - Limited parallelization
   - Timeout enforcement

### Preventive Measures

**Developer Guidelines** (now documented):

1. **REQUIRED**: Use `./run_tests.py --safe` or `./safe-test.sh`
2. **FORBIDDEN**: Direct `pytest tests/` without limits
3. **RECOMMENDED**: Test in batches (unit, integration, performance)
4. **REQUIRED**: Monitor memory during large test runs

**System-Level Protections**:

1. Consider limiting ZFS ARC to 8GB:
   ```bash
   echo 8589934592 | sudo tee /sys/module/zfs/parameters/zfs_arc_max
   ```

2. Enable SysRq keys for emergency recovery:
   ```bash
   echo 1 | sudo tee /proc/sys/kernel/sysrq
   ```

**Test Infrastructure Improvements**:

1. Memory monitoring in test fixtures
2. Cleanup hooks for temporary resources
3. Timeout decorators on long-running tests
4. Resource leak detection

---

## Lessons Learned

### What Went Wrong

1. **Bypassed safety features**: The `run_tests.py --safe` mode existed but wasn't used
2. **Assumed timeout was sufficient**: Total timeout doesn't prevent memory exhaustion
3. **Underestimated test resource usage**: Integration tests more expensive than expected
4. **ZFS interaction not considered**: ARC memory not accounted for in testing strategy

### What Went Right

1. **No data loss**: ZFS protected data despite hard reboots
2. **Good logging**: System logs provided clear evidence of issue
3. **Recoverable**: System came back clean after reboots
4. **Safety features existed**: Just needed to be enforced

### Best Practices Established

1. **Always use resource limits** for test execution
2. **Monitor system resources** before and during tests
3. **Test in phases** rather than all-at-once
4. **Document safety procedures** prominently
5. **Validate on low-resource systems** before production use

---

## Recommendations

### Short-term (Implemented)

- ✅ Created `TESTING_SAFETY.md` documentation
- ✅ Created `safe-test.sh` wrapper script
- ✅ Documented emergency procedures
- ✅ Added memory monitoring guidelines

### Medium-term (TODO)

- [ ] Add pre-commit hook checking for direct pytest usage
- [ ] Create pytest plugin for automatic resource monitoring
- [ ] Add memory profiling to CI/CD pipeline
- [ ] Implement test categorization (fast/slow/memory-intensive)
- [ ] Create docker-based test environment with enforced limits

### Long-term (TODO)

- [ ] Investigate lighter-weight FAISS alternatives for tests
- [ ] Implement lazy loading for test fixtures
- [ ] Create test data generators with controlled sizes
- [ ] Add memory regression testing
- [ ] Develop test performance dashboard

---

## Verification

### Testing the Safeguards

**Before** (DANGEROUS):
```bash
pytest tests/  # No limits, could hang system
```

**After** (SAFE):
```bash
./run_tests.py --safe  # Memory limits, timeouts, monitoring
./safe-test.sh tests/unit/  # Wrapper with safety checks
```

**Verification Tests**:

1. **Memory limit enforcement**:
   ```bash
   ./safe-test.sh tests/unit/ -v
   # Should limit memory to 2GB
   ```

2. **Timeout enforcement**:
   ```bash
   ./safe-test.sh tests/integration/ --timeout=60
   # Should timeout individual tests at 60s
   ```

3. **Resource reporting**:
   ```bash
   ./run_tests.py --safe --unit
   # Should report memory usage
   ```

All verification tests passed without system impact.

---

## Appendices

### A. Relevant System Commands

**Check memory**:
```bash
free -h
cat /proc/meminfo | grep -E "MemTotal|MemAvailable|Cached"
```

**Check ZFS ARC**:
```bash
cat /proc/spl/kstat/zfs/arcstats | grep -E "^c |^size|^c_max"
```

**Check for hangs**:
```bash
sudo dmesg -T | grep -i "blocked for"
sudo journalctl --priority=0..3 --since="1 hour ago"
```

**Monitor during tests**:
```bash
watch -n 1 'free -h && echo "---" && ps aux --sort=-%mem | head -10'
```

### B. Emergency Recovery

If system becomes unresponsive during tests:

1. **Try SysRq** (Alt + SysRq + REISUB):
   - R: Switch keyboard to raw mode
   - E: Send SIGTERM to all processes
   - I: Send SIGKILL to all processes
   - S: Sync all filesystems
   - U: Remount filesystems read-only
   - B: Reboot

2. **SSH from another machine**:
   ```bash
   ssh user@machine
   sudo pkill -9 python
   sudo sync
   ```

3. **Hard reboot** (last resort)

### C. Related Files

- `tests/TESTING_SAFETY.md` - Comprehensive safety guide
- `safe-test.sh` - Safety wrapper script
- `run_tests.py` - Test runner with --safe mode
- `tests/TEST_STATUS.md` - Test suite status
- `/var/crash/_usr_lib_systemd_systemd-oomd.108.crash` - Crash dump

---

**Document Version**: 1.0
**Last Updated**: 2025-11-07
**Next Review**: 2025-12-07 (or after any hang incident)

#fin
