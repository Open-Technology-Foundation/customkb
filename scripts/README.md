# CustomKB Scripts Directory

This directory contains utility scripts for managing and optimizing CustomKB installations.

## Scripts Overview

### Performance Optimization
- **show_optimization_tiers.py** - Displays optimization settings for different memory tiers
  - Usage: `python scripts/show_optimization_tiers.py`
  - Shows settings for: Low (<16GB), Medium (16-64GB), High (64-128GB), Very High (>128GB)

- **emergency_optimize.py** - Applies conservative settings to prevent crashes
  - Usage: `python scripts/emergency_optimize.py <kb_config_path>`
  - Use when experiencing memory issues or crashes

- **emergency_cleanup.sh** - Emergency cleanup of stale resources
  - Usage: `./scripts/emergency_cleanup.sh`

Note: Primary optimization is now via `customkb optimize <kb_name>`.

### GPU Support
- **benchmark_gpu.py** - Benchmarks GPU vs CPU performance for reranking
  - Usage: `python scripts/benchmark_gpu.py`
  - Tests reranking model performance on both CPU and GPU

- **gpu_monitor.sh** - Monitors GPU usage during CustomKB operations
  - Usage: `./scripts/gpu_monitor.sh`
  - Real-time GPU memory and utilization monitoring

- **gpu_env.sh** - GPU environment variable setup
  - Usage: `source scripts/gpu_env.sh`

- **test_cuda.sh** - CUDA installation verification
  - Usage: `./scripts/test_cuda.sh`

### Benchmarking
- **benchmark_vectordb.py** - Vector database benchmarking
  - Usage: `python scripts/benchmark_vectordb.py`

### BM25 Management
- **rebuild_bm25_filtered.py** - Creates filtered BM25 indexes with keyword filtering
  - Usage: `python scripts/rebuild_bm25_filtered.py <kb_config> [--keywords ...] [--min-score N]`
  - Reduces BM25 index size by filtering to relevant documents

- **upgrade_bm25_tokens.py** - Upgrade database schema for BM25 token storage
  - Usage: `python scripts/upgrade_bm25_tokens.py`

### Diagnostics
- **diagnose_crashes.py** - Analyze crash logs and system state
  - Usage: `python scripts/diagnose_crashes.py <kb_name>`

- **clean_corrupted_cache.py** - Clean corrupted cache files
  - Usage: `python scripts/clean_corrupted_cache.py`

### Testing
- **Note:** Safe testing functionality has been integrated into `run_tests.py --safe`
  - Use `python run_tests.py --safe` for memory-limited test execution
  - See `tests/` directory for full testing documentation

### Security
- **security-check.sh** - Run security scans on dependencies and code
  - Usage: `./scripts/security-check.sh`
  - Checks for known vulnerabilities in dependencies (Safety)
  - Scans code for security issues (Bandit)

## Quick Start

1. **Optimize a knowledgebase:**
   ```bash
   customkb optimize myproject
   ```

2. **Check optimization settings:**
   ```bash
   python scripts/show_optimization_tiers.py
   ```

3. **Run tests safely:**
   ```bash
   python run_tests.py --safe --unit
   ```

4. **Monitor GPU usage:**
   ```bash
   ./scripts/gpu_monitor.sh
   ```

5. **Emergency optimization after crash:**
   ```bash
   python scripts/emergency_optimize.py /var/lib/vectordbs/myproject/myproject.cfg
   ```

## Notes
- All Python scripts require the CustomKB virtual environment to be activated
- Shell scripts should be run from the CustomKB project root directory
- GPU scripts require NVIDIA GPU with CUDA support

#fin
