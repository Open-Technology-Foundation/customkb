# CustomKB Scripts Directory

This directory contains utility scripts for managing and optimizing CustomKB installations.

## Scripts Overview

### Performance Optimization
- **optimize_kb_performance.py** - Main optimization script that applies memory-based performance tiers
  - Usage: `python scripts/optimize_kb_performance.py [kb_name] [--memory-gb N]`
  - Creates optimized configurations based on available system memory
  
- **show_optimization_tiers.py** - Displays optimization settings for different memory tiers
  - Usage: `python scripts/show_optimization_tiers.py`
  - Shows settings for: Low (<16GB), Medium (16-64GB), High (64-128GB), Very High (>128GB)

- **emergency_optimize.py** - Applies conservative settings to prevent crashes
  - Usage: `python scripts/emergency_optimize.py <kb_config_path>`
  - Use when experiencing memory issues or crashes

### GPU Support
- **benchmark_gpu.py** - Benchmarks GPU vs CPU performance for reranking
  - Usage: `python scripts/benchmark_gpu.py`
  - Tests reranking model performance on both CPU and GPU
  
- **gpu_monitor.sh** - Monitors GPU usage during CustomKB operations
  - Usage: `./scripts/gpu_monitor.sh`
  - Real-time GPU memory and utilization monitoring

### BM25 Management  
- **rebuild_bm25_filtered.py** - Creates filtered BM25 indexes with keyword filtering
  - Usage: `python scripts/rebuild_bm25_filtered.py <kb_config> [--keywords ...] [--min-score N]`
  - Reduces BM25 index size by filtering to relevant documents

### Testing
- **Note:** Safe testing functionality has been integrated into `run_tests.py --safe`
  - Use `python run_tests.py --safe` for memory-limited test execution
  - See `/tests/README.md` for full testing documentation

### Security
- **security-check.sh** - Run security scans on dependencies and code
  - Usage: `./scripts/security-check.sh`
  - Checks for known vulnerabilities in dependencies (Safety)
  - Scans code for security issues (Bandit)

## Quick Start

1. **Optimize a knowledgebase:**
   ```bash
   python scripts/optimize_kb_performance.py myproject
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