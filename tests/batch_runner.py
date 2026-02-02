#!/usr/bin/env python
"""
Batch test runner for CustomKB to prevent system crashes.
Runs tests in controlled batches with memory monitoring and limits.
"""

import argparse
import contextlib
import gc
import json
import subprocess
import sys
import time
from pathlib import Path

import psutil

# Test batch definitions
TEST_BATCHES = {
  'unit_core': {
    'description': 'Core unit tests (config, models, security)',
    'paths': [
      'tests/unit/test_config_manager.py',
      'tests/unit/test_model_manager.py',
      'tests/unit/utils/test_security_utils.py',
      'tests/unit/utils/test_text_utils.py',
      'tests/unit/utils/test_logging_utils.py',
    ],
    'memory_limit_gb': 2,
    'timeout': 300,  # 5 minutes
  },
  'unit_database': {
    'description': 'Database and storage unit tests',
    'paths': [
      'tests/unit/test_db_manager.py',
      'tests/unit/test_index_manager.py',
      'tests/unit/test_optimization_manager.py',
    ],
    'memory_limit_gb': 2,
    'timeout': 300,
  },
  'unit_processing': {
    'description': 'Processing and algorithm unit tests',
    'paths': [
      'tests/unit/test_embed_manager.py',
      'tests/unit/test_query_manager.py',
      'tests/unit/test_rerank_manager.py',
      'tests/unit/test_bm25_manager.py',
      'tests/unit/test_bm25_backward_compatibility.py',
    ],
    'memory_limit_gb': 4,
    'timeout': 600,  # 10 minutes
  },
  'integration_small': {
    'description': 'Small integration tests',
    'paths': [
      'tests/integration/test_bm25_integration.py',
      'tests/integration/test_reranking_integration.py',
    ],
    'memory_limit_gb': 4,
    'timeout': 600,
  },
  'integration_large': {
    'description': 'Large integration tests',
    'paths': [
      'tests/integration/test_end_to_end.py',
    ],
    'memory_limit_gb': 8,
    'timeout': 1200,  # 20 minutes
  },
  'performance': {
    'description': 'Performance tests (run separately)',
    'paths': [
      'tests/performance/',
    ],
    'memory_limit_gb': 8,
    'timeout': 1800,  # 30 minutes
  },
}


class MemoryMonitor:
  """Monitor memory usage during test execution."""

  def __init__(self, limit_gb: float):
    self.limit_gb = limit_gb
    self.limit_bytes = limit_gb * 1024 * 1024 * 1024
    self.process = psutil.Process()
    self.initial_memory = self.get_memory_usage()

  def get_memory_usage(self) -> dict[str, float]:
    """Get current memory usage statistics."""
    mem_info = self.process.memory_info()
    return {
      'rss_mb': mem_info.rss / 1024 / 1024,
      'vms_mb': mem_info.vms / 1024 / 1024,
      'percent': self.process.memory_percent(),
    }

  def check_memory(self) -> tuple[bool, str]:
    """Check if memory usage is within limits."""
    current = self.get_memory_usage()

    if current['rss_mb'] > self.limit_gb * 1024:
      return False, f"Memory limit exceeded: {current['rss_mb']:.1f}MB > {self.limit_gb * 1024}MB"

    if current['percent'] > 80:
      return False, f"System memory usage critical: {current['percent']:.1f}%"

    return True, f"Memory OK: {current['rss_mb']:.1f}MB ({current['percent']:.1f}%)"

  def get_summary(self) -> str:
    """Get memory usage summary."""
    current = self.get_memory_usage()
    delta = current['rss_mb'] - self.initial_memory['rss_mb']
    return (f"Memory: {current['rss_mb']:.1f}MB "
            f"(+{delta:.1f}MB from start, "
            f"{current['percent']:.1f}% of system)")


def run_test_batch(batch_name: str, batch_config: dict,
                   verbose: bool = False, coverage: bool = False) -> dict[str, any]:
  """Run a batch of tests with memory monitoring."""
  print(f"\n{'='*60}")
  print(f"Running batch: {batch_name}")
  print(f"Description: {batch_config['description']}")
  print(f"Memory limit: {batch_config['memory_limit_gb']}GB")
  print(f"Timeout: {batch_config['timeout']}s")
  print('='*60)

  # Start memory monitoring
  monitor = MemoryMonitor(batch_config['memory_limit_gb'])

  # Build pytest command
  cmd = [sys.executable, '-m', 'pytest']
  cmd.extend(batch_config['paths'])

  if verbose:
    cmd.append('-v')
  else:
    cmd.append('-q')

  # Add explicit stdout flushing for subprocess
  cmd.extend(['--capture=no'])

  if coverage:
    cmd.extend(['--cov=.', '--cov-append', '--cov-report='])

  # Disable parallel execution for safety
  cmd.extend(['--tb=short', '--no-header'])

  # Run tests
  start_time = time.time()
  result = {
    'batch': batch_name,
    'success': False,
    'duration': 0,
    'memory_peak': 0,
    'tests_run': 0,
    'tests_passed': 0,
    'tests_failed': 0,
    'error': None,
  }

  if verbose:
    print(f"Running command: {' '.join(cmd)}")

  try:
    # Run with timeout
    proc = subprocess.Popen(
      cmd,
      stdout=subprocess.PIPE,
      stderr=subprocess.PIPE,
      text=True,
    )

    # Monitor memory during execution
    memory_peak = 0
    while proc.poll() is None:
      time.sleep(1)
      mem_ok, mem_msg = monitor.check_memory()
      current_mem = monitor.get_memory_usage()['rss_mb']
      memory_peak = max(memory_peak, current_mem)

      if not mem_ok:
        print(f"\n❌ {mem_msg}")
        proc.terminate()
        proc.wait(timeout=5)
        result['error'] = mem_msg
        return result

      # Check timeout
      if time.time() - start_time > batch_config['timeout']:
        print(f"\n❌ Timeout exceeded ({batch_config['timeout']}s)")
        proc.terminate()
        proc.wait(timeout=5)
        result['error'] = 'Timeout exceeded'
        return result

    # Get results
    stdout, stderr = proc.communicate()
    duration = time.time() - start_time

    # Parse test results from output
    if proc.returncode == 0:
      result['success'] = True

    # Extract test counts from pytest output
    for line in stdout.split('\n'):
      if 'passed' in line or 'failed' in line:
        parts = line.split()
        for i, part in enumerate(parts):
          if part == 'passed':
            with contextlib.suppress(ValueError, IndexError):
              result['tests_passed'] = int(parts[i-1])
          elif part == 'failed':
            with contextlib.suppress(ValueError, IndexError):
              result['tests_failed'] = int(parts[i-1])

    result['tests_run'] = result['tests_passed'] + result['tests_failed']
    result['duration'] = duration
    result['memory_peak'] = memory_peak

    # Print results
    print(f"\n{'✅' if result['success'] else '❌'} Batch completed in {duration:.1f}s")
    print(f"   Tests: {result['tests_run']} run, "
          f"{result['tests_passed']} passed, "
          f"{result['tests_failed']} failed")
    print(f"   {monitor.get_summary()}")
    print(f"   Peak memory: {memory_peak:.1f}MB")

    if stderr and verbose:
      print("\nErrors:")
      print(stderr)

  except (OSError, subprocess.SubprocessError, ValueError) as e:
    result['error'] = str(e)
    print(f"\n❌ Batch failed with error: {e}")

  finally:
    # Force garbage collection
    gc.collect()
    time.sleep(2)  # Allow system to settle

  return result


def main():
  """Main batch runner function."""
  parser = argparse.ArgumentParser(
    description="Run CustomKB tests in safe batches with memory limits"
  )
  parser.add_argument(
    '--batch',
    choices=list(TEST_BATCHES.keys()) + ['all'],
    help='Batch to run (or "all" for all batches)'
  )
  parser.add_argument(
    '--list',
    action='store_true',
    help='List available batches'
  )
  parser.add_argument(
    '--memory-limit',
    type=float,
    help='Override memory limit in GB'
  )
  parser.add_argument(
    '--verbose',
    '-v',
    action='store_true',
    help='Verbose output'
  )
  parser.add_argument(
    '--coverage',
    action='store_true',
    help='Generate coverage report'
  )
  parser.add_argument(
    '--stop-on-failure',
    action='store_true',
    help='Stop running batches after first failure'
  )
  parser.add_argument(
    '--force',
    action='store_true',
    help='Force run even if system memory is high'
  )

  args = parser.parse_args()

  # List batches
  if args.list:
    print("Available test batches:")
    print("=" * 60)
    for name, config in TEST_BATCHES.items():
      print(f"\n{name}:")
      print(f"  Description: {config['description']}")
      print(f"  Memory limit: {config['memory_limit_gb']}GB")
      print(f"  Timeout: {config['timeout']}s")
      print(f"  Paths: {len(config['paths'])} test file(s)")
    return 0

  # Determine which batches to run
  if args.batch == 'all':
    batches_to_run = list(TEST_BATCHES.keys())
  elif args.batch:
    batches_to_run = [args.batch]
  else:
    print("Please specify --batch or use --list to see available batches")
    return 1

  # Override memory limits if specified
  if args.memory_limit:
    for batch_config in TEST_BATCHES.values():
      batch_config['memory_limit_gb'] = min(
        args.memory_limit,
        batch_config['memory_limit_gb']
      )

  # Check system memory
  total_memory_gb = psutil.virtual_memory().total / 1024 / 1024 / 1024
  print(f"\nSystem memory: {total_memory_gb:.1f}GB")
  print(f"Current memory usage: {psutil.virtual_memory().percent:.1f}%")

  if psutil.virtual_memory().percent > 50:
    print("\n⚠️  Warning: System memory usage is already high!")
    if not args.force:
      response = input("Continue anyway? (y/N): ")
      if response.lower() != 'y':
        return 1
    else:
      print("Continuing due to --force flag")

  # Initialize coverage if requested
  if args.coverage:
    # Clear previous coverage data
    subprocess.run([sys.executable, '-m', 'coverage', 'erase'], check=False)

  # Run batches
  results = []
  overall_success = True

  for batch_name in batches_to_run:
    if batch_name not in TEST_BATCHES:
      print(f"\n❌ Unknown batch: {batch_name}")
      continue

    batch_config = TEST_BATCHES[batch_name].copy()
    result = run_test_batch(batch_name, batch_config, args.verbose, args.coverage)
    results.append(result)

    if not result['success']:
      overall_success = False
      if args.stop_on_failure:
        print("\n❌ Stopping due to batch failure")
        break

  # Print summary
  print("\n" + "="*60)
  print("SUMMARY")
  print("="*60)

  total_tests = sum(r['tests_run'] for r in results)
  total_passed = sum(r['tests_passed'] for r in results)
  total_failed = sum(r['tests_failed'] for r in results)
  total_duration = sum(r['duration'] for r in results)
  max_memory = max((r['memory_peak'] for r in results), default=0)

  print(f"\nTotal tests run: {total_tests}")
  print(f"Passed: {total_passed}")
  print(f"Failed: {total_failed}")
  print(f"Total duration: {total_duration:.1f}s")
  print(f"Peak memory usage: {max_memory:.1f}MB")

  print("\nBatch results:")
  for result in results:
    status = "✅" if result['success'] else "❌"
    print(f"  {status} {result['batch']}: "
          f"{result['tests_passed']}/{result['tests_run']} passed, "
          f"{result['duration']:.1f}s, "
          f"{result['memory_peak']:.1f}MB peak")
    if result['error']:
      print(f"     Error: {result['error']}")

  # Generate coverage report if requested
  if args.coverage and overall_success:
    print("\n" + "="*60)
    print("Generating coverage report...")
    subprocess.run([
      sys.executable, '-m', 'coverage', 'report',
      '--skip-covered', '--skip-empty'
    ])
    subprocess.run([
      sys.executable, '-m', 'coverage', 'html'
    ])
    print("HTML coverage report: htmlcov/index.html")

  # Save results to file
  results_file = Path('test_results.json')
  with open(results_file, 'w') as f:
    json.dump({
      'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
      'overall_success': overall_success,
      'total_tests': total_tests,
      'total_passed': total_passed,
      'total_failed': total_failed,
      'total_duration': total_duration,
      'max_memory_mb': max_memory,
      'batches': results,
    }, f, indent=2)
  print(f"\nResults saved to: {results_file}")

  return 0 if overall_success else 1


if __name__ == '__main__':
  sys.exit(main())

#fin
