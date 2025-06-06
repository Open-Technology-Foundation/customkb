#!/usr/bin/env python
"""
Test runner script for CustomKB test suite.
Provides convenient commands for running different types of tests.
"""

import sys
import subprocess
import argparse
import os
from pathlib import Path


def run_command(cmd, description=""):
  """Run a command and return the result."""
  print(f"\n{'='*60}")
  if description:
    print(f"Running: {description}")
  print(f"Command: {' '.join(cmd)}")
  print('='*60)
  
  try:
    result = subprocess.run(cmd, check=True, capture_output=False)
    return result.returncode == 0
  except subprocess.CalledProcessError as e:
    print(f"Command failed with exit code {e.returncode}")
    return False
  except FileNotFoundError:
    print(f"Command not found: {cmd[0]}")
    print("Make sure pytest is installed: pip install -r requirements-test.txt")
    return False


def main():
  """Main test runner function."""
  parser = argparse.ArgumentParser(description="CustomKB Test Runner")
  parser.add_argument('--unit', action='store_true', help='Run unit tests only')
  parser.add_argument('--integration', action='store_true', help='Run integration tests only')
  parser.add_argument('--performance', action='store_true', help='Run performance tests only')
  parser.add_argument('--fast', action='store_true', help='Run fast tests only (exclude slow tests)')
  parser.add_argument('--coverage', action='store_true', help='Generate coverage report')
  parser.add_argument('--html', action='store_true', help='Generate HTML coverage report')
  parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
  parser.add_argument('--parallel', '-n', type=int, help='Run tests in parallel (number of workers)')
  parser.add_argument('--markers', help='Run tests with specific markers')
  parser.add_argument('--keyword', '-k', help='Run tests matching keyword expression')
  parser.add_argument('--file', help='Run specific test file')
  parser.add_argument('--install-deps', action='store_true', help='Install test dependencies first')
  
  args = parser.parse_args()
  
  # Install dependencies if requested
  if args.install_deps:
    print("Installing test dependencies...")
    cmd = [sys.executable, '-m', 'pip', 'install', '-r', 'requirements-test.txt']
    if not run_command(cmd, "Installing test dependencies"):
      return 1
  
  # Build pytest command
  cmd = ['python', '-m', 'pytest']
  
  # Add test selection
  if args.unit:
    cmd.extend(['-m', 'unit'])
  elif args.integration:
    cmd.extend(['-m', 'integration'])
  elif args.performance:
    cmd.extend(['-m', 'performance'])
  
  # Add speed filters
  if args.fast:
    cmd.extend(['-m', 'not slow'])
  
  # Add custom markers
  if args.markers:
    cmd.extend(['-m', args.markers])
  
  # Add keyword filter
  if args.keyword:
    cmd.extend(['-k', args.keyword])
  
  # Add specific file
  if args.file:
    cmd.append(args.file)
  
  # Add verbosity
  if args.verbose:
    cmd.append('-v')
  
  # Add parallel execution
  if args.parallel:
    cmd.extend(['-n', str(args.parallel)])
  
  # Add coverage options
  if args.coverage or args.html:
    cmd.extend(['--cov=.', '--cov-report=term-missing'])
    if args.html:
      cmd.extend(['--cov-report=html'])
  
  # Run the tests
  success = run_command(cmd, "Running CustomKB tests")
  
  if success:
    print("\n" + "="*60)
    print("✅ All tests passed!")
    if args.coverage or args.html:
      print("\n📊 Coverage report generated")
      if args.html:
        print("   HTML report: htmlcov/index.html")
    print("="*60)
    return 0
  else:
    print("\n" + "="*60)
    print("❌ Some tests failed!")
    print("="*60)
    return 1


def run_quick_check():
  """Run a quick smoke test to verify basic functionality."""
  print("Running quick smoke test...")
  
  cmd = [
    'python', '-m', 'pytest', 
    'tests/unit/test_config_manager.py::TestKnowledgeBase::test_init_with_kwargs',
    '-v'
  ]
  
  return run_command(cmd, "Quick smoke test")


def run_full_suite():
  """Run the complete test suite with coverage."""
  print("Running full test suite...")
  
  cmd = [
    'python', '-m', 'pytest',
    '--cov=.',
    '--cov-report=term-missing',
    '--cov-report=html',
    '-v'
  ]
  
  return run_command(cmd, "Full test suite with coverage")


def run_ci_tests():
  """Run tests suitable for CI/CD pipeline."""
  print("Running CI/CD test suite...")
  
  cmd = [
    'python', '-m', 'pytest',
    '-m', 'not requires_api and not requires_data',
    '--cov=.',
    '--cov-report=xml',
    '--cov-report=term',
    '--tb=short'
  ]
  
  return run_command(cmd, "CI/CD test suite")


if __name__ == '__main__':
  # Check if being called with special functions
  if len(sys.argv) == 2:
    if sys.argv[1] == 'quick':
      sys.exit(0 if run_quick_check() else 1)
    elif sys.argv[1] == 'full':
      sys.exit(0 if run_full_suite() else 1)
    elif sys.argv[1] == 'ci':
      sys.exit(0 if run_ci_tests() else 1)
  
  # Otherwise run main argument parser
  sys.exit(main())

#fin