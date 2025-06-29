#!/bin/bash
# Safe test runner with memory limits to prevent system hangs

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}CustomKB Safe Test Runner${NC}"
echo "This script runs tests with memory limits to prevent system hangs."
echo

# Function to run tests with memory limit
run_tests_safely() {
  local test_path="$1"
  local description="$2"
  local memory_limit="${3:-4000000}"  # Default 4GB in KB
  
  echo -e "${YELLOW}Running: $description${NC}"
  echo "Memory limit: $((memory_limit / 1024 / 1024))GB"
  
  # Set memory limit for this shell session
  ulimit -v "$memory_limit"
  
  # Activate virtual environment
  source .venv/bin/activate
  
  # Run tests with limited parallelization
  if pytest "$test_path" -v -n 2 --tb=short; then
    echo -e "${GREEN}✓ $description passed${NC}\n"
  else
    echo -e "${RED}✗ $description failed${NC}\n"
  fi
}

# Check if we're in the correct directory
if [[ ! -f "customkb.py" ]]; then
  echo -e "${RED}Error: Must run from CustomKB project root${NC}"
  exit 1
fi

# Parse command line arguments
TEST_TYPE="${1:-all}"

case "$TEST_TYPE" in
  unit)
    echo "Running unit tests only (safest)..."
    run_tests_safely "tests/unit/" "Unit Tests" 2000000  # 2GB limit
    ;;
  integration)
    echo "Running integration tests (moderate risk)..."
    run_tests_safely "tests/integration/" "Integration Tests" 4000000  # 4GB limit
    ;;
  performance)
    echo "Running performance tests (highest risk)..."
    run_tests_safely "tests/performance/" "Performance Tests" 6000000  # 6GB limit
    ;;
  all)
    echo "Running all tests with safety limits..."
    run_tests_safely "tests/unit/" "Unit Tests" 2000000
    run_tests_safely "tests/integration/" "Integration Tests" 4000000
    echo -e "${YELLOW}Skipping performance tests for safety. Run './scripts/safe_test_runner.sh performance' to run them.${NC}"
    ;;
  *)
    echo "Usage: $0 [unit|integration|performance|all]"
    echo "  unit         - Run unit tests only (safest)"
    echo "  integration  - Run integration tests"
    echo "  performance  - Run performance tests (highest memory usage)"
    echo "  all          - Run unit and integration tests (skips performance)"
    exit 1
    ;;
esac

echo -e "${GREEN}Test run completed safely!${NC}"

#fin