#!/bin/bash
# Main test runner for citation system test suite

set -euo pipefail

# Get test directory
TEST_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Default values
RUN_UNIT=1
RUN_INTEGRATION=1
RUN_PERFORMANCE=0
RUN_EDGE_CASES=0
RUN_ALL=0
PARALLEL_TESTS=0
COVERAGE=0
VERBOSE=0
FILTER=""
PRESERVE_ENV=0

# Help message
show_help() {
  cat << EOF
Usage: $(basename "$0") [OPTIONS]

Run test suite for citation system.

OPTIONS:
    --all              Run all test suites (default: unit + integration)
    --unit             Run only unit tests
    --integration      Run only integration tests
    --performance      Run only performance tests
    --edge-cases       Run only edge case tests
    --quick            Run quick smoke tests only
    --parallel         Run tests in parallel
    --coverage         Generate coverage report
    --verbose, -v      Verbose output
    --filter PATTERN   Run only tests matching pattern
    --preserve         Preserve test workspace after completion
    -h, --help         Show this help message

EXAMPLES:
    # Run all tests
    ./run_tests.sh --all

    # Run only unit tests with verbose output
    ./run_tests.sh --unit -v

    # Run tests matching "api" pattern
    ./run_tests.sh --filter "api"

    # Run quick smoke tests
    ./run_tests.sh --quick

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --all)
      RUN_ALL=1
      RUN_UNIT=1
      RUN_INTEGRATION=1
      RUN_PERFORMANCE=1
      RUN_EDGE_CASES=1
      ;;
    --unit)
      RUN_UNIT=1
      RUN_INTEGRATION=0
      ;;
    --integration)
      RUN_UNIT=0
      RUN_INTEGRATION=1
      ;;
    --performance)
      RUN_UNIT=0
      RUN_INTEGRATION=0
      RUN_PERFORMANCE=1
      ;;
    --edge-cases)
      RUN_UNIT=0
      RUN_INTEGRATION=0
      RUN_EDGE_CASES=1
      ;;
    --quick)
      # Quick smoke tests - minimal subset
      RUN_UNIT=1
      RUN_INTEGRATION=0
      export TEST_QUICK_MODE=1
      ;;
    --parallel)
      PARALLEL_TESTS=1
      ;;
    --coverage)
      COVERAGE=1
      ;;
    -v|--verbose)
      VERBOSE=1
      export TEST_VERBOSE=1
      ;;
    --filter)
      shift
      FILTER="$1"
      ;;
    --preserve)
      PRESERVE_ENV=1
      export PRESERVE_TEST_ENV=1
      ;;
    -h|--help)
      show_help
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      show_help
      exit 1
      ;;
  esac
  shift
done

# Export test settings
export TEST_VERBOSE="$VERBOSE"

# Colors for output
if [[ -t 1 ]]; then
  RED=$'\033[0;31m'
  GREEN=$'\033[0;32m'
  YELLOW=$'\033[0;33m'
  BLUE=$'\033[0;34m'
  BOLD=$'\033[1m'
  NOCOLOR=$'\033[0m'
else
  RED='' GREEN='' YELLOW='' BLUE='' BOLD='' NOCOLOR=''
fi

# Test results tracking
declare -i TOTAL_SUITES=0
declare -i PASSED_SUITES=0
declare -i FAILED_SUITES=0
declare -a FAILED_TESTS=()

# Function to run a test file
run_test_file() {
  local test_file="$1"
  local test_name=$(basename "$test_file" .sh)
  
  echo
  echo "${BOLD}${BLUE}Running test suite: $test_name${NOCOLOR}"
  echo "${BLUE}────────────────────────────────────────${NOCOLOR}"
  
  ((TOTAL_SUITES++))
  
  local start_time=$(date +%s)
  
  if "$test_file"; then
    local elapsed=$(($(date +%s) - start_time))
    echo "${GREEN}✓ Suite passed${NOCOLOR} (${elapsed}s)"
    ((PASSED_SUITES++))
    return 0
  else
    local elapsed=$(($(date +%s) - start_time))
    echo "${RED}✗ Suite failed${NOCOLOR} (${elapsed}s)"
    ((FAILED_SUITES++))
    FAILED_TESTS+=("$test_name")
    return 1
  fi
}

# Function to run tests in a directory
run_test_directory() {
  local dir="$1"
  local pattern="${2:-test_*.sh}"
  
  if [[ ! -d "$dir" ]]; then
    echo "${YELLOW}Warning: Test directory not found: $dir${NOCOLOR}"
    return
  fi
  
  # Find test files
  local test_files=()
  while IFS= read -r -d '' file; do
    # Apply filter if specified
    if [[ -z "$FILTER" ]] || [[ "$file" =~ $FILTER ]]; then
      test_files+=("$file")
    fi
  done < <(find "$dir" -name "$pattern" -type f -executable -print0 | sort -z)
  
  if [[ ${#test_files[@]} -eq 0 ]]; then
    echo "${YELLOW}No test files found in: $dir${NOCOLOR}"
    return
  fi
  
  echo
  echo "${BOLD}${BLUE}===== $(basename "$dir") Tests =====${NOCOLOR}"
  
  # Run tests (parallel or sequential)
  if ((PARALLEL_TESTS)); then
    # Run tests in parallel
    local pids=()
    for test_file in "${test_files[@]}"; do
      run_test_file "$test_file" &
      pids+=($!)
    done
    
    # Wait for all tests
    local failed=0
    for pid in "${pids[@]}"; do
      wait $pid || ((failed++))
    done
    
    return $failed
  else
    # Run tests sequentially
    local failed=0
    for test_file in "${test_files[@]}"; do
      run_test_file "$test_file" || ((failed++))
    done
    
    return $failed
  fi
}

# Main test execution
main() {
  local start_time=$(date +%s)
  
  echo "${BOLD}${BLUE}Citation System Test Suite${NOCOLOR}"
  echo "${BLUE}════════════════════════════════${NOCOLOR}"
  echo "Test directory: $TEST_DIR"
  echo "Start time: $(date)"
  
  # Check for mock API if needed
  if [[ -f "$TEST_DIR/test_config.sh" ]]; then
    source "$TEST_DIR/test_config.sh"
    if ! check_mock_api 2>/dev/null; then
      echo
      echo "${YELLOW}Note: Mock API server not running. Some tests may be skipped.${NOCOLOR}"
      echo "${YELLOW}Start it with: $TEST_DIR/mocks/mock_openai_api.sh${NOCOLOR}"
    fi
  fi
  
  # Run selected test suites
  local exit_code=0
  
  if ((RUN_UNIT)); then
    run_test_directory "$TEST_DIR/unit" || exit_code=1
  fi
  
  if ((RUN_INTEGRATION)); then
    run_test_directory "$TEST_DIR/integration" || exit_code=1
  fi
  
  if ((RUN_PERFORMANCE)); then
    run_test_directory "$TEST_DIR/performance" || exit_code=1
  fi
  
  if ((RUN_EDGE_CASES)); then
    run_test_directory "$TEST_DIR/edge_cases" || exit_code=1
  fi
  
  # Generate coverage report if requested
  if ((COVERAGE)); then
    echo
    echo "${BOLD}${BLUE}Generating coverage report...${NOCOLOR}"
    # This would integrate with a bash coverage tool if available
    echo "${YELLOW}Coverage reporting not yet implemented${NOCOLOR}"
  fi
  
  # Final summary
  local elapsed=$(($(date +%s) - start_time))
  
  echo
  echo "${BOLD}${BLUE}════════════════════════════════${NOCOLOR}"
  echo "${BOLD}${BLUE}Test Suite Summary${NOCOLOR}"
  echo "${BOLD}${BLUE}════════════════════════════════${NOCOLOR}"
  echo "Total test suites:  $TOTAL_SUITES"
  echo "Passed:            ${GREEN}$PASSED_SUITES${NOCOLOR}"
  echo "Failed:            ${RED}$FAILED_SUITES${NOCOLOR}"
  
  if [[ ${#FAILED_TESTS[@]} -gt 0 ]]; then
    echo
    echo "${RED}Failed test suites:${NOCOLOR}"
    for test in "${FAILED_TESTS[@]}"; do
      echo "  - $test"
    done
  fi
  
  echo
  echo "Total time: ${elapsed}s"
  echo "End time: $(date)"
  
  if ((PRESERVE_ENV)); then
    echo
    echo "${YELLOW}Test workspace preserved at: ${TEST_WORKSPACE:-/tmp/citation-tests-*}${NOCOLOR}"
  fi
  
  # Exit with appropriate code
  if ((FAILED_SUITES > 0)); then
    echo
    echo "${RED}${BOLD}TEST SUITE FAILED${NOCOLOR}"
    exit 1
  else
    echo
    echo "${GREEN}${BOLD}ALL TESTS PASSED${NOCOLOR}"
    exit 0
  fi
}

# Run main
main

#fin