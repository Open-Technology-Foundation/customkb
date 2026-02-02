#!/bin/bash
# Test configuration for citation system test suite
# This file is sourced by all test scripts to provide common configuration

# Test environment setup
set -euo pipefail

# Get absolute path to test directory
declare -r TEST_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
declare -r CITATION_ROOT="$(cd "$TEST_ROOT/.." && pwd)"

# Test paths
declare -r TEST_FIXTURES="$TEST_ROOT/fixtures"
declare -r TEST_DOCUMENTS="$TEST_FIXTURES/documents"
declare -r TEST_DATABASES="$TEST_FIXTURES/databases"
declare -r TEST_API_RESPONSES="$TEST_FIXTURES/api_responses"
declare -r TEST_MOCKS="$TEST_ROOT/mocks"

# Create temporary test workspace
declare -r TEST_WORKSPACE="${TEST_WORKSPACE:-/tmp/citation-tests-$$}"
declare -r TEST_WORK_DIR="$TEST_WORKSPACE/work"
declare -r TEST_DB_DIR="$TEST_WORKSPACE/db"
declare -r TEST_LOG_DIR="$TEST_WORKSPACE/logs"
declare -r TEST_RESULTS_DIR="$TEST_WORKSPACE/results"

# Scripts under test
declare -r GEN_CITATIONS_SCRIPT="$CITATION_ROOT/gen-citations.sh"
declare -r APPEND_CITATIONS_SCRIPT="$CITATION_ROOT/append-citations.sh"

# Library paths
declare -r LIB_DIR="$CITATION_ROOT/lib"
declare -r DB_FUNCTIONS="$LIB_DIR/db_functions.sh"
declare -r API_FUNCTIONS="$LIB_DIR/api_functions.sh"
declare -r PARALLEL_FUNCTIONS="$LIB_DIR/parallel_functions.sh"

# Test database configuration
declare -r TEST_DB_NAME="test_citations.db"
declare -r TEST_DB_PATH="$TEST_DB_DIR/$TEST_DB_NAME"

# Mock API configuration
declare -r MOCK_API_PORT="${MOCK_API_PORT:-8888}"
declare -r MOCK_API_URL="http://localhost:$MOCK_API_PORT"
declare -r MOCK_API_ENDPOINT="$MOCK_API_URL/v1/chat/completions"

# Test timing configuration
declare -r TEST_TIMEOUT="${TEST_TIMEOUT:-300}"  # 5 minutes default
declare -r PARALLEL_TEST_WORKERS="${PARALLEL_TEST_WORKERS:-4}"

# Test output configuration
declare -r TEST_VERBOSE="${TEST_VERBOSE:-0}"
declare -r TEST_DEBUG="${TEST_DEBUG:-0}"
declare -r TEST_COLOR="${TEST_COLOR:-1}"

# Colors for test output
if [[ -t 1 ]] && ((TEST_COLOR)); then
  declare -r RED=$'\033[0;31m'
  declare -r GREEN=$'\033[0;32m'
  declare -r YELLOW=$'\033[0;33m'
  declare -r BLUE=$'\033[0;34m'
  declare -r MAGENTA=$'\033[0;35m'
  declare -r CYAN=$'\033[0;36m'
  declare -r BOLD=$'\033[1m'
  declare -r NOCOLOR=$'\033[0m'
else
  declare -r RED='' GREEN='' YELLOW='' BLUE='' MAGENTA='' CYAN='' BOLD='' NOCOLOR=''
fi

# Test counters (global)
declare -gi TESTS_TOTAL=0
declare -gi TESTS_PASSED=0
declare -gi TESTS_FAILED=0
declare -gi TESTS_SKIPPED=0
declare -gi ASSERTIONS_TOTAL=0
declare -gi ASSERTIONS_PASSED=0
declare -gi ASSERTIONS_FAILED=0

# Test timing
declare -g TEST_SUITE_START_TIME=""
declare -g TEST_START_TIME=""
declare -g CURRENT_TEST_NAME=""
declare -g CURRENT_TEST_FILE=""

# Setup test environment
setup_test_env() {
  local test_name="${1:-test}"
  
  # Create workspace directories
  mkdir -p "$TEST_WORK_DIR" "$TEST_DB_DIR" "$TEST_LOG_DIR" "$TEST_RESULTS_DIR"
  
  # Set up test-specific subdirectory
  local test_dir="$TEST_WORK_DIR/$test_name"
  mkdir -p "$test_dir"
  
  # Export for use in tests
  export TEST_CURRENT_DIR="$test_dir"
  export TEST_CURRENT_DB="$TEST_DB_DIR/${test_name}.db"
  
  # Set up mock API key if needed
  export OPENAI_API_KEY="${OPENAI_API_KEY:-test-api-key-12345}"
  
  # Disable actual API calls unless explicitly enabled
  export CITATION_USE_MOCK_API="${CITATION_USE_MOCK_API:-1}"
  
  ((TEST_DEBUG)) && echo "${BLUE}[DEBUG]${NOCOLOR} Test environment set up in: $test_dir"
}

# Cleanup test environment
cleanup_test_env() {
  local preserve="${1:-0}"
  
  # Only cleanup if we're the main process and not being sourced
  if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    if ((preserve)); then
      echo "${YELLOW}[INFO]${NOCOLOR} Preserving test workspace at: $TEST_WORKSPACE"
    else
      ((TEST_DEBUG)) && echo "${BLUE}[DEBUG]${NOCOLOR} Cleaning up test workspace: $TEST_WORKSPACE"
      rm -rf "$TEST_WORKSPACE"
    fi
  fi
}

# Initialize test suite
init_test_suite() {
  TEST_SUITE_START_TIME=$(date +%s)
  
  # Ensure cleanup on exit
  trap 'cleanup_test_env ${PRESERVE_TEST_ENV:-0}' EXIT
  
  # Create main workspace and subdirectories
  mkdir -p "$TEST_WORKSPACE" "$TEST_RESULTS_DIR" "$TEST_LOG_DIR"
  
  # Initialize results file
  local results_file="$TEST_RESULTS_DIR/test_results_$(date +%Y%m%d_%H%M%S).txt"
  echo "Test Suite Started: $(date)" > "$results_file"
  echo "Test Root: $TEST_ROOT" >> "$results_file"
  echo "Workspace: $TEST_WORKSPACE" >> "$results_file"
  echo "---" >> "$results_file"
  
  export TEST_RESULTS_FILE="$results_file"
}

# Check if mock API is available
check_mock_api() {
  if ((CITATION_USE_MOCK_API)); then
    if ! curl -s -f "$MOCK_API_URL/health" >/dev/null 2>&1; then
      echo "${YELLOW}[WARN]${NOCOLOR} Mock API server not running on port $MOCK_API_PORT"
      echo "${YELLOW}[WARN]${NOCOLOR} Start it with: $TEST_MOCKS/mock_openai_api.sh"
      return 1
    fi
  fi
  return 0
}

# Export test configuration for scripts
export_test_config() {
  # Export paths
  export TEST_ROOT TEST_FIXTURES TEST_WORKSPACE
  export GEN_CITATIONS_SCRIPT APPEND_CITATIONS_SCRIPT
  
  # Export test settings
  export TEST_VERBOSE TEST_DEBUG TEST_TIMEOUT
  
  # Export mock settings
  export MOCK_API_URL MOCK_API_ENDPOINT CITATION_USE_MOCK_API
}

# Source test helpers if available
if [[ -f "$TEST_ROOT/test_helpers.sh" ]]; then
  source "$TEST_ROOT/test_helpers.sh"
fi

#fin