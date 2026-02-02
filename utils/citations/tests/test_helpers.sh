#!/bin/bash
# Test helper functions for citation system test suite

# Test execution helpers
start_test() {
  local test_name="$1"
  local test_file="${2:-${BASH_SOURCE[1]}}"
  
  CURRENT_TEST_NAME="$test_name"
  CURRENT_TEST_FILE="$test_file"
  TEST_START_TIME=$(date +%s)
  ((TESTS_TOTAL++))
  
  echo
  echo "${BOLD}${BLUE}=== Test: $test_name ===${NOCOLOR}"
  ((TEST_VERBOSE)) && echo "${CYAN}File: $test_file${NOCOLOR}"
}

end_test() {
  local status="${1:-0}"
  local elapsed=$(($(date +%s) - TEST_START_TIME))
  
  if ((status == 0)); then
    ((TESTS_PASSED++))
    echo "${GREEN}✓ PASS${NOCOLOR} $CURRENT_TEST_NAME (${elapsed}s)"
  else
    ((TESTS_FAILED++))
    echo "${RED}✗ FAIL${NOCOLOR} $CURRENT_TEST_NAME (${elapsed}s)"
  fi
  
  # Log to results file
  if [[ -n "${TEST_RESULTS_FILE:-}" ]]; then
    local result=$((status == 0 ? "PASS" : "FAIL"))
    echo "[$result] $CURRENT_TEST_NAME - ${elapsed}s" >> "$TEST_RESULTS_FILE"
  fi
  
  return $status
}

skip_test() {
  local reason="${1:-No reason given}"
  ((TESTS_TOTAL++))
  ((TESTS_SKIPPED++))
  
  echo "${YELLOW}⊘ SKIP${NOCOLOR} $CURRENT_TEST_NAME - $reason"
  
  # Log to results file
  if [[ -n "${TEST_RESULTS_FILE:-}" ]]; then
    echo "[SKIP] $CURRENT_TEST_NAME - $reason" >> "$TEST_RESULTS_FILE"
  fi
}

# Assertion functions
assert_equals() {
  local expected="$1"
  local actual="$2"
  local message="${3:-Values should be equal}"
  
  ((ASSERTIONS_TOTAL++))
  
  if [[ "$expected" == "$actual" ]]; then
    ((ASSERTIONS_PASSED++))
    ((TEST_VERBOSE)) && echo "${GREEN}  ✓${NOCOLOR} $message"
    return 0
  else
    ((ASSERTIONS_FAILED++))
    echo "${RED}  ✗${NOCOLOR} $message"
    echo "${RED}    Expected:${NOCOLOR} '$expected'"
    echo "${RED}    Actual:${NOCOLOR}   '$actual'"
    return 1
  fi
}

assert_not_equals() {
  local value1="$1"
  local value2="$2"
  local message="${3:-Values should not be equal}"
  
  ((ASSERTIONS_TOTAL++))
  
  if [[ "$value1" != "$value2" ]]; then
    ((ASSERTIONS_PASSED++))
    ((TEST_VERBOSE)) && echo "${GREEN}  ✓${NOCOLOR} $message"
    return 0
  else
    ((ASSERTIONS_FAILED++))
    echo "${RED}  ✗${NOCOLOR} $message"
    echo "${RED}    Both values:${NOCOLOR} '$value1'"
    return 1
  fi
}

assert_contains() {
  local haystack="$1"
  local needle="$2"
  local message="${3:-String should contain substring}"
  
  ((ASSERTIONS_TOTAL++))
  
  if [[ "$haystack" == *"$needle"* ]]; then
    ((ASSERTIONS_PASSED++))
    ((TEST_VERBOSE)) && echo "${GREEN}  ✓${NOCOLOR} $message"
    return 0
  else
    ((ASSERTIONS_FAILED++))
    echo "${RED}  ✗${NOCOLOR} $message"
    echo "${RED}    String:${NOCOLOR} '$haystack'"
    echo "${RED}    Should contain:${NOCOLOR} '$needle'"
    return 1
  fi
}

assert_not_contains() {
  local haystack="$1"
  local needle="$2"
  local message="${3:-String should not contain substring}"
  
  ((ASSERTIONS_TOTAL++))
  
  if [[ "$haystack" != *"$needle"* ]]; then
    ((ASSERTIONS_PASSED++))
    ((TEST_VERBOSE)) && echo "${GREEN}  ✓${NOCOLOR} $message"
    return 0
  else
    ((ASSERTIONS_FAILED++))
    echo "${RED}  ✗${NOCOLOR} $message"
    echo "${RED}    String:${NOCOLOR} '$haystack'"
    echo "${RED}    Should not contain:${NOCOLOR} '$needle'"
    return 1
  fi
}

assert_true() {
  local condition="$1"
  local message="${2:-Condition should be true}"
  
  ((ASSERTIONS_TOTAL++))
  
  if ((condition)); then
    ((ASSERTIONS_PASSED++))
    ((TEST_VERBOSE)) && echo "${GREEN}  ✓${NOCOLOR} $message"
    return 0
  else
    ((ASSERTIONS_FAILED++))
    echo "${RED}  ✗${NOCOLOR} $message"
    return 1
  fi
}

assert_false() {
  local condition="$1"
  local message="${2:-Condition should be false}"
  
  ((ASSERTIONS_TOTAL++))
  
  if ((! condition)); then
    ((ASSERTIONS_PASSED++))
    ((TEST_VERBOSE)) && echo "${GREEN}  ✓${NOCOLOR} $message"
    return 0
  else
    ((ASSERTIONS_FAILED++))
    echo "${RED}  ✗${NOCOLOR} $message"
    return 1
  fi
}

assert_file_exists() {
  local file="$1"
  local message="${2:-File should exist}"
  
  ((ASSERTIONS_TOTAL++))
  
  if [[ -f "$file" ]]; then
    ((ASSERTIONS_PASSED++))
    ((TEST_VERBOSE)) && echo "${GREEN}  ✓${NOCOLOR} $message: $file"
    return 0
  else
    ((ASSERTIONS_FAILED++))
    echo "${RED}  ✗${NOCOLOR} $message: $file"
    return 1
  fi
}

assert_file_not_exists() {
  local file="$1"
  local message="${2:-File should not exist}"
  
  ((ASSERTIONS_TOTAL++))
  
  if [[ ! -f "$file" ]]; then
    ((ASSERTIONS_PASSED++))
    ((TEST_VERBOSE)) && echo "${GREEN}  ✓${NOCOLOR} $message: $file"
    return 0
  else
    ((ASSERTIONS_FAILED++))
    echo "${RED}  ✗${NOCOLOR} $message: $file"
    return 1
  fi
}

assert_dir_exists() {
  local dir="$1"
  local message="${2:-Directory should exist}"
  
  ((ASSERTIONS_TOTAL++))
  
  if [[ -d "$dir" ]]; then
    ((ASSERTIONS_PASSED++))
    ((TEST_VERBOSE)) && echo "${GREEN}  ✓${NOCOLOR} $message: $dir"
    return 0
  else
    ((ASSERTIONS_FAILED++))
    echo "${RED}  ✗${NOCOLOR} $message: $dir"
    return 1
  fi
}

assert_command_success() {
  local command="$1"
  local message="${2:-Command should succeed}"
  
  ((ASSERTIONS_TOTAL++))
  
  if eval "$command" >/dev/null 2>&1; then
    ((ASSERTIONS_PASSED++))
    ((TEST_VERBOSE)) && echo "${GREEN}  ✓${NOCOLOR} $message"
    return 0
  else
    local exit_code=$?
    ((ASSERTIONS_FAILED++))
    echo "${RED}  ✗${NOCOLOR} $message"
    echo "${RED}    Command:${NOCOLOR} $command"
    echo "${RED}    Exit code:${NOCOLOR} $exit_code"
    return 1
  fi
}

assert_command_fails() {
  local command="$1"
  local message="${2:-Command should fail}"
  
  ((ASSERTIONS_TOTAL++))
  
  if ! eval "$command" >/dev/null 2>&1; then
    ((ASSERTIONS_PASSED++))
    ((TEST_VERBOSE)) && echo "${GREEN}  ✓${NOCOLOR} $message"
    return 0
  else
    ((ASSERTIONS_FAILED++))
    echo "${RED}  ✗${NOCOLOR} $message"
    echo "${RED}    Command succeeded unexpectedly:${NOCOLOR} $command"
    return 1
  fi
}

assert_output_contains() {
  local command="$1"
  local expected="$2"
  local message="${3:-Command output should contain string}"
  
  ((ASSERTIONS_TOTAL++))
  
  local output
  output=$(eval "$command" 2>&1)
  
  if [[ "$output" == *"$expected"* ]]; then
    ((ASSERTIONS_PASSED++))
    ((TEST_VERBOSE)) && echo "${GREEN}  ✓${NOCOLOR} $message"
    return 0
  else
    ((ASSERTIONS_FAILED++))
    echo "${RED}  ✗${NOCOLOR} $message"
    echo "${RED}    Command:${NOCOLOR} $command"
    echo "${RED}    Expected in output:${NOCOLOR} '$expected'"
    echo "${RED}    Actual output:${NOCOLOR} '$output'"
    return 1
  fi
}

# File comparison helpers
assert_files_equal() {
  local file1="$1"
  local file2="$2"
  local message="${3:-Files should be equal}"
  
  ((ASSERTIONS_TOTAL++))
  
  if diff -q "$file1" "$file2" >/dev/null 2>&1; then
    ((ASSERTIONS_PASSED++))
    ((TEST_VERBOSE)) && echo "${GREEN}  ✓${NOCOLOR} $message"
    return 0
  else
    ((ASSERTIONS_FAILED++))
    echo "${RED}  ✗${NOCOLOR} $message"
    echo "${RED}    Files differ:${NOCOLOR} $file1 vs $file2"
    ((TEST_VERBOSE)) && diff -u "$file1" "$file2" | head -20
    return 1
  fi
}

# Database helpers
assert_db_has_table() {
  local db="$1"
  local table="$2"
  local message="${3:-Database should have table}"
  
  ((ASSERTIONS_TOTAL++))
  
  if sqlite3 "$db" ".tables" | grep -q "\\b$table\\b"; then
    ((ASSERTIONS_PASSED++))
    ((TEST_VERBOSE)) && echo "${GREEN}  ✓${NOCOLOR} $message: $table"
    return 0
  else
    ((ASSERTIONS_FAILED++))
    echo "${RED}  ✗${NOCOLOR} $message: $table"
    echo "${RED}    Tables in database:${NOCOLOR} $(sqlite3 "$db" ".tables")"
    return 1
  fi
}

assert_db_row_count() {
  local db="$1"
  local table="$2"
  local expected="$3"
  local message="${4:-Table should have expected row count}"
  
  ((ASSERTIONS_TOTAL++))
  
  local actual
  actual=$(sqlite3 "$db" "SELECT COUNT(*) FROM $table;")
  
  if [[ "$actual" == "$expected" ]]; then
    ((ASSERTIONS_PASSED++))
    ((TEST_VERBOSE)) && echo "${GREEN}  ✓${NOCOLOR} $message: $expected rows"
    return 0
  else
    ((ASSERTIONS_FAILED++))
    echo "${RED}  ✗${NOCOLOR} $message"
    echo "${RED}    Expected:${NOCOLOR} $expected rows"
    echo "${RED}    Actual:${NOCOLOR} $actual rows"
    return 1
  fi
}

# Test data creation helpers
create_test_file() {
  local path="$1"
  local content="${2:-# Test Document\n\nThis is a test document.}"
  
  mkdir -p "$(dirname "$path")"
  echo -e "$content" > "$path"
}

create_test_database() {
  local db_path="$1"
  
  mkdir -p "$(dirname "$db_path")"
  
  sqlite3 "$db_path" << 'EOF'
CREATE TABLE IF NOT EXISTS citations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sourcefile VARCHAR(255) UNIQUE,
    title TEXT,
    author TEXT,
    year TEXT,
    raw_citation TEXT
);
CREATE INDEX IF NOT EXISTS idx_sourcefile ON citations(sourcefile);
EOF
}

add_test_citation() {
  local db_path="$1"
  local sourcefile="$2"
  local title="${3:-Test Title}"
  local author="${4:-Test Author}"
  local year="${5:-2024}"
  local raw="${6:-"$title", $author, $year}"
  
  sqlite3 "$db_path" "INSERT OR REPLACE INTO citations (sourcefile, title, author, year, raw_citation) VALUES ('$sourcefile', '$title', '$author', '$year', '$raw');"
}

# Mock helpers
start_mock_api() {
  local port="${1:-$MOCK_API_PORT}"
  local responses_dir="${2:-$TEST_API_RESPONSES}"
  
  if [[ -x "$TEST_MOCKS/mock_openai_api.sh" ]]; then
    "$TEST_MOCKS/mock_openai_api.sh" --port "$port" --responses "$responses_dir" &
    local pid=$!
    
    # Wait for server to start
    local retries=30
    while ((retries > 0)); do
      if curl -s -f "http://localhost:$port/health" >/dev/null 2>&1; then
        echo "${GREEN}[INFO]${NOCOLOR} Mock API started on port $port (PID: $pid)"
        return 0
      fi
      sleep 0.1
      ((retries--))
    done
    
    echo "${RED}[ERROR]${NOCOLOR} Failed to start mock API"
    kill $pid 2>/dev/null
    return 1
  else
    echo "${YELLOW}[WARN]${NOCOLOR} Mock API script not found"
    return 1
  fi
}

stop_mock_api() {
  local port="${1:-$MOCK_API_PORT}"
  
  # Find and kill process listening on port
  local pid
  pid=$(lsof -ti :$port 2>/dev/null)
  
  if [[ -n "$pid" ]]; then
    kill $pid 2>/dev/null
    echo "${GREEN}[INFO]${NOCOLOR} Stopped mock API (PID: $pid)"
  fi
}

# Progress monitoring helpers
monitor_progress() {
  local progress_file="$1"
  local expected_total="$2"
  local timeout="${3:-30}"
  
  local start_time=$(date +%s)
  local last_count=0
  
  while true; do
    if [[ -f "$progress_file" ]]; then
      local current_count
      current_count=$(awk -F: '{sum+=$1} END {print sum+0}' "$progress_file" 2>/dev/null || echo "0")
      
      if ((current_count != last_count)); then
        echo "${CYAN}[PROGRESS]${NOCOLOR} $current_count / $expected_total"
        last_count=$current_count
      fi
      
      if ((current_count >= expected_total)); then
        return 0
      fi
    fi
    
    if (($(date +%s) - start_time > timeout)); then
      echo "${RED}[TIMEOUT]${NOCOLOR} Progress monitoring timed out"
      return 1
    fi
    
    sleep 0.5
  done
}

# Test summary
print_test_summary() {
  local elapsed=$(($(date +%s) - TEST_SUITE_START_TIME))
  
  echo
  echo "${BOLD}${BLUE}===== Test Summary =====${NOCOLOR}"
  echo "Total tests:      $TESTS_TOTAL"
  echo "Passed:          ${GREEN}$TESTS_PASSED${NOCOLOR}"
  echo "Failed:          ${RED}$TESTS_FAILED${NOCOLOR}"
  echo "Skipped:         ${YELLOW}$TESTS_SKIPPED${NOCOLOR}"
  echo
  echo "Total assertions: $ASSERTIONS_TOTAL"
  echo "Passed:          ${GREEN}$ASSERTIONS_PASSED${NOCOLOR}"
  echo "Failed:          ${RED}$ASSERTIONS_FAILED${NOCOLOR}"
  echo
  echo "Time elapsed:     ${elapsed}s"
  
  if [[ -n "${TEST_RESULTS_FILE:-}" ]]; then
    echo
    echo "Results saved to: $TEST_RESULTS_FILE"
  fi
  
  # Return failure if any tests failed
  ((TESTS_FAILED == 0))
}

#fin