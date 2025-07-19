# Citation System Test Suite

Comprehensive test suite for the citation extraction and application system.

## Overview

This test suite provides thorough testing coverage for:
- `gen-citations.sh` - Citation extraction from documents
- `append-citations.sh` - Citation application to documents
- All supporting library functions

## Quick Start

```bash
# Run all tests
./run_tests.sh

# Run only unit tests
./run_tests.sh --unit

# Run with verbose output
./run_tests.sh -v

# Run quick smoke tests
./run_tests.sh --quick
```

## Test Structure

```
tests/
├── run_tests.sh              # Main test runner
├── test_config.sh            # Shared configuration
├── test_helpers.sh           # Test utilities and assertions
├── fixtures/                 # Test data
│   ├── documents/           # Sample documents
│   ├── databases/           # Pre-populated databases
│   └── api_responses/       # Mock API responses
├── unit/                    # Unit tests
│   ├── test_db_functions.sh
│   └── test_api_functions.sh
├── integration/             # Integration tests
│   └── test_workflow.sh
├── performance/             # Performance tests (optional)
├── edge_cases/              # Edge case tests (optional)
└── mocks/                   # Mock servers
    └── mock_openai_api.sh
```

## Running Tests

### Basic Usage

```bash
# Run all tests
./run_tests.sh

# Run specific test suites
./run_tests.sh --unit              # Unit tests only
./run_tests.sh --integration       # Integration tests only
./run_tests.sh --performance       # Performance tests only

# Run with options
./run_tests.sh -v                  # Verbose output
./run_tests.sh --filter "api"      # Run tests matching pattern
./run_tests.sh --preserve          # Keep test workspace after completion
```

### Mock API Server

Some tests require the mock OpenAI API server to be running:

```bash
# Start mock API server (in separate terminal)
./mocks/mock_openai_api.sh

# The server listens on port 8888 by default
# Stop with Ctrl+C
```

### Individual Test Execution

You can run individual test files directly:

```bash
# Run specific unit test
./unit/test_db_functions.sh

# Run with verbose output
TEST_VERBOSE=1 ./unit/test_api_functions.sh
```

## Writing Tests

### Test Structure

```bash
#!/bin/bash
# Test description

# Source test framework
TEST_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$TEST_DIR/test_config.sh"
source "$TEST_DIR/test_helpers.sh"

# Initialize test suite
init_test_suite
setup_test_env "test_name"

# Define test function
test_example() {
  start_test "Example test case"
  
  # Test code here
  assert_equals "expected" "actual" "Values should match"
  
  end_test 0  # 0 for pass, 1 for fail
}

# Run tests
test_example

# Print summary
print_test_summary
```

### Available Assertions

```bash
# Basic assertions
assert_equals "expected" "actual" "message"
assert_not_equals "val1" "val2" "message"
assert_true $condition "message"
assert_false $condition "message"

# String assertions
assert_contains "haystack" "needle" "message"
assert_not_contains "haystack" "needle" "message"

# File assertions
assert_file_exists "/path/to/file" "message"
assert_file_not_exists "/path/to/file" "message"
assert_dir_exists "/path/to/dir" "message"
assert_files_equal "file1" "file2" "message"

# Command assertions
assert_command_success "command" "message"
assert_command_fails "command" "message"
assert_output_contains "command" "expected" "message"

# Database assertions
assert_db_has_table "db.db" "table_name" "message"
assert_db_row_count "db.db" "table" "5" "message"
```

### Test Helpers

```bash
# Create test data
create_test_file "/path/to/file" "content"
create_test_database "/path/to/db.db"
add_test_citation "db.db" "file.md" "Title" "Author" "Year"

# Mock API management
start_mock_api "port" "responses_dir"
stop_mock_api "port"
check_mock_api  # Returns 0 if available

# Progress monitoring
monitor_progress "progress_file" "expected_total" "timeout"
```

## Test Coverage

### Unit Tests

- **Database Functions** (`test_db_functions.sh`)
  - Database initialization
  - Citation UPSERT operations
  - Special character handling
  - File existence checking
  - Error handling

- **API Functions** (`test_api_functions.sh`)
  - API key validation
  - Rate limiting
  - API calls and retries
  - Result extraction
  - Error handling

### Integration Tests

- **Complete Workflow** (`test_workflow.sh`)
  - End-to-end citation extraction and application
  - Dry run mode
  - Force mode
  - Subdirectory handling
  - Error recovery
  - Unicode support

### Performance Tests (Optional)

- Large dataset processing
- Parallel vs sequential performance
- Resource usage monitoring

## Environment Variables

```bash
# Test configuration
TEST_VERBOSE=1              # Enable verbose output
TEST_DEBUG=1                # Enable debug output
TEST_TIMEOUT=300            # Test timeout in seconds
PRESERVE_TEST_ENV=1         # Keep test workspace

# Mock API configuration
CITATION_USE_MOCK_API=1     # Use mock instead of real API
MOCK_API_PORT=8888          # Mock API port
```

## Troubleshooting

### Common Issues

1. **Mock API not running**
   ```
   Note: Mock API server not running. Some tests may be skipped.
   Start it with: ./mocks/mock_openai_api.sh
   ```

2. **Permission denied**
   ```bash
   # Make scripts executable
   chmod +x run_tests.sh
   chmod +x unit/*.sh
   chmod +x integration/*.sh
   ```

3. **Test failures**
   - Check test output for specific assertion failures
   - Run with `-v` for verbose output
   - Check log files in test workspace

### Debug Mode

```bash
# Run with debug output
TEST_DEBUG=1 ./run_tests.sh

# Preserve test workspace for inspection
./run_tests.sh --preserve

# Check test logs
ls -la /tmp/citation-tests-*/logs/
```

## Continuous Integration

The test suite is designed to work with CI/CD systems:

```bash
# Exit codes
# 0 - All tests passed
# 1 - One or more tests failed

# Machine-readable output
./run_tests.sh > test_results.txt 2>&1
echo "Exit code: $?"
```

## Contributing

When adding new functionality:

1. Add unit tests for new functions
2. Add integration tests for new features
3. Update existing tests if behavior changes
4. Ensure all tests pass before committing

## Future Enhancements

- [ ] Coverage reporting integration
- [ ] Parallel test execution optimization
- [ ] Performance benchmarking suite
- [ ] Stress testing scenarios
- [ ] Additional mock services

#fin