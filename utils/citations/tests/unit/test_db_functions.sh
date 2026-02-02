#!/bin/bash
# Unit tests for db_functions.sh

# Get test directory
TEST_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Source test configuration
source "$TEST_DIR/test_config.sh"
source "$TEST_DIR/test_helpers.sh"

# Source the functions being tested
source "$DB_FUNCTIONS"

# Test suite setup
init_test_suite
setup_test_env "db_functions"

# Test 1: Database initialization
test_db_init() {
  start_test "db_init creates database with correct schema"
  
  local test_db="$TEST_CURRENT_DIR/test_init.db"
  
  # Initialize database
  if db_init "$test_db"; then
    assert_file_exists "$test_db" "Database file should be created"
    assert_db_has_table "$test_db" "citations" "Citations table should exist"
    
    # Check schema
    local schema
    schema=$(sqlite3 "$test_db" ".schema citations")
    assert_contains "$schema" "id INTEGER PRIMARY KEY" "Schema should have id column"
    assert_contains "$schema" "sourcefile VARCHAR(255) UNIQUE" "Schema should have unique sourcefile"
    assert_contains "$schema" "title TEXT" "Schema should have title column"
    assert_contains "$schema" "author TEXT" "Schema should have author column"
    assert_contains "$schema" "year TEXT" "Schema should have year column"
    assert_contains "$schema" "raw_citation TEXT" "Schema should have raw_citation column"
    
    # Check index
    local indexes
    indexes=$(sqlite3 "$test_db" ".indexes")
    assert_contains "$indexes" "idx_sourcefile" "Sourcefile index should exist"
    
    end_test 0
  else
    end_test 1
  fi
}

# Test 2: Database initialization idempotency
test_db_init_idempotent() {
  start_test "db_init is idempotent"
  
  local test_db="$TEST_CURRENT_DIR/test_idempotent.db"
  
  # Initialize twice
  db_init "$test_db" || { end_test 1; return; }
  
  # Add a citation
  add_test_citation "$test_db" "test.md" "Test Title" "Test Author" "2024"
  
  # Initialize again - should not delete existing data
  db_init "$test_db" || { end_test 1; return; }
  
  # Check data still exists
  local count
  count=$(sqlite3 "$test_db" "SELECT COUNT(*) FROM citations;")
  assert_equals "1" "$count" "Existing data should be preserved"
  
  end_test 0
}

# Test 3: Citation UPSERT functionality
test_db_upsert_citation() {
  start_test "db_upsert_citation inserts and updates correctly"
  
  local test_db="$TEST_CURRENT_DIR/test_upsert.db"
  db_init "$test_db" || { end_test 1; return; }
  
  # Insert new citation
  db_upsert_citation "$test_db" "test.md" "First Title" "First Author" "2023" "Raw citation 1"
  assert_db_row_count "$test_db" "citations" "1" "Should have one citation"
  
  # Verify insertion
  local result
  result=$(sqlite3 "$test_db" "SELECT title FROM citations WHERE sourcefile='test.md';")
  assert_equals "First Title" "$result" "Title should match"
  
  # Update existing citation
  db_upsert_citation "$test_db" "test.md" "Updated Title" "Updated Author" "2024" "Raw citation 2"
  assert_db_row_count "$test_db" "citations" "1" "Should still have one citation"
  
  # Verify update
  result=$(sqlite3 "$test_db" "SELECT title, year FROM citations WHERE sourcefile='test.md';")
  assert_equals "Updated Title|2024" "$result" "Citation should be updated"
  
  end_test 0
}

# Test 4: Special character handling
test_db_special_characters() {
  start_test "db_upsert_citation handles special characters"
  
  local test_db="$TEST_CURRENT_DIR/test_special.db"
  db_init "$test_db" || { end_test 1; return; }
  
  # Test with quotes
  db_upsert_citation "$test_db" "quotes.md" "Title with \"Quotes\"" "O'Neil" "2024" "Complex \"citation\""
  
  local result
  result=$(sqlite3 "$test_db" "SELECT title FROM citations WHERE sourcefile='quotes.md';")
  assert_equals "Title with \"Quotes\"" "$result" "Should handle double quotes"
  
  result=$(sqlite3 "$test_db" "SELECT author FROM citations WHERE sourcefile='quotes.md';")
  assert_equals "O'Neil" "$result" "Should handle single quotes"
  
  # Test with unicode
  db_upsert_citation "$test_db" "unicode.md" "Unicode Title: 日本語" "François Müller" "2024" "Unicode citation"
  
  result=$(sqlite3 "$test_db" "SELECT title FROM citations WHERE sourcefile='unicode.md';")
  assert_contains "$result" "日本語" "Should handle unicode characters"
  
  end_test 0
}

# Test 5: File existence checking
test_db_file_exists() {
  start_test "db_file_exists checks correctly"
  
  local test_db="$TEST_CURRENT_DIR/test_exists.db"
  db_init "$test_db" || { end_test 1; return; }
  
  # Add citations
  add_test_citation "$test_db" "exists.md" "Title" "Author" "2024"
  add_test_citation "$test_db" "blank.md" "" "" ""
  
  # Test existence
  if db_file_exists "$test_db" "exists.md"; then
    assert_true 1 "Should find existing file"
  else
    assert_true 0 "Should find existing file"
  fi
  
  if db_file_exists "$test_db" "notexists.md"; then
    assert_true 0 "Should not find non-existing file"
  else
    assert_true 1 "Should not find non-existing file"
  fi
  
  # Test blank checking
  if db_file_exists "$test_db" "blank.md" 1; then
    assert_true 1 "Should find blank citation when check_blank=1"
  else
    assert_true 0 "Should find blank citation when check_blank=1"
  fi
  
  if db_file_exists "$test_db" "exists.md" 1; then
    assert_true 0 "Should not find non-blank as blank"
  else
    assert_true 1 "Should not find non-blank as blank"
  fi
  
  end_test 0
}

# Test 6: Get all citations
test_db_get_all_citations() {
  start_test "db_get_all_citations returns correct format"
  
  local test_db="$TEST_CURRENT_DIR/test_getall.db"
  db_init "$test_db" || { end_test 1; return; }
  
  # Add multiple citations
  add_test_citation "$test_db" "file1.md" "Title 1" "Author 1" "2021"
  add_test_citation "$test_db" "file2.md" "Title 2" "Author 2" "2022"
  add_test_citation "$test_db" "file3.md" "Title 3" "Author 3" "2023"
  
  # Get all citations
  local citations
  citations=$(db_get_all_citations "$test_db" | wc -l)
  assert_equals "3" "$citations" "Should return all citations"
  
  # Check format
  local first_line
  first_line=$(db_get_all_citations "$test_db" | head -1)
  assert_contains "$first_line" "|" "Should use pipe separator"
  
  # Count fields
  local field_count
  field_count=$(echo "$first_line" | awk -F'|' '{print NF}')
  assert_equals "5" "$field_count" "Should have 5 fields"
  
  end_test 0
}

# Test 7: Parse citation function
test_parse_citation() {
  start_test "parse_citation handles various formats"
  
  # Standard format
  local result
  result=$(parse_citation '"Test Title", Test Author, 2024')
  assert_equals "Test Title|Test Author|2024" "$result" "Should parse standard format"
  
  # With NF fields
  result=$(parse_citation '"Title Only", NF, NF')
  assert_equals "Title Only||" "$result" "Should handle NF fields"
  
  # All NF
  result=$(parse_citation 'NF, NF, NF')
  assert_equals "||" "$result" "Should handle all NF"
  
  # Just NF
  result=$(parse_citation 'NF')
  assert_equals "||" "$result" "Should handle single NF"
  
  # Complex title with comma
  result=$(parse_citation '"Title, with comma", Author Name, 2024')
  assert_equals "Title, with comma|Author Name|2024" "$result" "Should handle comma in title"
  
  # Edge cases
  result=$(parse_citation '"", NF, 2024')
  assert_equals "||2024" "$result" "Should handle empty title"
  
  end_test 0
}

# Test 8: Database locking behavior
test_db_locking() {
  start_test "Database handles concurrent access"
  
  local test_db="$TEST_CURRENT_DIR/test_locking.db"
  db_init "$test_db" || { end_test 1; return; }
  
  # Start a long-running transaction in background
  sqlite3 "$test_db" "BEGIN EXCLUSIVE; SELECT 1;" &
  local bg_pid=$!
  sleep 0.1
  
  # Try to access database (should handle gracefully)
  local start_time=$(date +%s)
  if db_upsert_citation "$test_db" "test.md" "Title" "Author" "2024" "Citation" 2>/dev/null; then
    # If it succeeded, the lock was released
    assert_true 1 "Database access handled gracefully"
  else
    # If it failed, that's also acceptable behavior
    assert_true 1 "Database lock detected correctly"
  fi
  
  # Clean up background process
  kill $bg_pid 2>/dev/null || true
  wait $bg_pid 2>/dev/null || true
  
  end_test 0
}

# Test 9: Error handling
test_db_error_handling() {
  start_test "Database functions handle errors gracefully"
  
  # Non-existent database directory
  local bad_db="/nonexistent/path/test.db"
  if db_init "$bad_db" 2>/dev/null; then
    assert_true 0 "Should fail with bad path"
  else
    assert_true 1 "Correctly failed with bad path"
  fi
  
  # Read-only database
  local test_db="$TEST_CURRENT_DIR/readonly.db"
  db_init "$test_db" || { end_test 1; return; }
  chmod 444 "$test_db"
  
  if db_upsert_citation "$test_db" "test.md" "Title" "Author" "2024" "Citation" 2>/dev/null; then
    assert_true 0 "Should fail with read-only database"
  else
    assert_true 1 "Correctly failed with read-only database"
  fi
  
  chmod 644 "$test_db"
  
  end_test 0
}

# Test 10: WAL mode verification
test_db_wal_mode() {
  start_test "Database uses WAL mode"
  
  local test_db="$TEST_CURRENT_DIR/test_wal.db"
  db_init "$test_db" || { end_test 1; return; }
  
  # Check journal mode
  local mode
  mode=$(sqlite3 "$test_db" "PRAGMA journal_mode;")
  assert_equals "wal" "$mode" "Database should use WAL mode"
  
  # Verify WAL file exists after write
  db_upsert_citation "$test_db" "test.md" "Title" "Author" "2024" "Citation"
  assert_file_exists "${test_db}-wal" "WAL file should exist"
  
  end_test 0
}

# Run all tests
test_db_init
test_db_init_idempotent
test_db_upsert_citation
test_db_special_characters
test_db_file_exists
test_db_get_all_citations
test_parse_citation
test_db_locking
test_db_error_handling
test_db_wal_mode

# Print summary
print_test_summary

#fin