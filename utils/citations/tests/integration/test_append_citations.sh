#!/bin/bash
# Integration tests for append-citations.sh

# Get test directory
TEST_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Source test configuration
source "$TEST_DIR/test_config.sh"
source "$TEST_DIR/test_helpers.sh"

# Test suite setup
init_test_suite
setup_test_env "append_citations"

echo "Starting append-citations.sh integration tests..."

# Test 1: Basic append functionality
test_basic_append() {
  start_test "Basic citation append to files without frontmatter"
  
  local test_docs="$TEST_CURRENT_DIR/basic"
  local test_db="$TEST_CURRENT_DIR/basic.db"
  
  # Create test documents
  mkdir -p "$test_docs"
  create_test_file "$test_docs/doc1.md" "# Document One\n\nThis is the first document."
  create_test_file "$test_docs/doc2.md" "# Document Two\n\nThis is the second document."
  create_test_file "$test_docs/doc3.md" "# Document Three\n\nThis has no citation in DB."
  
  # Create database with citations
  create_test_database "$test_db"
  add_test_citation "$test_db" "doc1.md" "Document One Title" "Author One" "2023"
  add_test_citation "$test_db" "doc2.md" "Document Two Title" "Author Two" "2024"
  
  # Run append-citations
  local output
  output=$("$APPEND_CITATIONS_SCRIPT" -d "$test_db" "$test_docs" 2>&1)
  local status=$?
  
  assert_equals "0" "$status" "Script should exit successfully"
  
  # Verify frontmatter was added
  assert_file_exists "$test_docs/doc1.md.bak" "Backup should be created for doc1"
  assert_file_exists "$test_docs/doc2.md.bak" "Backup should be created for doc2"
  assert_file_not_exists "$test_docs/doc3.md.bak" "No backup for doc3 (no citation)"
  
  # Check frontmatter content
  local content
  content=$(head -5 "$test_docs/doc1.md")
  assert_contains "$content" "---" "Should have frontmatter markers"
  assert_contains "$content" "Document One Title" "Should have correct title"
  assert_contains "$content" "Author One" "Should have correct author"
  assert_contains "$content" "2023" "Should have correct year"
  
  # Verify original content preserved
  assert_contains "$(cat "$test_docs/doc1.md")" "This is the first document" "Original content preserved"
  
  # Check statistics in output
  assert_contains "$output" "Updated:   2" "Should update 2 files"
  assert_contains "$output" "Skipped:   0" "Should skip 0 files"
  
  end_test 0
}

# Test 2: Dry run mode
test_dry_run() {
  start_test "Dry run mode shows changes without modifying files"
  
  local test_docs="$TEST_CURRENT_DIR/dryrun"
  local test_db="$TEST_CURRENT_DIR/dryrun.db"
  
  # Create test documents
  mkdir -p "$test_docs"
  create_test_file "$test_docs/test.md" "# Test Document\n\nOriginal content."
  
  # Save original content MD5
  local orig_md5
  orig_md5=$(md5sum "$test_docs/test.md" | cut -d' ' -f1)
  
  # Create database
  create_test_database "$test_db"
  add_test_citation "$test_db" "test.md" "Test Title" "Test Author" "2024"
  
  # Run in dry-run mode
  local output
  output=$("$APPEND_CITATIONS_SCRIPT" -d "$test_db" -n "$test_docs" 2>&1)
  
  # Check output
  assert_contains "$output" "Would update: test.md" "Should show what would be updated"
  assert_contains "$output" "Title:  Test Title" "Should show title"
  assert_contains "$output" "Author: Test Author" "Should show author"
  assert_contains "$output" "Year:   2024" "Should show year"
  
  # Verify file unchanged
  local new_md5
  new_md5=$(md5sum "$test_docs/test.md" | cut -d' ' -f1)
  assert_equals "$orig_md5" "$new_md5" "File should not be modified"
  
  assert_file_not_exists "$test_docs/test.md.bak" "No backup in dry-run"
  
  end_test 0
}

# Test 3: Force mode with existing frontmatter
test_force_mode() {
  start_test "Force mode overwrites existing frontmatter"
  
  local test_docs="$TEST_CURRENT_DIR/force"
  local test_db="$TEST_CURRENT_DIR/force.db"
  
  # Create document with existing frontmatter
  mkdir -p "$test_docs"
  cat > "$test_docs/existing.md" << 'EOF'
---
title: "Old Title"
author: "Old Author"
date: "2020"
tags: ["old", "tag"]
---

# Existing Document

Content that should be preserved.
EOF

  # Create database with new citation
  create_test_database "$test_db"
  add_test_citation "$test_db" "existing.md" "New Title" "New Author" "2024"
  
  # Run without force - should skip
  local output
  output=$("$APPEND_CITATIONS_SCRIPT" -d "$test_db" "$test_docs" 2>&1)
  assert_contains "$output" "Skipped:   1" "Should skip file with existing frontmatter"
  assert_contains "$(cat "$test_docs/existing.md")" "Old Title" "Old frontmatter preserved"
  
  # Run with force
  output=$("$APPEND_CITATIONS_SCRIPT" -d "$test_db" -f "$test_docs" 2>&1)
  assert_contains "$output" "Updated:   1" "Should update with force"
  
  # Verify new frontmatter
  local content
  content=$(cat "$test_docs/existing.md")
  assert_contains "$content" "New Title" "Should have new title"
  assert_contains "$content" "New Author" "Should have new author"
  assert_contains "$content" "2024" "Should have new year"
  assert_not_contains "$content" "Old Title" "Old title should be gone"
  assert_not_contains "$content" "tags:" "Extra fields should be removed"
  
  # Verify content preserved
  assert_contains "$content" "Content that should be preserved" "Document content preserved"
  
  end_test 0
}

# Test 4: No backup mode
test_no_backup() {
  start_test "No backup mode skips backup creation"
  
  local test_docs="$TEST_CURRENT_DIR/nobackup"
  local test_db="$TEST_CURRENT_DIR/nobackup.db"
  
  # Create test document
  mkdir -p "$test_docs"
  create_test_file "$test_docs/test.md" "# Test\n\nContent."
  
  # Create database
  create_test_database "$test_db"
  add_test_citation "$test_db" "test.md" "Title" "Author" "2024"
  
  # Run with --no-backup
  "$APPEND_CITATIONS_SCRIPT" -d "$test_db" -b "$test_docs" >/dev/null 2>&1
  
  # Check no backup created
  assert_file_not_exists "$test_docs/test.md.bak" "No backup file should exist"
  
  # But citation should still be added
  assert_contains "$(head -1 "$test_docs/test.md")" "---" "Frontmatter should be added"
  
  end_test 0
}

# Test 5: Multiple directories
test_multiple_directories() {
  start_test "Process multiple directories in one run"
  
  local test_db="$TEST_CURRENT_DIR/multi.db"
  local dir1="$TEST_CURRENT_DIR/dir1"
  local dir2="$TEST_CURRENT_DIR/dir2"
  
  # Create multiple directories
  mkdir -p "$dir1" "$dir2"
  create_test_file "$dir1/file1.md" "# File 1"
  create_test_file "$dir2/file2.md" "# File 2"
  
  # Create database
  create_test_database "$test_db"
  add_test_citation "$test_db" "file1.md" "Title 1" "Author 1" "2023"
  add_test_citation "$test_db" "file2.md" "Title 2" "Author 2" "2024"
  
  # Run on both directories
  local output
  output=$("$APPEND_CITATIONS_SCRIPT" -d "$test_db" "$dir1" "$dir2" 2>&1)
  
  # Both should be processed
  assert_contains "$output" "Processing directory: $dir1" "Should process dir1"
  assert_contains "$output" "Processing directory: $dir2" "Should process dir2"
  assert_contains "$output" "Updated:   2" "Should update 2 files total"
  
  # Check both files updated
  assert_file_exists "$dir1/file1.md.bak" "Dir1 file should have backup"
  assert_file_exists "$dir2/file2.md.bak" "Dir2 file should have backup"
  
  end_test 0
}

# Test 6: Subdirectory handling
test_subdirectories() {
  start_test "Handle files in subdirectories correctly"
  
  local test_docs="$TEST_CURRENT_DIR/subdirs"
  local test_db="$TEST_CURRENT_DIR/subdirs.db"
  
  # Create nested structure
  mkdir -p "$test_docs/level1/level2"
  create_test_file "$test_docs/root.md" "# Root"
  create_test_file "$test_docs/level1/sub1.md" "# Sub1"
  create_test_file "$test_docs/level1/level2/sub2.md" "# Sub2"
  
  # Create database with nested paths
  create_test_database "$test_db"
  add_test_citation "$test_db" "root.md" "Root Title" "Root Author" "2021"
  add_test_citation "$test_db" "level1/sub1.md" "Sub1 Title" "Sub1 Author" "2022"
  add_test_citation "$test_db" "level1/level2/sub2.md" "Sub2 Title" "Sub2 Author" "2023"
  
  # Run append
  "$APPEND_CITATIONS_SCRIPT" -d "$test_db" "$test_docs" >/dev/null 2>&1
  
  # Verify all files updated
  assert_file_exists "$test_docs/root.md.bak" "Root file should have backup"
  assert_file_exists "$test_docs/level1/sub1.md.bak" "Sub1 should have backup"
  assert_file_exists "$test_docs/level1/level2/sub2.md.bak" "Sub2 should have backup"
  
  # Check nested file content
  assert_contains "$(cat "$test_docs/level1/level2/sub2.md")" "Sub2 Title" "Nested file should be updated"
  
  end_test 0
}

# Test 7: Empty citation handling
test_empty_citations() {
  start_test "Handle empty/partial citations gracefully"
  
  local test_docs="$TEST_CURRENT_DIR/empty"
  local test_db="$TEST_CURRENT_DIR/empty.db"
  
  # Create test documents
  mkdir -p "$test_docs"
  create_test_file "$test_docs/partial.md" "# Partial"
  create_test_file "$test_docs/empty.md" "# Empty"
  create_test_file "$test_docs/full.md" "# Full"
  
  # Create database with various citation states
  create_test_database "$test_db"
  add_test_citation "$test_db" "partial.md" "Partial Title" "" "2024"  # No author
  add_test_citation "$test_db" "empty.md" "" "" ""  # All empty
  add_test_citation "$test_db" "full.md" "Full Title" "Full Author" "2024"
  
  # Run append
  local output
  output=$("$APPEND_CITATIONS_SCRIPT" -d "$test_db" -v "$test_docs" 2>&1)
  
  # Empty citation should be skipped
  assert_contains "$output" "Skipped:   1" "Should skip empty citation"
  assert_file_not_exists "$test_docs/empty.md.bak" "Empty should not be processed"
  
  # Partial should be processed
  assert_file_exists "$test_docs/partial.md.bak" "Partial should be processed"
  local content
  content=$(cat "$test_docs/partial.md")
  assert_contains "$content" "Partial Title" "Partial title should be added"
  assert_contains "$content" 'author: ""' "Empty author should be empty string"
  
  end_test 0
}

# Test 8: Special characters and unicode
test_special_characters() {
  start_test "Handle special characters and unicode correctly"
  
  local test_docs="$TEST_CURRENT_DIR/unicode"
  local test_db="$TEST_CURRENT_DIR/unicode.db"
  
  # Create test document
  mkdir -p "$test_docs"
  create_test_file "$test_docs/unicode.md" "# Unicode Test\n\nContent with émojis 🎉"
  
  # Create database with special characters
  create_test_database "$test_db"
  sqlite3 "$test_db" << 'EOF'
INSERT INTO citations (sourcefile, title, author, year, raw_citation) VALUES
  ('unicode.md', 'Title with "quotes" & special chars', 'François O''Neil-Müller', '2024', 'Raw citation');
EOF
  
  # Run append
  "$APPEND_CITATIONS_SCRIPT" -d "$test_db" "$test_docs" >/dev/null 2>&1
  
  # Check content
  local content
  content=$(cat "$test_docs/unicode.md")
  assert_contains "$content" '"quotes"' "Should preserve quotes in title"
  assert_contains "$content" "François" "Should preserve unicode characters"
  assert_contains "$content" "O'Neil-Müller" "Should handle apostrophes and umlauts"
  assert_contains "$content" "émojis 🎉" "Should preserve content with emojis"
  
  end_test 0
}

# Test 9: Error handling
test_error_handling() {
  start_test "Handle various error conditions gracefully"
  
  local test_docs="$TEST_CURRENT_DIR/errors"
  local test_db="$TEST_CURRENT_DIR/errors.db"
  
  # Test 1: Missing database
  local output
  output=$("$APPEND_CITATIONS_SCRIPT" -d "/tmp/nonexistent.db" "$test_docs" 2>&1)
  local status=$?
  assert_not_equals "0" "$status" "Should fail with missing database"
  assert_contains "$output" "Database not found" "Should show database error"
  
  # Test 2: Missing directory
  create_test_database "$test_db"
  output=$("$APPEND_CITATIONS_SCRIPT" -d "$test_db" "/tmp/nonexistent-dir" 2>&1)
  status=$?
  assert_not_equals "0" "$status" "Should fail with missing directory"
  assert_contains "$output" "does not exist" "Should show directory error"
  
  # Test 3: Read-only file
  mkdir -p "$test_docs"
  create_test_file "$test_docs/readonly.md" "# Read Only"
  chmod 444 "$test_docs/readonly.md"
  add_test_citation "$test_db" "readonly.md" "Title" "Author" "2024"
  
  output=$("$APPEND_CITATIONS_SCRIPT" -d "$test_db" "$test_docs" 2>&1)
  assert_contains "$output" "Errors:    1" "Should report error for read-only file"
  
  # Cleanup
  chmod 644 "$test_docs/readonly.md" 2>/dev/null || true
  
  end_test 0
}

# Test 10: Quiet and verbose modes
test_output_modes() {
  start_test "Quiet and verbose output modes"
  
  local test_docs="$TEST_CURRENT_DIR/output"
  local test_db="$TEST_CURRENT_DIR/output.db"
  
  # Create test setup
  mkdir -p "$test_docs"
  create_test_file "$test_docs/test.md" "# Test"
  create_test_database "$test_db"
  add_test_citation "$test_db" "test.md" "Title" "Author" "2024"
  
  # Test quiet mode
  local output
  output=$("$APPEND_CITATIONS_SCRIPT" -d "$test_db" -q "$test_docs" 2>&1)
  local line_count
  line_count=$(echo "$output" | wc -l)
  assert_true "$((line_count < 3))" "Quiet mode should have minimal output"
  
  # Test verbose mode
  output=$("$APPEND_CITATIONS_SCRIPT" -d "$test_db" -v -v "$test_docs" 2>&1)
  assert_contains "$output" "Processing directory:" "Verbose should show directory"
  assert_contains "$output" "Found .* citations in database" "Verbose should show count"
  assert_contains "$output" "Processing:" "Verbose should show progress"
  
  # Test debug mode
  output=$("$APPEND_CITATIONS_SCRIPT" -d "$test_db" -D "$test_docs" 2>&1)
  assert_contains "$output" "debug:" "Debug mode should show debug messages"
  
  end_test 0
}

# Test 11: Combined options
test_combined_options() {
  start_test "Combined command line options (e.g., -nfv)"
  
  local test_docs="$TEST_CURRENT_DIR/combined"
  local test_db="$TEST_CURRENT_DIR/combined.db"
  
  # Create test with existing frontmatter
  mkdir -p "$test_docs"
  cat > "$test_docs/test.md" << 'EOF'
---
title: "Old"
---
Content
EOF
  
  create_test_database "$test_db"
  add_test_citation "$test_db" "test.md" "New" "Author" "2024"
  
  # Test combined options: -nfv (dry-run, force, verbose)
  local output
  output=$("$APPEND_CITATIONS_SCRIPT" -d "$test_db" -nfv "$test_docs" 2>&1)
  
  # Should show verbose dry-run with force
  assert_contains "$output" "Would update: test.md" "Should show dry-run update"
  assert_contains "$output" "Processing directory:" "Should be verbose"
  assert_contains "$(cat "$test_docs/test.md")" "Old" "File should not be modified (dry-run)"
  
  # Test combined: -fb (force, no-backup)
  output=$("$APPEND_CITATIONS_SCRIPT" -d "$test_db" -fb "$test_docs" 2>&1)
  assert_file_not_exists "$test_docs/test.md.bak" "Should not create backup"
  assert_contains "$(cat "$test_docs/test.md")" "New" "Should update with force"
  
  end_test 0
}

# Test 12: File path edge cases
test_file_path_edge_cases() {
  start_test "Handle file paths with spaces and special characters"
  
  local test_docs="$TEST_CURRENT_DIR/paths with spaces"
  local test_db="$TEST_CURRENT_DIR/paths.db"
  
  # Create directory and files with spaces
  mkdir -p "$test_docs/sub dir"
  create_test_file "$test_docs/file with spaces.md" "# Spaces"
  create_test_file "$test_docs/sub dir/nested file.md" "# Nested"
  
  # Create database entries
  create_test_database "$test_db"
  add_test_citation "$test_db" "file with spaces.md" "Title Spaces" "Author" "2024"
  add_test_citation "$test_db" "sub dir/nested file.md" "Title Nested" "Author" "2024"
  
  # Run append
  "$APPEND_CITATIONS_SCRIPT" -d "$test_db" "$test_docs" >/dev/null 2>&1
  
  # Verify files were processed
  assert_file_exists "$test_docs/file with spaces.md.bak" "Should handle spaces in filename"
  assert_file_exists "$test_docs/sub dir/nested file.md.bak" "Should handle spaces in path"
  
  end_test 0
}

# Run all tests
test_basic_append
test_dry_run
test_force_mode
test_no_backup
test_multiple_directories
test_subdirectories
test_empty_citations
test_special_characters
test_error_handling
test_output_modes
test_combined_options
test_file_path_edge_cases

# Print summary
print_test_summary

#fin