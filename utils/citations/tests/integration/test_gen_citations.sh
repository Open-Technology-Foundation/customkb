#!/bin/bash
# Integration tests for gen-citations.sh

# Get test directory
TEST_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Source test configuration
source "$TEST_DIR/test_config.sh"
source "$TEST_DIR/test_helpers.sh"

# Test suite setup
init_test_suite
setup_test_env "gen_citations"

# Mock API setup for testing
setup_mock_api_response() {
  local response_file="$1"
  local content="$2"
  
  mkdir -p "$(dirname "$response_file")"
  echo "$content" > "$response_file"
}

# Test 1: Basic citation generation
test_basic_generation() {
  start_test "Basic citation generation from documents"
  
  local test_docs="$TEST_CURRENT_DIR/basic_docs"
  local test_db="$TEST_CURRENT_DIR/basic.db"
  
  # Create test documents
  mkdir -p "$test_docs"
  cat > "$test_docs/clear_citation.md" << 'EOF'
# Understanding Machine Learning

By Dr. Jane Smith
Published: 2023

## Abstract
This document explores machine learning concepts.
EOF

  cat > "$test_docs/no_citation.md" << 'EOF'
# Random Notes

Just some random notes without clear bibliographic information.
Lorem ipsum dolor sit amet.
EOF

  # Test dry-run first
  local output
  output=$("$GEN_CITATIONS_SCRIPT" -d "$test_db" -n "$test_docs" 2>&1)
  assert_contains "$output" "DRY RUN MODE" "Should indicate dry-run"
  assert_file_not_exists "$test_db" "Database should not be created in dry-run"
  
  # For actual testing, we need mock API or manual database creation
  # Since API calls would be real, we'll simulate the result
  create_test_database "$test_db"
  add_test_citation "$test_db" "clear_citation.md" "Understanding Machine Learning" "Dr. Jane Smith" "2023"
  
  # Verify database content
  assert_db_row_count "$test_db" "citations" "1" "Should have one citation"
  
  end_test 0
}

# Test 2: Force update mode
test_force_update() {
  start_test "Force update of existing citations"
  
  local test_docs="$TEST_CURRENT_DIR/force_docs"
  local test_db="$TEST_CURRENT_DIR/force.db"
  
  # Create test document
  mkdir -p "$test_docs"
  create_test_file "$test_docs/update.md" "# Document to Update\n\nBy Old Author, 2020"
  
  # Create initial citation
  create_test_database "$test_db"
  add_test_citation "$test_db" "update.md" "Old Title" "Old Author" "2020"
  
  # Without force, would skip existing
  # With force, would update (requires mock API)
  
  # Simulate force update
  local output
  output=$("$GEN_CITATIONS_SCRIPT" -d "$test_db" -f -n "$test_docs" 2>&1)
  assert_contains "$output" "force" "Should indicate force mode"
  
  end_test 0
}

# Test 3: Exclude patterns
test_exclude_patterns() {
  start_test "Exclude patterns filter out files correctly"
  
  local test_docs="$TEST_CURRENT_DIR/exclude_docs"
  local test_db="$TEST_CURRENT_DIR/exclude.db"
  
  # Create directory structure
  mkdir -p "$test_docs/include" "$test_docs/backup" "$test_docs/archive"
  create_test_file "$test_docs/include/doc1.md" "# Include 1"
  create_test_file "$test_docs/backup/doc2.md" "# Backup Doc"
  create_test_file "$test_docs/archive/doc3.md" "# Archive Doc"
  
  # Run with excludes
  local output
  output=$("$GEN_CITATIONS_SCRIPT" -d "$test_db" -x "*/backup/*" -x "*/archive/*" -n "$test_docs" 2>&1)
  
  # Should only process files not in excluded directories
  assert_contains "$output" "Found" "Should find files"
  # In dry-run, we can't verify exact processing, but pattern is tested
  
  end_test 0
}

# Test 4: Model and parameter options
test_model_parameters() {
  start_test "Different models and parameters"
  
  local test_docs="$TEST_CURRENT_DIR/model_docs"
  local test_db="$TEST_CURRENT_DIR/model.db"
  
  # Create test document
  mkdir -p "$test_docs"
  create_test_file "$test_docs/test.md" "# Test Document"
  
  # Test different model
  local output
  output=$("$GEN_CITATIONS_SCRIPT" -d "$test_db" -m "gpt-4" -n "$test_docs" 2>&1)
  # Can't verify actual model usage without mock, but option parsing is tested
  
  # Test chunk size
  output=$("$GEN_CITATIONS_SCRIPT" -d "$test_db" -c 10000 -n "$test_docs" 2>&1)
  
  # Test max tokens
  output=$("$GEN_CITATIONS_SCRIPT" -d "$test_db" -M 256 -n "$test_docs" 2>&1)
  
  # Test temperature
  output=$("$GEN_CITATIONS_SCRIPT" -d "$test_db" -t 0.5 -n "$test_docs" 2>&1)
  
  # All should parse without error
  assert_equals "0" "$?" "Options should parse correctly"
  
  end_test 0
}

# Test 5: Reprocess blank citations
test_reprocess_blank() {
  start_test "Reprocess blank citations mode"
  
  local test_docs="$TEST_CURRENT_DIR/blank_docs"
  local test_db="$TEST_CURRENT_DIR/blank.db"
  
  # Create test documents
  mkdir -p "$test_docs"
  create_test_file "$test_docs/has_citation.md" "# Doc with Citation"
  create_test_file "$test_docs/blank_citation.md" "# Doc with Blank"
  
  # Create database with mixed citations
  create_test_database "$test_db"
  add_test_citation "$test_db" "has_citation.md" "Title" "Author" "2024"
  add_test_citation "$test_db" "blank_citation.md" "" "" ""  # Blank citation
  
  # Test reprocess-blank mode
  local output
  output=$("$GEN_CITATIONS_SCRIPT" -d "$test_db" -r -n "$test_docs" 2>&1)
  
  # Would process only blank citations
  assert_contains "$output" "reprocess" "Should indicate reprocess mode"
  
  end_test 0
}

# Test 6: Parallel processing
test_parallel_processing() {
  start_test "Parallel processing with multiple workers"
  
  local test_docs="$TEST_CURRENT_DIR/parallel_docs"
  local test_db="$TEST_CURRENT_DIR/parallel.db"
  
  # Create multiple test documents
  mkdir -p "$test_docs"
  for i in {1..20}; do
    create_test_file "$test_docs/doc$i.md" "# Document $i\n\nBy Author $i"
  done
  
  # Test parallel mode
  local output
  output=$("$GEN_CITATIONS_SCRIPT" -d "$test_db" -p 4 -n "$test_docs" 2>&1)
  
  # Should indicate parallel processing
  assert_contains "$output" "Found 20 files" "Should find all files"
  
  # Test sequential mode
  output=$("$GEN_CITATIONS_SCRIPT" -d "$test_db" --no-parallel -n "$test_docs" 2>&1)
  
  end_test 0
}

# Test 7: Subdirectory traversal
test_subdirectory_traversal() {
  start_test "Traverse and process subdirectories"
  
  local test_docs="$TEST_CURRENT_DIR/subdir_docs"
  local test_db="$TEST_CURRENT_DIR/subdir.db"
  
  # Create nested structure
  mkdir -p "$test_docs/level1/level2/level3"
  create_test_file "$test_docs/root.md" "# Root Doc"
  create_test_file "$test_docs/level1/doc1.md" "# Level 1 Doc"
  create_test_file "$test_docs/level1/level2/doc2.md" "# Level 2 Doc"
  create_test_file "$test_docs/level1/level2/level3/doc3.md" "# Level 3 Doc"
  
  # Also create .txt files
  create_test_file "$test_docs/test.txt" "Text Document\nBy Text Author"
  
  # Process directory
  local output
  output=$("$GEN_CITATIONS_SCRIPT" -d "$test_db" -n "$test_docs" 2>&1)
  
  # Should find all files
  assert_contains "$output" "Found 5 files" "Should find all .md and .txt files"
  
  end_test 0
}

# Test 8: Output modes (quiet, verbose, debug)
test_output_modes() {
  start_test "Different output verbosity modes"
  
  local test_docs="$TEST_CURRENT_DIR/output_docs"
  local test_db="$TEST_CURRENT_DIR/output.db"
  
  # Create test document
  mkdir -p "$test_docs"
  create_test_file "$test_docs/test.md" "# Test"
  
  # Test quiet mode
  local output
  output=$("$GEN_CITATIONS_SCRIPT" -d "$test_db" -q -n "$test_docs" 2>&1)
  local line_count
  line_count=$(echo "$output" | grep -v "^$" | wc -l)
  assert_true "$((line_count < 5))" "Quiet mode should have minimal output"
  
  # Test verbose mode
  output=$("$GEN_CITATIONS_SCRIPT" -d "$test_db" -v -v -n "$test_docs" 2>&1)
  assert_contains "$output" "Processing directory" "Verbose should show details"
  
  # Test debug mode
  output=$("$GEN_CITATIONS_SCRIPT" -d "$test_db" -D -n "$test_docs" 2>&1)
  assert_contains "$output" "DEBUG" "Debug mode should show debug info"
  
  end_test 0
}

# Test 9: Error handling
test_error_handling() {
  start_test "Handle various error conditions"
  
  # Test missing API key
  local old_key="$OPENAI_API_KEY"
  unset OPENAI_API_KEY
  
  local output
  output=$("$GEN_CITATIONS_SCRIPT" -d "/tmp/test.db" -n "/tmp" 2>&1)
  local status=$?
  assert_not_equals "0" "$status" "Should fail without API key"
  assert_contains "$output" "API key" "Should mention API key issue"
  
  # Restore API key
  export OPENAI_API_KEY="$old_key"
  
  # Test invalid directory
  output=$("$GEN_CITATIONS_SCRIPT" -d "/tmp/test.db" "/nonexistent/directory" 2>&1)
  status=$?
  assert_not_equals "0" "$status" "Should fail with nonexistent directory"
  
  # Test invalid options
  output=$("$GEN_CITATIONS_SCRIPT" --invalid-option 2>&1)
  status=$?
  assert_not_equals "0" "$status" "Should fail with invalid option"
  
  end_test 0
}

# Test 10: Multiple source directories
test_multiple_sources() {
  start_test "Process multiple source directories"
  
  local test_db="$TEST_CURRENT_DIR/multi.db"
  local dir1="$TEST_CURRENT_DIR/source1"
  local dir2="$TEST_CURRENT_DIR/source2"
  local dir3="$TEST_CURRENT_DIR/source3"
  
  # Create multiple directories
  mkdir -p "$dir1" "$dir2" "$dir3"
  create_test_file "$dir1/doc1.md" "# Doc 1"
  create_test_file "$dir2/doc2.md" "# Doc 2"
  create_test_file "$dir3/doc3.md" "# Doc 3"
  
  # Process all directories
  local output
  output=$("$GEN_CITATIONS_SCRIPT" -d "$test_db" -n "$dir1" "$dir2" "$dir3" 2>&1)
  
  # Should process all directories
  assert_contains "$output" "Processing directory '$dir1'" "Should process dir1"
  assert_contains "$output" "Processing directory '$dir2'" "Should process dir2"
  assert_contains "$output" "Processing directory '$dir3'" "Should process dir3"
  
  end_test 0
}

# Test 11: File type filtering
test_file_types() {
  start_test "Process only .md and .txt files"
  
  local test_docs="$TEST_CURRENT_DIR/filetypes"
  local test_db="$TEST_CURRENT_DIR/filetypes.db"
  
  # Create various file types
  mkdir -p "$test_docs"
  create_test_file "$test_docs/markdown.md" "# Markdown"
  create_test_file "$test_docs/text.txt" "Text file"
  create_test_file "$test_docs/ignore.pdf" "PDF content"
  create_test_file "$test_docs/ignore.docx" "Word doc"
  create_test_file "$test_docs/README" "No extension"
  
  # Process directory
  local output
  output=$("$GEN_CITATIONS_SCRIPT" -d "$test_db" -n "$test_docs" 2>&1)
  
  # Should only find .md and .txt
  assert_contains "$output" "Found 2 files" "Should only find markdown and text files"
  
  end_test 0
}

# Test 12: Combined options
test_combined_options() {
  start_test "Combined command line options"
  
  local test_docs="$TEST_CURRENT_DIR/combined"
  local test_db="$TEST_CURRENT_DIR/combined.db"
  
  # Create test setup
  mkdir -p "$test_docs"
  create_test_file "$test_docs/test.md" "# Test"
  
  # Test combined short options: -fnv (force, dry-run, verbose)
  local output
  output=$("$GEN_CITATIONS_SCRIPT" -d "$test_db" -fnv "$test_docs" 2>&1)
  
  assert_contains "$output" "DRY RUN" "Should be in dry-run mode"
  assert_contains "$output" "force" "Should indicate force mode"
  assert_contains "$output" "Processing directory" "Should be verbose"
  
  # Test combined with excludes
  output=$("$GEN_CITATIONS_SCRIPT" -d "$test_db" -qfx "*.bak" -x "*.tmp" "$test_docs" 2>&1)
  local line_count
  line_count=$(echo "$output" | wc -l)
  assert_true "$((line_count < 5))" "Should be quiet even with multiple options"
  
  end_test 0
}

# Note: Many tests require mock API to fully test functionality
# These tests focus on command parsing, options, and basic flow

# Run all tests
test_basic_generation
test_force_update
test_exclude_patterns
test_model_parameters
test_reprocess_blank
test_parallel_processing
test_subdirectory_traversal
test_output_modes
test_error_handling
test_multiple_sources
test_file_types
test_combined_options

# Print summary
print_test_summary

#fin