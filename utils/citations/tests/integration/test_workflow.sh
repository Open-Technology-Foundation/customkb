#!/bin/bash
# Integration tests for complete citation workflow

# Get test directory
TEST_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Source test configuration
source "$TEST_DIR/test_config.sh"
source "$TEST_DIR/test_helpers.sh"

# Test suite setup
init_test_suite
setup_test_env "workflow"

# Create test documents
create_test_documents() {
  local doc_dir="$1"
  
  mkdir -p "$doc_dir/subdir"
  
  # Simple document
  cat > "$doc_dir/simple.md" << 'EOF'
# Simple Document

By John Doe
Published: 2023

This is a simple test document.
EOF

  # Document with existing frontmatter
  cat > "$doc_dir/with_frontmatter.md" << 'EOF'
---
title: "Old Title"
author: "Old Author"
date: "2020"
---

# Document with Frontmatter

This document already has frontmatter.
EOF

  # Document in subdirectory
  cat > "$doc_dir/subdir/nested.md" << 'EOF'
# Nested Document

Author: Jane Smith
Year: 2024

Document in a subdirectory.
EOF

  # Document without clear citation
  cat > "$doc_dir/no_citation.md" << 'EOF'
# Random Notes

Just some random notes without bibliographic information.
Lorem ipsum dolor sit amet.
EOF
}

# Test 1: Basic workflow - generate and append citations
test_basic_workflow() {
  start_test "Basic workflow: generate and append citations"
  
  local test_docs="$TEST_CURRENT_DIR/documents"
  local test_db="$TEST_CURRENT_DIR/workflow.db"
  
  # Create test documents
  create_test_documents "$test_docs"
  
  # Skip if no mock API available
  if ! check_mock_api; then
    skip_test "Mock API not available"
    return
  fi
  
  # Generate citations
  echo "Generating citations..."
  if "$GEN_CITATIONS_SCRIPT" -d "$test_db" -n "$test_docs" >/dev/null 2>&1; then
    assert_true 1 "Dry run should succeed"
  else
    assert_true 0 "Dry run should succeed"
  fi
  
  # For actual test, we need to mock the API responses
  # This would require the mock API server to be running
  # For now, we'll manually add test citations
  
  echo "Adding test citations to database..."
  create_test_database "$test_db"
  add_test_citation "$test_db" "simple.md" "Simple Document" "John Doe" "2023"
  add_test_citation "$test_db" "subdir/nested.md" "Nested Document" "Jane Smith" "2024"
  
  assert_db_row_count "$test_db" "citations" "2" "Should have 2 citations"
  
  # Append citations to documents
  echo "Appending citations..."
  if "$APPEND_CITATIONS_SCRIPT" -d "$test_db" "$test_docs" >/dev/null 2>&1; then
    assert_true 1 "Append should succeed"
    
    # Verify frontmatter was added
    if head -1 "$test_docs/simple.md" | grep -q "^---$"; then
      assert_true 1 "Simple.md should have frontmatter"
    else
      assert_true 0 "Simple.md should have frontmatter"
    fi
    
    # Verify existing frontmatter preserved
    if grep -q "Old Title" "$test_docs/with_frontmatter.md"; then
      assert_true 1 "Existing frontmatter should be preserved"
    else
      assert_true 0 "Existing frontmatter should be preserved"
    fi
    
    # Check backup files created
    assert_file_exists "$test_docs/simple.md.bak" "Backup should be created"
  else
    assert_true 0 "Append should succeed"
  fi
  
  end_test 0
}

# Test 2: Dry run workflow
test_dry_run_workflow() {
  start_test "Dry run workflow doesn't modify files"
  
  local test_docs="$TEST_CURRENT_DIR/dryrun_docs"
  local test_db="$TEST_CURRENT_DIR/dryrun.db"
  
  # Create test documents
  create_test_documents "$test_docs"
  
  # Get original content
  local orig_md5
  orig_md5=$(find "$test_docs" -name "*.md" -exec md5sum {} \; | sort)
  
  # Set up database with citations
  create_test_database "$test_db"
  add_test_citation "$test_db" "simple.md" "Test Title" "Test Author" "2024"
  
  # Run append in dry-run mode
  local output
  output=$("$APPEND_CITATIONS_SCRIPT" -d "$test_db" -n "$test_docs" 2>&1)
  
  assert_contains "$output" "Would update" "Dry run should show what would be done"
  
  # Verify files unchanged
  local new_md5
  new_md5=$(find "$test_docs" -name "*.md" -exec md5sum {} \; | sort)
  
  assert_equals "$orig_md5" "$new_md5" "Files should not be modified in dry run"
  
  end_test 0
}

# Test 3: Force mode workflow
test_force_mode_workflow() {
  start_test "Force mode overwrites existing frontmatter"
  
  local test_docs="$TEST_CURRENT_DIR/force_docs"
  local test_db="$TEST_CURRENT_DIR/force.db"
  
  # Create test documents
  create_test_documents "$test_docs"
  
  # Set up database
  create_test_database "$test_db"
  add_test_citation "$test_db" "with_frontmatter.md" "New Title" "New Author" "2024"
  
  # Run append without force - should skip
  "$APPEND_CITATIONS_SCRIPT" -d "$test_db" "$test_docs" >/dev/null 2>&1
  
  if grep -q "Old Title" "$test_docs/with_frontmatter.md"; then
    assert_true 1 "Without force, old frontmatter preserved"
  else
    assert_true 0 "Without force, old frontmatter preserved"
  fi
  
  # Run with force
  "$APPEND_CITATIONS_SCRIPT" -d "$test_db" -f "$test_docs" >/dev/null 2>&1
  
  if grep -q "New Title" "$test_docs/with_frontmatter.md"; then
    assert_true 1 "With force, new frontmatter applied"
  else
    assert_true 0 "With force, new frontmatter applied"
  fi
  
  if ! grep -q "Old Title" "$test_docs/with_frontmatter.md"; then
    assert_true 1 "Old frontmatter should be gone"
  else
    assert_true 0 "Old frontmatter should be gone"
  fi
  
  end_test 0
}

# Test 4: Subdirectory handling
test_subdirectory_workflow() {
  start_test "Workflow handles subdirectories correctly"
  
  local test_docs="$TEST_CURRENT_DIR/subdir_docs"
  local test_db="$TEST_CURRENT_DIR/subdir.db"
  
  # Create nested structure
  mkdir -p "$test_docs/level1/level2/level3"
  echo "# Deep Document" > "$test_docs/level1/level2/level3/deep.md"
  
  # Set up database with nested path
  create_test_database "$test_db"
  add_test_citation "$test_db" "level1/level2/level3/deep.md" "Deep Document" "Deep Author" "2024"
  
  # Run append
  "$APPEND_CITATIONS_SCRIPT" -d "$test_db" "$test_docs" >/dev/null 2>&1
  
  assert_file_exists "$test_docs/level1/level2/level3/deep.md.bak" "Should process deeply nested files"
  
  if head -1 "$test_docs/level1/level2/level3/deep.md" | grep -q "^---$"; then
    assert_true 1 "Deep file should have frontmatter"
  else
    assert_true 0 "Deep file should have frontmatter"
  fi
  
  end_test 0
}

# Test 5: Error recovery workflow
test_error_recovery_workflow() {
  start_test "Workflow handles errors gracefully"
  
  local test_docs="$TEST_CURRENT_DIR/error_docs"
  local test_db="$TEST_CURRENT_DIR/error.db"
  
  # Create test documents
  create_test_documents "$test_docs"
  
  # Create read-only file
  chmod 444 "$test_docs/simple.md"
  
  # Set up database
  create_test_database "$test_db"
  add_test_citation "$test_db" "simple.md" "Test Title" "Test Author" "2024"
  
  # Run append - should handle read-only file gracefully
  local output
  output=$("$APPEND_CITATIONS_SCRIPT" -d "$test_db" "$test_docs" 2>&1)
  
  # Should show error but continue
  assert_contains "$output" "rror" "Should report error for read-only file"
  
  # Fix permissions
  chmod 644 "$test_docs/simple.md"
  
  # Test with non-existent database
  output=$("$APPEND_CITATIONS_SCRIPT" -d "/tmp/nonexistent.db" "$test_docs" 2>&1)
  assert_contains "$output" "not found" "Should report missing database"
  
  end_test 0
}

# Test 6: Performance with many files
test_performance_workflow() {
  start_test "Workflow handles many files efficiently"
  
  local test_docs="$TEST_CURRENT_DIR/perf_docs"
  local test_db="$TEST_CURRENT_DIR/perf.db"
  local num_files=50
  
  # Create many test files
  mkdir -p "$test_docs"
  for i in $(seq 1 $num_files); do
    echo "# Document $i" > "$test_docs/doc$i.md"
  done
  
  # Set up database with citations for half the files
  create_test_database "$test_db"
  for i in $(seq 1 $((num_files / 2))); do
    add_test_citation "$test_db" "doc$i.md" "Document $i" "Author $i" "2024"
  done
  
  # Time the append operation
  local start_time=$(date +%s)
  "$APPEND_CITATIONS_SCRIPT" -d "$test_db" -q "$test_docs" >/dev/null 2>&1
  local elapsed=$(($(date +%s) - start_time))
  
  echo "  Processed $num_files files in ${elapsed}s"
  
  # Should complete reasonably quickly
  if ((elapsed < 10)); then
    assert_true 1 "Should process $num_files files in under 10 seconds"
  else
    assert_true 0 "Should process $num_files files in under 10 seconds (took ${elapsed}s)"
  fi
  
  # Verify correct number of files updated
  local updated_count
  updated_count=$(find "$test_docs" -name "*.bak" | wc -l)
  assert_equals "$((num_files / 2))" "$updated_count" "Should update correct number of files"
  
  end_test 0
}

# Test 7: Unicode and special characters workflow
test_unicode_workflow() {
  start_test "Workflow handles unicode and special characters"
  
  local test_docs="$TEST_CURRENT_DIR/unicode_docs"
  local test_db="$TEST_CURRENT_DIR/unicode.db"
  
  # Create documents with unicode
  mkdir -p "$test_docs"
  
  cat > "$test_docs/japanese.md" << 'EOF'
# 日本語のドキュメント

著者: 山田太郎
発行: 2024年

これはテストドキュメントです。
EOF

  cat > "$test_docs/special.md" << 'EOF'
# Document with "Special" Characters

By: François O'Neil-Müller
Year: 2024

Testing various special characters.
EOF

  # Set up database
  create_test_database "$test_db"
  add_test_citation "$test_db" "japanese.md" "日本語のドキュメント" "山田太郎" "2024"
  add_test_citation "$test_db" "special.md" "Document with \"Special\" Characters" "François O'Neil-Müller" "2024"
  
  # Run append
  "$APPEND_CITATIONS_SCRIPT" -d "$test_db" "$test_docs" >/dev/null 2>&1
  
  # Verify unicode preserved
  if grep -q "山田太郎" "$test_docs/japanese.md"; then
    assert_true 1 "Japanese characters preserved"
  else
    assert_true 0 "Japanese characters preserved"
  fi
  
  if grep -q "François" "$test_docs/special.md"; then
    assert_true 1 "Special characters preserved"
  else
    assert_true 0 "Special characters preserved"
  fi
  
  end_test 0
}

# Test 8: Parallel processing (if available)
test_parallel_workflow() {
  start_test "Parallel processing workflow"
  
  # This would test gen-citations.sh with parallel processing
  # Requires mock API server to be running
  
  if ! check_mock_api; then
    skip_test "Mock API not available for parallel test"
    return
  fi
  
  local test_docs="$TEST_CURRENT_DIR/parallel_docs"
  local test_db="$TEST_CURRENT_DIR/parallel.db"
  
  # Create test documents
  mkdir -p "$test_docs"
  for i in {1..20}; do
    echo "# Document $i" > "$test_docs/doc$i.md"
  done
  
  # Test parallel processing
  local output
  output=$("$GEN_CITATIONS_SCRIPT" -d "$test_db" -p 4 -n "$test_docs" 2>&1)
  
  assert_contains "$output" "parallel" "Should indicate parallel processing"
  
  end_test 0
}

# Run all tests
test_basic_workflow
test_dry_run_workflow
test_force_mode_workflow
test_subdirectory_workflow
test_error_recovery_workflow
test_performance_workflow
test_unicode_workflow
test_parallel_workflow

# Print summary
print_test_summary

#fin