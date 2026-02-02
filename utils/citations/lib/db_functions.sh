#!/bin/bash
# db_functions.sh - SQLite database operations for the citations system
#
# This library provides all database-related functions for storing and
# retrieving citation data. It handles database initialization, CRUD
# operations, and citation parsing.
#
# Database Schema:
#   - id: Auto-incrementing primary key
#   - sourcefile: Unique file path (relative to source directory)
#   - title: Extracted document title
#   - author: Extracted author name(s)
#   - year: Publication year
#   - raw_citation: Original citation string from API
#   - processed_date: Timestamp of first processing
#   - last_modified: Timestamp of last update
#
# Functions:
#   - db_init: Initialize database and create tables
#   - db_file_exists: Check if file has been processed
#   - db_upsert_citation: Insert or update citation data
#   - db_get_all_citations: Retrieve all citations
#   - db_get_citation: Get citation for specific file
#   - db_get_processed_count: Count total citations
#   - db_get_blank_citations: Find entries with missing data
#   - parse_citation: Parse citation string into components
#
set -euo pipefail

# Escape input for safe SQL interpolation
# Args:
#   $1: Input string to escape
# Returns:
#   0 on success, 1 on invalid input (null bytes)
# Output:
#   Escaped string safe for SQL single-quoted literals
sql_escape() {
  local -- input=${1:-}
  # Check for null bytes using grep (bash can't handle \0 in strings directly)
  if printf '%s' "$input" | grep -qP '\x00' 2>/dev/null; then
    >&2 echo "ERROR: Input contains null bytes"
    return 1
  fi
  # Remove control characters (keep tab, newline, CR)
  input=$(printf '%s' "$input" | tr -d '\001-\010\013\014\016-\037')
  # Escape single quotes (SQL standard) and backslashes
  input="${input//\'/\'\'}"
  input="${input//\\/\\\\}"
  printf '%s' "$input"
}

# Validate file path for security
# Args:
#   $1: Input path to validate
#   $2: Base directory for containment check (optional)
# Returns:
#   0 if path is valid, 1 on security violation
validate_file_path() {
  local -- input_path=$1 base_dir=${2:-}
  # Reject empty paths
  [[ -n "$input_path" ]] || { >&2 echo "ERROR: Empty path"; return 1; }
  # Reject null bytes using grep (bash can't handle \0 in strings directly)
  if printf '%s' "$input_path" | grep -qP '\x00' 2>/dev/null; then
    >&2 echo "ERROR: Null bytes in path"
    return 1
  fi
  # Reject path traversal attempts - only when '..' is a path component
  # This allows filenames like 'n.e.c..txt' while blocking '../etc/passwd'
  if [[ "$input_path" =~ (^|/)\.\.(/|$) ]]; then
    >&2 echo "ERROR: Path traversal detected"
    return 1
  fi
  # If base_dir provided, verify containment using realpath
  if [[ -n "$base_dir" ]]; then
    local -- resolved_path resolved_base
    resolved_base=$(realpath -m -- "$base_dir" 2>/dev/null) || return 1
    resolved_path=$(realpath -m -- "$input_path" 2>/dev/null) || return 1
    [[ "$resolved_path" == "$resolved_base"* ]] || { >&2 echo "ERROR: Path escapes base directory"; return 1; }
  fi
  return 0
}

# Initialize database and create tables
# Args:
#   $1: Database path (default: ./citations.db)
# Returns:
#   0 on success
db_init() {
  local db_path="${1:-./citations.db}"

  # Create database directory if needed
  local db_dir
  db_dir=$(dirname -- "$db_path")
  [[ -d "$db_dir" ]] || mkdir -p "$db_dir"

  # Create citations table with performance pragmas
  sqlite3 "$db_path" <<'EOF'
PRAGMA synchronous=NORMAL;
PRAGMA cache_size=10000;
PRAGMA temp_store=MEMORY;

CREATE TABLE IF NOT EXISTS citations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sourcefile TEXT UNIQUE NOT NULL,
    title TEXT,
    author TEXT,
    year TEXT,
    raw_citation TEXT,
    broad_context TEXT,
    processed_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_modified TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_sourcefile ON citations(sourcefile);
CREATE INDEX IF NOT EXISTS idx_processed_date ON citations(processed_date);
EOF

  # Check if broad_context column exists and add if missing (for existing databases)
  if ! sqlite3 "$db_path" "PRAGMA table_info(citations);" | grep -q "broad_context"; then
    sqlite3 "$db_path" "ALTER TABLE citations ADD COLUMN broad_context TEXT;"
  fi

  return 0
}

# Check if a file has already been processed
# Args:
#   $1: Database path
#   $2: Source file path (relative)
#   $3: Check for blank entries (optional, default: 0)
#       If 1, returns true only if file exists with blank title AND author
# Returns:
#   0 if file exists (or exists with blanks if $3=1), 1 otherwise
db_file_exists() {
  local db_path="$1"
  local sourcefile="$2"
  local -i check_blank="${3:-0}"  # Default to 0 (normal check)
  local -i count

  # Escape input for SQL safety
  sourcefile=$(sql_escape "$sourcefile") || return 1

  if ((check_blank)); then
    # Check if file exists AND has blank title/author
    count=$(sqlite3 "$db_path" "SELECT COUNT(*) FROM citations WHERE sourcefile = '$sourcefile' AND (title IS NULL OR title = '') AND (author IS NULL OR author = '');")
  else
    # Normal check - just see if file exists
    count=$(sqlite3 "$db_path" "SELECT COUNT(*) FROM citations WHERE sourcefile = '$sourcefile';")
  fi

  ((count > 0))
}

# Insert or update a citation
# Args:
#   $1: Database path
#   $2: Source file path (relative)
#   $3: Title (can be empty)
#   $4: Author (can be empty)
#   $5: Year (can be empty)
#   $6: Raw citation string
#   $7: Broad context (can be empty)
# Returns:
#   SQLite exit code
db_upsert_citation() {
  local db_path="$1"
  local sourcefile="$2"
  local title="$3"
  local author="$4"
  local year="$5"
  local raw_citation="$6"
  local broad_context="${7:-}"

  # Escape all inputs for SQL safety
  sourcefile=$(sql_escape "$sourcefile") || return 1
  title=$(sql_escape "$title") || return 1
  author=$(sql_escape "$author") || return 1
  year=$(sql_escape "$year") || return 1
  raw_citation=$(sql_escape "$raw_citation") || return 1
  broad_context=$(sql_escape "$broad_context") || return 1

  sqlite3 "$db_path" <<EOF
INSERT INTO citations (sourcefile, title, author, year, raw_citation, broad_context)
VALUES ('$sourcefile', '$title', '$author', '$year', '$raw_citation', '$broad_context')
ON CONFLICT(sourcefile) DO UPDATE SET
    title = excluded.title,
    author = excluded.author,
    year = excluded.year,
    raw_citation = excluded.raw_citation,
    broad_context = excluded.broad_context,
    last_modified = CURRENT_TIMESTAMP;
EOF
}

# Get all citations from database
db_get_all_citations() {
  local db_path="$1"
  
  sqlite3 -separator '|' "$db_path" <<'EOF'
SELECT sourcefile, title, author, year, raw_citation, broad_context
FROM citations
ORDER BY sourcefile;
EOF
}

# Get citation for a specific file
db_get_citation() {
  local db_path="$1"
  local sourcefile="$2"

  # Escape input for SQL safety
  sourcefile=$(sql_escape "$sourcefile") || return 1

  sqlite3 -separator '|' "$db_path" <<EOF
SELECT title, author, year, raw_citation, broad_context
FROM citations
WHERE sourcefile = '$sourcefile';
EOF
}

# Get count of processed files
db_get_processed_count() {
  local db_path="$1"
  
  sqlite3 "$db_path" "SELECT COUNT(*) FROM citations;"
}

# Get list of files with blank title AND author
db_get_blank_citations() {
  local db_path="$1"

  sqlite3 "$db_path" "SELECT sourcefile FROM citations WHERE (title IS NULL OR title = '') AND (author IS NULL OR author = '');"
}

# Export all existing sourcefiles to a file for fast lookup
# This enables O(1) lookups instead of O(n) database queries
# Args:
#   $1: Database path
#   $2: Output file path
# Returns:
#   0 on success
db_export_existing_files() {
  local db_path="$1" output_file="$2"

  sqlite3 "$db_path" "SELECT sourcefile FROM citations ORDER BY sourcefile;" > "$output_file"
}

# Check if file exists using pre-exported lookup file
# Much faster than db_file_exists() when checking many files
# Args:
#   $1: Lookup file (from db_export_existing_files)
#   $2: Source file path to check
# Returns:
#   0 if file exists in lookup, 1 otherwise
db_file_exists_cached() {
  local lookup_file="$1" sourcefile="$2"

  grep -Fxq "$sourcefile" "$lookup_file" 2>/dev/null
}

# Parse citation string into components
# Expected format: "title", author, year
# Args:
#   $1: Citation string from API
# Returns:
#   0 on successful parse, 1 on error
# Output:
#   Pipe-delimited string: title|author|year
#   Fields may be empty if not found or "NF"
parse_citation() {
  local citation="$1"
  local title=""
  local author=""
  local year=""
  
  # Remove leading/trailing whitespace
  citation=$(echo "$citation" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
  
  # Check if it's "NF" (not found) - with or without quotes
  if [[ $citation == NF || $citation == '"NF"' || -z $citation ]]; then
    echo "||"
    return 1
  fi
  
  # Expected format: "title", author(s), year
  # First, check if the last field is a year
  if [[ "$citation" =~ [[:space:]]*([0-9]{4}|NF)[[:space:]]*$ ]]; then
    year="${BASH_REMATCH[1]}"
    # Remove the year from the end
    local citation_without_year="${citation%,*}"
    
    # Check if title is quoted
    if [[ "$citation_without_year" =~ ^\"([^\"]+)\"[[:space:]]*,[[:space:]]*(.+)[[:space:]]*$ ]]; then
      title="${BASH_REMATCH[1]}"
      author="${BASH_REMATCH[2]}"
    elif [[ "$citation_without_year" =~ ^([^,]+)[[:space:]]*,[[:space:]]*(.+)[[:space:]]*$ ]]; then
      # Unquoted title
      title="${BASH_REMATCH[1]}"
      author="${BASH_REMATCH[2]}"
    else
      # Only title and year, no author
      if [[ "$citation_without_year" =~ ^\"([^\"]+)\"[[:space:]]*$ ]]; then
        title="${BASH_REMATCH[1]}"
      else
        title="$citation_without_year"
      fi
    fi
  else
    # No valid year found, try to parse what we can
    if [[ "$citation" =~ ^\"([^\"]+)\"[[:space:]]*,[[:space:]]*(.+)[[:space:]]*$ ]]; then
      title="${BASH_REMATCH[1]}"
      author="${BASH_REMATCH[2]}"
    elif [[ "$citation" =~ ^([^,]+)[[:space:]]*,[[:space:]]*(.+)[[:space:]]*$ ]]; then
      title="${BASH_REMATCH[1]}"
      author="${BASH_REMATCH[2]}"
    else
      # Unable to parse, return raw citation in title field
      title="$citation"
    fi
  fi
  
  # Trim whitespace
  title=$(echo "$title" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
  author=$(echo "$author" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
  year=$(echo "$year" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
  
  # Replace "NF" with empty string in individual fields
  # Also check for quoted "NF"
  [[ "$title" == "NF" || "$title" == '"NF"' ]] && title=""
  [[ "$author" == "NF" || "$author" == '"NF"' ]] && author=""
  [[ "$year" == "NF" || "$year" == '"NF"' ]] && year=""
  
  echo "${title}|${author}|${year}"
  return 0
}

# Clean up database connection on exit
db_cleanup() {
  # SQLite handles cleanup automatically
  return 0
}

#fin
