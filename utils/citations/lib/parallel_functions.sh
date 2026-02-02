#!/bin/bash
# parallel_functions.sh - Multi-worker parallel processing for citation extraction
#
# This library implements a robust parallel processing framework for the
# citations system, enabling high-throughput citation extraction using
# multiple concurrent workers.
#
# Architecture:
#   - Queue-based work distribution with atomic operations
#   - Worker processes with independent rate limiting
#   - Thread-safe progress tracking and result collection
#   - SQLite WAL mode for concurrent database access
#   - Comprehensive error handling and recovery
#
# Features:
#   - Configurable number of workers (1-20+)
#   - Atomic queue operations using file locking
#   - Worker health monitoring and automatic cleanup
#   - Batch result processing to minimize database locks
#   - Progress tracking with real-time updates
#   - Graceful shutdown on interruption
#
# Functions:
#   - parallel_init: Initialize work environment
#   - parallel_cleanup: Clean up resources
#   - parallel_queue_add: Add files to work queue
#   - parallel_queue_get_next: Atomic queue retrieval
#   - parallel_worker: Worker process function
#   - parallel_update_progress: Thread-safe progress updates
#   - parallel_launch_workers: Start worker processes
#   - parallel_wait_workers: Wait for completion
#   - parallel_process_results: Batch process results
#
set -euo pipefail

# Global variables for parallel processing
declare -gi MAX_WORKERS=5
declare -gi ACTIVE_WORKERS=0
declare -ga WORKER_PIDS=()
declare -g WORK_DIR=""
declare -g RESULTS_DIR=""
declare -g QUEUE_FILE=""
declare -g PROGRESS_FILE=""
declare -g LOCK_FILE=""
declare -gi QUEUE_SPLIT=0  # Flag indicating queue has been split into per-worker files

# Initialize parallel processing environment
# Creates temporary work directory structure and prepares database
# Args:
#   $1: Database path (must exist)
#   $2: Number of workers (default: 5, min: 1)
# Returns:
#   0 on success, 1 on failure
# Sets global variables: WORK_DIR, RESULTS_DIR, QUEUE_FILE, PROGRESS_FILE, LOCK_FILE
parallel_init() {
  local db_path="$1"
  local -i num_workers="${2:-5}"
  
  # Validate inputs
  if [[ -z "$db_path" ]]; then
    >&2 echo "ERROR: Database path not provided to parallel_init"
    return 1
  fi
  
  if [[ ! -f "$db_path" ]]; then
    >&2 echo "ERROR: Database file not found: $db_path"
    return 1
  fi
  
  if ((num_workers < 1)); then
    >&2 echo "ERROR: Invalid number of workers: $num_workers"
    return 1
  fi
  
  # Validate required environment variables
  if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    >&2 echo "ERROR: OPENAI_API_KEY environment variable is not set"
    return 1
  fi
  
  MAX_WORKERS=$num_workers
  
  # Create temporary work directory
  WORK_DIR=$(mktemp -d /tmp/citations-parallel-XXXXXX)
  if [[ ! -d "$WORK_DIR" ]]; then
    >&2 echo "ERROR: Failed to create temporary work directory"
    return 1
  fi
  
  RESULTS_DIR="$WORK_DIR/results"
  QUEUE_FILE="$WORK_DIR/queue"
  PROGRESS_FILE="$WORK_DIR/progress"
  LOCK_FILE="$WORK_DIR/lock"
  
  if ! mkdir -p "$RESULTS_DIR"; then
    >&2 echo "ERROR: Failed to create results directory"
    rm -rf "$WORK_DIR"
    return 1
  fi
  
  # Initialize progress counters
  if ! echo "0:0:0:0" > "$PROGRESS_FILE"; then
    >&2 echo "ERROR: Failed to create progress file"
    rm -rf "$WORK_DIR"
    return 1
  fi
  
  # Enable SQLite WAL mode for better concurrency
  if ! sqlite3 "$db_path" "PRAGMA journal_mode=WAL;" >/dev/null 2>&1; then
    >&2 echo "WARNING: Failed to enable WAL mode for database"
    # This is not fatal, continue anyway
  fi
  
  return 0
}

# Cleanup parallel processing environment
parallel_cleanup() {
  local -i i

  # Kill any remaining workers
  for ((i=0; i<${#WORKER_PIDS[@]}; i++)); do
    local pid="${WORKER_PIDS[i]}"
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      kill -TERM "$pid" 2>/dev/null || true
    fi
  done

  # Wait for workers to finish
  for ((i=0; i<${#WORKER_PIDS[@]}; i++)); do
    local pid="${WORKER_PIDS[i]}"
    if [[ -n "$pid" ]]; then
      wait "$pid" 2>/dev/null || true
    fi
  done

  # Remove per-worker queue files if they exist
  if [[ -n "$QUEUE_FILE" ]]; then
    rm -f "$QUEUE_FILE".worker* 2>/dev/null || true
    rm -f "$QUEUE_FILE".cursor 2>/dev/null || true
  fi

  # Remove work directory
  [[ -d "$WORK_DIR" ]] && rm -rf "$WORK_DIR"

  # Reset queue split flag
  QUEUE_SPLIT=0

  return 0
}

# Add file to work queue
# Args:
#   $1: Full file path
#   $2: Relative file path (for database)
#   $3: Source directory
# Appends to QUEUE_FILE in format: file|file_relative|src_dir
parallel_queue_add() {
  local file="$1"
  local file_relative="$2"
  local src_dir="$3"

  echo "${file}|${file_relative}|${src_dir}" >> "$QUEUE_FILE"
}

# Split queue into per-worker files for lock-free O(1) access
# This eliminates the O(n²) problem where sed scans from line 1 for each read
# Args:
#   $1: Queue file path
#   $2: Number of workers
# Creates: ${queue_file}.worker00, ${queue_file}.worker01, etc.
parallel_split_queue() {
  local queue_file="$1"
  local -i num_workers="$2"
  local -i total_lines worker_lines

  [[ -f "$queue_file" ]] || return 1

  total_lines=$(wc -l < "$queue_file")
  ((total_lines > 0)) || return 0

  # Calculate lines per worker (round up)
  worker_lines=$(( (total_lines + num_workers - 1) / num_workers ))
  ((worker_lines > 0)) || worker_lines=1

  # Split queue into per-worker files
  # -l N: N lines per file
  # -d: numeric suffixes (00, 01, 02...)
  # -a 2: 2-digit suffix
  split -l "$worker_lines" -d -a 2 "$queue_file" "${queue_file}.worker"

  QUEUE_SPLIT=1
  return 0
}

# Get next item from queue atomically using cursor-based approach
# O(1) per call instead of O(n) rewriting - critical for large queues
parallel_queue_get_next() {
  local queue_file="$1"
  local lock_file="$2"
  local cursor_file="${queue_file}.cursor"

  # Use flock to ensure atomic access
  (
    flock -x 200

    # Initialize cursor file if not present
    [[ -f "$cursor_file" ]] || echo "1" > "$cursor_file"

    # Read current cursor position
    local -i cursor
    cursor=$(< "$cursor_file")

    # Check if queue file exists
    [[ -f "$queue_file" ]] || return 0

    # Read line at cursor position (sed is O(n) but only runs once per item)
    local queue_entry
    queue_entry=$(sed -n "${cursor}p" "$queue_file" 2>/dev/null || true)

    # If we got an entry, advance cursor and output it
    if [[ -n "$queue_entry" ]]; then
      cursor+=1
      echo "$cursor" > "$cursor_file"
      echo "$queue_entry"
    fi
  ) 200>"$lock_file.queue"
}

# Worker function to process files from queue
# Runs in separate process, processes files until queue is empty
# When QUEUE_SPLIT=1, reads from dedicated per-worker queue file (O(1), no locking)
# Otherwise falls back to shared queue with locking (O(n) but safe)
# Args:
#   $1: Worker ID (for logging)
#   $2: Database path
#   $3: Model name
#   $4: System prompt
#   $5: Chunk size
#   $6: Max tokens
#   $7: Temperature
#   $8: Force flag (0/1)
#   $9: Reprocess blank flag (0/1)
#   $10: Verbose level
#   $11: Debug flag (0/1)
#   $12: Broad context (optional)
# Writes results to RESULTS_DIR/worker${id}_${RANDOM}.result
parallel_worker() {
  local -i worker_id="$1"
  local database="$2"
  local model="$3"
  local system_prompt="$4"
  local -i chunk_size="$5"
  local -i max_tokens="$6"
  local temperature="$7"
  local -i force="$8"
  local -i reprocess_blank="$9"
  local -i verbose="${10}"
  local -i debug="${11}"
  local broad_context="${12:-}"

  # Worker-specific rate limiting
  local -i last_api_call=0

  # Batch progress tracking - flush every 10 files to reduce I/O
  local -i batch_processed=0 batch_skipped=0 batch_errors=0 batch_notfound=0
  local -i files_since_flush=0
  local -i BATCH_FLUSH_THRESHOLD=10

  ((debug)) && echo "DEBUG: Worker $worker_id starting" >&2

  # Helper to flush accumulated progress
  flush_progress() {
    if ((batch_processed + batch_skipped + batch_errors + batch_notfound > 0)); then
      parallel_update_progress "$batch_processed" "$batch_skipped" "$batch_errors" "$batch_notfound"
      batch_processed=0 batch_skipped=0 batch_errors=0 batch_notfound=0
      files_since_flush=0
    fi
  }

  # Helper to process a single queue entry
  process_queue_entry() {
    local queue_entry="$1"

    [[ -n "$queue_entry" ]] || return 0

    ((debug)) && echo "DEBUG: Worker $worker_id processing: $queue_entry" >&2

    # Parse queue entry
    local file file_relative src_dir
    IFS='|' read -r file file_relative src_dir <<< "$queue_entry"

    # Skip if already processed (unless force mode or reprocess-blank mode)
    if ((! force)); then
      if ((reprocess_blank)); then
        # In reprocess-blank mode: skip if file exists but is NOT blank
        if db_file_exists "$database" "$file_relative" && ! db_file_exists "$database" "$file_relative" 1; then
          batch_skipped+=1
          files_since_flush+=1
          ((files_since_flush >= BATCH_FLUSH_THRESHOLD)) && flush_progress
          return 0
        fi
      else
        # Normal mode: skip if file exists
        if db_file_exists "$database" "$file_relative"; then
          batch_skipped+=1
          files_since_flush+=1
          ((files_since_flush >= BATCH_FLUSH_THRESHOLD)) && flush_progress
          return 0
        fi
      fi
    fi

    # Prepare file content
    local filetitle
    filetitle=$(basename -- "$file")
    filetitle=${filetitle%.*}
    filetitle="${filetitle//-/ }"
    filetitle="${filetitle//_/ - }"

    # Read first chunk of file
    local file_content
    file_content=$(head -c "$chunk_size" "$file" 2>/dev/null || true)

    # Skip empty files
    if [[ -z "$file_content" ]]; then
      batch_skipped+=1
      files_since_flush+=1
      ((files_since_flush >= BATCH_FLUSH_THRESHOLD)) && flush_progress
      return 0
    fi

    # Prepare content with file title hint
    local user_content="---
file-title: $filetitle
---

$file_content"

    # Worker-specific rate limiting
    local -i now current_ms elapsed needed_delay
    now=$(date +%s)
    local nanos
    nanos=$(date +%N)
    current_ms=$((now * 1000 + 10#$nanos / 1000000))

    if ((last_api_call > 0)); then
      elapsed=$((current_ms - last_api_call))
      if ((elapsed < 1000)); then  # 1 second minimum between calls
        needed_delay=$((1000 - elapsed))
        # Sleep for at least 1 second if there's any delay needed
        if ((needed_delay > 0)); then
          sleep 1
        fi
      fi
    fi
    last_api_call=$current_ms

    # Call API (routes to correct provider based on model)
    local response result
    if response=$(api_call "$model" "$system_prompt" "$user_content" "$max_tokens" "$temperature" 2>/dev/null); then
      result=$(api_extract_result "$response")

      if [[ -z "$result" || "$result" == "NF" || "$result" == '"NF"' || "$result" == "Error" ]]; then
        batch_notfound+=1
      else
        # Write result to temporary file for batch processing - use mktemp for security
        local result_file
        result_file=$(mktemp "$RESULTS_DIR/worker${worker_id}_XXXXXX.result")
        echo "${file_relative}|${result}|${broad_context}" > "$result_file"
        batch_processed+=1
      fi
    else
      batch_errors+=1
    fi

    files_since_flush+=1
    ((files_since_flush >= BATCH_FLUSH_THRESHOLD)) && flush_progress
    return 0
  }

  # Check if we're using pre-split queues (O(1) lock-free access)
  local worker_queue="${QUEUE_FILE}.worker$(printf '%02d' "$worker_id")"

  if [[ -f "$worker_queue" ]]; then
    # Pre-split queue mode: read our dedicated file sequentially (no locking needed)
    ((debug)) && echo "DEBUG: Worker $worker_id using dedicated queue: $worker_queue" >&2

    while IFS= read -r queue_entry || [[ -n "$queue_entry" ]]; do
      [[ -n "$queue_entry" ]] || continue
      process_queue_entry "$queue_entry"
    done < "$worker_queue"

    flush_progress  # Flush remaining progress before exit
    ((debug)) && echo "DEBUG: Worker $worker_id finished processing dedicated queue" >&2
  else
    # Fallback: shared queue with locking (O(n) per sed call - legacy mode)
    ((debug)) && echo "DEBUG: Worker $worker_id using shared queue (fallback mode)" >&2

    while true; do
      # Get next item from queue
      local queue_entry
      queue_entry=$(parallel_queue_get_next "$QUEUE_FILE" "$LOCK_FILE")

      # Exit if queue is empty
      if [[ -z "$queue_entry" ]]; then
        ((debug)) && echo "DEBUG: Worker $worker_id found empty queue, exiting" >&2
        flush_progress  # Flush remaining progress before exit
        break
      fi

      process_queue_entry "$queue_entry"
    done
  fi
}

# Update progress counters atomically
parallel_update_progress() {
  local -i processed="$1"
  local -i skipped="$2"
  local -i errors="$3"
  local -i notfound="$4"
  
  # Use flock for atomic updates
  (
    flock -x 200
    
    # Read current values
    local current
    current=$(cat "$PROGRESS_FILE")
    IFS=':' read -r cur_processed cur_skipped cur_errors cur_notfound <<< "$current"
    
    # Update values
    cur_processed=$((cur_processed + processed))
    cur_skipped=$((cur_skipped + skipped))
    cur_errors=$((cur_errors + errors))
    cur_notfound=$((cur_notfound + notfound))
    
    # Write back
    echo "${cur_processed}:${cur_skipped}:${cur_errors}:${cur_notfound}" > "$PROGRESS_FILE"
    
  ) 200>"$LOCK_FILE"
}

# Get current progress
parallel_get_progress() {
  cat "$PROGRESS_FILE"
}

# Launch parallel workers
# Creates worker scripts and starts background processes
# Args:
#   $1-$12: Same as parallel_worker arguments
# Returns:
#   0 if at least one worker started, 1 on total failure
# Sets: WORKER_PIDS array with process IDs
parallel_launch_workers() {
  local database="$1"
  local model="$2"
  local system_prompt="$3"
  local -i chunk_size="$4"
  local -i max_tokens="$5"
  local temperature="$6"
  local -i force="$7"
  local -i reprocess_blank="$8"
  local -i verbose="$9"
  local -i debug="${10}"
  local lib_dir="${11}"
  local broad_context="${12:-}"

  # Worker count safety limit to prevent resource exhaustion
  local -i max_safe_workers
  max_safe_workers=$(( $(nproc 2>/dev/null || echo 4) * 2 ))
  if ((MAX_WORKERS > max_safe_workers)); then
    >&2 echo "WARNING: Limiting workers from $MAX_WORKERS to $max_safe_workers (2x CPU cores)"
    MAX_WORKERS=$max_safe_workers
  fi

  # Validate required parameters
  if [[ -z "$database" || ! -f "$database" ]]; then
    >&2 echo "ERROR: Invalid database path: $database"
    return 1
  fi
  
  if [[ -z "$lib_dir" || ! -d "$lib_dir" ]]; then
    >&2 echo "ERROR: Invalid library directory: $lib_dir"
    return 1
  fi
  
  # Verify required library files exist
  local lib_file
  for lib_file in "db_functions.sh" "api_functions.sh" "parallel_functions.sh"; do
    if [[ ! -f "$lib_dir/$lib_file" ]]; then
      >&2 echo "ERROR: Required library not found: $lib_dir/$lib_file"
      return 1
    fi
  done
  
  # Verify work directory structure exists
  if [[ ! -d "$WORK_DIR" || ! -d "$RESULTS_DIR" || ! -f "$QUEUE_FILE" ]]; then
    >&2 echo "ERROR: Work directory not properly initialized"
    return 1
  fi

  # Split queue into per-worker files for O(1) lock-free access
  # This eliminates the O(n²) problem with the cursor-based approach
  if parallel_split_queue "$QUEUE_FILE" "$MAX_WORKERS"; then
    ((debug)) && >&2 echo "DEBUG: Queue split into $MAX_WORKERS worker files"
  else
    >&2 echo "WARNING: Queue split failed, falling back to shared queue mode"
  fi

  # Export necessary functions and variables for subshells
  export -f parallel_worker
  export -f parallel_queue_get_next
  export -f parallel_split_queue
  export -f parallel_update_progress
  export -f api_call
  export -f api_call_openai
  export -f api_call_anthropic
  export -f api_rate_limit
  export -f api_extract_result
  export -f db_file_exists
  export -f parse_citation
  export QUEUE_FILE LOCK_FILE RESULTS_DIR PROGRESS_FILE QUEUE_SPLIT
  
  local -i i
  for ((i=0; i<MAX_WORKERS; i++)); do
    # Create a temporary script file to avoid quoting issues
    local worker_script="$WORK_DIR/worker_$i.sh"
    
    # Create worker script with error handling
    if ! cat > "$worker_script" <<EOF
#!/bin/bash
set -euo pipefail

# Set up signal handling
trap 'exit 0' TERM INT

# Source required libraries
source '$lib_dir/db_functions.sh' || { echo "ERROR: Failed to source db_functions.sh" >&2; exit 1; }
source '$lib_dir/api_functions.sh' || { echo "ERROR: Failed to source api_functions.sh" >&2; exit 1; }
source '$lib_dir/parallel_functions.sh' || { echo "ERROR: Failed to source parallel_functions.sh" >&2; exit 1; }

# Export required variables
export QUEUE_FILE='$QUEUE_FILE'
export LOCK_FILE='$LOCK_FILE'
export RESULTS_DIR='$RESULTS_DIR'
export PROGRESS_FILE='$PROGRESS_FILE'
export QUEUE_SPLIT='$QUEUE_SPLIT'
# Export all available API keys
[[ -n '${OPENAI_API_KEY:-}' ]] && export OPENAI_API_KEY='$OPENAI_API_KEY'
[[ -n '${ANTHROPIC_API_KEY:-}' ]] && export ANTHROPIC_API_KEY='${ANTHROPIC_API_KEY:-}'
[[ -n '${XAI_API_KEY:-}' ]] && export XAI_API_KEY='${XAI_API_KEY:-}'
[[ -n '${GOOGLE_API_KEY:-}' ]] && export GOOGLE_API_KEY='${GOOGLE_API_KEY:-}'

# Verify required variables are set
[[ -f "\$QUEUE_FILE" ]] || { echo "ERROR: Queue file not found: \$QUEUE_FILE" >&2; exit 1; }
[[ -d "\$RESULTS_DIR" ]] || { echo "ERROR: Results directory not found: \$RESULTS_DIR" >&2; exit 1; }

# Start worker
parallel_worker '$i' '$database' '$model' '$system_prompt' \
  '$chunk_size' '$max_tokens' '$temperature' '$force' '$reprocess_blank' '$verbose' '$debug' '$broad_context' || {
  echo "ERROR: parallel_worker returned non-zero exit code: \$?" >&2
  exit 1
}
EOF
    then
      >&2 echo "ERROR: Failed to create worker script $i"
      continue
    fi
    
    # Make script executable
    if ! chmod +x "$worker_script"; then
      >&2 echo "ERROR: Failed to make worker script $i executable"
      rm -f "$worker_script"
      continue
    fi
    
    # Launch worker and verify it started
    if "$worker_script" 2>"$WORK_DIR/worker_$i.err" & then
      local pid=$!
      # Give worker a moment to start and verify it's running
      sleep 0.1
      if kill -0 "$pid" 2>/dev/null; then
        WORKER_PIDS+=($pid)
        ACTIVE_WORKERS+=1
        ((debug)) && echo "DEBUG: Launched worker $i with PID $pid" >&2
      else
        >&2 echo "ERROR: Worker $i (PID $pid) failed to start"
        # Check if worker left any error output
        if [[ -f "$WORK_DIR/worker_$i.err" ]]; then
          >&2 echo "Worker $i error output:"
          >&2 cat "$WORK_DIR/worker_$i.err"
        else
          >&2 echo "No error file found at $WORK_DIR/worker_$i.err"
        fi
        # Also check the worker script itself
        if [[ -f "$worker_script" ]]; then
          >&2 echo "Worker script exists at $worker_script"
        else
          >&2 echo "Worker script missing at $worker_script"
        fi
      fi
    else
      >&2 echo "ERROR: Failed to launch worker $i"
      local exit_code=$?
      >&2 echo "Exit code: $exit_code"
    fi
  done
  
  # Verify at least one worker started
  if ((ACTIVE_WORKERS == 0)); then
    >&2 echo "ERROR: No workers could be started"
    return 1
  fi
  
  # Warn if not all requested workers could start
  if ((ACTIVE_WORKERS < MAX_WORKERS)); then
    >&2 echo "WARNING: Only $ACTIVE_WORKERS of $MAX_WORKERS requested workers started successfully"
    >&2 echo "         This may be due to system resource limits. The script will continue with available workers."
  fi
}

# Wait for all workers to complete
parallel_wait_workers() {
  local -i i
  
  for ((i=0; i<${#WORKER_PIDS[@]}; i++)); do
    local pid="${WORKER_PIDS[i]}"
    if [[ -n "$pid" ]]; then
      wait "$pid" 2>/dev/null || true
    fi
  done
  
  ACTIVE_WORKERS=0
  WORKER_PIDS=()
}

# Process results from all workers
# Batch inserts all results into database with retry logic
# Args:
#   $1: Database path
#   $2: Verbose flag
#   $3: Debug flag
# Returns:
#   0 on success, 1 on database error
parallel_process_results() {
  local database="$1"
  local -i verbose="$2"
  local -i debug="$3"
  
  # Collect all result files
  local result_file
  local -i count=0
  local -i max_retries=3
  local -i retry_delay=1
  local -i attempt
  
  # Function to execute SQL with retry logic
  execute_with_retry() {
    local sql="$1"
    for ((attempt=1; attempt<=max_retries; attempt++)); do
      if echo "$sql" | sqlite3 "$database" 2>/dev/null; then
        return 0
      fi
      
      # Check if database is locked
      local error_msg
      error_msg=$(echo "$sql" | sqlite3 "$database" 2>&1 || true)
      if [[ "$error_msg" == *"database is locked"* ]]; then
        ((debug)) && echo "DEBUG: Database locked, retry $attempt/$max_retries after ${retry_delay}s"
        sleep "$retry_delay"
        retry_delay=$((retry_delay * 2))  # Exponential backoff
      else
        # Non-recoverable error
        >&2 echo "ERROR: Database error: $error_msg"
        return 1
      fi
    done
    >&2 echo "ERROR: Failed to execute SQL after $max_retries attempts"
    return 1
  }
  
  # Create temporary SQL file for atomic batch processing
  local sql_file="$WORK_DIR/batch_insert.sql"
  
  # Generate SQL statements
  {
    echo "PRAGMA busy_timeout = 5000;"  # 5 second timeout for locks
    echo "BEGIN TRANSACTION;"
    
    # Process each result file
    for result_file in "$RESULTS_DIR"/*.result; do
      [[ -f "$result_file" ]] || continue
      
      while IFS='|' read -r file_relative result result_broad_context; do
        [[ -z "$file_relative" || -z "$result" ]] && continue
        
        # Parse citation
        if parsed=$(parse_citation "$result"); then
          IFS='|' read -r title author year <<< "$parsed"
          
          # Clean up the raw citation
          local clean_result="$result"
          if [[ -z "$title" && -z "$author" && -z "$year" ]]; then
            clean_result=""
          else
            clean_result="${clean_result//, NF/}"
            clean_result="${clean_result//NF, /}"
            clean_result="${clean_result//NF/}"
          fi
          
          # Escape single quotes for SQL
          file_relative="${file_relative//\'/\'\'}"
          title="${title//\'/\'\'}"
          author="${author//\'/\'\'}"
          year="${year//\'/\'\'}"
          clean_result="${clean_result//\'/\'\'}"
          result_broad_context="${result_broad_context//\'/\'\'}"
          
          # Generate SQL
          cat <<EOF
INSERT INTO citations (sourcefile, title, author, year, raw_citation, broad_context)
VALUES ('$file_relative', '$title', '$author', '$year', '$clean_result', '$result_broad_context')
ON CONFLICT(sourcefile) DO UPDATE SET
    title = excluded.title,
    author = excluded.author,
    year = excluded.year,
    raw_citation = excluded.raw_citation,
    broad_context = excluded.broad_context,
    last_modified = CURRENT_TIMESTAMP;
EOF
          count+=1
        fi
      done < "$result_file"
    done
    
    echo "COMMIT;"
  } > "$sql_file"
  
  # Execute batch with retry logic
  if execute_with_retry "$(cat "$sql_file")"; then
    ((debug)) && echo "DEBUG: Successfully batch inserted $count citations"
    rm -f "$sql_file"
    return 0
  else
    >&2 echo "ERROR: Failed to process results"
    rm -f "$sql_file"
    return 1
  fi
}

#fin