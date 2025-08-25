#!/bin/bash
# gen-citations.sh - Extract bibliographic citations from documents using AI
#
# This script processes markdown and text documents to automatically extract
# citation information (title, author, year) using OpenAI's API. It supports
# both sequential and parallel processing modes for efficient handling of
# large document collections.
#
# Features:
#   - Parallel processing with configurable workers (1-20+)
#   - Smart citation extraction using file name hints
#   - SQLite database storage with deduplication
#   - Progress tracking and performance metrics
#   - Dry-run mode for preview
#   - Comprehensive error handling and retry logic
#
# Usage: gen-citations.sh [OPTIONS] [SOURCE_DIRECTORY...]
# See --help for detailed options
#
set -euo pipefail
declare -- PRG0 PRGDIR LIB_DIR PRG
PRG0=$(readlink -fn -- "$0")
PRGDIR=$(dirname -- "$PRG0")
LIB_DIR="$PRGDIR"/lib
PRG=$(basename -s .sh -- "$0")

# Logging functions
#shellcheck disable=SC2015
[[ -t 2 ]] && declare -r RED=$'\033[0;31m' YELLOW=$'\033[0;33m' GREEN=$'\033[0;32m' NOCOLOR=$'\033[0m' || declare -r RED='' YELLOW='' GREEN='' NOCOLOR=''
vecho() { ((VERBOSE)) || return 0; local msg; for msg in "$@"; do >&2 printf '%s: %s\n' "$PRG" "$msg"; done; }
vwarn() { ((VERBOSE)) || return 0; local msg; for msg in "$@"; do >&2 printf '%s: %swarn%s: %s\n' "$PRG" "$YELLOW" "$NOCOLOR" "$msg"; done; }
error() { local msg; for msg in "$@"; do >&2 printf '%s: %serror%s: %s\n' "$PRG" "$RED" "$NOCOLOR" "$msg"; done; }
success() { local msg; for msg in "$@"; do >&2 printf '%s: %ssuccess%s: %s\n' "$PRG" "$GREEN" "$NOCOLOR" "$msg"; done; }
debug() { ((DEBUG)) || return 0; local msg; for msg in "$@"; do >&2 printf '%s: %sdebug%s: %s\n' "$PRG" "$YELLOW" "$NOCOLOR" "$msg"; done; }
die() { (($# < 2)) || error "${@:2}"; (($# < 1)) || exit "$1"; exit 1; }
decp() { ((DEBUG)) || return 0; >&2 declare -p "$@"; }
trim() { local v="$*"; v="${v#"${v%%[![:blank:]]*}"}"; echo -n "${v%"${v##*[![:blank:]]}"}"; }

# Libraries are always in the script directory
[[ -f "$LIB_DIR"/db_functions.sh ]] || die 2 "Error: Cannot find citation library files in '$LIB_DIR'"
# Source helper libraries
source "$LIB_DIR"/db_functions.sh
source "$LIB_DIR"/api_functions.sh
source "$LIB_DIR"/parallel_functions.sh

# Flag to prevent duplicate cleanup
declare -i CLEANUP_DONE=0

# Initialize statistics variables early for cleanup function
declare -i PROCESSED=0 SKIPPED=0 ERRORS=0 NOTFOUND=0

# Cleanup function
xcleanup() { 
  local -i exitcode=${1:-0}
  
  # Prevent duplicate execution
  ((CLEANUP_DONE)) && return 0
  CLEANUP_DONE=1
  
  [[ -t 0 ]] && >&2 printf '\e[?25h'  # Show cursor (was hidden during progress display)
  
  # Cleanup parallel processing if used
  if ((${PARALLEL:-0})) && [[ -n "${WORK_DIR:-}" ]]; then
    parallel_cleanup
  fi
  
  # Display final statistics
  if ((${VERBOSE:-0} && (PROCESSED + SKIPPED + ERRORS > 0))); then
    >&2 echo
    success "Processing complete:" \
        "  Processed: $PROCESSED" \
        "  Skipped:   $SKIPPED" \
        "  Errors:    $ERRORS" \
        "  Not Found: $NOTFOUND"
  fi
  
  db_cleanup
  exit "$exitcode"
}
# Set up signal handlers for cleanup on interrupt (Ctrl+C) or normal exit
trap 'xcleanup $?' SIGINT EXIT

# Optional: Source environment configuration if it exists
[[ -f "$PRGDIR/citation_env.sh" ]] && source "$PRGDIR/citation_env.sh"

# Configuration with environment variable overrides
declare -- MODEL=${CITATION_MODEL:-gpt-4o-mini}
declare -i CHUNK_SIZE=${CITATION_CHUNK_SIZE:-7420}
declare -i MAX_TOKENS=${CITATION_MAX_TOKENS:-128}
declare -- TEMPERATURE=${CITATION_TEMPERATURE:-0.1}
declare -- BROAD_CONTEXT=${CITATION_BROAD_CONTEXT:-}
declare -a SRC_DIRS=()
declare -a EXCLUDES=( ${CITATION_EXCLUDES:-} )
declare -i PARALLEL=${CITATION_PARALLEL:-1}
declare -i PARALLEL_WORKERS=${CITATION_PARALLEL_WORKERS:-10}
declare -i REPROCESS_BLANK=${CITATION_REPROCESS_BLANK:-0}
# Default source directory if none specified
declare -- DEFAULT_SRC=${CITATION_DEFAULT_SRC:-}
# Database path (absolute path recommended)
declare -- DATABASE=${CITATION_DATABASE:-"$PWD"/citations.db}

# Flags
declare -i VERBOSE=1 DEBUG=0 FORCE=0 DRY_RUN=0
declare -i HAS_SUDO=1
[[ $(id -nG) =~ sudo ]] || HAS_SUDO=0

# System prompt for citation extraction
declare -- SYSTEM_PROMPT='Extract bibliographic information from the provided text. Pay attention to the file-title hint which often contains the actual title.

REQUIRED OUTPUT FORMAT:
"title", author, year

RULES:
1. Always output exactly three comma-separated fields
2. Title must be in double quotes
3. Use "NF" (without quotes) for any field that cannot be determined
4. Consider the file-title hint as a strong signal for the actual title or author
5. Look for author names in headers, bylines, or copyright notices
6. Years should be 4-digit numbers only
7. If the title is just a single numeric value, treat it as NF (no title)

EXAMPLES:
Input: file-title: machines-of-loving-grace
Text: "Machines of Loving Grace by Dario Amodei October 2024"
Output: "Machines of Loving Grace", Dario Amodei, 2024

Input: file-title: What-if-Buddhists-Ran-the-World
Text: "What if Buddhists Ran the World? Stephen Batchelor"
Output: "What if Buddhists Ran the World", Stephen Batchelor, NF

Input: file-title: anonymous-report
Text: "Annual Report 2023"
Output: "Annual Report 2023", NF, 2023

Input: file-title: untitled-doc
Text: [Random text with no clear bibliographic info]
Output: NF

Input: file-title: report-2023
Text: "2023 Company Performance Review"
Output: NF, NF, 2023

IMPORTANT: Output ONLY the citation line. No explanations or additional text.'

# Show help
show_help() {
  cat <<EOF
Usage: $PRG [OPTIONS] [SOURCE_DIRECTORY...]

Generate citations from markdown and text documents using OpenAI API.
Extracts title, author, and year information from document headers.

SOURCE DIRECTORIES:
  One or more directories containing .md or .txt files to process.
  If none specified, uses DEFAULT_SRC if configured.

OPTIONS:
  -d, --database PATH     Database file path (default: $DATABASE)
  -f, --force             Force update existing citations in database
  -r, --reprocess-blank   Reprocess files where both title and author are blank
  -n, --dry-run           Show what would be done without making changes
  -m, --model MODEL       OpenAI model to use (default: $MODEL)
  -C, --chunk-size SIZE   Characters to read from each file (default: $CHUNK_SIZE)
  -M, --max-tokens N      Maximum tokens in API response (default: $MAX_TOKENS)
  -t, --temperature T     AI temperature 0-1 (default: $TEMPERATURE)
  -c, --context DOMAINS   Broad context domains (comma-delimited)
  -x, --exclude PATTERN   Exclude files matching pattern (can use multiple times)
  -q, --quiet             Suppress verbose output
  -v, --verbose           Increase verbosity (can be used multiple times)
  -D, --debug             Enable debug output
  -h, --help              Show this help message

PARALLEL PROCESSING:
  -p, --parallel N        Number of parallel workers (default: $PARALLEL_WORKERS)
  --no-parallel           Disable parallel processing
  --sequential            Same as --no-parallel

ENVIRONMENT VARIABLES:
  OPENAI_API_KEY          Required: OpenAI API key
  CITATION_MODEL          Override default model (default: gpt-4o-mini)
  CITATION_DATABASE       Override default database path
  CITATION_CHUNK_SIZE     Override chunk size in characters
  CITATION_MAX_TOKENS     Override max response tokens
  CITATION_TEMPERATURE    Override temperature (0=deterministic, 1=creative)
  CITATION_BROAD_CONTEXT  Override broad context domains (comma-delimited)
  CITATION_EXCLUDES       Space-separated list of exclude patterns
  VECTORDBS               Base directory for knowledgebases

DATABASE SCHEMA:
  The SQLite database contains:
    - sourcefile: Relative path to the source file
    - title: Extracted document title (or empty)
    - author: Extracted author name (or empty)
    - year: Extracted publication year (or empty)
    - raw_citation: Full citation string
    - broad_context: Broad context domains (or empty)

EXAMPLES:
  # Process current directory
  $PRG .

  # Process specific directory
  $PRG /path/to/documents

  # Process multiple directories
  $PRG /docs/papers /docs/articles /docs/books

  # Force update all citations
  $PRG --force /path/to/documents

  # Dry run to preview processing
  $PRG --dry-run /path/to/documents

  # Use different model with higher token limit
  $PRG -m gpt-4 -M 256 /path/to/documents

  # Exclude backup directories
  $PRG -x "*/backup/*" -x "*/archive/*" /path/to/documents

  # High-performance parallel processing
  $PRG -p 20 /var/lib/vectordbs/large-project/

  # Quiet mode with custom database
  $PRG -q -d /var/lib/citations.db /path/to/documents

  # Debug mode for troubleshooting
  $PRG -D -v -v /path/to/documents
  
  # Add broad context domains
  $PRG --context "anthropology,history" /path/to/documents

PERFORMANCE TIPS:
  - Sequential mode: ~60 files/minute
  - 5 workers: ~250-300 files/minute
  - 10 workers: ~450-550 files/minute
  - 20 workers: ~800-1000 files/minute

NOTES:
  - Citations are extracted from the first ${CHUNK_SIZE} characters of each file
  - Files already in the database are skipped unless --force is used
  - The script uses file names as hints to improve citation accuracy
  - "NF" in results means "Not Found" for that field

EOF
}

# Parse command line arguments
while (($#)); do 
  case "$1" in
    -d|--database) 
      shift; DATABASE="${1:-$DATABASE}"
      ;;
    -f|--force|--force-update)
      FORCE=1
      ;;
    -r|--reprocess-blank)
      REPROCESS_BLANK=1
      ;;
    -C|--chunk-size)
      shift; CHUNK_SIZE=${1:-$CHUNK_SIZE}
      ;;
    -x|--exclude)
      shift; (($# == 0)) || EXCLUDES+=(-not -path "$1")
      ;;
    -m|--model)
      shift; MODEL="${1:-$MODEL}"
      ;;
    -M|--max-tokens)
      shift; MAX_TOKENS=${1:-$MAX_TOKENS}
      ;;
    -t|--temperature)
      shift; TEMPERATURE="${1:-$TEMPERATURE}"
      ;;
    -c|--context)
      shift; BROAD_CONTEXT="${1:-$BROAD_CONTEXT}"
      ;;

    -p|--parallel)
      shift; PARALLEL_WORKERS="${1:-$PARALLEL_WORKERS}"
      [[ "$PARALLEL_WORKERS" =~ ^[0-9]+$ ]] || ((PARALLEL_WORKERS < 1)) || \
          die 22 "Invalid parallel workers count: $PARALLEL_WORKERS"
      PARALLEL=1
      ;;
    -s|--no-parallel|--sequential)
      PARALLEL=0
      ;;

    -N|--not-dry-run|--no-dry-run|--notdryrun|--nodryrun) DRY_RUN=0 ;;
    -n|--dry-run|--dryrun) DRY_RUN=1 ;;
    -q|--quiet) VERBOSE=0 ;;
    -v|--verbose) VERBOSE+=1 ;;
    -D|--debug) DEBUG=1; VERBOSE=2 ;;
    -h|--help) show_help; exit 0 ;;
    -[dfrCxMmtpcsNnqvDh]*) #shellcheck disable=SC2046 #split up single options
      set -- '' $(printf -- "-%c " $(grep -o . <<<"${1:1}")) "${@:2}"
      ;;
    -*) die 22 "Invalid option '$1'" ;;
    *) # Validate directory exists
      [[ -d "$1" ]] || die 2 "Directory '$1' does not exist"
      SRC_DIRS+=("$(readlink -en -- "$1")")
      ;;
  esac
  shift
done

# Use default source directory if none specified
if ((${#SRC_DIRS[@]} == 0)); then
  if [[ -n "$DEFAULT_SRC" && -d "$DEFAULT_SRC" ]]; then
    SRC_DIRS+=("$DEFAULT_SRC")
  else
    die 1 "No source directories specified and no default available. Use --help for usage."
  fi
fi

# Validate environment
api_validate_key || die 1 "API key validation failed"

# Initialize database (unless dry run)
[[ -f "${DATABASE}" ]] || {
  [[ -f "${DATABASE}".db ]] || die 2 "'$DATABASE' does not exist"
  DATABASE+=.db
}
DATABASE=$(readlink -en -- "$DATABASE")
if ((DRY_RUN)); then
  vecho "DRY RUN MODE - No changes will be made"
else
  vecho "Initializing database: $DATABASE"
  db_init "$DATABASE" || die 1 "Failed to initialize database"
fi

# Process each source directory
declare -a files=()
for src_dir in "${SRC_DIRS[@]}"; do
  vecho "Processing directory '$src_dir'"
  
  # Find all markdown and text files
  readarray -t files < <(
      find -L "$src_dir" -type f ${EXCLUDES[@]+"${EXCLUDES[@]}"} \( -name '*.md' -o -name '*.txt' \)  \
          | sort)
  if ((${#files[@]} == 0)); then
    vwarn "No files found in '$src_dir'"
    continue
  fi
  
  vecho "Found ${#files[@]} files to process"
  
  # Choose processing mode
  if ((PARALLEL && ! DRY_RUN)); then
    # Parallel processing mode
    vecho "Using parallel processing with $PARALLEL_WORKERS workers"
    
    # Initialize parallel environment
    if ! parallel_init "$DATABASE" "$PARALLEL_WORKERS"; then
      error "Failed to initialize parallel processing environment"
      continue
    fi
    
    # Pre-filter files and add to queue
    declare -i queued=0
    for file in "${files[@]}"; do
      # Check if file exists
      if [[ ! -f "$file" ]]; then
        NOTFOUND+=1
        vwarn "File not found: $file"
        continue
      fi
      
      # Get relative path for database storage
      declare -- src_dir_normalized="${src_dir%/}/"
      file_relative="${file#$src_dir_normalized}"
      
      # Add to work queue
      parallel_queue_add "$file" "$file_relative" "$src_dir"
      queued=$((queued + 1))
    done
    
    vecho "Queued $queued files for processing"
    
    # Debug: check queue file
    if ((DEBUG)); then
      debug "Queue file: $QUEUE_FILE"
      debug "Queue size: $(wc -l < "$QUEUE_FILE") lines"
      debug "First 5 queue entries:"
      head -5 "$QUEUE_FILE" | while read -r line; do
        debug "  $line"
      done
    fi
    
    # Launch workers
    vecho "About to launch workers with:"
    vecho "  DATABASE=$DATABASE"
    vecho "  MODEL=$MODEL"
    vecho "  CHUNK_SIZE=$CHUNK_SIZE"
    vecho "  MAX_TOKENS=$MAX_TOKENS"
    vecho "  TEMPERATURE=$TEMPERATURE"
    vecho "  FORCE=$FORCE"
    vecho "  REPROCESS_BLANK=$REPROCESS_BLANK"
    vecho "  VERBOSE=$VERBOSE"
    vecho "  DEBUG=$DEBUG"
    vecho "  LIB_DIR=$LIB_DIR"
    vecho "  BROAD_CONTEXT=$BROAD_CONTEXT"
    
    if ! parallel_launch_workers "$DATABASE" "$MODEL" "$SYSTEM_PROMPT" \
      "$CHUNK_SIZE" "$MAX_TOKENS" "$TEMPERATURE" "$FORCE" "$REPROCESS_BLANK" "$VERBOSE" "$DEBUG" "$LIB_DIR" "$BROAD_CONTEXT"; then
      error "Failed to launch parallel workers"
      # Continue with error message already printed by parallel_launch_workers
    fi
    
    debug "Workers launched, ACTIVE_WORKERS=$ACTIVE_WORKERS"
    debug "WORKER_PIDS: ${WORKER_PIDS[*]:-}"
    
    # Progress monitoring with adaptive sleep
    declare -i last_total=0
    declare -i total start_time elapsed eta
    declare -i no_progress_count=0
    declare -- sleep_interval="0.1"
    start_time=$(date +%s)
    
    # Ensure WORKER_PIDS array exists
    if [[ -z "${WORKER_PIDS+x}" ]]; then
      error "WORKER_PIDS array not defined"
      continue
    fi
    
    # Initial delay to allow workers to start processing
    sleep 0.5  # Give workers time to start
    
    while true; do
      # Check if any workers are still running
      declare -i still_running=0
      if [[ -n "${WORKER_PIDS[*]:-}" ]] && ((${#WORKER_PIDS[@]} > 0)); then
        for pid in "${WORKER_PIDS[@]}"; do
          # kill -0 checks if process exists without sending a signal
          if kill -0 "$pid" 2>/dev/null; then
            still_running+=1
          fi
        done
      fi
      
      # Exit loop if no workers are running
      if ((still_running == 0)); then
        debug "All workers have finished"
        break
      fi
      
      ACTIVE_WORKERS=$still_running
      
      # Get progress
      declare -- progress_data
      if progress_data=$(parallel_get_progress 2>/dev/null); then
        IFS=':' read -r cur_processed cur_skipped cur_errors cur_notfound <<< "$progress_data"
        total=$((cur_processed + cur_skipped + cur_errors + cur_notfound))
      else
        debug "Failed to read progress file"
        cur_processed=0
        cur_skipped=0
        cur_errors=0
        cur_notfound=0
        total=0
      fi
      
      # Update display if progress changed
      if ((total != last_total)); then
        last_total=$total
        no_progress_count=0
        sleep_interval=0.1  # Reset to fast polling when progress detected
        
        if ((VERBOSE)); then
          elapsed=$(($(date +%s) - start_time))
          if ((elapsed > 0 && total > 0)); then
            # Calculate processing rate in files per minute
            declare -i rate=$((total * 60 / elapsed))  # files per minute
            eta=$(( (queued - total) * elapsed / total ))
            >&2 printf '\r[%d/%d] %d workers | Rate: %d/min | ETA: %ds | P:%d S:%d E:%d N:%d' \
              "$total" "$queued" "$ACTIVE_WORKERS" "$rate" "$eta" \
              "$cur_processed" "$cur_skipped" "$cur_errors" "$cur_notfound"
          else
            >&2 printf '\r[%d/%d] %d workers active' "$total" "$queued" "$ACTIVE_WORKERS"
          fi
        fi
      else
        # No progress - gradually increase sleep interval
        no_progress_count+=1
        # Simple sleep interval increase without bc
        if ((no_progress_count > 10)); then
          # Increase sleep by 0.1 seconds each time, max 1 second
          case "$sleep_interval" in
            0.1) sleep_interval=0.2 ;;
            0.2) sleep_interval=0.3 ;;
            0.3) sleep_interval=0.4 ;;
            0.4) sleep_interval=0.5 ;;
            0.5) sleep_interval=0.6 ;;
            0.6) sleep_interval=0.7 ;;
            0.7) sleep_interval=0.8 ;;
            0.8) sleep_interval=0.9 ;;
            0.9) sleep_interval=1.0 ;;
            *) sleep_interval=1.0 ;;
          esac
          no_progress_count=0
        fi
      fi
      
      # Adaptive sleep to reduce CPU usage
      sleep "$sleep_interval"
    done
    
    # Clear progress line by overwriting with spaces
    ((VERBOSE)) && >&2 printf '\r%*s\r' 100 ''
    
    # Wait for all workers
    parallel_wait_workers
    
    # Process results from workers
    vecho "Processing results from workers..."
    parallel_process_results "$DATABASE" "$VERBOSE" "$DEBUG"
    
    # Update global counters
    IFS=':' read -r PROCESSED SKIPPED ERRORS NOTFOUND < <(parallel_get_progress)
    
  # Sequential processing mode (or dry run) ==========================================
  else
    ((PARALLEL)) && ((DRY_RUN)) && vecho "Parallel processing disabled in dry-run mode"
    
    # Process each file
    declare -i file_num=0
    for file in "${files[@]}"; do
      file_num+=1

      # Check if file exists
      if [[ ! -f "$file" ]]; then
        NOTFOUND+=1
        vwarn "File not found: $file"
        continue
      fi

      # Get relative path for database storage
      # Ensure src_dir has a trailing slash for consistent removal
      # This creates paths like "subdir/file.md" instead of "/full/path/subdir/file.md"
      declare -- src_dir_normalized="${src_dir%/}/"
      file_relative="${file#$src_dir_normalized}"

      # Progress indicator with filename truncation for long paths
      if ((VERBOSE)); then
        >&2 printf '\r[%d/%d] Processing: %s' "$file_num" "${#files[@]}" "$file_relative"
        # Truncate long filenames to fit terminal width
        if ((${#file_relative} > 50)); then
          >&2 printf '\r[%d/%d] Processing: ...%s' "$file_num" "${#files[@]}" "${file_relative: -47}"
        fi
      fi

      # Check if already processed (unless force mode or reprocess-blank mode)
      if ! ((FORCE)); then
        if ((REPROCESS_BLANK)); then
          # In reprocess-blank mode: skip if file exists but is NOT blank
          if ! ((DRY_RUN)) && db_file_exists "$DATABASE" "$file_relative" && ! db_file_exists "$DATABASE" "$file_relative" 1; then
            SKIPPED+=1
            ((VERBOSE>1)) && vwarn "Skipping, has valid citation: $file_relative"
            continue
          fi
        else
          # Normal mode: skip if file exists
          if ! ((DRY_RUN)) && db_file_exists "$DATABASE" "$file_relative"; then
            SKIPPED+=1
            ((VERBOSE>1)) && vwarn "Skipping, already processed: $file_relative"
            continue
          fi
        fi
      fi

      # Prepare file title context string
      # Extract and transform filename to create title hint
      filetitle=$(basename -- "$file")
      filetitle=${filetitle%.*} # remove last extension (md, txt, etc)
      filetitle="${filetitle//-/ }"     # Replace hyphens with spaces
      filetitle="${filetitle//_/ - }"   # Replace underscores with " - "

      # Read first chunk of file
      file_content=$(head -c "$CHUNK_SIZE" "$file" 2>/dev/null || true)

      # Skip empty files
      if [[ -z "$file_content" ]]; then
        SKIPPED+=1
        debug "Skipping empty file: $file_relative"
        continue
      fi

      # Prepare content with file title hint
      user_content="---
file-title: $filetitle
"
      (( ! ${#BROAD_CONTEXT})) || user_content+="broad-context: $BROAD_CONTEXT
"
      user_content+="---

$file_content
"
      # Dry run - just show what would be done
      if ((DRY_RUN)); then
        >&2 echo
        vecho "Would process: $file_relative"
        continue
      fi

      # Call API
      debug "Calling API for: $file_relative"
      if response=$(api_call_openai "$MODEL" "$SYSTEM_PROMPT" "$user_content" "$MAX_TOKENS" "$TEMPERATURE"); then
        # Extract result
        result=$(api_extract_result "$response")

        if [[ -z "$result" || "$result" == NF || "$result" == Error ]]; then
          NOTFOUND+=1
          debug "No citation found for: $file_relative"
          continue
        fi

        # Parse citation into components
        if parsed=$(parse_citation "$result"); then
          IFS='|' read -r title author year <<< "$parsed"

          # Clean up the raw citation - remove any "NF" values
          declare -- clean_result="$result"
          # If all fields are empty, don't store the raw citation
          if [[ -z "$title" && -z "$author" && -z "$year" ]]; then
            clean_result=""
          else
            # Replace ", NF" with empty string in the raw citation
            clean_result="${clean_result//, NF/}"
            # Also handle "NF" at the beginning or middle of citation
            clean_result="${clean_result//NF, /}"
            clean_result="${clean_result/NF/}"
          fi

          # Store in database
          if db_upsert_citation "$DATABASE" "$file_relative" "$title" "$author" "$year" "$clean_result" "$BROAD_CONTEXT"; then
            PROCESSED+=1
            debug "Stored citation: $file_relative -> $result"

            # Show result in verbose mode
            if ((VERBOSE > 1)); then
              >&2 echo
              success "Citation found: $file_relative" \
                "  Title:  $title" \
                "  Author: $author" \
                "  Year:   $year"
            fi
          else
            ERRORS+=1
            error "Failed to store citation for: $file_relative"
          fi
        else
          ERRORS+=1
          error "Failed to parse citation for: $file_relative: $result"
        fi
      else
        ERRORS+=1
        error "API call failed for: $file_relative"
      fi
    done
    
    # Clear progress line by overwriting with spaces
    ((VERBOSE)) && >&2 printf '\r%*s\r' 80 ''
  fi  # End of parallel/sequential choice
done

# Show summary
if ((VERBOSE)); then
  if ((DRY_RUN)); then
    >&2 echo
    vecho '' "Dry run complete. Use without --dry-run to process files."
  else
    declare -i total_in_db
    total_in_db=$(db_get_processed_count "$DATABASE")
    >&2 echo
    success "Citation extraction complete"
    vecho "Total citations in database: $total_in_db"
  fi
fi

#fin

