#!/bin/bash
# append-citations.sh - Add bibliographic citations as YAML frontmatter to documents
#
# This script reads citation data from the SQLite database (populated by
# gen-citations.sh) and applies it as YAML frontmatter to the original
# source documents. It supports dry-run mode, backup creation, and can
# handle existing frontmatter.
#
# Features:
#   - Automatic backup creation before modifications (.bak files)
#   - Preserves existing document content
#   - Handles existing frontmatter (skip or overwrite with --force)
#   - Dry-run mode to preview changes
#   - Removes empty frontmatter when all citation fields are blank
#   - Supports broad context tagging for domain orientation
#
# Usage: append-citations.sh [OPTIONS] [SOURCE_DIRECTORY...]
# See --help for detailed options
#
set -euo pipefail
declare -- PRG0 PRGDIR LIB_DIR PRG
PRG0=$(readlink -en -- "$0")
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
trim() { local v="$*"; v="${v#"${v%%[![:blank:]]*}"}"; echo -n "${v%"${v##*[![:blank:]]}"}" ; }

# Libraries are always in the script directory
[[ -f "$LIB_DIR"/db_functions.sh ]] || die 2 "Error: Cannot find citation library files in '$LIB_DIR'"
# Source helper libraries
source "$LIB_DIR"/db_functions.sh

# Flag to prevent duplicate cleanup
declare -i CLEANUP_DONE=0

# Initialize statistics variables early for cleanup function
declare -i UPDATED=0 SKIPPED=0 ERRORS=0

# Cleanup function
xcleanup() { 
  local -i exitcode=${1:-0}
  # Prevent duplicate execution
  ((CLEANUP_DONE)) && return 0
  CLEANUP_DONE=1
  
  # Display final statistics
  if ((${VERBOSE:-0} && (UPDATED + SKIPPED + ERRORS > 0))); then
    >&2 echo
    success "Processing complete:" \
        "  Updated:   $UPDATED" \
        "  Skipped:   $SKIPPED" \
        "  Errors:    $ERRORS"
  fi
  [[ -t 0 ]] && >&2 printf '\e[?25h'  # Show cursor
  
  exit "$exitcode"
}
trap 'xcleanup $?' SIGINT EXIT

# Configuration
declare -- DATABASE="${CITATION_DATABASE:-"$PWD"/citations.db}"
declare -a SRC_DIRS=()

# Broad context for domain orientation (comma-delimited)
declare -- BROAD_CONTEXT="${BROAD_CONTEXT:-}"

# Default source directory if none specified
declare -- DEFAULT_SRC=""
if [[ -n "${VECTORDBS:-}" ]]; then
  DEFAULT_SRC="$VECTORDBS/appliedanthropology/embed_data.text/"
fi

# Flags
declare -i VERBOSE=1 DEBUG=0 DRY_RUN=0 BACKUP=1 FORCE=0
declare -- BROAD_CONTEXT_CLI=""
declare -i HAS_SUDO=1
[[ $(id -nG) =~ sudo ]] || HAS_SUDO=0

# Show help
show_help() {
  cat <<EOF
Usage: $PRG [OPTIONS] [SOURCE_DIRECTORY...]

Append citations from database to source documents as YAML frontmatter.

SOURCE DIRECTORIES:
  One or more directories containing files to update with citations.
  If none specified, uses DEFAULT_SRC if configured.

OPTIONS:
  -d, --database PATH     Database file path (default: $DATABASE)
  -c, --context DOMAINS   Broad context domains (comma-delimited)
  -n, --dry-run           Show what would be done without making changes
  -b, --no-backup         Don't create backup files (*.bak)
  -f, --force             Overwrite existing frontmatter
  -q, --quiet             Suppress verbose output
  -v, --verbose           Increase verbosity
  -D, --debug             Enable debug output
  -h, --help              Show this help message

ENVIRONMENT VARIABLES:
  CITATION_DATABASE       Override default database path
  VECTORDBS              Base directory for knowledge bases
  BROAD_CONTEXT          Default broad context domains (comma-delimited)

EXAMPLES:
  # Apply citations to default directory
  $PRG

  # Apply citations to specific directory
  $PRG /path/to/documents

  # Dry run to see what would be changed
  $PRG --dry-run /path/to/documents

  # Force overwrite existing frontmatter
  $PRG --force /path/to/documents

  # Add broad context domains
  $PRG --context "anthropology, history" /path/to/documents

NOTES:
  - Creates .bak backup files before modification (unless --no-backup)
  - Preserves existing content after frontmatter
  - Skips files with existing frontmatter unless --force is used
  - Only adds frontmatter for files with valid citation data

EOF
}

# Check if file has YAML frontmatter
has_frontmatter() {
  local file="$1"
  local first_line
  
  first_line=$(head -n1 "$file" 2>/dev/null || true)
  [[ "$first_line" == "---" ]]
}

# Extract existing frontmatter
extract_frontmatter() {
  local file="$1"
  local -i in_frontmatter=0
  local -i line_num=0
  
  while IFS= read -r line; do
    line_num+=1
    
    if ((line_num == 1)) && [[ "$line" == "---" ]]; then
      in_frontmatter=1
      echo "$line"
    elif ((in_frontmatter)) && [[ "$line" == "---" ]]; then
      echo "$line"
      break
    elif ((in_frontmatter)); then
      echo "$line"
    else
      break
    fi
  done < "$file"
}

# Get content after frontmatter
get_content_after_frontmatter() {
  local file="$1"
  local -i in_frontmatter=0
  local -i found_end=0
  local -i line_num=0
  
  while IFS= read -r line; do
    line_num+=1
    
    if ((line_num == 1)) && [[ "$line" == "---" ]]; then
      in_frontmatter=1
    elif ((in_frontmatter)) && [[ "$line" == "---" ]]; then
      found_end=1
      in_frontmatter=0
    elif ((found_end)) || ((! in_frontmatter)); then
      echo "$line"
    fi
  done < "$file"
}

# Create YAML frontmatter from citation
create_frontmatter() {
  local title="$1"
  local author="$2"
  local year="$3"
  local db_broad_context="$4"
  
  # Use CLI context if provided, otherwise use database context
  local effective_context="${BROAD_CONTEXT:-$db_broad_context}"
  
  # Only include non-empty fields
  echo "---"
  [[ -n "$title" ]] && echo "title: \"$title\""
  [[ -n "$author" ]] && echo "author: \"$author\""
  [[ -n "$year" ]] && echo "date: \"$year\""
  [[ -n "$effective_context" ]] && echo "broad_context: \"$effective_context\""
  echo "---"
}

# Parse command line arguments
while (($#)); do 
  case "$1" in
    -d|--database) 
      shift; DATABASE="${1:-$DATABASE}"
      ;;
    -c|--context)
      shift; BROAD_CONTEXT_CLI="${1:-}"
      ;;
    -n|--dry-run) DRY_RUN=1 ;;
    -b|--no-backup) BACKUP=0 ;;
    -f|--force) FORCE=1 ;;
    -q|--quiet) VERBOSE=0 ;;
    -v|--verbose) VERBOSE+=1 ;;
    -D|--debug) DEBUG=1; VERBOSE=2 ;;
    -h|--help) show_help; exit 0 ;;
    -[dcnbfqvDh]*) #shellcheck disable=SC2046 #split up single options
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

# Validate database exists
[[ -f "$DATABASE" ]] || die 2 "Database not found: $DATABASE. Run gen-citations.sh first."

# Use CLI context if provided, otherwise use environment variable
if [[ -n "$BROAD_CONTEXT_CLI" ]]; then
  BROAD_CONTEXT="$BROAD_CONTEXT_CLI"
fi

# Process each source directory
for src_dir in "${SRC_DIRS[@]}"; do
  vecho "Processing directory: $src_dir"
  ((HAS_SUDO==0)) || sudo chown $USER:$USER "${src_dir:=jUnK}"/ -R

  # Get all citations from database
  declare -a citations=()
  readarray -t citations < <(db_get_all_citations "$DATABASE")
  if ((${#citations[@]} == 0)); then
    vwarn "No citations found in database"
    continue
  fi
  vecho "Found ${#citations[@]} citations in database"
  
  # Process each citation
  declare -i citation_num=0
  for citation_line in "${citations[@]}"; do
    citation_num+=1
    
    # Parse citation data
    IFS='|' read -r sourcefile title author year _ db_broad_context <<< "$citation_line"
    
    # Build full file path
    # Ensure src_dir has a trailing slash for consistent concatenation
    declare -- src_dir_normalized="${src_dir%/}/"
    file_path="${src_dir_normalized}${sourcefile}"
    
    # Progress indicator with filename truncation for long paths
    if ((VERBOSE)); then
      >&2 printf '\r[%d/%d] Processing: %s' "$citation_num" "${#citations[@]}" "$sourcefile"
      # Truncate long filenames to fit terminal width
      if ((${#sourcefile} > 50)); then
        >&2 printf '\r[%d/%d] Processing: ...%s' "$citation_num" "${#citations[@]}" "${sourcefile: -47}"
      fi
    fi
    
    # Check if file exists
    if [[ ! -f "$file_path" ]]; then
      debug "File not found: $file_path"
      continue
    fi
    
    # Check if file already has frontmatter
    if has_frontmatter "$file_path" && ((! FORCE)); then
      SKIPPED+=1
      debug "Skipping file with existing frontmatter: $sourcefile"
      continue
    fi
    
    # Handle case where all citation fields are empty
    if [[ -z "$title" && -z "$author" && -z "$year" ]]; then
      # If file has existing frontmatter and force mode is on, remove it
      if has_frontmatter "$file_path" && ((FORCE)); then
        if ((DRY_RUN)); then
          echo
          echo "Would remove frontmatter from: $sourcefile (all fields empty)"
          continue
        fi
        
        # Create backup if requested
        if ((BACKUP)); then
          cp "$file_path" "${file_path}.bak" || {
            ERRORS+=1
            error "Failed to create backup for: $sourcefile"
            continue
          }
        fi
        
        # Get content after frontmatter and write it back
        content=$(get_content_after_frontmatter "$file_path")
        if echo "$content" > "$file_path"; then
          UPDATED+=1
          debug "Removed frontmatter from: $sourcefile"
          if ((VERBOSE > 1)); then
            echo
            success "Removed frontmatter from: $sourcefile (all citation fields empty)"
          fi
        else
          ERRORS+=1
          error "Failed to remove frontmatter from: $sourcefile"
          # Restore from backup if update failed
          if ((BACKUP)) && [[ -f "${file_path}.bak" ]]; then
            mv "${file_path}.bak" "$file_path"
          fi
        fi
      else
        SKIPPED+=1
        debug "Skipping file with no citation data: $sourcefile"
      fi
      continue
    fi
    
    # Dry run - just show what would be done
    if ((DRY_RUN)); then
      echo
      echo "Would update: $sourcefile"
      [[ -n "$title" ]] && echo "  Title:      $title"
      [[ -n "$author" ]] && echo "  Author:     $author"
      [[ -n "$year" ]] && echo "  Year:       $year"
      continue
    fi
    
    # Create backup if requested
    if ((BACKUP)); then
      cp "$file_path" "${file_path}.bak" || {
        ERRORS+=1
        error "Failed to create backup for: $sourcefile"
        continue
      }
    fi
    
    # Create new frontmatter
    new_frontmatter=$(create_frontmatter "$title" "$author" "$year" "$db_broad_context")
    
    # Get content after existing frontmatter (if any)
    if has_frontmatter "$file_path"; then
      content=$(get_content_after_frontmatter "$file_path")
    else
      content=$(cat "$file_path")
    fi
    
    # Write updated file
    if {
      echo "$new_frontmatter"
      echo "$content"
    } > "$file_path.tmp" && mv "$file_path.tmp" "$file_path"; then
      UPDATED+=1
      debug "Updated: $sourcefile"
      
      # Show result in verbose mode
      if ((VERBOSE > 1)); then
        echo
        success "Updated: $sourcefile"
        [[ -n "$title" ]] && echo "  Title:      $title"
        [[ -n "$author" ]] && echo "  Author:     $author"
        [[ -n "$year" ]] && echo "  Year:       $year"
      fi
    else
      ERRORS+=1
      error "Failed to update: $sourcefile"
      
      # Restore from backup if update failed
      if ((BACKUP)) && [[ -f "${file_path}.bak" ]]; then
        mv "${file_path}.bak" "$file_path"
      fi
    fi
  done
  
  # Clear progress line by overwriting with spaces
  ((VERBOSE)) && >&2 printf '\r%*s\r' 80 ''
done

# Show summary
if ((VERBOSE)); then
  if ((DRY_RUN)); then
    >&2 echo
    vecho '' "Dry run complete. Use without --dry-run to apply changes."
  else
    >&2 echo
    success "Citation insertion complete"
  fi
fi

#fin
