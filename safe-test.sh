#!/usr/bin/env bash
#
# Safe Test Wrapper for CustomKB - DEPRECATED
#
# ⚠️  WARNING: This script is DEPRECATED and can cause system hangs!
# ⚠️  Use ./run_tests.py --safe instead
#
# REASON: ulimit -v memory restrictions cause pytest collection to hang,
#         which can lead to system crashes. See CRASH_INCIDENT_4.md for details.
#
# RECOMMENDED ALTERNATIVE:
#   ./run_tests.py --safe [options]
#
# Or use pytest directly with timeout:
#   source .venv/bin/activate && timeout 180 pytest tests/unit/ -v
#
# This script is kept for reference only.
#

echo "⚠️  WARNING: safe-test.sh is deprecated!"
echo "⚠️  This script can cause system hangs due to ulimit -v restrictions."
echo ""
echo "Please use instead:"
echo "  ./run_tests.py --safe [options]"
echo ""
echo "Or press Ctrl+C to cancel, Enter to continue anyway (not recommended)..."
read -r

set -euo pipefail

# Configuration
DEFAULT_MEMORY_LIMIT_MB=2048
DEFAULT_TIMEOUT_SECONDS=300  # 5 minutes total
DEFAULT_TEST_TIMEOUT=120     # 2 minutes per test

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored messages
print_info() {
  echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
  echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
  echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
  echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're in the right directory
if [[ ! -f "customkb.py" ]]; then
  print_error "Must be run from CustomKB root directory"
  exit 1
fi

# Check if virtual environment is activated
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  print_warning "Virtual environment not activated"
  print_info "Activating .venv automatically..."
  source .venv/bin/activate || {
    print_error "Failed to activate virtual environment"
    exit 1
  }
fi

# Show system resources
print_info "System Resources:"
echo "---"
free -h | grep -E "Mem:|Swap:"
echo "---"

# Check available memory
AVAILABLE_MEM_MB=$(free -m | awk '/^Mem:/{print $7}')
if [[ $AVAILABLE_MEM_MB -lt 4096 ]]; then
  print_warning "Available memory: ${AVAILABLE_MEM_MB}MB (recommend 4GB+ for safe testing)"
fi

# Check if ZFS ARC is consuming too much memory
if [[ -f /proc/spl/kstat/zfs/arcstats ]]; then
  ARC_SIZE=$(awk '/^size/{print $3}' /proc/spl/kstat/zfs/arcstats)
  ARC_SIZE_MB=$((ARC_SIZE / 1024 / 1024))
  if [[ $ARC_SIZE_MB -gt 8192 ]]; then
    print_warning "ZFS ARC using ${ARC_SIZE_MB}MB (consider limiting if tests hang)"
  fi
fi

# Build pytest command with safety features
PYTEST_CMD=(
  "timeout" "${DEFAULT_TIMEOUT_SECONDS}"
  "python" "-m" "pytest"
  "--timeout=${DEFAULT_TEST_TIMEOUT}"
  "--timeout-method=thread"
)

# Add user arguments
PYTEST_CMD+=("$@")

# Print command
print_info "Running tests with safety limits:"
echo "  - Total timeout: ${DEFAULT_TIMEOUT_SECONDS}s"
echo "  - Per-test timeout: ${DEFAULT_TEST_TIMEOUT}s"
echo "  - Memory limit: ${DEFAULT_MEMORY_LIMIT_MB}MB (via ulimit)"
echo ""
print_info "Command: ${PYTEST_CMD[*]}"
echo ""

# Set memory limit using ulimit (soft limit)
ulimit -v $((DEFAULT_MEMORY_LIMIT_MB * 1024)) 2>/dev/null || {
  print_warning "Could not set memory limit (may require different method)"
}

# Record start time and memory
START_TIME=$(date +%s)
START_MEM=$(free -m | awk '/^Mem:/{print $3}')

# Run tests
"${PYTEST_CMD[@]}" || TEST_EXIT_CODE=$?

# Record end time and memory
END_TIME=$(date +%s)
END_MEM=$(free -m | awk '/^Mem:/{print $3}')
DURATION=$((END_TIME - START_TIME))
MEM_DELTA=$((END_MEM - START_MEM))

echo ""
echo "======================================================================"
if [[ ${TEST_EXIT_CODE:-0} -eq 0 ]]; then
  print_success "Tests completed successfully"
elif [[ ${TEST_EXIT_CODE:-0} -eq 124 ]]; then
  print_error "Tests timed out after ${DEFAULT_TIMEOUT_SECONDS}s"
else
  print_warning "Tests completed with exit code: ${TEST_EXIT_CODE:-0}"
fi

print_info "Duration: ${DURATION}s"
print_info "Memory delta: ${MEM_DELTA}MB"
echo "======================================================================"

# Exit with pytest's exit code
exit ${TEST_EXIT_CODE:-0}

#fin
