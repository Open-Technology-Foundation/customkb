#!/bin/bash
#
# Test Runner Script for Okusi Associates Email Auto-Reply System
# ================================================================
#
# This script activates the virtual environment and runs the complete
# test suite with appropriate coverage reporting and output formatting.
#
# Usage:
#   ./run_tests.sh [pytest options]
#   ./run_tests.sh --verbose
#   ./run_tests.sh tests/unit/
#   ./run_tests.sh -k test_config
#   ./run_tests.sh --cov-report=html
#

set -euo pipefail

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Email Processor Test Suite Runner${NC}"
echo -e "${BLUE}========================================${NC}"
echo

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
  echo -e "${RED}Error: Virtual environment not found!${NC}"
  echo "Please run: python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements-dev.txt"
  exit 1
fi

# Activate virtual environment
echo -e "${YELLOW}Activating virtual environment...${NC}"
source .venv/bin/activate

# Check if pytest is installed
if ! python -c "import pytest" 2>/dev/null; then
  echo -e "${RED}Error: pytest not installed!${NC}"
  echo "Please run: pip install -r requirements-dev.txt"
  exit 1
fi

# Check if we're running as vmail user (warn but don't fail)
if [ "$(whoami)" != "vmail" ]; then
  echo -e "${YELLOW}Warning: Not running as vmail user. Some file permission tests may fail.${NC}"
  echo -e "${YELLOW}For full test compatibility, run: sudo -u vmail ./run_tests.sh${NC}"
  echo
fi

# Default pytest arguments
PYTEST_ARGS=(
  "--verbose"
  "--tb=short"
  "--cov=."
  "--cov-report=term-missing"
  "--cov-report=html:htmlcov"
  "--cov-fail-under=75"
  "-x"  # Stop on first failure for faster debugging
)

# Add any command line arguments
PYTEST_ARGS+=("$@")

echo -e "${YELLOW}Running pytest with coverage...${NC}"
echo "Arguments: ${PYTEST_ARGS[*]}"
echo

# Run tests
if python -m pytest "${PYTEST_ARGS[@]}" tests/; then
  echo
  echo -e "${GREEN}========================================${NC}"
  echo -e "${GREEN}  All tests passed! ✅${NC}"
  echo -e "${GREEN}========================================${NC}"
  echo
  
  # Show coverage summary
  if [ -f "htmlcov/index.html" ]; then
    echo -e "${BLUE}Coverage report generated: htmlcov/index.html${NC}"
    
    # Try to extract coverage percentage
    if command -v grep >/dev/null 2>&1; then
      coverage_line=$(grep -o "pc_cov\">[0-9]*%" htmlcov/index.html 2>/dev/null | head -1 || true)
      if [ -n "$coverage_line" ]; then
        coverage_pct=$(echo "$coverage_line" | grep -o "[0-9]*%")
        echo -e "${BLUE}Overall coverage: $coverage_pct${NC}"
      fi
    fi
  fi
  
  echo
  echo -e "${GREEN}Test suite completed successfully!${NC}"
  exit 0
else
  echo
  echo -e "${RED}========================================${NC}"
  echo -e "${RED}  Some tests failed! ❌${NC}"
  echo -e "${RED}========================================${NC}"
  echo
  echo -e "${YELLOW}Tips for debugging:${NC}"
  echo "  - Run specific test: ./run_tests.sh tests/unit/test_config_loader.py::TestEmailConfigInitialization::test_init_with_default_config_file"
  echo "  - Run with more verbose output: ./run_tests.sh -vv"
  echo "  - Run without stopping on first failure: ./run_tests.sh --continue-on-collection-errors"
  echo "  - See coverage report: open htmlcov/index.html"
  echo
  exit 1
fi